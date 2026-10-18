# SPDX-License-Identifier: Apache-2.0
"""Isolated S256 MoE grouping candidate; production and B1 serving unchanged."""

import copy
import math

import torch

import ttnn
from models.autoports.qwen_qwen3_8_flash_next.tt.multichip_decoder import (
    _EXPERT_COMPACT_PACKED,
    _EXPERT_DISPATCH_PACKED,
    ParallelMoE,
    _expert_kernel,
)


def replace_once(source, old, new):
    if source.count(old) != 1:
        raise ValueError(f"Expected exactly one qualified kernel fragment: {old}")
    return source.replace(old, new)


DISPATCH256 = replace_once(_EXPERT_DISPATCH_PACKED, "e*128+count", "e*256+count")
COMPACT256 = replace_once(_EXPERT_COMPACT_PACKED, "ASSERT(packed<128)", "ASSERT(packed<256)")
COMPACT256 = replace_once(COMPACT256, "(e%128)*4+packed/32", "(e%128)*8+packed/32")


def make_candidate(owner):
    result = copy.copy(owner)
    result.offsets256 = owner.ops.upload(torch.arange(128, dtype=torch.int32) * 256, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
    return result


def group256(self, x):
    n = x.shape[2]
    padded = math.ceil(n / 32) * 32
    router = self.ops.linear(x, self.router, compute_kernel_config=self.router_compute)
    logits = router[..., :512]
    values, indices = ttnn.topk(
        ttnn.pad(logits, [(0, 0), (0, 0), (0, 0), (0, 512)], float("-inf")), k=32, dim=-1, sorted=True
    )
    selected = ttnn.softmax(
        ttnn.add(values, self.mask), dim=-1, numeric_stable=True, compute_kernel_config=self.ops.compute
    )
    ids = ttnn.to_layout(ttnn.typecast(indices, ttnn.uint32), ttnn.ROW_MAJOR_LAYOUT)
    scores = ttnn.to_layout(selected, ttnn.ROW_MAJOR_LAYOUT)
    counts = ttnn.empty((128,), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.ops.mesh)
    mapping = ttnn.empty((1, 1, n, 32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.ops.mesh)
    capacity = ttnn.empty(
        (1, 1, 128 * 256, 2560), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.ops.mesh
    )
    counts = ttnn.to_memory_config(counts, ttnn.DRAM_MEMORY_CONFIG)
    source = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
    _expert_kernel(
        self.ops.mesh,
        [source, ids, scores, self.rank, counts, mapping, capacity],
        DISPATCH256,
        [n],
        workers=32,
        cb_size=5120 + n * 192 + 64,
    )
    down = ttnn.experimental.deepseek_prefill.unified_routed_expert_moe(
        capacity,
        self.offsets256,
        counts,
        self.ids,
        self.gates,
        self.ups,
        self.downs,
        256,
        compute_kernel_config=self.routed_compute,
    )
    compact = ttnn.empty(
        (1, 10, padded, 2560),
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=self.ops.mesh,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    _expert_kernel(
        self.ops.mesh,
        [down, ids, scores, self.rank, mapping, compact],
        COMPACT256,
        [n],
        workers=80,
        cb_size=2176 + n * 320 + 64,
    )
    scores = (
        ttnn.pad(selected[..., :10], [(0, 0), (0, 0), (0, padded - n), (0, 0)], 0) if n < padded else selected[..., :10]
    )
    weighted = ttnn.multiply(compact, ttnn.permute(scores, (0, 3, 2, 1)), output_tensor=compact)
    result = ttnn.sum(weighted, dim=1, keepdim=True)[:, :, :n, :]
    shared = self.ops.linear(x, self.shared_gu, compute_kernel_config=self.shared_compute)
    shared = self.ops.linear(
        ttnn.multiply(shared[..., :160], shared[..., 160:], input_tensor_a_activations=[ttnn.UnaryOpType.SILU]),
        self.shared_down,
        compute_kernel_config=self.shared_compute,
    )
    return ttnn.add(
        result, ttnn.multiply(shared, router[..., 512:513], input_tensor_b_activations=[ttnn.UnaryOpType.SIGMOID])
    )


def forward(model, value):
    batch, sequence = tuple(value.shape)[1:3]
    if batch != 1 or sequence <= 128:
        return ParallelMoE.__call__(model, value)
    if sequence != 256:
        raise ValueError("This discriminating candidate supports only B1 S256")
    return model.ops.scatter_reduce(group256(model, value))
