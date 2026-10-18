# SPDX-License-Identifier: Apache-2.0
"""Isolated S512 MoE grouping; does not modify production constructors/views."""

import math

import torch

import ttnn
from models.autoports.qwen_qwen3_8_flash_next.tests import benchmark_prefill_moe_group_candidate as group256_candidate
from models.autoports.qwen_qwen3_8_flash_next.tt.multichip_decoder import (
    _EXPERT_COMPACT_PACKED,
    _EXPERT_DISPATCH_PACKED,
    _expert_kernel,
)

DISPATCH512 = group256_candidate.replace_once(_EXPERT_DISPATCH_PACKED, "e*128+count", "e*512+count")
COMPACT512 = group256_candidate.replace_once(_EXPERT_COMPACT_PACKED, "ASSERT(packed<128)", "ASSERT(packed<512)")
COMPACT512 = group256_candidate.replace_once(COMPACT512, "(e%128)*4+packed/32", "(e%128)*16+packed/32")


def make_candidate(owner):
    result = group256_candidate.make_candidate(owner)
    result.offsets512 = owner.ops.upload(torch.arange(128, dtype=torch.int32) * 512, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
    return result


def group512(self, x):
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
        (1, 1, 128 * 512, 2560), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.ops.mesh
    )
    counts = ttnn.to_memory_config(counts, ttnn.DRAM_MEMORY_CONFIG)
    source = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
    _expert_kernel(
        self.ops.mesh,
        [source, ids, scores, self.rank, counts, mapping, capacity],
        DISPATCH512,
        [n],
        workers=32,
        cb_size=5120 + n * 192 + 64,
    )
    down = ttnn.experimental.deepseek_prefill.unified_routed_expert_moe(
        capacity,
        self.offsets512,
        counts,
        self.ids,
        self.gates,
        self.ups,
        self.downs,
        512,
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
        COMPACT512,
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


def forward256(model, value):
    sequence = value.shape[2]
    if sequence <= 256:
        return group256_candidate.forward(model, value)
    if tuple(value.shape)[1:3] != (1, 512):
        raise ValueError("Expected B1 S512")
    chunks = [group256_candidate.group256(model, value[:, :, start : start + 256]) for start in (0, 256)]
    return model.ops.scatter_reduce(ttnn.concat(chunks, dim=2))


def forward512(model, value):
    if value.shape[2] <= 256:
        return group256_candidate.forward(model, value)
    if tuple(value.shape)[1:3] != (1, 512):
        raise ValueError("Expected B1 S512")
    return model.ops.scatter_reduce(group512(model, value))
