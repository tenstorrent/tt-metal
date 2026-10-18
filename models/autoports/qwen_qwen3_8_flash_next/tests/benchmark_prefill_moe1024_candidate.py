# SPDX-License-Identifier: Apache-2.0
"""Isolated MoE capacity1024; production/native B1 remain unchanged."""

import math

import torch

import ttnn
from models.autoports.qwen_qwen3_8_flash_next.tests import benchmark_prefill_moe512_candidate as base
from models.autoports.qwen_qwen3_8_flash_next.tt.multichip_decoder import (
    _EXPERT_COMPACT_PACKED,
    _EXPERT_DISPATCH_PACKED,
    _expert_kernel,
)

DISPATCH1024 = base.group256_candidate.replace_once(_EXPERT_DISPATCH_PACKED, "e*128+count", "e*1024+count")
COMPACT1024 = base.group256_candidate.replace_once(_EXPERT_COMPACT_PACKED, "ASSERT(packed<128)", "ASSERT(packed<1024)")
COMPACT1024 = base.group256_candidate.replace_once(COMPACT1024, "(e%128)*4+packed/32", "(e%128)*32+packed/32")


def make_candidate(owner):
    result = base.make_candidate(owner)
    result.offsets1024 = owner.ops.upload(
        torch.arange(128, dtype=torch.int32) * 1024, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT
    )
    return result


def group1024(self, x):
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
        (1, 1, 128 * 1024, 2560), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.ops.mesh
    )
    counts = ttnn.to_memory_config(counts, ttnn.DRAM_MEMORY_CONFIG)
    source = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
    _expert_kernel(
        self.ops.mesh,
        [source, ids, scores, self.rank, counts, mapping, capacity],
        DISPATCH1024,
        [n],
        workers=32,
        cb_size=5120 + n * 192 + 64,
    )
    down = ttnn.experimental.deepseek_prefill.unified_routed_expert_moe(
        capacity,
        self.offsets1024,
        counts,
        self.ids,
        self.gates,
        self.ups,
        self.downs,
        1024,
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
        COMPACT1024,
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


def forward512(model, value):
    if tuple(value.shape)[1:3] not in ((1, 512), (1, 1024)):
        raise ValueError("Expected B1 S512 or S1024")
    chunks = [base.group512(model, value[:, :, start : start + 512]) for start in range(0, value.shape[2], 512)]
    return model.ops.scatter_reduce(chunks[0] if len(chunks) == 1 else ttnn.concat(chunks, dim=2))


def forward1024(model, value):
    if value.shape[2] == 512:
        return forward512(model, value)
    if tuple(value.shape)[1:3] != (1, 1024):
        raise ValueError("Expected B1 S1024")
    return model.ops.scatter_reduce(group1024(model, value))
