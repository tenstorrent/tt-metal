# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Experimental EP4 combine reduction for an explicitly selected probe path.

This changes native sequential BF16 L1 accumulation to FP32 matmul accumulation
with a single BF16 output boundary. It needs component and whole-decoder PCC
validation; it is not an assertion of numerical or performance equivalence.
"""

import math

import ttnn


def make_matmul_reducer(mesh):
    """Allocate rank bounds during setup and return a MeshMoe method candidate.

    The returned callable has signature ``(self, combined, weights, ids_rm, n)``.
    Inputs may contain either n logical tokens or the native reducer's 32-token
    zero padding. No device readback, host routing, or setup allocation occurs
    inside the returned callable.
    """
    import torch

    def rank_bound(offset):
        parts = [torch.full((1, 1, 1, 1), rank * 128 + offset, dtype=torch.float32) for rank in range(4)]
        return ttnn.from_torch(
            torch.cat(parts, dim=0).contiguous(),
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    rank_lo, rank_hi = rank_bound(0), rank_bound(128)

    def reduce(self, combined, weights, ids_rm, n):
        if not 1 <= n <= 32:
            raise ValueError("The reduction candidate requires 1..32 logical tokens")
        if tuple(combined.shape)[-2:] != (10, 2560):
            raise ValueError("Expected combine output [..., tokens, 10, 2560]")
        tokens = math.prod(tuple(combined.shape)[:-2])
        if tokens not in (n, 32):
            raise ValueError("Combine output must contain logical tokens or native 32-token padding")
        if math.prod(weights.shape) != tokens * 10 or math.prod(ids_rm.shape) != tokens * 10:
            raise ValueError("Routing weights and IDs must match the combine token and expert dimensions")

        # Slice in ROW_MAJOR layout before tilization. The native 32-row padding
        # is an internal reduction requirement, not extra requests or state.
        values = ttnn.reshape(combined, (tokens * 10, 2560))
        gate_weights = ttnn.reshape(weights, (tokens, 10))
        expert_ids = ttnn.reshape(ids_rm, (tokens, 10))
        if tokens != n:
            values = values[: n * 10, :]
            gate_weights = gate_weights[:n, :]
            expert_ids = expert_ids[:n, :]

        expert_ids = ttnn.to_layout(
            ttnn.typecast(ttnn.reshape(expert_ids, (1, 1, n, 10)), ttnn.float32), ttnn.TILE_LAYOUT
        )
        owned = ttnn.logical_and(ttnn.ge(expert_ids, rank_lo), ttnn.lt(expert_ids, rank_hi))
        owned = ttnn.reshape(ttnn.typecast(owned, ttnn.bfloat16), (1, n, 10, 1))
        values = ttnn.to_layout(ttnn.reshape(values, (1, n, 10, 2560)), ttnn.TILE_LAYOUT)
        # Nonowned combine slots can contain stale NaNs. A zero weight alone
        # cannot mask those: select literal zero before any multiplication.
        values = ttnn.where(owned, values, 0.0)
        gate_weights = ttnn.to_layout(ttnn.reshape(gate_weights, (1, n, 1, 10)), ttnn.TILE_LAYOUT)

        # BF16 operands, HiFi4/FP32 destination through the existing self.compute,
        # and BF16 output. The padded K=32 lanes originate from zero-filled
        # tilization of logical K=10; they must never become selected experts.
        result = ttnn.matmul(
            gate_weights,
            values,
            dtype=ttnn.bfloat16,
            compute_kernel_config=self.compute,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        return ttnn.reshape(result, (1, 1, n, 2560))

    return reduce
