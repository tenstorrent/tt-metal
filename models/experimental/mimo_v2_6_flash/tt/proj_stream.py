# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Decode projections of one MiMo layer on a TP=8 row: qkv and o (BFP8, HiFi2) and the router (bf16, HiFi4, fp32
out), for one tile row of decode users.

This is the implementation the model's decode path uses for these matmuls (attention.py: 1D in0-multicast linears
from common.decode_linear_config; moe.py: DramShardedMatmul). It is kept behind a small API so faster programs
(e.g. bank-local streaming generic_op kernels in tt/kernels/proj/) can replace it in one place.
"""

import ttnn

from .common import DramShardedMatmul, compute_config, decode_linear_config


class DecodeProjections:
    """``qkv(x)``, ``o(x)`` and ``router(x)`` for x [1, 1, 32, K] bf16 (users in the leading rows)."""

    def __init__(self, plan, hidden, qkv_width, o_in, num_experts):
        self.plan = plan
        self.hidden, self.qkv_width, self.o_in, self.num_experts = hidden, qkv_width, o_in, num_experts
        self.mm_config = compute_config(ttnn.MathFidelity.HiFi2)
        self.router_config = compute_config(ttnn.MathFidelity.HiFi4, fp32_acc=True)
        self.qkv_program = decode_linear_config(plan.mesh_device, hidden, qkv_width)
        self.o_program = decode_linear_config(plan.mesh_device, o_in, hidden)
        self.router_mm = DramShardedMatmul(
            plan, hidden, num_experts, out_dtype=ttnn.float32, compute_kernel_config=self.router_config
        )

    def prepare_weights(self, w_qkv, w_o, w_router):
        """Device weights as the model keeps them (DRAM interleaved BFP8 qkv [K, qkv_width] and o [o_in, hidden]
        per chip; the router [hidden, E] bf16 in the DRAM-sharded layout). Returns the tensors this class reads."""
        return w_qkv, w_o, w_router

    def qkv(self, x, w):
        return ttnn.linear(
            x,
            w,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            compute_kernel_config=self.mm_config,
            program_config=self.qkv_program,
        )

    def o(self, x, w):
        return ttnn.linear(
            x,
            w,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            compute_kernel_config=self.mm_config,
            program_config=self.o_program,
        )

    def router(self, x, w):
        return self.router_mm(x, w)
