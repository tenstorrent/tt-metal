# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""MiMo MoE decode routing after the sigmoid: top-k, normalized weights and token mask in one small program.

``ttnn.topk`` + ``ttnn.gather`` + the normalize / reshape / typecast glue cost ~120 us per decode layer on a
Galaxy row (top-k 44 us and gather 33 us run on one core each). Here one core per decode user selects its
top-k experts from the biased fp32 scores by exact integer compares of the fp32 bit patterns (the biased
scores are positive), gathers the unbiased scores, normalizes them in IEEE fp32 (eps 1e-20, as the
reference) and applies the token mask. It writes two persistent row-major sticks in L1: expert ids
(uint16, pair ``u * k + i`` = user u's i-th expert, descending score) and weights (fp32), plus the
per-pair row weights tile tensor the sparse_matmul expert path multiplies its SwiGLU output with.
"""

import torch

import ttnn

KERNEL = "models/experimental/mimo_v2_6_flash/tt/kernels/moe_route.cpp"
SCRATCH_CB = 0


class MoERoute:
    def __init__(self, plan, num_experts, k, users):
        assert num_experts % 32 == 0 and k <= 16 and users <= 16
        self.plan, self.num_experts, self.k, self.users = plan, num_experts, k, users
        device = plan.mesh_device
        grid = device.compute_with_storage_grid_size()
        cores = [ttnn.CoreCoord(u % grid.x, u // grid.x) for u in range(users)]
        self.cores = cores
        self.core_ranges = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in cores])
        pairs = users * k

        def stick(dtype, torch_dtype):
            return ttnn.from_torch(
                torch.zeros(1, 1, 1, pairs, dtype=torch_dtype),
                device=device,
                dtype=dtype,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                memory_config=ttnn.L1_MEMORY_CONFIG,
                mesh_mapper=plan.replicate(),
            )

        self.ids = stick(ttnn.uint16, torch.int32)
        self.weights = stick(ttnn.float32, torch.float32)
        # Per-pair row weights for the expert path: pair p's tile holds its weight at (its user, 0), zeros elsewhere.
        self.row_weight = ttnn.from_torch(
            torch.zeros(1, pairs, 32, 1),
            device=device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            mesh_mapper=plan.replicate(),
        )
        self.scratch_bytes = num_experts * 8 + 512 + 32 + 64 + k * 16

    def _program(self, scores, biased, mask):
        rt = ttnn.RuntimeArgs()
        for u, core in enumerate(self.cores):
            rt[core.x][core.y] = [
                scores.buffer_address(),
                biased.buffer_address(),
                mask.buffer_address(),
                self.ids.buffer_address(),
                self.weights.buffer_address(),
                u,
                self.row_weight.buffer_address(),
            ]
        pairs = self.users * self.k
        kernel = ttnn.KernelDescriptor(
            kernel_source=KERNEL,
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=self.core_ranges,
            compile_time_args=[self.num_experts, self.k, SCRATCH_CB, pairs * 2, pairs * 4, 32 * 32 * 2],
            runtime_args=rt,
            config=ttnn.WriterConfigDescriptor(),
        )
        scratch = ttnn.CBDescriptor(
            total_size=self.scratch_bytes,
            core_ranges=self.core_ranges,
            format_descriptors=[ttnn.CBFormatDescriptor(SCRATCH_CB, ttnn.float32, self.scratch_bytes)],
        )
        return ttnn.ProgramDescriptor(kernels=[kernel], semaphores=[], cbs=[scratch])

    def __call__(self, scores, biased, mask):
        """scores, biased: [1, 1, 32, E] fp32 TILE (DRAM interleaved); mask [1, 1, 32, 1] bf16 TILE (DRAM).

        Returns the persistent (ids uint16, weights fp32) sticks, [1, 1, 1, users * k] row-major in L1, and fills
        ``self.row_weight`` [1, users * k, 32, 1] bf16; all are overwritten by the next call."""
        for t in (scores, biased, mask):
            assert t.memory_config().buffer_type == ttnn.BufferType.DRAM and t.layout == ttnn.TILE_LAYOUT
        program = self._program(scores, biased, mask)
        ttnn.generic_op([scores, biased, mask, self.ids, self.weights, self.row_weight], program)
        return self.ids, self.weights
