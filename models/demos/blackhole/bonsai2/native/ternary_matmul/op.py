# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""``ternary_matmul``: y = x @ (T * s).T through the streaming BFP2 engine.

The op's job is to be cheap per call. Everything that depends only on the weight
-- the packed tile image, the scale table, the per-core column offsets, the CB
sizes -- is built once and held, so a call dispatches one program and does no
metadata work.

That distinction is the point of A0: the DeepSeek op rebuilt its per-core
metadata on every call and spent 90% of wall clock there, so its 1.3 GB/s was a
host measurement. The device kernel itself reached 162 GB/s (31.7% of peak) on
these shapes with the metadata already in place, which is what this op preserves.

The weight lives in DRAM and streams per call. That is what batch-1 decode needs
(the weights cannot fit in 32 GB) and what prefill needs too: the caller passes
an M-row chunk and the weight streams once per chunk, so a 4k prompt costs 32
weight passes rather than 4096.
"""

from __future__ import annotations

import numpy as np
import torch

import ttnn
from models.demos.deepseek_v3_b1.unified_kernel_descriptor import (
    PerCoreRuntimeArgsDescriptor,
    UnifiedKernelDescriptor,
)

from ..linear_weight import TernaryWeight

TILE = 32
BFP2_TILE_BYTES = 320
#: Rows per activation tile. The MVMUL takes in0 tiles of {1, 2, 4, 8} rows.
M_TILE = 8
#: Weight tiles per streamed subblock. A0: 32 tiles is within 1% of the best
#: measured bandwidth, and it must be a whole number of scale groups.
DEFAULT_SUBBLOCK_K = 32

_KERNEL = "models/demos/blackhole/bonsai2/native/ternary_matmul/kernels/ternary_matmul_kernel.cpp"


def choose_subblock(k: int, group: int, preferred: int = DEFAULT_SUBBLOCK_K) -> int:
    """Largest subblock <= ``preferred`` that divides Kt and spans whole groups."""
    kt = k // TILE
    gpt = group // TILE
    sbk = min(preferred, kt)
    while sbk >= gpt and (kt % sbk or sbk % gpt):
        sbk -= 1
    if sbk < gpt or kt % sbk or sbk % gpt:
        raise ValueError(f"no usable subblock for Kt={kt} with {gpt} tiles per group")
    return sbk


class TernaryMatmul:
    """One prepared ternary projection, callable once per token."""

    def __init__(self, device, weight: TernaryWeight, subblock_k: int | None = None, m_tile: int = M_TILE):
        self.device = device
        self.w = weight
        self.m = m_tile
        self.subblock_k = subblock_k or choose_subblock(weight.K, weight.group)
        kt = weight.K // TILE
        if kt % self.subblock_k:
            raise ValueError(f"subblock_k={self.subblock_k} does not divide Kt={kt}")
        self.num_subblocks = kt // self.subblock_k
        self.groups_per_subblock = self.subblock_k // weight.subblock_k
        if self.groups_per_subblock < 1:
            raise ValueError(f"subblock_k={self.subblock_k} is smaller than one scale group")

        self._setup_cores()
        self._upload_weights()
        self._alloc_io()

    # -- one-time setup ---------------------------------------------------

    def _setup_cores(self):
        w = self.w
        primary = self.device.get_optimal_dram_bank_to_logical_worker_assignment(ttnn.NOC.NOC_0)
        self.banks = len(primary)
        if self.banks != w.num_banks:
            raise ValueError(f"weight was packed for {w.num_banks} banks, device has {self.banks}")
        self.cores = [ttnn.CoreCoord(p.x + off, p.y) for p in primary for off in range(w.cores_per_bank)]
        if len(self.cores) != w.num_cores:
            raise ValueError(f"weight expects {w.num_cores} cores, got {len(self.cores)}")
        self.core_set = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in self.cores])

    def _upload_weights(self):
        """The weight image and scale table, uploaded once as flat byte buffers.

        They are read by byte offset from DRAM, so they stay ROW_MAJOR uint8:
        putting them in a tile layout would only obscure the addressing the
        kernel does.
        """
        w = self.w
        self.w_tensor = ttnn.from_torch(
            torch.from_numpy(w.tile_bytes.copy()),
            dtype=ttnn.uint8,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self.s_tensor = ttnn.from_torch(
            torch.from_numpy(w.scale_bytes.view(np.uint16).copy()),
            dtype=ttnn.uint16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def _alloc_io(self):
        """Activation and output tensors, allocated once and refilled per call."""
        w = self.w
        m = self.m
        self.x_tensor = ttnn.from_torch(
            torch.zeros((m * len(self.cores), w.K), dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(self.core_set, [m, w.K], ttnn.ShardOrientation.ROW_MAJOR),
            ),
            tile=ttnn.Tile([m, TILE]),
        )
        self.out_tensor = ttnn.from_torch(
            torch.zeros((m, w.n_padded), dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(self.core_set, [m, w.tiles_per_core * TILE], ttnn.ShardOrientation.ROW_MAJOR),
            ),
            tile=ttnn.Tile([m, TILE]),
        )

    # -- per call ---------------------------------------------------------

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        w = self.w
        x = x.to(torch.bfloat16)
        rows = x.shape[0]
        if x.shape[-1] != w.K:
            raise ValueError(f"x has K={x.shape[-1]}, weight has K={w.K}")
        if rows > self.m:
            raise ValueError(f"x has {rows} rows, the tile holds {self.m}")

        xp = torch.zeros((self.m, w.K), dtype=torch.bfloat16)
        xp[:rows] = x
        # A host-side ttnn tensor shaped like the sharded activation, then one
        # device copy. Building it per call is the only host work; it is a few
        # hundred KB and does not scale with the weight.
        host_x = ttnn.from_torch(
            xp.repeat(len(self.cores), 1),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            tile=ttnn.Tile([self.m, TILE]),
        )
        ttnn.copy_host_to_device_tensor(host_x, self.x_tensor)

        self._dispatch()
        return ttnn.to_torch(self.out_tensor)[:rows, : w.N]

    def _cb_descriptors(self):
        w = self.w
        sbk = self.subblock_k
        kt = w.K // TILE

        def raw(cb, page_bytes, pages):
            return ttnn.CBDescriptor(
                total_size=page_bytes * pages,
                core_ranges=self.core_set,
                format_descriptors=[
                    ttnn.CBFormatDescriptor(buffer_index=cb, data_format=ttnn.uint8, page_size=page_bytes)
                ],
            )

        def bf16(cb, page_bytes, pages):
            return ttnn.CBDescriptor(
                total_size=page_bytes * pages,
                core_ranges=self.core_set,
                format_descriptors=[
                    ttnn.CBFormatDescriptor(buffer_index=cb, data_format=ttnn.bfloat16, page_size=page_bytes)
                ],
            )

        # cb_w: 2 subblocks of BFP2 weight bytes; cb_s: 2 blocks of scale rows;
        # cb_p: 2 single-tile scratch slots.
        weight_page = sbk * BFP2_TILE_BYTES
        scale_page = self.groups_per_subblock * w.tiles_per_core * 2
        tile_page = TILE * TILE * 4  # fp32 scratch tile
        return [
            ttnn.cb_descriptor_from_sharded_tensor(0, self.x_tensor),
            raw(1, weight_page, 2),
            bf16(2, scale_page, 2),
            bf16(3, tile_page, 2),
            # The kernel emits one tile per output column, so the output CB has to
            # hold all of this core's columns: it is drained only when the program
            # finishes, and a single-slot CB deadlocks on the second column.
            ttnn.CBDescriptor(
                total_size=self.m * TILE * 4 * w.tiles_per_core,
                core_ranges=self.core_set,
                format_descriptors=[
                    ttnn.CBFormatDescriptor(
                        buffer_index=4,
                        data_format=ttnn.bfloat16,
                        page_size=self.m * TILE * 4,
                    )
                ],
            ),
        ]

    def _dispatch(self):
        w = self.w
        sbk = self.subblock_k
        ct = [0, 1, 2, 3, 4, w.tiles_per_core, sbk, self.num_subblocks, w.subblock_k]

        # Per-core runtime args: where this core's slice of the weight starts, and
        # which DRAM bank to read it from. Constants for the weight, so they are
        # computed here and reused by the descriptor.
        reader_args = []
        for i, core in enumerate(self.cores):
            reader_args.append(
                (
                    core,
                    [
                        self.w_tensor.buffer_address(),
                        self.s_tensor.buffer_address(),
                        i * w.tiles_per_core,
                        i % self.banks,
                    ],
                )
            )

        uk = UnifiedKernelDescriptor(
            kernel_source=_KERNEL,
            core_ranges=self.core_set,
            ncrisc_compile_time_args=ct,
            brisc_compile_time_args=ct,
            trisc_compile_time_args=ct,
            trisc_compute_config=ttnn.ComputeConfigDescriptor(
                math_fidelity=ttnn.MathFidelity.LoFi,
                math_approx_mode=False,
                fp32_dest_acc_en=True,
                dst_full_sync_en=False,
            ),
            per_core_runtime_args_descriptor=PerCoreRuntimeArgsDescriptor(ncrisc_args=reader_args),
        )

        descriptors = uk.get_kernel_descriptors()
        program = ttnn.ProgramDescriptor(
            kernels=descriptors.kernels, cbs=self._cb_descriptors(), semaphores=[]
        )
        ttnn.generic_op([self.x_tensor, self.w_tensor, self.s_tensor, self.out_tensor], program)
