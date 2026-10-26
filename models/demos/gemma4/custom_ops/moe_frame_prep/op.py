# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""``MoeFramePrep``: the decode MoE kernel's three dispatch inputs straight from the lane-gathered frame, one program.

The lane engine gathers every lane's decode rows on every chip (``hidden`` ``[1, 1, R, H]`` bf16 TILE DRAM, R = lanes x 32,
replicated over the mesh) together with the packed routing of ``custom_ops/router_topk.op_packed`` (``packed``
``[1, 1, R, 2k]`` bf16 ROW_MAJOR DRAM: weights then global expert ids per row). The kernel's dispatch splits the frame
over the mesh rows (cluster axis 0): row-chip ``r`` owns frame rows ``[r*B, r*B + B)`` with ``B = 32`` tokens per device,
and rows beyond ``R`` are the frame's zero-weight padding. ``rowbase`` ``[1, 1, 1, 16]`` uint32 ROW_MAJOR holds
``r*B`` on chip ``(r, c)`` (``make_rowbase``), so the same program on every chip reads its own rows.

Core ``u`` (``B`` cores, one per token; ``B`` = 32 on the Galaxy frame, 64 on the (2,4) block's frame 128) produces
token ``u`` of this chip:

* ``x`` ``[B, 1, 1, H]`` bf16 ROW_MAJOR in ``x_memory_config`` (the dispatch input config, L1 interleaved): frame row
  ``rowbase + u`` untilized on the fly (two 32-byte face rows per column tile), or zeros for a padding row;
* ``ids`` ``[B, 1, 1, k]`` uint16 ROW_MAJOR in ``x_memory_config``: the row's expert ids (bf16 -> uint16), or ``0..k-1``;
* ``scores`` ``[B, 1, 1, k]`` bf16 ROW_MAJOR in ``scores_memory_config`` (the dispatch's height-sharded L1 config, one
  ``[1, k]`` shard per core, or any interleaved config): the row's weights, or zeros.

This replaces, per layer, ``pad (hidden), pad (values), to_layout x3, concat (pad ids), mesh_partition x3, reshape x3,
typecast, to_memory_config x3`` of ``tt/experts/moe_decode.py`` + ``models/common/moe/tt_moe_decode.py``
(``_format_dispatch_inputs``): ~15 data-movement programs -> 1.
"""

from pathlib import Path

import torch

import ttnn
from models.demos.gemma4.custom_ops.grid import compute_grid

KERNEL_DIR = Path(__file__).resolve().parent / "kernels"
TILE = 32


class MoeFramePrep:
    CB_SCRATCH = 0  # one row of hidden + the packed row + the ids / scores rows

    @staticmethod
    def golden(hidden: torch.Tensor, packed: torch.Tensor, rowbase: int, k: int, B: int = TILE):
        """torch reference: hidden [1,1,R,H], packed [1,1,R,2k] float -> (x [B,1,1,H], ids [B,1,1,k] int,
        scores [B,1,1,k]) for the chip owning frame rows [rowbase, rowbase + B)."""
        R, H = hidden.shape[2], hidden.shape[3]
        x = torch.zeros(B, 1, 1, H, dtype=hidden.dtype)
        ids = torch.arange(k, dtype=torch.int64).repeat(B, 1).reshape(B, 1, 1, k)
        scores = torch.zeros(B, 1, 1, k, dtype=packed.dtype)
        for u in range(B):
            f = rowbase + u
            if f < R:
                x[u, 0, 0] = hidden[0, 0, f]
                scores[u, 0, 0] = packed[0, 0, f, :k]
                ids[u, 0, 0] = packed[0, 0, f, k:].round().to(torch.int64)
        return x, ids, scores

    @staticmethod
    def make_rowbase(mesh_device, B: int = TILE):
        """``[1, 1, 1, 16]`` uint32 per chip holding ``r * B`` for mesh row ``r`` (the dispatch axis is axis 0)."""
        is_mesh = hasattr(mesh_device, "shape")
        rows, cols = tuple(mesh_device.shape) if is_mesh else (1, 1)
        vals = torch.zeros(rows, cols, 1, 16, dtype=torch.int32)
        for r in range(rows):
            vals[r, :, 0, :] = r * B
        mapper = ttnn.ShardTensor2dMesh(mesh_device, mesh_device.shape, dims=(0, 1)) if is_mesh else None
        return ttnn.from_torch(
            vals,
            device=mesh_device,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=mapper,
        )

    @classmethod
    def op(
        cls,
        hidden: ttnn.Tensor,
        packed: ttnn.Tensor,
        rowbase: ttnn.Tensor,
        k: int,
        x_memory_config=None,
        scores_memory_config=None,
        B: int = TILE,
    ):
        assert hidden.layout == ttnn.TILE_LAYOUT and hidden.dtype == ttnn.bfloat16, "hidden: bf16 TILE"
        assert not hidden.is_sharded() and not packed.is_sharded(), "hidden / packed: interleaved"
        _, _, R, H = (int(v) for v in hidden.shape)
        assert R % TILE == 0 and H % TILE == 0, (R, H)
        assert packed.layout == ttnn.ROW_MAJOR_LAYOUT and packed.dtype == ttnn.bfloat16, "packed: bf16 ROW_MAJOR"
        assert tuple(int(v) for v in packed.shape) == (1, 1, R, 2 * k), (packed.shape, R, k)
        assert rowbase.layout == ttnn.ROW_MAJOR_LAYOUT and rowbase.dtype == ttnn.uint32, "rowbase: uint32 RM"
        assert B % TILE == 0, "whole row tiles of tokens per device"
        device = hidden.device()
        x_mem = x_memory_config or ttnn.L1_MEMORY_CONFIG
        s_mem = scores_memory_config or x_mem
        x = ttnn.allocate_tensor_on_device(
            ttnn.Shape([B, 1, 1, H]), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, device, x_mem
        )
        ids = ttnn.allocate_tensor_on_device(
            ttnn.Shape([B, 1, 1, k]), ttnn.uint16, ttnn.ROW_MAJOR_LAYOUT, device, x_mem
        )
        scores = ttnn.allocate_tensor_on_device(
            ttnn.Shape([B, 1, 1, k]), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, device, s_mem
        )
        gx, gy = compute_grid(device)  # (x, y), capped by GEMMA4_TEST_GRID
        assert gx * gy >= B, f"one token per core: {B} tokens need {B} cores, grid {gx}x{gy}"
        cores = [ttnn.CoreCoord(i % gx, i // gx) for i in range(B)]
        core_set = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in cores])
        cols_t = H // TILE
        row_bytes = H * 2
        packed_bytes = packed.buffer_page_size()
        # scratch layout (from a 64-byte aligned base, 64 B of slack): hidden row | face-row staging (2 x cols_t
        # slots of 64 B) | packed row (64 B slot) | ids row (64 B slot) | scores row (64 B slot)
        scratch_bytes = 64 + row_bytes + 2 * cols_t * 64 + 3 * 64
        cbs = [
            ttnn.CBDescriptor(
                total_size=scratch_bytes,
                core_ranges=core_set,
                format_descriptors=[
                    ttnn.CBFormatDescriptor(
                        buffer_index=cls.CB_SCRATCH, data_format=ttnn.bfloat16, page_size=scratch_bytes
                    )
                ],
            )
        ]
        ct = [
            cls.CB_SCRATCH,
            R,
            cols_t,
            k,
            packed_bytes,
            rowbase.buffer_page_size(),
            x.buffer_page_size(),
            ids.buffer_page_size(),
            scores.buffer_page_size(),
        ]
        for t in (hidden, packed, rowbase, x, ids, scores):
            ct += ttnn.TensorAccessorArgs(t).get_compile_time_args()
        rt = ttnn.RuntimeArgs()
        for u, core in enumerate(cores):
            rt[core.x][core.y] = [
                hidden.buffer_address(),
                packed.buffer_address(),
                rowbase.buffer_address(),
                x.buffer_address(),
                ids.buffer_address(),
                scores.buffer_address(),
                u,
            ]
        kernels = [
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "moe_frame_prep.cpp"),
                core_ranges=core_set,
                compile_time_args=ct,
                runtime_args=rt,
                config=ttnn.ReaderConfigDescriptor(),
            )
        ]
        program = ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs)
        ttnn.generic_op([hidden, packed, rowbase, x, ids, scores], program)
        return x, ids, scores
