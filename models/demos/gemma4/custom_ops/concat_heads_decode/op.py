# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""``ConcatHeadsDecode``: the decode SDPA output ``[1, B, H(pad 32), D]`` (bf16, TILE, DRAM interleaved: one row
tile per user, the heads on its rows) to the o_proj input ``[1, 1, B, H*D]`` (bf16, TILE, DRAM interleaved: user
``b`` on row ``b``, head ``h`` at columns ``h*D..``) as one program: output tile ``(h, c)`` row ``b`` = input tile
``(b, c)`` row ``h``; rows ``>= B`` are zero. A work unit is one column tile ``c`` of the head dim and a group of
``heads_per_unit`` heads (default: the smallest group that fits the units on the worker grid, so ``H * D/32``
units of one head on the 26B's 4 local heads): the core reads the ``B`` input tiles ``(b, c)`` once and assembles
its output tiles with 32-byte face-row copies. This replaces ``to_memory_config (interleaved -> height sharded),
nlp_concat_heads_decode, sharded_to_interleaved`` (and the ``[:, :, :B]`` slice at B < 32): 3-4 programs -> 1.
The one-unit-per-column form (``heads_per_unit = H``, 8 / 16 cores) took 27.8 us per layer on the 26B; the
per-head split spreads the same copies over 32 / 64 cores.
"""

import os
from pathlib import Path

import torch

import ttnn

KERNEL_DIR = Path(__file__).resolve().parent / "kernels"
TILE = 32


class ConcatHeadsDecode:
    CB_IN = 0  # the B input tiles of this core's column
    CB_OUT = 1  # the H output tiles

    @staticmethod
    def golden(x: torch.Tensor, num_heads: int) -> torch.Tensor:
        """x [1, B, H, D] -> [1, 1, B, H*D]."""
        B, H, D = x.shape[1], num_heads, x.shape[3]
        return x[:, :, :H, :].reshape(1, 1, B, H * D)

    @classmethod
    def op(cls, x: ttnn.Tensor, num_heads: int, heads_per_unit: int = None) -> ttnn.Tensor:
        assert (
            x.layout == ttnn.TILE_LAYOUT and x.dtype == ttnn.bfloat16 and not x.is_sharded()
        ), "x: bf16 TILE interleaved"
        _, B, Hp, D = (int(v) for v in x.shape)
        assert B <= TILE and x.padded_shape[2] == TILE and num_heads <= Hp, (x.shape, num_heads)
        assert D % TILE == 0
        cols = D // TILE
        device = x.device()
        out = ttnn.allocate_tensor_on_device(
            ttnn.Shape([1, 1, B, num_heads * D]), x.dtype, x.layout, device, x.memory_config()
        )
        grid = device.compute_with_storage_grid_size()
        n_grid = grid.x * grid.y
        if heads_per_unit is None:
            heads_per_unit = int(os.environ.get("GEMMA4_CONCAT_HEADS_PER_UNIT", "0")) or next(
                hg for hg in range(1, num_heads + 1) if cols * ((num_heads + hg - 1) // hg) <= n_grid
            )
        heads_per_unit = max(1, min(num_heads, heads_per_unit))
        groups = (num_heads + heads_per_unit - 1) // heads_per_unit
        n_units = cols * groups  # unit id = c * groups + g: column tile c, heads g*hg .. min(H, (g+1)*hg)
        n_cores = min(n_units, n_grid)
        per = [(n_units // n_cores) + (1 if i < n_units % n_cores else 0) for i in range(n_cores)]
        starts = [sum(per[:i]) for i in range(n_cores)]
        cores = [ttnn.CoreCoord(i % grid.x, i // grid.x) for i in range(n_cores)]
        core_set = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in cores])
        tile_bytes = TILE * TILE * 2

        def cb(idx, pages):
            return ttnn.CBDescriptor(
                total_size=pages * tile_bytes,
                core_ranges=core_set,
                format_descriptors=[
                    ttnn.CBFormatDescriptor(buffer_index=idx, data_format=ttnn.bfloat16, page_size=tile_bytes)
                ],
            )

        cbs = [cb(cls.CB_IN, TILE), cb(cls.CB_OUT, heads_per_unit)]
        ct = [cls.CB_IN, cls.CB_OUT, B, num_heads, cols, heads_per_unit, groups]
        ct += ttnn.TensorAccessorArgs(x).get_compile_time_args()
        ct += ttnn.TensorAccessorArgs(out).get_compile_time_args()
        rt = ttnn.RuntimeArgs()
        for i, core in enumerate(cores):
            rt[core.x][core.y] = [x.buffer_address(), out.buffer_address(), starts[i], per[i]]
        kernels = [
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "concat_heads_decode.cpp"),
                core_ranges=core_set,
                compile_time_args=ct,
                runtime_args=rt,
                config=ttnn.ReaderConfigDescriptor(),
            )
        ]
        program = ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs)
        return ttnn.generic_op([x, out], program)
