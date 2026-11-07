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
from models.demos.gemma4.custom_ops.grid import compute_grid, core_range_set, plan_concat_heads_units, row_major_cores

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
    def op(cls, x: ttnn.Tensor, num_heads: int, heads_per_unit: int = None, out_memory_config=None) -> ttnn.Tensor:
        """``out_memory_config``: the output's memory config (default: the input's interleaved one). An L1
        WIDTH_SHARDED config -- the activation layout of the DRAM-sharded o_proj decode matmul, a shard of whole
        column tiles per core -- makes the op write the o_proj input in place of that matmul's interleaved ->
        sharded copy (the page accessor maps output tile ``(h, c)`` to its shard core)."""
        assert (
            x.layout == ttnn.TILE_LAYOUT and x.dtype == ttnn.bfloat16 and not x.is_sharded()
        ), "x: bf16 TILE interleaved"
        _, B, Hp, D = (int(v) for v in x.shape)
        assert B <= TILE and x.padded_shape[2] == TILE and num_heads <= Hp, (x.shape, num_heads)
        assert D % TILE == 0
        cols = D // TILE
        device = x.device()
        out_mem = x.memory_config() if out_memory_config is None else out_memory_config
        if out_mem.is_sharded():
            shard_w = int(out_mem.shard_spec.shape[1])
            assert (
                out_mem.memory_layout == ttnn.TensorMemoryLayout.WIDTH_SHARDED
                and int(out_mem.shard_spec.shape[0]) == TILE
                and shard_w % TILE == 0
                and (num_heads * D) % shard_w == 0
            ), f"width-sharded output must hold whole column tiles of one row tile: {out_mem}"
        out = ttnn.allocate_tensor_on_device(ttnn.Shape([1, 1, B, num_heads * D]), x.dtype, x.layout, device, out_mem)
        grid = compute_grid(device)
        if heads_per_unit is None:
            heads_per_unit = int(os.environ.get("GEMMA4_CONCAT_HEADS_PER_UNIT", "0")) or None
        n_cores, heads_per_unit, groups, per = plan_concat_heads_units(grid, num_heads, cols, heads_per_unit)
        # unit id = c * groups + g: column tile c, heads g*hg .. min(H, (g+1)*hg)
        starts = [sum(per[:i]) for i in range(n_cores)]
        cores = row_major_cores(grid, n_cores)
        core_set = core_range_set(cores)
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
