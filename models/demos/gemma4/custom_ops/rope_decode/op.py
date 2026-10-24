# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""``RopeDecode``: per-user decode RoPE ``x * cos + rotate_half(x) * sin`` as one program.

Input ``x`` is the decode head tensor ``[1, B, heads, head_dim]`` (bf16, TILE, interleaved DRAM: users on dim 1,
heads on dim 2). In TILE layout the heads dim is padded to a multiple of 32, so every user owns ``ht = ceil(heads/32)``
row tiles (rows ``heads..31`` of the last one are padding): the buffer has ``B * ht * head_dim/32`` pages, page
``rt * cols_t + c`` for row tile ``rt`` and column tile ``c``. ``cos`` / ``sin`` are the per-user tables
``[1, B, 1, head_dim]`` (bf16, TILE): one tile per (user, column tile), the user's row in row 0, page ``u * cols_t + c``.

A work unit is one row tile and one rotate_half column pair ``(c, c + half_t)`` (NeoX pairs column ``t`` with
``t +/- head_dim/64``). The reader pushes the two x tiles and the user's two cos and two sin tiles; compute
row-broadcasts cos / sin over the tile (FPU ``mul_tiles_bcast_rows``) and combines the products on fp32 dest
(``out_a = x_a*cos_a - x_b*sin_a``, ``out_b = x_b*cos_b + x_a*sin_b``), so no ``repeat`` over heads is materialized
and no slice / neg / concat runs. The composite this replaces (``attention/operations.py::apply_rope_decode_peruser``)
is the golden.
"""

from pathlib import Path

import torch

import ttnn

KERNEL_DIR = Path(__file__).resolve().parent / "kernels"
TILE = 32


class RopeDecode:
    CB_X = 0  # the x tile pair (c, c + half)
    CB_COS = 1  # the user's cos tiles of the pair's columns (row 0 valid)
    CB_SIN = 2  # the user's sin tiles of the pair's columns (row 0 valid)
    CB_OUT = 16

    @staticmethod
    def golden(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """torch reference: x [1,B,H,D], cos/sin [1,B,1,D] (NeoX rotate_half)."""
        d = x.shape[-1]
        x1, x2 = x[..., : d // 2], x[..., d // 2 :]
        rot = torch.cat([-x2, x1], dim=-1)
        return x * cos + rot * sin

    @classmethod
    def op(cls, x: ttnn.Tensor, cos: ttnn.Tensor, sin: ttnn.Tensor, core_grid=None, _mode: int = 2) -> ttnn.Tensor:
        """``_mode`` selects the compute kernel's isolation ladder (see compute_rope_decode.cpp); 2 is the op."""
        _, B, heads, D = x.shape
        assert D % (2 * TILE) == 0, f"head_dim {D} must be a multiple of 64 (tile-pair rotate_half)"
        assert x.layout == ttnn.TILE_LAYOUT and x.dtype == ttnn.bfloat16, "x must be a bf16 TILE tensor"
        for name, t in (("cos", cos), ("sin", sin)):
            assert tuple(t.shape) == (1, B, 1, D), f"{name} must be [1, {B}, 1, {D}], got {tuple(t.shape)}"
            assert t.padded_shape[-2] == TILE and t.layout == ttnn.TILE_LAYOUT, f"{name} must be one TILE row per user"
        hp = x.padded_shape[-2]  # heads padded to the tile height
        assert hp % TILE == 0 and x.padded_shape[1] == B, f"unexpected padded shape {tuple(x.padded_shape)}"
        ht = hp // TILE  # row tiles per user
        rows_t = B * ht
        cols_t = D // TILE
        half_t = cols_t // 2
        n_units = rows_t * half_t  # (row tile, column pair)
        out = ttnn.allocate_tensor_on_device(x.shape, x.dtype, x.layout, x.device(), x.memory_config())

        device = x.device()
        grid = core_grid or device.compute_with_storage_grid_size()
        n_cores = min(grid.x * grid.y, n_units)
        # contiguous unit ranges per core (unit id = rt * half_t + c)
        per_core = [(n_units // n_cores) + (1 if i < n_units % n_cores else 0) for i in range(n_cores)]
        starts = [sum(per_core[:i]) for i in range(n_cores)]
        cores = [ttnn.CoreCoord(i % grid.x, i // grid.x) for i in range(n_cores)]
        core_set = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in cores])

        tile_bytes = TILE * TILE * 2  # bf16 tile
        fmt = ttnn.bfloat16

        def cb(idx, pages):
            return ttnn.CBDescriptor(
                total_size=pages * tile_bytes,
                core_ranges=core_set,
                format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=idx, data_format=fmt, page_size=tile_bytes)],
            )

        # 2 tiles per unit, double-buffered
        cbs = [cb(cls.CB_X, 4), cb(cls.CB_COS, 4), cb(cls.CB_SIN, 4), cb(cls.CB_OUT, 4)]

        reader_ct = [cls.CB_X, cls.CB_COS, cls.CB_SIN, ht, cols_t, half_t]
        reader_ct += ttnn.TensorAccessorArgs(x).get_compile_time_args()
        reader_ct += ttnn.TensorAccessorArgs(cos).get_compile_time_args()
        reader_ct += ttnn.TensorAccessorArgs(sin).get_compile_time_args()
        writer_ct = [cls.CB_OUT, cols_t, half_t] + ttnn.TensorAccessorArgs(out).get_compile_time_args()
        compute_ct = [cls.CB_X, cls.CB_COS, cls.CB_SIN, cls.CB_OUT, int(_mode)]

        reader_rt = [
            [x.buffer_address(), cos.buffer_address(), sin.buffer_address(), starts[i], per_core[i]]
            for i in range(n_cores)
        ]
        writer_rt = [[out.buffer_address(), starts[i], per_core[i]] for i in range(n_cores)]
        compute_rt = [[per_core[i]] for i in range(n_cores)]

        def rt(values):
            args = ttnn.RuntimeArgs()
            for core, v in zip(cores, values):
                args[core.x][core.y] = [int(a) for a in v]
            return args

        # fp32 dest: the FPU products land in fp32 and the sub / add (binary SFPU on dest) runs in fp32; the packer
        # narrows once to bf16 (the composite rounds after every op).
        compute_cfg = ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True)
        kernels = [
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "reader_rope_decode.cpp"),
                core_ranges=core_set,
                compile_time_args=reader_ct,
                runtime_args=rt(reader_rt),
                config=ttnn.ReaderConfigDescriptor(),
            ),
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "writer_rope_decode.cpp"),
                core_ranges=core_set,
                compile_time_args=writer_ct,
                runtime_args=rt(writer_rt),
                config=ttnn.WriterConfigDescriptor(),
            ),
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "compute_rope_decode.cpp"),
                core_ranges=core_set,
                compile_time_args=compute_ct,
                runtime_args=rt(compute_rt),
                config=compute_cfg,
            ),
        ]
        program = ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs)
        return ttnn.generic_op([x, cos, sin, out], program)
