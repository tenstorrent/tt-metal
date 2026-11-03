# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""``GegluDecode``: ``up * gelu(gate)`` over a fused ``[up | gate]`` activation as one program.

Input ``x`` is ``[1, E, R, 2*I]`` (bf16, TILE, DRAM or L1 interleaved): the fused gate_up projection output of
the shared MLP (``E = 1``) or of the routed experts' ``sparse_matmul`` (``E`` local experts on dim 1, viewed 4-D),
``up`` in the first ``I`` columns and ``gate`` in the last ``I`` (the GLU convention of ``experts/weights.py``).
``I`` must be a multiple of 32. Output ``[1, E, R, I]`` bf16 TILE in the input's memory config.

A work unit is one output tile ``(e, r, c)``: the reader pushes the ``up`` tile ``(e, r, c)`` and the ``gate`` tile
``(e, r, c + I/32)``; compute applies the exact (erf) GELU to the gate on fp32 dest and multiplies by ``up`` on
dest (one bf16 rounding instead of the composite's two). This replaces ``slice, slice, gelu, mul`` (and the
``transpose`` of the experts' 6-D sparse output, which becomes a free reshape) with one program.
"""

from pathlib import Path

import torch

import ttnn
from models.demos.gemma4.custom_ops.grid import compute_grid

KERNEL_DIR = Path(__file__).resolve().parent / "kernels"
TILE = 32


class GegluDecode:
    CB_IN = 0  # [up, gate] tile pair
    CB_OUT = 16

    @staticmethod
    def golden(x: torch.Tensor) -> torch.Tensor:
        """torch reference on ``[..., 2*I]``: ``up * gelu(gate)`` (exact erf GELU)."""
        half = x.shape[-1] // 2
        up, gate = x[..., :half], x[..., half:]
        return up * torch.nn.functional.gelu(gate.float()).to(x.dtype)

    @classmethod
    def op(cls, x: ttnn.Tensor, core_grid=None) -> ttnn.Tensor:
        assert x.layout == ttnn.TILE_LAYOUT and x.dtype == ttnn.bfloat16, "x must be a bf16 TILE tensor"
        shape = list(x.shape)
        assert len(shape) == 4 and shape[0] == 1, f"x must be [1, E, R, 2I], got {shape}"
        _, E, R, N2 = shape
        I = N2 // 2
        assert N2 == 2 * I and I % TILE == 0, f"the GLU half {I} must be a multiple of 32"
        rt = x.padded_shape[-2] // TILE  # row tiles per e
        cols2 = N2 // TILE
        cols = I // TILE
        n_units = E * rt * cols
        out = ttnn.allocate_tensor_on_device(ttnn.Shape([1, E, R, I]), x.dtype, x.layout, x.device(), x.memory_config())
        device = x.device()
        gx, gy = (int(core_grid.x), int(core_grid.y)) if core_grid is not None else compute_grid(device)
        n_cores = min(gx * gy, n_units)
        per_core = [(n_units // n_cores) + (1 if i < n_units % n_cores else 0) for i in range(n_cores)]
        starts = [sum(per_core[:i]) for i in range(n_cores)]
        cores = [ttnn.CoreCoord(i % gx, i // gx) for i in range(n_cores)]
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

        cbs = [cb(cls.CB_IN, 4), cb(cls.CB_OUT, 2)]
        reader_ct = [cls.CB_IN, cols2, cols] + ttnn.TensorAccessorArgs(x).get_compile_time_args()
        writer_ct = [cls.CB_OUT] + ttnn.TensorAccessorArgs(out).get_compile_time_args()
        compute_ct = [cls.CB_IN, cls.CB_OUT]

        def rt_args(values):
            args = ttnn.RuntimeArgs()
            for core, v in zip(cores, values):
                args[core.x][core.y] = [int(a) for a in v]
            return args

        reader_rt = rt_args([[x.buffer_address(), starts[i], per_core[i]] for i in range(n_cores)])
        writer_rt = rt_args([[out.buffer_address(), starts[i], per_core[i]] for i in range(n_cores)])
        compute_rt = rt_args([[per_core[i]] for i in range(n_cores)])
        compute_cfg = ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True)
        kernels = [
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "reader_geglu_decode.cpp"),
                core_ranges=core_set,
                compile_time_args=reader_ct,
                runtime_args=reader_rt,
                config=ttnn.ReaderConfigDescriptor(),
            ),
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "writer_geglu_decode.cpp"),
                core_ranges=core_set,
                compile_time_args=writer_ct,
                runtime_args=writer_rt,
                config=ttnn.WriterConfigDescriptor(),
            ),
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "compute_geglu_decode.cpp"),
                core_ranges=core_set,
                compile_time_args=compute_ct,
                runtime_args=compute_rt,
                config=compute_cfg,
            ),
        ]
        program = ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs)
        return ttnn.generic_op([x, out], program)
