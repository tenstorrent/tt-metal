# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""``GegluPrefill``: ``gelu(gate) * up`` over the fused ``[up | gate]`` gate_up output of a prefill as one program.

Input ``x`` is ``[1, 1, R, 2*I]`` (bf16, TILE, DRAM interleaved): the shared MLP's gate_up projection output, ``up``
in the first ``I`` columns and ``gate`` in the last ``I``. Output ``[1, 1, R, I]`` bf16 TILE in the input's memory
config. It replaces the composite ``slice, slice, gelu, mul`` of ``shared_mlp.SharedMLP.__call__`` (two 2I-wide
reads and two I-wide writes by the slices, an I-wide read / write by the gelu, two reads and a write by the mul)
with one read of the input and one write of the output.

The arithmetic is that of ``ttnn.mul(gate, up, input_tensor_a_activations=[GELU])`` (``shared_mlp``'s prefill
GeGLU, itself bitwise ``ttnn.mul(ttnn.gelu(gate), up)``), step for step (``kernels/compute_geglu_prefill.cpp``):
the exact (erf) GELU of the gate on a 16-bit dest packed to bf16, then binary_ng's SFPU multiply with the up tile
on a 16-bit dest packed to bf16, under the op defaults' compute config (HiFi4, no fp32 dest, no approx mode).

A work unit is ``B`` consecutive output tiles of one tile row (``B`` the largest divisor of the I tiles up to 4:
the multiply holds a gelu and an up tile per output tile in the 8-tile 16-bit dest window); units are dealt in
contiguous runs over the device grid (``min(units, grid)`` cores), one read barrier and one write barrier per unit.
The shared MLP takes it where ``B`` is 4 (the 12B / 31B GLU halves); a narrower half (the 26B's 17 tiles) keeps the
two-program form. One Galaxy chip, [4096, 2 x 3840]: 303 us vs 662 us for slice, slice and the activation multiply.
"""

from pathlib import Path

import torch

import ttnn
from models.demos.gemma4.custom_ops.grid import compute_grid

KERNEL_DIR = Path(__file__).resolve().parent / "kernels"
TILE = 32
MAX_BLOCK = 4  # two dest slots per tile (gelu, up) in the 8-tile 16-bit half-sync window


def unit_block(cols: int) -> int:
    """Output tiles per work unit: the largest divisor of the I tiles up to ``MAX_BLOCK``."""
    return max(d for d in range(1, MAX_BLOCK + 1) if cols % d == 0)


class GegluPrefill:
    CB_IN = 0  # [up x B, gate x B]
    CB_MID = 1  # gelu(gate) x B, bf16
    CB_OUT = 16

    @staticmethod
    def golden(x: torch.Tensor) -> torch.Tensor:
        """torch reference on ``[..., 2*I]`` with the composite's bf16 roundings: ``bf16(gelu(gate)) * up``."""
        half = x.shape[-1] // 2
        up, gate = x[..., :half].bfloat16(), x[..., half:].bfloat16()
        g = torch.nn.functional.gelu(gate.float()).bfloat16()
        return (g.float() * up.float()).bfloat16()

    @staticmethod
    def supports(x) -> bool:
        """A bf16 TILE ``[1, 1, R, 2I]`` interleaved tensor with I a tile multiple."""
        shape = list(x.shape)
        return (
            x.layout == ttnn.TILE_LAYOUT
            and x.dtype == ttnn.bfloat16
            and not x.is_sharded()
            and len(shape) == 4
            and shape[0] == 1
            and shape[1] == 1
            and shape[-1] % (2 * TILE) == 0
        )

    @classmethod
    def op(cls, x: ttnn.Tensor, core_grid=None) -> ttnn.Tensor:
        assert cls.supports(x), f"GegluPrefill takes a bf16 TILE interleaved [1, 1, R, 2I], got {list(x.shape)}"
        _, _, R, N2 = list(x.shape)
        I = N2 // 2
        rows = x.padded_shape[-2] // TILE
        cols2 = N2 // TILE
        cols = I // TILE
        B = unit_block(cols)
        n_units = rows * (cols // B)
        out = ttnn.allocate_tensor_on_device(ttnn.Shape([1, 1, R, I]), x.dtype, x.layout, x.device(), x.memory_config())
        gx, gy = (int(core_grid.x), int(core_grid.y)) if core_grid is not None else compute_grid(x.device())
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

        # cb_in double-buffers a unit's 2B input tiles, cb_out a unit's B output tiles; cb_mid holds one unit
        cbs = [cb(cls.CB_IN, 4 * B), cb(cls.CB_MID, B), cb(cls.CB_OUT, 2 * B)]
        reader_ct = [cls.CB_IN, cols2, cols, B] + ttnn.TensorAccessorArgs(x).get_compile_time_args()
        writer_ct = [cls.CB_OUT, B] + ttnn.TensorAccessorArgs(out).get_compile_time_args()
        compute_ct = [cls.CB_IN, cls.CB_MID, cls.CB_OUT, B]

        def rt_args(values):
            args = ttnn.RuntimeArgs()
            for core, v in zip(cores, values):
                args[core.x][core.y] = [int(a) for a in v]
            return args

        kernels = [
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "reader_geglu_prefill.cpp"),
                core_ranges=core_set,
                compile_time_args=reader_ct,
                runtime_args=rt_args([[x.buffer_address(), starts[i], per_core[i]] for i in range(n_cores)]),
                config=ttnn.ReaderConfigDescriptor(),
            ),
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "writer_geglu_prefill.cpp"),
                core_ranges=core_set,
                compile_time_args=writer_ct,
                runtime_args=rt_args([[out.buffer_address(), starts[i], per_core[i]] for i in range(n_cores)]),
                config=ttnn.WriterConfigDescriptor(),
            ),
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "compute_geglu_prefill.cpp"),
                core_ranges=core_set,
                compile_time_args=compute_ct,
                runtime_args=rt_args([[per_core[i]] for i in range(n_cores)]),
                config=ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=False),
            ),
        ]
        program = ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs)
        return ttnn.generic_op([x, out], program)
