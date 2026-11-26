# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""``MoeCombine``: the routed experts' weighted sum ``out[u, :] = sum_e w[e, u] * down[e, u, :]`` as one program.

``down`` is the local experts' down-projection output ``[1, E, R, H]`` (bf16, TILE, interleaved; ``R <= 32``
users on one row tile, the 6-D ``sparse_matmul`` output viewed 4-D), ``w`` the routing weights ``[1, 1, E, 32]``
(bf16, ROW_MAJOR: row ``e`` holds the users' weights of local expert ``e``, zeros for users that did not select
it and for users ``>= R``), the form ``custom_ops/router_topk`` emits. Output ``[1, 1, R, H]`` bf16 TILE in
``down``'s memory config.

A work unit is one output column tile ``c``. The reader takes the 2 KB weight table once and, for every local
expert ``e`` (a fixed count: the math core cannot read a data-dependent one), pushes a weight tile it builds in L1
(column 0 = the expert's user weights, zeros elsewhere) and the ``down`` tile ``(e, c)`` -- for an expert no user
selected (a zero row: ``sparse_matmul`` skipped it) the ``down`` read is skipped and a zero tile goes in instead.
Compute multiplies each tile by its weight column (FPU column broadcast) and accumulates the products in fp32,
the running sum passing through an fp32 L1 buffer between experts (an FPU write to a dest tile races an SFPU read
of it inside one acquire window, measured: wrong sums from 3 experts on), one bf16 pack. This replaces ``permute, mul, fast_reduce_nc`` and the
routing-side glue (the ``[E, R, 1]`` weight permute) with one program, and reads only the selected experts.
"""

from pathlib import Path

import torch

import ttnn
from models.demos.gemma4.custom_ops.grid import compute_grid

KERNEL_DIR = Path(__file__).resolve().parent / "kernels"
TILE = 32


class MoeCombine:
    CB_D = 0  # down tiles (zeros for the experts nobody selected)
    CB_W = 1  # weight tiles built by the reader (column 0 valid)
    CB_TAB = 3  # the [E, 32] bf16 weight table (one 64-byte row per expert), read once per core
    CB_ACC = 4  # the fp32 running sum between experts (one acquire / release per expert: see the compute kernel)
    CB_OUT = 16

    @staticmethod
    def golden(down: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """down [1,E,R,H], w [1,1,E,32] -> [1,1,R,H]: sum over experts of w[e, u] * down[e, u, :]."""
        R = down.shape[2]
        wt = w[0, 0, :, :R].float()  # [E, R]
        return (down.float() * wt[None, :, :, None]).sum(dim=1, keepdim=True)

    @classmethod
    def op(cls, down: ttnn.Tensor, w: ttnn.Tensor, core_grid=None) -> ttnn.Tensor:
        assert down.layout == ttnn.TILE_LAYOUT and down.dtype == ttnn.bfloat16, "down must be a bf16 TILE tensor"
        _, E, R, H = (int(v) for v in down.shape)
        assert R <= TILE and down.padded_shape[-2] == TILE, f"one row tile of users expected, got {down.shape}"
        assert H % TILE == 0, f"hidden {H} must be a multiple of 32"
        assert w.layout == ttnn.ROW_MAJOR_LAYOUT and w.dtype == ttnn.bfloat16, "w must be a bf16 ROW_MAJOR tensor"
        assert tuple(int(v) for v in w.shape) == (1, 1, E, TILE), f"w must be [1, 1, {E}, 32], got {tuple(w.shape)}"
        cols = H // TILE
        out = ttnn.allocate_tensor_on_device(
            ttnn.Shape([1, 1, R, H]), down.dtype, down.layout, down.device(), down.memory_config()
        )
        device = down.device()
        gx, gy = (int(core_grid.x), int(core_grid.y)) if core_grid is not None else compute_grid(device)
        n_cores = min(gx * gy, cols)
        per_core = [(cols // n_cores) + (1 if i < cols % n_cores else 0) for i in range(n_cores)]
        starts = [sum(per_core[:i]) for i in range(n_cores)]
        cores = [ttnn.CoreCoord(i % gx, i // gx) for i in range(n_cores)]
        core_set = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in cores])
        tile_bytes = TILE * TILE * 2
        w_page = w.buffer_page_size()  # one 64-byte row per expert

        def cb(idx, pages, page_size=tile_bytes, fmt=ttnn.bfloat16):
            return ttnn.CBDescriptor(
                total_size=pages * page_size,
                core_ranges=core_set,
                format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=idx, data_format=fmt, page_size=page_size)],
            )

        tab_bytes = E * w_page
        # one expert per dest window of the compute kernel (its EPW compile-time arg; the validated form, the
        # running sum passing through cb_acc between windows); the d / w CBs hold two windows
        epw = 1
        slots = 2 * epw
        cbs = [
            cb(cls.CB_D, slots),  # zeroed once by the reader (a skipped expert leaves its slot zero)
            cb(cls.CB_W, slots),  # zeroed once by the reader (only column 0 is rewritten per expert)
            cb(cls.CB_TAB, 1, page_size=tab_bytes),
            cb(cls.CB_ACC, 2, page_size=TILE * TILE * 4, fmt=ttnn.float32),
            cb(cls.CB_OUT, 2),
        ]
        reader_ct = [cls.CB_D, cls.CB_W, cls.CB_TAB, E, R, cols, w_page, slots]
        reader_ct += ttnn.TensorAccessorArgs(down).get_compile_time_args()
        reader_ct += ttnn.TensorAccessorArgs(w).get_compile_time_args()
        writer_ct = [cls.CB_OUT] + ttnn.TensorAccessorArgs(out).get_compile_time_args()
        compute_ct = [cls.CB_D, cls.CB_W, cls.CB_OUT, E, cls.CB_ACC, epw]

        def rt_args(values):
            args = ttnn.RuntimeArgs()
            for core, v in zip(cores, values):
                args[core.x][core.y] = [int(a) for a in v]
            return args

        reader_rt = rt_args(
            [[down.buffer_address(), w.buffer_address(), starts[i], per_core[i]] for i in range(n_cores)]
        )
        writer_rt = rt_args([[out.buffer_address(), starts[i], per_core[i]] for i in range(n_cores)])
        compute_rt = rt_args([[per_core[i]] for i in range(n_cores)])
        compute_cfg = ttnn.ComputeConfigDescriptor(
            math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, dst_full_sync_en=False
        )
        kernels = [
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "reader_moe_combine.cpp"),
                core_ranges=core_set,
                compile_time_args=reader_ct,
                runtime_args=reader_rt,
                config=ttnn.ReaderConfigDescriptor(),
            ),
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "writer_moe_combine.cpp"),
                core_ranges=core_set,
                compile_time_args=writer_ct,
                runtime_args=writer_rt,
                config=ttnn.WriterConfigDescriptor(),
            ),
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "compute_moe_combine.cpp"),
                core_ranges=core_set,
                compile_time_args=compute_ct,
                runtime_args=compute_rt,
                config=compute_cfg,
            ),
        ]
        program = ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs)
        return ttnn.generic_op([down, w, out], program)
