# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""``GluSplit``: the shared expert's / dense MLP's ``slice(gate) + slice(up) + clamped_silu_glu`` as ONE model-local
``ttnn.generic_op`` (decode ledger C11, 2026-10-26; kernels in ``kernels/``).

    h[t, c] = silu(min(gu[t, c], limit)) * clamp(gu[t, I + c], -limit, limit)        gu [1,1,T,2I] -> h [1,1,T,I]

The fused gate|up matmul writes ``[gate_c | up_c]`` (tt/mlp.py); today's path cuts it into two tensors with two
slice ops (1.6 us each + their launch gaps) before the binary activation (4.8 us). This op reads the gate tile
(r, c) and the up tile (r, It + c) straight from the matmul output and applies the same fused SFPU op the expert
stream kernels use (``clamped_silu_glu_tile``, limit 10 compile-time = GLM-5.3-Flash ``swiglu_limit``). Work unit =
one output tile, dealt over the compute grid row-major; bf16 in / bf16 out (fp32 DST). Knob: ``GLM53_SHARED_GLU``
in tt/mlp.py (``composite`` default | ``fused`` for T <= 32 | ``fused_all``).
"""

from __future__ import annotations

from pathlib import Path

import torch

import ttnn

KERNEL_DIR = Path(__file__).resolve().parent / "kernels"
TILE = 32
BF16_TILE_BYTES = TILE * TILE * 2
LIMIT = 10.0  # the SFPU op's compile-time limit (api/compute/clamped_silu_glu.h: ClampedSiluGluConfigDsV4)


def split_work(total: int, grid) -> list[tuple[ttnn.CoreCoord, int, int]]:
    """``(core, start, count)`` per core, row-major over the grid (x fastest), the first ``total % ncores`` cores
    taking one unit more."""
    ncores = min(total, grid.x * grid.y)
    base, rem = divmod(total, ncores)
    out, start = [], 0
    for i in range(ncores):
        count = base + (1 if i < rem else 0)
        out.append((ttnn.CoreCoord(i % grid.x, i // grid.x), start, count))
        start += count
    return out


def core_range_set(n: int, grid) -> ttnn.CoreRangeSet:
    """The row-major prefix of the grid as (at most) two rectangles."""
    full_rows, rem = divmod(n, grid.x)
    ranges = []
    if full_rows:
        ranges.append(ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, full_rows - 1)))
    if rem:
        ranges.append(ttnn.CoreRange(ttnn.CoreCoord(0, full_rows), ttnn.CoreCoord(rem - 1, full_rows)))
    return ttnn.CoreRangeSet(ranges)


class GluSplit:
    CB_G, CB_U, CB_OUT = 0, 1, 16

    def __init__(self, device, limit: float = LIMIT):
        assert float(limit) == LIMIT, f"glu_split: the fused SFPU op's limit is {LIMIT} (compile-time), got {limit}"
        self.device = device
        self.limit = float(limit)
        self._grid = device.compute_with_storage_grid_size()

    @staticmethod
    def golden(gu: torch.Tensor, limit: float = LIMIT) -> torch.Tensor:
        """gu [..., 2I] -> [..., I] in fp32: silu(min(g, limit)) * clamp(u, -limit, limit)."""
        I = gu.shape[-1] // 2
        g, u = gu[..., :I].float(), gu[..., I:].float()
        return torch.nn.functional.silu(g.clamp(max=limit)) * u.clamp(-limit, limit)

    def allocate_output(self, gu: ttnn.Tensor) -> ttnn.Tensor:
        shape = list(gu.shape)
        shape[-1] //= 2
        return ttnn.allocate_tensor_on_device(
            ttnn.Shape(shape), ttnn.bfloat16, ttnn.TILE_LAYOUT, self.device, ttnn.DRAM_MEMORY_CONFIG
        )

    def __call__(self, gu: ttnn.Tensor, out: ttnn.Tensor) -> ttnn.Tensor:
        """gu [1,1,T,2I] bf16 TILE interleaved -> out [1,1,T,I] bf16 TILE interleaved (pre-allocated)."""
        assert gu.dtype == ttnn.bfloat16 and gu.layout == ttnn.TILE_LAYOUT and not gu.memory_config().is_sharded()
        assert out.dtype == ttnn.bfloat16 and out.layout == ttnn.TILE_LAYOUT and not out.memory_config().is_sharded()
        T2, W2 = gu.padded_shape[-2], gu.padded_shape[-1]
        T, I = out.padded_shape[-2], out.padded_shape[-1]
        assert W2 == 2 * I and T2 == T and I % TILE == 0 and T % TILE == 0, (gu.padded_shape, out.padded_shape)
        Rt, It = T // TILE, I // TILE
        work = split_work(Rt * It, self._grid)
        cores = core_range_set(len(work), self._grid)

        def cb(index):
            return ttnn.CBDescriptor(
                total_size=2 * BF16_TILE_BYTES,
                core_ranges=cores,
                format_descriptors=[
                    ttnn.CBFormatDescriptor(buffer_index=index, data_format=ttnn.bfloat16, page_size=BF16_TILE_BYTES)
                ],
            )

        cbs = [cb(self.CB_G), cb(self.CB_U), cb(self.CB_OUT)]
        acc = lambda t: ttnn.TensorAccessorArgs(t).get_compile_time_args()
        reader_ct = [self.CB_G, self.CB_U, It, BF16_TILE_BYTES] + acc(gu)
        writer_ct = [self.CB_OUT, BF16_TILE_BYTES] + acc(out)
        compute_ct = [self.CB_G, self.CB_U, self.CB_OUT]
        reader_rt, writer_rt, compute_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        for core, start, count in work:
            reader_rt[core.x][core.y] = [start, count]
            writer_rt[core.x][core.y] = [start, count]
            compute_rt[core.x][core.y] = [count]
        kernels = [
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "reader_glu_split.cpp"),
                core_ranges=cores,
                compile_time_args=reader_ct,
                runtime_args=reader_rt,
                common_runtime_args=[gu.buffer_address()],
                config=ttnn.ReaderConfigDescriptor(),
            ),
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "writer_glu_split.cpp"),
                core_ranges=cores,
                compile_time_args=writer_ct,
                runtime_args=writer_rt,
                common_runtime_args=[out.buffer_address()],
                config=ttnn.WriterConfigDescriptor(),
            ),
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "compute_glu_split.cpp"),
                core_ranges=cores,
                compile_time_args=compute_ct,
                runtime_args=compute_rt,
                config=ttnn.ComputeConfigDescriptor(
                    math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, math_approx_mode=False
                ),
            ),
        ]
        program = ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs)
        return ttnn.generic_op([gu, out], program)
