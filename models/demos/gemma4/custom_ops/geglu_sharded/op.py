# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""``GegluShardedDecode``: ``up * gelu(gate)`` straight from the DRAM-sharded gate_up decode matmul's L1
width-sharded output into the DRAM-sharded down matmul's L1 width-sharded activation layout, as one program.

Input ``x`` is the gate_up projection output ``[1, 1, R, 2I]`` (bf16, TILE, L1 WIDTH_SHARDED over the matmul's
core grid, ``[up | gate]`` columns, ``R <= 32`` rows on one row tile); ``out_memory_config`` the down matmul's
activation layout (``DramShardedLinear.act_memory_config``: L1 WIDTH_SHARDED over its own grid, whole column
tiles per shard). The program runs on the OUTPUT shard grid: every core computes its own shard's column tiles
(``up`` tile ``j`` and ``gate`` tile ``I/32 + j`` fetched over the NoC from the gate_up shard cores), the exact
(erf) GELU on fp32 dest and the product (the ``custom_ops/geglu_decode`` compute, reused), packed straight into
the tensor-backed output shard. This replaces ``sharded_to_interleaved (gate_up), geglu_decode (or slice, slice,
gelu, mul), to_memory_config (down's reshard)``: 3 (6) programs -> 1, and the 2I-wide activation never leaves L1.
"""

from pathlib import Path

import torch

import ttnn

KERNEL_DIR = Path(__file__).resolve().parent / "kernels"
GEGLU_KERNEL_DIR = Path(__file__).resolve().parent.parent / "geglu_decode" / "kernels"
TILE = 32


def _shard_cores(mem):
    cores = []
    for rng in mem.shard_spec.grid.ranges():
        for y in range(rng.start.y, rng.end.y + 1):
            for x in range(rng.start.x, rng.end.x + 1):
                cores.append(ttnn.CoreCoord(x, y))
    return cores


class GegluShardedDecode:
    CB_IN = 0
    CB_OUT = 16

    @staticmethod
    def golden(x: torch.Tensor) -> torch.Tensor:
        half = x.shape[-1] // 2
        up, gate = x[..., :half], x[..., half:]
        return up * torch.nn.functional.gelu(gate.float()).to(x.dtype)

    @classmethod
    def op(cls, x: ttnn.Tensor, out_memory_config) -> ttnn.Tensor:
        assert x.layout == ttnn.TILE_LAYOUT and x.dtype == ttnn.bfloat16, "x must be a bf16 TILE tensor"
        xm = x.memory_config()
        assert (
            x.is_sharded()
            and xm.buffer_type == ttnn.BufferType.L1
            and xm.memory_layout == ttnn.TensorMemoryLayout.WIDTH_SHARDED
        ), f"x must be L1 width-sharded (the gate_up decode matmul output): {xm}"
        shape = [int(v) for v in x.shape]
        assert len(shape) == 4 and shape[0] == 1 and shape[1] == 1 and 1 <= shape[2] <= TILE, shape
        R, N2 = shape[2], shape[3]
        I = N2 // 2
        assert N2 == 2 * I and I % TILE == 0, f"the GLU half {I} must be a multiple of 32"
        cols = I // TILE
        in_w = int(xm.shard_spec.shape[1])
        assert in_w % TILE == 0 and (2 * I) % in_w == 0 and int(xm.shard_spec.shape[0]) == TILE, xm
        per_in = in_w // TILE
        in_cores = _shard_cores(xm)
        assert len(in_cores) * per_in == 2 * cols, (len(in_cores), per_in, cols)
        om = out_memory_config
        assert (
            om.is_sharded()
            and om.buffer_type == ttnn.BufferType.L1
            and om.memory_layout == ttnn.TensorMemoryLayout.WIDTH_SHARDED
            and int(om.shard_spec.shape[0]) == TILE
        ), f"output must be an L1 width-sharded one-row-tile layout: {om}"
        out_w = int(om.shard_spec.shape[1])
        assert out_w % TILE == 0 and I % out_w == 0, (out_w, I)
        per_out = out_w // TILE
        out_cores = _shard_cores(om)
        assert len(out_cores) * per_out == cols, (len(out_cores), per_out, cols)
        device = x.device()
        out = ttnn.allocate_tensor_on_device(ttnn.Shape([1, 1, R, I]), x.dtype, x.layout, device, om)
        out_grid = om.shard_spec.grid
        tile_bytes = TILE * TILE * 2
        cbs = [
            ttnn.CBDescriptor(
                total_size=2 * per_out * tile_bytes,  # the core's whole unit: (up, gate) per output tile
                core_ranges=out_grid,
                format_descriptors=[
                    ttnn.CBFormatDescriptor(buffer_index=cls.CB_IN, data_format=ttnn.bfloat16, page_size=tile_bytes)
                ],
            ),
            ttnn.cb_descriptor_from_sharded_tensor(cls.CB_OUT, out),
        ]
        in_noc = []
        for c in in_cores:
            n = device.worker_core_from_logical_core(c)
            in_noc += [n.x, n.y]
        reader_ct = [cls.CB_IN, cols, per_in]
        writer_ct = [cls.CB_OUT]
        compute_ct = [cls.CB_IN, cls.CB_OUT]
        reader_rt, writer_rt, compute_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        for i, core in enumerate(out_cores):
            reader_rt[core.x][core.y] = [x.buffer_address(), i * per_out, per_out, len(in_cores)] + in_noc
            writer_rt[core.x][core.y] = [per_out]
            compute_rt[core.x][core.y] = [per_out]
        compute_cfg = ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True)
        kernels = [
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "reader_geglu_sharded.cpp"),
                core_ranges=out_grid,
                compile_time_args=reader_ct,
                runtime_args=reader_rt,
                config=ttnn.ReaderConfigDescriptor(),
            ),
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "writer_geglu_sharded.cpp"),
                core_ranges=out_grid,
                compile_time_args=writer_ct,
                runtime_args=writer_rt,
                config=ttnn.WriterConfigDescriptor(),
            ),
            ttnn.KernelDescriptor(
                kernel_source=str(GEGLU_KERNEL_DIR / "compute_geglu_decode.cpp"),
                core_ranges=out_grid,
                compile_time_args=compute_ct,
                runtime_args=compute_rt,
                config=compute_cfg,
            ),
        ]
        program = ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs)
        return ttnn.generic_op([x, out], program)
