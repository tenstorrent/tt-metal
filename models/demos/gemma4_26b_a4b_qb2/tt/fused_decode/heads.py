# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Batch-1 decode head split + per-head RMSNorm + NeoX RoPE (kernels/heads_*.cpp), one head per core.

Same contract as tt/ws2_ops.py qkv_heads_norm_rope_b1 (q / k on `home`, v on `v_core`, one-core height-sharded
[1, 1, heads, D] outputs, heads on the rows): the output the cache update and SDPA take unchanged. The difference is
the work layout: there each tensor ran on one core over its D / 32 column tiles (all heads in the rows of every tile:
the compute cost is per column tile, 8 / 16 tiles at D 256 / 512); here every head packs into ONE tile (value
d = 32 r + c at row r, column c), so each core normalizes and rotates a single tile.
"""

import struct
from pathlib import Path

import torch

import ttnn
from models.demos.gemma4_26b_a4b_qb2.tt.ws2_ops import _one_core_heads_memory

KERNELS = Path(__file__).resolve().parent / "kernels"
TILE = 32


def _f32_bits(v):
    return struct.unpack("<I", struct.pack("<f", float(v)))[0]


def pack_head(v):
    """[D] -> one [32, 32] tile: value 32 r + c at (r, c)."""
    R = v.numel() // TILE
    t = torch.zeros(TILE, TILE)
    t[:R] = v.float().reshape(R, TILE)
    return t


def pack_rotated(v):
    """The packed tile of rotate_half's operand without the rotation: row r <- row (r + R / 2) % R, rows r < R / 2
    negated (NeoX: rotate_half(t)[d] = -t[d + D / 2] for d < D / 2, t[d - D / 2] after)."""
    R = v.numel() // TILE
    rows = v.float().reshape(R, TILE)
    t = torch.zeros(TILE, TILE)
    t[:R] = torch.roll(rows, -(R // 2), 0)
    t[: R // 2] *= -1
    return t


class HeadsB1:
    def __init__(self, mesh, hq, hkv, head_dim, gq, gk, eps, *, home=(0, 0), v_core=(1, 0), first_core=(2, 1)):
        """gq / gk: [D] torch q / k norm gammas (v: no gamma)."""
        assert head_dim % 64 == 0 and head_dim <= 512
        self.mesh, self.hq, self.hkv, self.D, self.eps = mesh, hq, hkv, head_dim, eps
        self.R = head_dim // TILE
        self.home, self.v_core = ttnn.CoreCoord(*home), ttnn.CoreCoord(*v_core)
        packs = torch.cat([pack_head(gq), pack_rotated(gq), pack_head(gk), pack_rotated(gk)])  # 4 tiles
        self.gammas = ttnn.from_torch(
            packs[None, None].bfloat16(),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )
        grid = mesh.compute_with_storage_grid_size()
        n = hq + 2 * hkv
        start = first_core[1] * grid.x + first_core[0]
        self.cores = [ttnn.CoreCoord(i % grid.x, i // grid.x) for i in range(start, start + n)]
        # unit u: tensor (0 q, 1 k, 2 v), head
        self.units = [(0, h) for h in range(hq)] + [(1, h) for h in range(hkv)] + [(2, h) for h in range(hkv)]

    def __call__(self, x, cos, sin):
        """x: the QKV projection output [1, 1, 1, (hq + 2 hkv) D] BF16 TILE, L1; cos / sin: the position-gathered
        [1, 1, 32, D] rows (row 0). Returns (q, k, v)."""
        D, R, mesh = self.D, self.R, self.mesh
        assert x.memory_config().buffer_type == ttnn.BufferType.L1 and x.dtype == ttnn.bfloat16
        outs = [
            ttnn.allocate_tensor_on_device(
                ttnn.Shape([1, 1, h, D]), ttnn.bfloat16, ttnn.TILE_LAYOUT, mesh, _one_core_heads_memory(core, D)
            )
            for h, core in ((self.hq, self.home), (self.hkv, self.home), (self.hkv, self.v_core))
        ]
        targets = [mesh.worker_core_from_logical_core(c) for c in (self.home, self.home, self.v_core)]
        cb_x, cb_xr, cb_cs, cb_g, cb_scaler, cb_stage, cb_sq, cb_rs, cb_t, cb_out = 0, 1, 2, 3, 4, 5, 6, 7, 8, 16
        groups = {True: [], False: []}  # rope -> cores
        for core, (t, _) in zip(self.cores, self.units):
            groups[t < 2].append(core)
        all_set = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in self.cores])

        def cb(index, pages, dtype=ttnn.bfloat16, page=2048):
            fmt = [ttnn.CBFormatDescriptor(buffer_index=index, data_format=dtype, page_size=page)]
            return ttnn.CBDescriptor(total_size=pages * page, core_ranges=all_set, format_descriptors=fmt)

        cbs = [
            cb(cb_x, 1),
            cb(cb_xr, 1),
            cb(cb_cs, 2),
            cb(cb_g, 2),
            cb(cb_scaler, 1),
            cb(cb_stage, 1, page=256 * R),
            cb(cb_sq, 1, ttnn.float32, 4096),
            cb(cb_rs, 1, ttnn.float32, 4096),
            cb(cb_t, 2, ttnn.float32, 4096),
            cb(cb_out, 1),
        ]
        acc = []
        for tsr in (x, cos, sin, self.gammas):
            acc += ttnn.TensorAccessorArgs(tsr).get_compile_time_args()
        reader_rts = {True: ttnn.RuntimeArgs(), False: ttnn.RuntimeArgs()}
        writer_rts = {True: ttnn.RuntimeArgs(), False: ttnn.RuntimeArgs()}
        heads_before = (0, self.hq, self.hq + self.hkv)
        for core, (t, h) in zip(self.cores, self.units):
            first = (heads_before[t] + h) * R
            reader_rt, writer_rt = reader_rts[t < 2], writer_rts[t < 2]
            reader_rt[core.x][core.y] = [
                x.buffer_address(),
                first,
                cos.buffer_address(),
                sin.buffer_address(),
                self.gammas.buffer_address(),
                t if t < 2 else 0,
            ]
            writer_rt[core.x][core.y] = [targets[t].x, targets[t].y, outs[t].buffer_address(), h]
        kernels = []
        for rope, cores in groups.items():
            if not cores:
                continue
            cs = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in cores])
            kernels += [
                ttnn.KernelDescriptor(
                    kernel_source=str(KERNELS / "heads_reader.cpp"),
                    core_ranges=cs,
                    compile_time_args=[cb_x, cb_xr, cb_cs, cb_g, cb_scaler, cb_stage, R, int(rope)] + acc,
                    runtime_args=reader_rts[rope],
                    config=ttnn.DataMovementConfigDescriptor(
                        processor=ttnn.DataMovementProcessor.RISCV_1, noc=ttnn.NOC.NOC_0
                    ),
                ),
                ttnn.KernelDescriptor(
                    kernel_source=str(KERNELS / "heads_compute.cpp"),
                    core_ranges=cs,
                    compile_time_args=[cb_x, cb_xr, cb_cs, cb_g, cb_scaler, cb_sq, cb_rs, cb_t, cb_out, int(rope)]
                    + [_f32_bits(1.0 / D), _f32_bits(self.eps)],
                    runtime_args=[],
                    config=ttnn.ComputeConfigDescriptor(
                        math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, math_approx_mode=False
                    ),
                ),
                ttnn.KernelDescriptor(
                    kernel_source=str(KERNELS / "heads_writer.cpp"),
                    core_ranges=cs,
                    compile_time_args=[cb_out, R],
                    runtime_args=writer_rts[rope],
                    config=ttnn.DataMovementConfigDescriptor(
                        processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.NOC_1
                    ),
                ),
            ]
        ttnn.generic_op(
            [x, cos, sin, self.gammas] + outs, ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs)
        )
        return tuple(outs)
