# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Prefill per-head RMSNorm + partial RoPE of a head-major q or k [1, H, T, D] (kernels/qkr_*.cpp), one DRAM pass:
replaces ttnn.rms_norm and apply_rotary_pos_emb_ttnn's ~10 slices / multiplies / concats per tensor. Rotary dim 64
(two tiles; rotate_half swaps 32-column halves). Products and the rotation in fp32, one bf16 rounding at the end."""

import struct

import torch

import ttnn
from models.demos.blackhole.bonsai2.tt.core_ranges import row_major_crs

KERNELS = "models/demos/blackhole/bonsai2/tt/tprefill/kernels/"


def _f32_bits(v):
    return struct.unpack("<I", struct.pack("<f", float(v)))[0]


class QKNormRope:
    def __init__(self, device, norm_w, eps):
        w = torch.as_tensor(norm_w, dtype=torch.float32).reshape(-1)
        D = w.numel()
        assert D % 32 == 0 and D // 32 >= 3, D
        tiles = [torch.ones(32, 32)] + [w[32 * j : 32 * j + 32].reshape(1, 32).expand(32, 32) for j in range(D // 32)]
        self.consts = ttnn.from_torch(torch.cat(tiles, 0), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
        self.D, self.eps, self.device = D, eps, device

    def __call__(self, x, cos, sin, kv_out=None):
        """x [1, H, T, D] bf16 TILE (head-major); cos, sin [..., T, 64] bf16 TILE -> rope(rms_norm(x) * w), same shape.
        T need not fill a tile (decode: T = 1, cos / sin [1, 1, 64]).
        kv_out = (k_sh, v, v_sh) (decode): write the result token-major into k_sh and copy v (head-major [1, H, 1, D])
        token-major into v_sh, both [1, 1, 32, D] L1 shards on one core (the paged cache update's inputs); returns k_sh.
        """
        device, D = self.device, self.D
        _, H, T, Dx = list(x.shape)
        assert Dx == D, (x.shape, D)
        assert list(cos.shape)[-1] == 64 and list(sin.shape)[-1] == 64, (cos.shape, sin.shape)
        # T < 32 (decode: one token): the padded tile rows go through too, only the logical rows are read later
        Mt, Dt, Rt = -(-T // 32), D // 32, 2
        if kv_out is not None:
            assert T == 1, T
            out, v, v_sh = kv_out
        else:
            out = ttnn.allocate_tensor_on_device(ttnn.Shape([1, H, T, D]), ttnn.bfloat16, ttnn.TILE_LAYOUT, device)
        grid = device.compute_with_storage_grid_size()
        items = H * Mt
        ncores = min(grid.x * grid.y, items)
        crs = row_major_crs(ncores, grid.x)
        span = [items, ncores, grid.x]

        def cb(index, fmt, pages):
            page = 4096 if fmt == ttnn.float32 else 2048
            return ttnn.CBDescriptor(
                total_size=page * pages,
                core_ranges=crs,
                format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=fmt, page_size=page)],
            )

        bf, f32 = ttnn.bfloat16, ttnn.float32
        rd_ct = []
        for t in (x, self.consts, cos, sin):
            rd_ct += ttnn.TensorAccessorArgs(t).get_compile_time_args()
        wr_ct = ttnn.TensorAccessorArgs(out).get_compile_time_args()
        wr_c = [out.buffer_address()] + span + [H, Mt, Dt]
        wr_def = []
        if kv_out is not None:
            sh = device.worker_core_from_logical_core(out.memory_config().shard_spec.grid.ranges()[0].start)
            wr_ct += ttnn.TensorAccessorArgs(v).get_compile_time_args()
            wr_c += [sh.x, sh.y, out.buffer_address(), v_sh.buffer_address(), v.buffer_address()]
            wr_def = [("TOKEN_MAJOR", "1")]
        wr_kernel = ttnn.KernelDescriptor(
            kernel_source=KERNELS + "qkr_writer.cpp",
            core_ranges=crs,
            compile_time_args=wr_ct,
            common_runtime_args=wr_c,
            defines=wr_def,
            config=ttnn.WriterConfigDescriptor(),
        )
        program = ttnn.ProgramDescriptor(
            kernels=[
                ttnn.KernelDescriptor(
                    kernel_source=KERNELS + "qkr_reader.cpp",
                    core_ranges=crs,
                    compile_time_args=rd_ct,
                    common_runtime_args=[
                        x.buffer_address(),
                        self.consts.buffer_address(),
                        cos.buffer_address(),
                        sin.buffer_address(),
                    ]
                    + span
                    + [H, Mt, Dt, Rt],  # fmt: skip
                    config=ttnn.ReaderConfigDescriptor(),
                ),
                wr_kernel,
                ttnn.KernelDescriptor(
                    kernel_source=KERNELS + "qkr_compute.cpp",
                    core_ranges=crs,
                    common_runtime_args=span + [Dt, _f32_bits(1.0 / D), _f32_bits(self.eps)],
                    config=ttnn.ComputeConfigDescriptor(
                        math_fidelity=ttnn.MathFidelity.HiFi4,
                        fp32_dest_acc_en=True,
                        math_approx_mode=False,
                        dst_full_sync_en=True,
                    ),
                ),
            ],
            semaphores=[],
            cbs=[cb(0, bf, 2 * Dt), cb(1, f32, 1 + Dt), cb(2, bf, 8), cb(3, f32, Dt), cb(4, f32, 1), cb(16, bf, 8)]
            + ([cb(17, bf, 1)] if kv_out is not None else []),
        )
        ttnn.generic_op([x, self.consts, cos, sin] + ([v, v_sh] if kv_out is not None else []) + [out], program)
        return out


_KV_SH = {}


def kv_shards(device, H, D):
    """Persistent token-major K and V decode buffers ([1, 1, H, D] bf16, one tile row, height-sharded on core (0, 0), as the paged
    cache update takes them), zero once: the norm + RoPE op writes only the head rows, the padding rows stay zero."""
    if (H, D) not in _KV_SH:
        mc = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))}),
                [32, D],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        )
        _KV_SH[(H, D)] = tuple(
            ttnn.from_torch(
                torch.zeros(1, 1, H, D), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mc
            )
            for _ in range(2)
        )
    return _KV_SH[(H, D)]
