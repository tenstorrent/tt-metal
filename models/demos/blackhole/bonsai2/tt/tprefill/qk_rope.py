# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Prefill per-head RMSNorm + partial RoPE of a head-major q or k [1, H, T, D] (kernels/qkr_*.cpp), one DRAM pass:
replaces ttnn.rms_norm and apply_rotary_pos_emb_ttnn's ~10 slices / multiplies / concats per tensor. Rotary dim 64
(two tiles; rotate_half swaps 32-column halves). Products and the rotation in fp32, one bf16 rounding at the end."""

import struct

import torch

import ttnn
from models.demos.blackhole.bonsai2.tt.core_ranges import core_xy, row_major_crs, row_major_range_crs

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
        self.c_off = 0  # the consts' first tile in self.consts (stream/decode.py packs every layer's into one tensor)

    @staticmethod
    def run_parts(*parts):
        """Several QKNormRope calls (parts=True, on disjoint cores: core_offset) as one op."""
        kernels, cbs, ios = [], [], []
        for k, c, t in parts:
            kernels += k
            cbs += c
            ios += [x for x in t if all(x is not y for y in ios)]
        ttnn.generic_op(ios, ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs))

    def __call__(self, x, cos, sin, kv_out=None, cache=None, kv_l1=None, q_hub=None, core_offset=0, parts=False):
        """x [1, H, T, D] bf16 TILE (head-major); cos, sin [..., T, 64] bf16 TILE -> rope(rms_norm(x) * w), same shape.
        T need not fill a tile (decode: T = 1, cos / sin [1, 1, 64]).
        kv_out = (k_sh, v, v_sh) (decode): write the result token-major into k_sh and copy v (head-major [1, H, 1, D])
        token-major into v_sh, both [1, 1, 32, D] L1 shards on one core (the paged cache update's inputs); returns k_sh.
        cache = (k_cache, v_cache, v, cur_pos, page_table) (decode, bf8 paged caches [blocks, H, bs, D]): write the
        result and v (head-major [1, H, 1, D]) straight into the caches at cur_pos[0] (no paged cache update ops);
        returns None. kv_l1 = (buffer, maxr, NG, cores): also into the decode SDPA's prefetched K / V
        (sdpa_dec/kernels/kv_prefetch.hpp) on the core holding the position's tile row.
        core_offset: on the cores from core_offset on; parts: return (kernels, CBs, tensors) for run_parts instead.
        """
        device, D = self.device, self.D
        _, H, T, Dx = list(x.shape)
        assert Dx == D, (x.shape, D)
        assert list(cos.shape)[-1] == 64 and list(sin.shape)[-1] == 64, (cos.shape, sin.shape)
        # T < 32 (decode: one token): the padded tile rows go through too, only the logical rows are read later
        Mt, Dt, Rt = -(-T // 32), D // 32, 2
        if cache is not None:
            assert T == 1, T
            kc, vc, v, pos, pt = cache
            out = kc
        elif kv_out is not None:
            assert T == 1, T
            out, v, v_sh = kv_out
        else:
            out = ttnn.allocate_tensor_on_device(ttnn.Shape([1, H, T, D]), ttnn.bfloat16, ttnn.TILE_LAYOUT, device)
        grid = device.compute_with_storage_grid_size()
        items = H * Mt
        ncores = min(grid.x * grid.y, items)
        if core_offset:
            crs = row_major_range_crs(core_offset, core_offset + ncores, grid.x)
        else:
            crs = row_major_crs(ncores, grid.x)
        span = [items, ncores, grid.x]
        off = [("CORE_OFFSET", str(core_offset))] if core_offset else []

        def cb(index, fmt, pages):
            page = {ttnn.float32: 4096, ttnn.bfloat8_b: 1088}.get(fmt, 2048)
            return ttnn.CBDescriptor(
                total_size=page * pages,
                core_ranges=crs,
                format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=fmt, page_size=page)],
            )

        bf, f32 = ttnn.bfloat16, ttnn.float32
        rd_ct = []
        for t in (x, self.consts, cos, sin) + ((v,) if cache is not None else ()):
            rd_ct += ttnn.TensorAccessorArgs(t).get_compile_time_args()
        wr_ct = ttnn.TensorAccessorArgs(out).get_compile_time_args()
        wr_c = [out.buffer_address()] + span + [H, Mt, Dt]
        wr_def = []
        if cache is not None:
            for t in (vc, pos, pt):
                wr_ct += ttnn.TensorAccessorArgs(t).get_compile_time_args()
            wr_c += [kc.buffer_address(), vc.buffer_address(), pos.buffer_address(), pt.buffer_address()]
            wr_c += [list(kc.shape)[2]]
            wr_def = [("KV_CACHE", "1")]
            if kv_l1 is not None:
                buf, maxr, NG, cores = kv_l1
                wr_c += [buf.buffer_address(), maxr, NG]
                for c in cores:
                    w = device.worker_core_from_logical_core(c)
                    wr_c.append((w.x << 16) | w.y)
                wr_def = wr_def + [("KV_L1", "1")]
        elif kv_out is not None:
            sh = device.worker_core_from_logical_core(out.memory_config().shard_spec.grid.ranges()[0].start)
            wr_ct += ttnn.TensorAccessorArgs(v if v is not None else out).get_compile_time_args()
            wr_c += [sh.x, sh.y, out.buffer_address()]
            wr_c += [v_sh.buffer_address(), v.buffer_address()] if v is not None else [0, 0]
            wr_def = [("TOKEN_MAJOR", "1")]
            if q_hub is not None:  # out: the SDPA's q hubs (one row block per kv group, at the same L1 address)
                for c in q_hub:
                    w = device.worker_core_from_logical_core(c)
                    wr_c.append((w.x << 16) | w.y)
                wr_def.append(("Q_HUB", "1"))
        wr_kernel = ttnn.KernelDescriptor(
            kernel_source=KERNELS + "qkr_writer.cpp",
            core_ranges=crs,
            compile_time_args=wr_ct,
            common_runtime_args=wr_c,
            defines=wr_def + off,
            config=ttnn.WriterConfigDescriptor(),
        )
        kernels, cbs = (
            [
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
                    + [H, Mt, Dt, Rt]
                    + ([v.buffer_address()] if cache is not None else [])
                    + [self.c_off],  # fmt: skip
                    defines=([("KV_CACHE", "1")] if cache is not None else []) + off,
                    config=ttnn.ReaderConfigDescriptor(),
                ),
                wr_kernel,
                ttnn.KernelDescriptor(
                    kernel_source=KERNELS + "qkr_compute.cpp",
                    core_ranges=crs,
                    common_runtime_args=span + [Dt, _f32_bits(1.0 / D), _f32_bits(self.eps)],
                    defines=([("KV_CACHE", "1")] if cache is not None else []) + off,
                    config=ttnn.ComputeConfigDescriptor(
                        math_fidelity=ttnn.MathFidelity.HiFi4,
                        fp32_dest_acc_en=True,
                        math_approx_mode=False,
                        dst_full_sync_en=True,
                    ),
                ),
            ],
            [cb(0, bf, 2 * Dt), cb(1, f32, 1 + Dt), cb(2, bf, 8), cb(3, f32, Dt), cb(4, f32, 1)]
            + ([cb(16, ttnn.bfloat8_b, Dt), cb(5, bf, Dt), cb(17, bf, 1)] if cache is not None else [cb(16, bf, 8)])
            + ([cb(17, bf, 1)] if kv_out is not None else []),
        )
        if cache is not None:
            ios = [x, self.consts, cos, sin, v, pos, pt, vc, kc]
        else:
            ios = [x, self.consts, cos, sin] + ([v, v_sh] if kv_out is not None and v is not None else []) + [out]
        if parts:
            return kernels, cbs, ios
        ttnn.generic_op(ios, ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs))
        return None if cache is not None else out


_KV_SH = {}


def kv_shards(device, H, D, n=2):
    """Persistent token-major K and V decode buffers ([1, 1, H, D] bf16, one tile row, height-sharded on core (0, 0), as the paged
    cache update takes them), zero once: the norm + RoPE op writes only the head rows, the padding rows stay zero."""
    key = (H, D, n, core_xy(0, 1).x, core_xy(0, 1).y)  # the shard core follows the core layout
    if key not in _KV_SH:
        mc = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet({ttnn.CoreRange(core_xy(0, 1), core_xy(0, 1))}),
                [32, D],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        )
        _KV_SH[key] = tuple(
            ttnn.from_torch(
                torch.zeros(1, 1, H, D), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mc
            )
            for _ in range(n)
        )
    return _KV_SH[key]
