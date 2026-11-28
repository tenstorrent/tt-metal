# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Decode SDPA for Bonsai's full attention (B = 1, 24 q heads, 4 kv heads, head_dim 256, bf8 paged K / V caches) in one
op on H * NG cores (kernels/sdpa_*.cpp), replacing paged_scaled_dot_product_attention_decode.

Core (g, c) takes kv head g's c-th share of the position tile rows: S = Q K^T for its group's 6 q heads (rows 0 .. 5),
then the group's 28 cores swap their row maxima (one NOC hop, no reduction tree) so all use the global maxima M:
P = exp(scale (S - M)), O = P V and l = P 1 sum directly across cores. Combiner d of the group (its first NC = 8 cores,
one per 32-column output tile) stacks the 28 partial row blocks (6 rows each, 5 per tile) and sums them with one
constant row-selection matmul per stacked tile, then out = O / l, into rows 6g .. 6g + 5 of the token-major output."""

import os
import struct

import torch

import ttnn

KERNELS = "models/demos/blackhole/bonsai2/tt/sdpa_dec/kernels/"


def _bf16_bits(v):
    return struct.unpack("<I", struct.pack("<f", float(v)))[0] >> 16


class DecodeSDPA:
    def __init__(self, device, cores, H=4, Hq=24, D=256, max_positions=4608, scale=None, maxr=None):
        assert Hq == 6 * H and D == 256, (H, Hq, D)
        assert len(cores) % H == 0, len(cores)
        self.device, self.cores, self.H, self.Hq, self.D = device, list(cores), H, Hq, D
        self.NG = len(cores) // H
        assert 8 <= self.NG, self.NG
        self.maxr = maxr or -(-(-(-max_positions // 32)) // self.NG)
        self.scale = D**-0.5 if scale is None else scale
        ones = torch.ones(32, 32)
        scaler = torch.zeros(32, 32)
        scaler[0, :] = 1
        scaler[16, :] = 1
        sel = torch.zeros(32, 32)
        for r in range(6):
            for k in range(5):
                sel[r, 6 * k + r] = 1
        self.consts = ttnn.from_torch(
            torch.cat([ones, scaler, sel], 0), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
        )

    def __call__(self, q, kc, vc, pos, pt, out=None, pf=None, q_hubs=None):
        """q token-major [1, 1, Hq, D] bf16 TILE (L1); kc, vc [blocks, H, bs, D] bf8; pos [1] int32 (the current position,
        its K / V already in the caches); pt [1, max_blocks] int32 -> token-major [1, 1, Hq, D] bf16 (L1).
        pf = (K / V buffer, record): K / V already prefetched into each core's L1 (kernels/kv_prefetch.hpp; its maxr,
        these cores in this order), no DRAM reads. q_hubs: q is kv group g's 6 heads in rows 0 .. 5 on core q_hubs[g]
        (qk_rope.py q_hub) instead of token-major."""
        device, H, NG, D = self.device, self.H, self.NG, self.D
        bs = list(kc.shape)[2]
        if out is None:
            out = ttnn.allocate_tensor_on_device(
                ttnn.Shape([1, 1, self.Hq, D]), ttnn.bfloat16, ttnn.TILE_LAYOUT, device, ttnn.L1_MEMORY_CONFIG
            )
        crs = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in self.cores])
        xy = []
        for c in self.cores:
            w = device.worker_core_from_logical_core(c)
            xy.append((w.x << 16) | w.y)
        rt_r, rt_w, rt_c = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        for i, c in enumerate(self.cores):
            rt_r[c.x][c.y] = [i // NG, i % NG]
            rt_w[c.x][c.y] = [i // NG, i % NG]
            rt_c[c.x][c.y] = [i // NG, i % NG]
        pt_bytes = list(pt.shape)[-1] * 4
        rd_ct = []
        for t in (pos, pt, kc, vc, q, self.consts):
            rd_ct += ttnn.TensorAccessorArgs(t).get_compile_time_args()
        nst = (NG + 4) // 5
        maxr = self.maxr
        bf, b8, f32 = ttnn.bfloat16, ttnn.bfloat8_b, ttnn.float32

        def cb(index, fmt, pages, page=None):
            page = page or {f32: 4096, b8: 1088}.get(fmt, 2048)
            return ttnn.CBDescriptor(
                total_size=page * pages,
                core_ranges=crs,
                format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=fmt, page_size=page)],
            )

        if pf is not None:
            kvb = maxr * 8 * 1088
            kv_cbs = [
                ttnn.cb_descriptor_from_sharded_tensor(1, pf[0], address_offset=0, total_size=kvb, core_ranges=crs),
                ttnn.cb_descriptor_from_sharded_tensor(2, pf[0], address_offset=kvb, total_size=kvb, core_ranges=crs),
            ]
        else:
            kv_cbs = [cb(1, b8, maxr * 8), cb(2, b8, maxr * 8)]
        cbs = kv_cbs + [
            cb(0, bf, 8),
            cb(3, bf, 3),
            cb(5, bf, 1, 64),
            cb(6, bf, 1, 128 + ((pt_bytes + 63) // 64) * 64),
            cb(7, bf, 1, 64),
            cb(8, f32, 1),
            cb(9, bf, 2 * nst),
            cb(10, bf, 1, (NG + 1) * 32),
            cb(11, bf, 1),
            cb(16, f32, maxr),
            cb(17, f32, 1),
            cb(18, bf, 9),
            cb(19, bf, 1),
            cb(20, bf, maxr),
        ]
        hub_xy = []
        for c in q_hubs or []:
            w = device.worker_core_from_logical_core(c)
            hub_xy.append((w.x << 16) | w.y)
        zones = [("SDPA_ZONES", "1")] if os.environ.get("SDPA_ZONES") == "1" else []
        program = ttnn.ProgramDescriptor(
            kernels=[
                ttnn.KernelDescriptor(
                    kernel_source=KERNELS + "sdpa_reader.cpp",
                    core_ranges=crs,
                    compile_time_args=rd_ct,
                    common_runtime_args=[
                        pos.buffer_address(),
                        pt.buffer_address(),
                        kc.buffer_address(),
                        vc.buffer_address(),
                        q.buffer_address(),
                        self.consts.buffer_address(),
                        bs,
                        pt_bytes,
                        NG,
                        H,
                        pf[1].buffer_address() if pf is not None else 0,
                    ]
                    + hub_xy,
                    runtime_args=rt_r,
                    defines=zones + ([("KV_PF", "1")] if pf is not None else []) + ([("Q_HUB", "1")] if q_hubs else []),
                    config=ttnn.ReaderConfigDescriptor(),
                ),
                ttnn.KernelDescriptor(
                    kernel_source=KERNELS + "sdpa_writer.cpp",
                    core_ranges=crs,
                    compile_time_args=ttnn.TensorAccessorArgs(out).get_compile_time_args(),
                    common_runtime_args=[out.buffer_address(), NG, 8] + xy,
                    runtime_args=rt_w,
                    defines=zones,
                    config=ttnn.WriterConfigDescriptor(),
                ),
                ttnn.KernelDescriptor(
                    kernel_source=KERNELS + "sdpa_compute.cpp",
                    core_ranges=crs,
                    common_runtime_args=[8, nst],
                    runtime_args=rt_c,
                    defines=[
                        ("SCALE_BF16", str(_bf16_bits(self.scale))),
                        ("EXP_ITERS", os.environ.get("SDPA_DEC_EXP_ITERS", "4")),
                    ],
                    config=ttnn.ComputeConfigDescriptor(
                        math_fidelity=getattr(ttnn.MathFidelity, os.environ.get("SDPA_DEC_FID", "HiFi4")),
                        fp32_dest_acc_en=True,
                        math_approx_mode=False,
                        dst_full_sync_en=True,
                    ),
                ),
            ],
            semaphores=[
                ttnn.SemaphoreDescriptor(id=0, core_ranges=crs, initial_value=0),
                ttnn.SemaphoreDescriptor(id=1, core_ranges=crs, initial_value=0),
            ],
            cbs=cbs,
        )
        extra = list(pf) if pf is not None else []
        ttnn.generic_op([pos, pt, kc, vc, q, self.consts] + extra + [out], program)
        return out
