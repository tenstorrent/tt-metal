# SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
# SPDX-License-Identifier: Apache-2.0
"""Prefill message-gate tail over T rows in one program: the part of prefill_model._Gate after the down_proj.

The op chain: typecast(z) to fp32, two rank-64 fp32 matmuls to [T, C] fp32, two tanh, two addcmul: ~8 passes
over [T, 2304] fp32. Here, per output tile (R, j): both projections (FPU, fp32 dest: the same tf32 operand
values and accumulation as ttnn.matmul), tanh_tile, and addcmul_tile twice, as the chain does them. Bit-equal.

Call: PfGateTail(device)(z, forget, gain, h, m) with z = the bf16 down_proj output [1,1,T,64] (no typecast),
forget / gain [1,1,64,C] fp32, h / m [1,1,T,C] fp32 -> h - tanh(z F) h + tanh(z G) m, fp32 [1,1,T,C].
"""

import struct

import ttnn
from models.demos.inkling2_mini.tt.fused import common as F

RANK = 64


def _bits(v):
    return struct.unpack("<I", struct.pack("<f", v))[0]


class PfGateTail:
    def __init__(self, device):
        self.device = device

    def __call__(self, z, forget, gain, h, m, out_dtype=ttnn.float32):
        """m may be bf16 and out_dtype bf16 (opt-in bf16 gate path: changes output); default all fp32."""
        t, c = h.shape[-2], h.shape[-1]
        kz = RANK // 32
        assert z.dtype == ttnn.bfloat16 and tuple(z.shape)[-2:] == (t, RANK), z.shape
        assert tuple(forget.shape)[-2:] == (RANK, c) and tuple(gain.shape)[-2:] == (RANK, c), forget.shape
        for a in (z, forget, gain, h, m):
            assert a.layout == ttnn.TILE_LAYOUT and a.memory_config().buffer_type == ttnn.BufferType.DRAM, a
        for a in (forget, gain, h):
            assert a.dtype == ttnn.float32, a.dtype
        assert m.dtype in (ttnn.float32, ttnn.bfloat16), m.dtype
        dram = ttnn.DRAM_MEMORY_CONFIG
        out = ttnn.allocate_tensor_on_device(ttnn.Shape([1, 1, t, c]), out_dtype, ttnn.TILE_LAYOUT, self.device, dram)
        wt, rows = c // 32, t // 32
        grid = self.device.compute_with_storage_grid_size()
        per = F.balanced_runs(wt, rows, grid.x * grid.y)  # units (j, R0, R1), the same tile count per core
        cores, crs = F.core_list(self.device, len(per))
        r_args, w_args, c_args = {}, {}, {}
        for core, us in zip(cores, per):
            flat = [v for u in us for v in u]
            r_args[core] = [z.buffer_address(), forget.buffer_address(), gain.buffer_address(), h.buffer_address()]
            r_args[core] += [m.buffer_address(), len(us)] + flat
            w_args[core] = [out.buffer_address(), F.tile_bytes(out_dtype), len(us)] + flat
            c_args[core] = [len(us)] + [r1 - r0 for _, r0, r1 in us]
        named = dict(wt=wt, kz=kz, minus_one=_bits(-1.0), one=_bits(1.0), m_page=F.tile_bytes(m.dtype))
        named = F.fixed_rt(named, r_args, w_args, c_args)
        kernels = [
            F.kernel("pf_gate_reader.cpp", ttnn.ReaderConfigDescriptor(), named, r_args, crs),
            F.kernel("pf_conv_writer.cpp", ttnn.WriterConfigDescriptor(), named, w_args, crs),
            F.kernel("pf_gate_compute.cpp", F.compute_config([3, 4], full_sync=True), named, c_args, crs),
        ]
        bf, fp = ttnn.bfloat16, ttnn.float32
        cbs = [F.cb(0, bf, 2 * kz, cores=crs), F.cb(1, fp, kz, cores=crs), F.cb(2, fp, kz, cores=crs),
               F.cb(3, fp, 2, cores=crs), F.cb(4, m.dtype, 2, cores=crs), F.cb(16, out_dtype, 2, cores=crs)]  # fmt: skip
        return F.run([z, forget, gain, h, m, out], kernels, cbs)
