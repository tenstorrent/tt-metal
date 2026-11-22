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

    def __call__(self, z, forget, gain, h, m):
        t, c = h.shape[-2], h.shape[-1]
        kz = RANK // 32
        assert z.dtype == ttnn.bfloat16 and tuple(z.shape)[-2:] == (t, RANK), z.shape
        assert tuple(forget.shape)[-2:] == (RANK, c) and tuple(gain.shape)[-2:] == (RANK, c), forget.shape
        for a in (z, forget, gain, h, m):
            assert a.layout == ttnn.TILE_LAYOUT and a.memory_config().buffer_type == ttnn.BufferType.DRAM, a
        for a in (forget, gain, h, m):
            assert a.dtype == ttnn.float32, a.dtype
        dram = ttnn.DRAM_MEMORY_CONFIG
        out = ttnn.allocate_tensor_on_device(
            ttnn.Shape([1, 1, t, c]), ttnn.float32, ttnn.TILE_LAYOUT, self.device, dram
        )
        wt, rows = c // 32, t // 32
        grid = self.device.compute_with_storage_grid_size()
        n_cores = grid.x * grid.y
        split = max(1, -(-4 * n_cores // wt))  # ~4 units per core: balance vs the per-unit F / G reads
        blk = -(-rows // min(split, rows))
        units = [(j, r0, min(rows, r0 + blk)) for j in range(wt) for r0 in range(0, rows, blk)]
        n_cores = min(n_cores, len(units))
        cores, crs = F.core_list(self.device, n_cores)
        per = [units[i::n_cores] for i in range(n_cores)]
        r_args, w_args, c_args = {}, {}, {}
        for core, us in zip(cores, per):
            flat = [v for u in us for v in u]
            r_args[core] = [z.buffer_address(), forget.buffer_address(), gain.buffer_address(), h.buffer_address()]
            r_args[core] += [m.buffer_address(), len(us)] + flat
            w_args[core] = [out.buffer_address(), 4096, len(us)] + flat
            c_args[core] = [len(us)] + [r1 - r0 for _, r0, r1 in us]
        named = dict(wt=wt, kz=kz, minus_one=_bits(-1.0), one=_bits(1.0))
        named = F.fixed_rt(named, r_args, w_args, c_args)
        kernels = [
            F.kernel("pf_gate_reader.cpp", ttnn.ReaderConfigDescriptor(), named, r_args, crs),
            F.kernel("pf_conv_writer.cpp", ttnn.WriterConfigDescriptor(), named, w_args, crs),
            F.kernel("pf_gate_compute.cpp", F.compute_config([3, 4], full_sync=True), named, c_args, crs),
        ]
        bf, fp = ttnn.bfloat16, ttnn.float32
        cbs = [F.cb(0, bf, 2 * kz, cores=crs), F.cb(1, fp, kz, cores=crs), F.cb(2, fp, kz, cores=crs),
               F.cb(3, fp, 2, cores=crs), F.cb(4, fp, 2, cores=crs), F.cb(16, fp, 2, cores=crs)]  # fmt: skip
        return F.run([z, forget, gain, h, m, out], kernels, cbs)
