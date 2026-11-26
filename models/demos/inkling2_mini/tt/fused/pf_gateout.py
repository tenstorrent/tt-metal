# SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
# SPDX-License-Identifier: Apache-2.0
"""Prefill attention output gate over T rows in one program: [1, n, T, 128] SDPA heads x sigmoid(gate rows) ->
[1, 1, T, n * 128] bf16 rows (the o-projection input).

Replaces nlp_concat_heads -> typecast fp32 -> (slice of the gate columns -> typecast fp32 -> sigmoid) -> multiply ->
typecast bf16. Same math and rounding (fp32 sigmoid and product, one bf16 rounding); the head merge is a tile-index
remap. Gate columns are read in place from the projection rows: tile column g0 / 32 + 4 h + c.
"""

import ttnn
from models.demos.inkling2_mini.tt.fused import common as F

HEAD_DIM = 128


class PfGateOut:
    def __init__(self, device):
        self.device = device

    def __call__(self, attn, gate_rows, g0):
        """attn [1, n, T, 128] bf16; gate_rows [1, 1, T, W] bf16 (gate of head h: columns g0 + 128 h ..)."""
        n, t = attn.shape[1], attn.shape[2]
        assert attn.dtype == gate_rows.dtype == ttnn.bfloat16 and t % 32 == 0 and g0 % 32 == 0, (attn.shape, g0)
        rt, gt = t // 32, gate_rows.padded_shape[-1] // 32
        out = ttnn.allocate_tensor_on_device(ttnn.Shape([1, 1, t, n * HEAD_DIM]), ttnn.bfloat16, ttnn.TILE_LAYOUT,
                                             self.device, ttnn.DRAM_MEMORY_CONFIG)  # fmt: skip
        units = [(r, h) for r in range(rt) for h in range(n)]
        grid = self.device.compute_with_storage_grid_size()
        n_cores = min(len(units), grid.x * grid.y)
        cores, crs = F.core_list(self.device, n_cores)
        per = [units[i::n_cores] for i in range(n_cores)]
        r_args, w_args, c_args = {}, {}, {}
        for core, us in zip(cores, per):
            flat = [v for u in us for v in u]
            r_args[core] = [attn.buffer_address(), gate_rows.buffer_address(), len(us)] + flat
            w_args[core] = [out.buffer_address(), len(us)] + flat
            c_args[core] = [len(us)]
        named = F.fixed_rt(dict(rt=rt, gt=gt, g0=g0 // 32, ot=n * 4), r_args, w_args, c_args)
        kernels = [
            F.kernel("pf_gateout_reader.cpp", ttnn.ReaderConfigDescriptor(), named, r_args, crs),
            F.kernel("pf_gateout_writer.cpp", ttnn.WriterConfigDescriptor(), named, w_args, crs),
            F.kernel("pf_gateout_compute.cpp", F.compute_config([]), named, c_args, crs),
        ]
        bf = ttnn.bfloat16
        cbs = [F.cb(0, bf, 8, cores=crs), F.cb(1, bf, 8, cores=crs), F.cb(16, bf, 8, cores=crs)]
        F.run([attn, gate_rows, out], kernels, cbs)
        return out
