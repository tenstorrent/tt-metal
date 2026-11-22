# SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
# SPDX-License-Identifier: Apache-2.0
"""Fused static short conv (k=4): one program per call instead of 6-7 ttnn ops (E2 residual.ShortConv).

Drop-in for residual.ShortConv: same ctor, alloc_state(c), __call__(x, ring, phase). The ring slots are dense
fp32 pages (see common.py) instead of rows; reset by copying zeros into them works as before.
conv_add() also folds the layer glue around the conv: out = [resid +] conv(x), in one program.
"""

import math

import torch

import ttnn
from models.demos.inkling2_mini.tt.fused import common as F

NUM_CACHE_SETS = 48


class FusedShortConv:
    PHASES = 4

    def __init__(self, rows, weights, name):
        w = weights[name]
        t = (ttnn.to_torch(w) if isinstance(w, ttnn.Tensor) else w).float().reshape(4, -1).clone()
        t[0] += 1  # fold the residual into the lag-0 tap
        t /= math.sqrt(5)
        self.device = rows.device
        self.c = t.shape[-1]
        assert self.c % 32 == 0
        self.T = F.pages(self.c)
        self.taps = F.dense_tensor(self.device, t)  # 4 * T pages, lag k at pages k*T ..
        self.out = F.OutCache(self.device)

    def alloc_state(self, c):
        """Zero ring [4 x dense fp32 vector] for HF cache index c."""
        assert 0 <= c < NUM_CACHE_SETS, c
        return [F.dense_tensor(self.device, torch.zeros(1, self.c)) for _ in range(self.PHASES)]

    def __call__(self, x, ring, phase):
        """x: row at position t, phase = t % 4. Writes fp32 x into ring[phase]; returns the output row in x's dtype."""
        return self.conv_add(x, ring, phase)

    def conv_add(self, x, ring, phase, resid=None, out_dtype=None, row_out=True):
        """out = conv(x) rounded to x's dtype, plus resid (fp32 add) if given, in out_dtype (default: x's dtype;
        with resid: resid's dtype)."""
        F.check_row_input(x)
        if resid is not None:
            F.check_row_input(resid)
        out_dtype = out_dtype or (resid.dtype if resid is not None else x.dtype)
        out, out_dense = self.out.get_twin(("conv", out_dtype), 1, self.c, out_dtype)
        T, ntc = self.T, self.c // 32
        stage2 = int(resid is not None or out_dtype != x.dtype)
        slots = [ring[(phase - k) % 4] for k in range(4)]
        res = resid if resid is not None else x
        in_args = (
            F.input_args(x) + [s.buffer_address() for s in slots[1:]] + [self.taps.buffer_address()] + F.input_args(res)
        )
        reader = F.kernel("conv_reader.cpp", ttnn.ReaderConfigDescriptor(), dict(ntc=ntc, pages=T), in_args[:5])
        cb_out = 17
        writer = F.kernel(
            "conv_writer.cpp",
            ttnn.WriterConfigDescriptor(),
            # row_out 0: dense twin only (fused consumers)
            dict(ntc=ntc, pages=T, cb_out=cb_out, row_out=int(row_out), has_resid=int(resid is not None)),
            [
                slots[0].buffer_address(),
                out.buffer_address(),
                F.tile_bytes(out_dtype),
                F.ES[out_dtype],
                out_dense.buffer_address(),
            ]
            + in_args,
        )
        x_in, res_in = F.input_dtype(x), F.input_dtype(res)
        fp32_cbs = [1, 2, 16] + ([0] if x_in == ttnn.float32 else []) + ([3] if res_in == ttnn.float32 else [])
        compute = F.kernel(
            "conv_compute.cpp",
            F.compute_config(fp32_cbs + ([18] if x.dtype == ttnn.float32 else [])),
            dict(pages=T, has_resid=int(resid is not None), stage2=stage2),
        )
        cbs = [
            F.cb(0, x_in, T),
            F.cb(1, ttnn.float32, 3 * T),
            F.cb(2, ttnn.float32, 4 * T),
            F.cb(3, res_in, T),
            F.cb(16, ttnn.float32, T),
            F.cb(17, out_dtype, T),
            F.cb(18, x.dtype, T),
            F.stage_cb(24, ntc),
            F.cb(25, ttnn.bfloat16, 1, 32),  # writer -> reader: second half of x landed
            F.stage_cb(26, ntc),
        ]
        tws = [t[0] for t in (F.twin(x), F.twin(res)) if t is not None]
        io = [x, res, *tws, self.taps, *slots, out_dense, out]
        F.run(io, [reader, writer, compute], cbs)
        return out  # the cached object itself: its dense twin is keyed by identity
