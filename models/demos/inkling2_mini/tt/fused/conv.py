# SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
# SPDX-License-Identifier: Apache-2.0
"""Fused static short conv (k=4): one program per call instead of 6-7 ttnn ops.

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
        self.pending = {}  # (ring, verify row) -> fp32 x of that row (pending_slot)

    def alloc_state(self, c):
        """Zero ring [4 x dense fp32 vector] for HF cache index c."""
        assert 0 <= c < NUM_CACHE_SETS, c
        return [F.dense_tensor(self.device, torch.zeros(1, self.c)) for _ in range(self.PHASES)]

    def pending_slot(self, ring, r):
        """Verify row r's fp32 x of the conv with this ring (conv_add(slot=...)), kept until commit."""
        key = (id(ring[0]), r)
        if key not in self.pending:
            self.pending[key] = F.dense_tensor(self.device, torch.zeros(1, self.c))
        return self.pending[key]

    def __call__(self, x, ring, phase):
        """x: row at position t, phase = t % 4. Writes fp32 x into ring[phase]; returns the output row in x's dtype."""
        return self.conv_add(x, ring, phase)

    def conv_add(self, x, ring, phase, resid=None, out_dtype=None, row_out=True, x_row=0, slot=None, signal=None):
        """out = conv(x) rounded to x's dtype, plus resid (fp32 add) if given, in out_dtype (default: x's dtype;
        with resid: resid's dtype).

        Verify row r of an R-row step (inside F.per_row, phase = (pos + r) % 4): x_row = r picks row r of an R-row x;
        lags 1..min(r, 3) are rows r-1.. of x itself (the ring does not hold them yet), older lags come from the
        ring, and fp32 x goes to `slot` (pending_slot(); commit() puts accepted rows into the ring) instead of
        ring[phase]: the ring is not written, so every row reads it as it was before the step.

        signal: (noc x, noc y) of a core in the same joined program (common.py joined) whose semaphore the writer
        increments once the output has landed."""
        F.check_row_input(x, l1_ok=True)  # the MoE output y straight from L1
        if resid is not None:
            F.check_row_input(resid)
        out_dtype = out_dtype or (resid.dtype if resid is not None else x.dtype)
        out, out_dense = self.out.get_twin(("conv", out_dtype), 1, self.c, out_dtype)
        T, ntc = self.T, self.c // 32
        stage2 = int(resid is not None or out_dtype != x.dtype)
        slots = [ring[(phase - k) % 4] for k in range(4)]
        if slot is not None:
            slots[0] = slot
        own = min(x_row, 3)
        res = resid if resid is not None else x
        res_row = x_row if resid is None else F.row_index() if resid.shape[-2] > 1 else 0
        in_args = (
            F.input_args(x, x_row)
            + [s.buffer_address() for s in slots[1:]]
            + [self.taps.buffer_address()]
            + F.input_args(res, res_row)
        )
        prev = [a for k in range(1, own + 1) for a in F.input_args(x, x_row - k)]
        cb_out = 17
        reader = F.kernel(
            "conv_reader.cpp",
            ttnn.ReaderConfigDescriptor(),
            dict(ntc=ntc, pages=T, row_out=int(row_out), cb_out=cb_out, own=own),
            in_args[:9] + [out.buffer_address(), F.tile_bytes(out_dtype), F.ES[out_dtype]] + prev,
        )
        writer = F.kernel(
            "conv_writer.cpp",
            ttnn.WriterConfigDescriptor(),
            # row_out 0: dense twin only (fused consumers)
            dict(
                ntc=ntc,
                pages=T,
                cb_out=cb_out,
                row_out=int(row_out),
                has_resid=int(resid is not None),
                sig=int(signal is not None),
                sig_x=signal[0] if signal else 0,
                sig_y=signal[1] if signal else 0,
            ),
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
        fp32_cbs += [4] if x_in == ttnn.float32 else []  # the own lags unpack as x does
        compute = F.kernel(
            "conv_compute.cpp",
            F.compute_config(fp32_cbs + ([18] if x.dtype == ttnn.float32 else [])),
            dict(pages=T, has_resid=int(resid is not None), stage2=stage2, tail=self.c - (T - 1) * F.PAGE, own=own),
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
        if own:
            cbs.append(F.cb(4, x_in, own * T))
        tws = [t[0] for t in (F.twin(x), F.twin(res)) if t is not None]
        io = [x, res, *tws, self.taps, *slots, out_dense, out]
        F.run(io, [reader, writer, compute], cbs)
        return out  # the cached object itself: its dense twin is keyed by identity


COMMIT_CORES = 8


def commit(device, pairs, pages):
    """Copy dense vectors src -> dst ([(src, dst)] of `pages` fp32 pages each) in one program over COMMIT_CORES
    cores: after a verify step, each accepted row's pending conv input into its ring slot."""
    cores, crs = F.core_list(device, min(COMMIT_CORES, len(pairs)))
    rt = {c: [0] for c in cores}
    for i, (src, dst) in enumerate(pairs):
        a = rt[cores[i % len(cores)]]
        a[0] += 1
        a += [src.buffer_address(), dst.buffer_address()]
    named = F.fixed_rt(dict(pages=pages), rt)
    k = F.kernel("rows_commit.cpp", ttnn.ReaderConfigDescriptor(), named, rt, crs)
    io = [t for p in pairs for t in p]
    F.run(io, [k], [F.cb(0, ttnn.float32, pages, cores=crs)])
