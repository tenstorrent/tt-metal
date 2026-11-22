# SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
# SPDX-License-Identifier: Apache-2.0
"""Prefill causal k=4 short conv over T rows in one program: drop-in for Inkling2Prefill._conv (+ its two uses).

The op chain: tail window (concat [32-row tail ++ chunk] + 3 unaligned row slices), 4 multiplies, 3 adds, typecast,
and at the call sites a typecast back to fp32 (post-attn conv) or a residual add + typecast (post-mlp conv): ~10
passes over [T, 2304]. Here, per output tile (R, j): the lagged rows come from two 0/1 shift-matrix matmuls of the
x tile and the tile above it (exact: one bf16 term per element), then the taps (row-broadcast, exact fp32) in the
chain's order. Modes: "bf16" (= _conv), "f32" (= _f32(_conv(..))), "resid" (= bf16(resid + _f32(_conv(..)))),
"hres" (= ttnn.add(_f32(resid), _f32(_conv(..))), fp32; resid bf16 or fp32: the post-attn residual).

Make the PfConv in Inkling2Prefill.__init__ (its shift constants are device tensors; the trace rule).
"""

import torch

import ttnn
from models.demos.inkling2_mini.tt.fused import common as F

HIDDEN = 2304
MODES = {"bf16": 0, "f32": 1, "resid": 2, "hres": 3}


def _shift_tiles():
    """[6 * 32, 32]: A_1..A_3 (row i <- row i - r of the same tile), B_1..B_3 (row i < r <- row 32 + i - r of the
    tile above)."""
    m = torch.zeros(6, 32, 32)
    for r in range(1, 4):
        for i in range(32):
            if i >= r:
                m[r - 1, i, i - r] = 1.0
            else:
                m[2 + r, i, 32 + i - r] = 1.0
    return m.reshape(6 * 32, 32)


class PfConv:
    def __init__(self, device):
        self.device = device
        self.shifts = ttnn.from_torch(
            _shift_tiles().view(1, 1, 6 * 32, 32), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
        )

    def __call__(self, taps, x, tail, mode="bf16", resid=None):
        """taps: 4 x [1,1,1,C] fp32 (lag 0..3); x [1,1,T,C] (bf16 or fp32, rounded to bf16 as _Tail.window does);
        tail: prefill_model._Tail (advanced here, and its `keep` hook honoured). Returns per `mode` (module doc)."""
        x = x if x.dtype == ttnn.bfloat16 else ttnn.typecast(x, ttnn.bfloat16)
        t, c = x.shape[-2], x.shape[-1]
        assert t % 32 == 0 and t >= 32 and c % 32 == 0, x.shape
        assert (mode in ("resid", "hres")) == (resid is not None), mode
        dram = ttnn.DRAM_MEMORY_CONFIG
        for a in [x, tail.t] + list(taps) + ([resid] if resid is not None else []):
            assert a.layout == ttnn.TILE_LAYOUT and a.memory_config().buffer_type == ttnn.BufferType.DRAM, a
        out_dtype = ttnn.float32 if mode in ("f32", "hres") else ttnn.bfloat16
        out = ttnn.allocate_tensor_on_device(ttnn.Shape([1, 1, t, c]), out_dtype, ttnn.TILE_LAYOUT, self.device, dram)
        wt, rows = c // 32, t // 32
        grid = self.device.compute_with_storage_grid_size()
        n_cores = grid.x * grid.y
        split = max(1, -(-n_cores // wt))  # row blocks per column so that units >= cores
        blk = -(-rows // split)
        units = [(j, r0, min(rows, r0 + blk)) for j in range(wt) for r0 in range(0, rows, blk)]
        n_cores = min(n_cores, len(units))
        cores, crs = F.core_list(self.device, n_cores)
        per = [units[i::n_cores] for i in range(n_cores)]
        res_addr = resid.buffer_address() if resid is not None else 0
        r_args, w_args, c_args = {}, {}, {}
        page = 4096 if out_dtype == ttnn.float32 else 2048
        for core, us in zip(cores, per):
            flat = [v for u in us for v in u]
            r_args[core] = [x.buffer_address(), tail.t.buffer_address()] + [p.buffer_address() for p in taps]
            r_args[core] += [self.shifts.buffer_address(), res_addr, len(us)] + flat
            w_args[core] = [out.buffer_address(), page, len(us)] + flat
            c_args[core] = [len(us)] + [r1 - r0 for _, r0, r1 in us]
        res_page = 4096 if resid is None or resid.dtype == ttnn.float32 else 2048
        named = dict(wt=wt, has_resid=int(resid is not None), res_page=res_page, mode=MODES[mode])
        named = F.fixed_rt(named, r_args, w_args, c_args)
        kernels = [
            F.kernel("pf_conv_reader.cpp", ttnn.ReaderConfigDescriptor(), named, r_args, crs),
            F.kernel("pf_conv_writer.cpp", ttnn.WriterConfigDescriptor(), named, w_args, crs),
            F.kernel("pf_conv_compute.cpp", F.compute_config([2, 4], full_sync=True),
                     named, c_args, crs),
        ]  # fmt: skip
        bf, fp = ttnn.bfloat16, ttnn.float32
        rd = resid.dtype if resid is not None else fp
        cbs = [F.cb(0, bf, 2, cores=crs), F.cb(1, bf, 2, cores=crs), F.cb(2, fp, 4, cores=crs), F.cb(3, bf, 6, cores=crs),
               F.cb(4, rd, 2, cores=crs), F.cb(16, out_dtype, 2, cores=crs), F.cb(17, bf, 2, cores=crs)]  # fmt: skip
        io = [x, tail.t, *taps, self.shifts] + ([resid] if resid is not None else []) + [out]
        F.run(io, kernels, cbs)
        # advance the tail as _Tail.window does: new tail = rows [t, t + 32) of [old tail ++ x] = last 32 rows of x
        keep = getattr(tail, "keep", None)
        if keep is not None:  # last chunk: the conv-history handoff reads the last 160 rows of [old tail ++ chunk]
            keep[0][keep[1]] = (
                ttnn.concat([tail.t, x], dim=2) if t + 32 <= 160 else ttnn.slice(x, [0, 0, t - 160, 0], [1, 1, t, c])
            )
        tail.t = ttnn.slice(x, [0, 0, t - 32, 0], [1, 1, t, c])
        return out
