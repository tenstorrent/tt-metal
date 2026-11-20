# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Page-level host model of the fused iHC site: what kernels/ihc_site.cpp leaves in every page, computed in torch
on the same page matrices and SFPU units (layout.py), core by core. No device.

It is the kernels' arithmetic written a second time, not a reference: reference/ihc.py is the reference. Its use
is tests/test_ihc_site_host.py, which runs the device cases of tests/test_ihc_site.py against it, so that the
layouts, the decoders and the checks themselves are exercised without the board, for both SFPU lane layouts.
Sums follow the kernels' order where that is cheap (core order, lane sums); everything is rounded to fp32 once per
stage rather than once per instruction, so results agree with the device to rounding, not bit for bit.
"""

from __future__ import annotations

import torch

import ttnn
from models.experimental.hy4_preview.fused.ihc_site import layout
from models.experimental.hy4_preview.fused.ihc_site.op import IhcSite, SiteRef, SiteTable
from models.experimental.hy4_preview.reference.ihc import IHCConfig

PAGE = layout.TILE


class HostTensor:
    """A page matrix standing in for a device tensor."""

    def __init__(self, matrix: torch.Tensor, dtype):
        self.dtype = dtype
        self.m = matrix.to(torch.bfloat16).float() if dtype == ttnn.bfloat16 else matrix.float().clone()

    def store(self, matrix: torch.Tensor):
        self.m = matrix.to(torch.bfloat16).float() if self.dtype == ttnn.bfloat16 else matrix.float().clone()


def _cell(g, h, p):
    return 2 * (2 * h + g) + p


def _face(page, j):
    r, c = 16 * (j // 2), 16 * (j % 2)
    return page[r : r + 16, c : c + 16]


class HostSite(IhcSite):
    """IhcSite with torch page matrices for tensors and this module for programs."""

    def __init__(
        self,
        device=None,
        cfg: IHCConfig = None,
        num_cores=24,
        origin=(0, 0),
        y_dtype=ttnn.float32,
        f_dtype=ttnn.float32,
        lanes=None,
        block_unpack=True,
        root_exchange=None,
        low_unpack=False,
        mm32=False,
        sfpu_proj=True,
        rotate=1,
    ):
        self.device = None
        self.cfg = cfg or IHCConfig()
        self.C, self.K = self.cfg.dim, num_cores
        assert self.C % (128 * num_cores) == 0
        self.px = self.C // 32 // num_cores
        self.py = self.px // 4
        self.y_dtype, self.f_dtype = y_dtype, f_dtype
        self.lanes = lanes or layout.SFPU_LANES
        self.root_exchange = num_cores >= 16 if root_exchange is None else root_exchange
        self.mm32 = mm32
        self.sfpu_proj = sfpu_proj
        self.rotate = rotate  # lane l of a rotated register holds lane (l + rotate) % 8
        self.coef = self._pages(torch.zeros(32, 32 * num_cores), ttnn.float32)
        self.stat = self._pages(torch.zeros(32, 32 * num_cores), ttnn.float32)
        self.dbg = None

    def _pages(self, matrix, dtype):
        return HostTensor(matrix, dtype)

    @staticmethod
    def _matrix(tensor):
        return tensor.m.clone()

    # ------------------------------------------------------------------ programs
    def _get(self, page, unit):
        return layout.unit_get(page, unit, self.lanes).double()

    def _set(self, page, unit, value):
        rows, cols = layout.unit_slices(unit, self.lanes)
        page[rows, cols] = value.to(page.dtype)

    def _post(self, X, F, coef):
        for b in range(self.C // 128):
            for k in range(4):
                t = 4 * b + k
                xp = X[:, 32 * t : 32 * t + 32]
                fp = F[:, 32 * b : 32 * b + 32]
                for g in range(2):
                    for h in range(2):
                        for p in range(2):
                            c = _cell(g, h, p)
                            f = self._get(fp, 8 * k + c)
                            for j in range(4):
                                x = self._get(xp, 8 * j + c)
                                self._set(xp, 8 * j + c, (f * self._get(coef, 2 * (4 + j) + g) + x).float())

    def site_table(self, sites) -> SiteTable:
        w_dtype = self._fn_pages(sites[0][0])[1]
        w = torch.cat([self._fn_pages(fn)[0] for fn, _, _ in sites], dim=1)
        const = torch.cat([layout.const_page(base, scale, self.cfg, self.lanes) for _, base, scale in sites], dim=1)
        return SiteTable(w=HostTensor(w, w_dtype), const=HostTensor(const, ttnn.float32), sites=len(sites))

    def _pre(self, X, params, y: HostTensor, debug: bool):
        K, px = self.K, self.px
        wc = (4 if self.sfpu_proj else 1) * self.C  # columns of one site's fn pages
        if isinstance(params, SiteRef):
            s, t = params.index, params.table
            W = t.w.m[:, s * wc : (s + 1) * wc].double()
            const = t.const.m[:, 32 * s : 32 * s + 32].double()
        else:
            W, const = params.w.m.double(), params.const.m[:, :32].double()
        scratch = [torch.zeros(32, 32, dtype=torch.float64) for _ in range(K)]
        for core in range(K):
            D = torch.zeros(16, 16, dtype=torch.float64)
            acc = [torch.zeros(4, 8, dtype=torch.float64) for _ in range(2)]
            pacc = [[torch.zeros(4, 8, dtype=torch.float64) for _ in range(2)] for _ in range(8)]  # [o][g]
            for t in range(core * px, (core + 1) * px):
                xp = X[:, 32 * t : 32 * t + 32].double()
                for j in range(4):
                    if self.sfpu_proj:
                        wp = W[:, 32 * (4 * t + j) : 32 * (4 * t + j) + 32]
                        for g in range(2):
                            for hp in range(4):
                                x = self._get(xp, 8 * j + _cell(g, hp >> 1, hp & 1))
                                for o in range(8):
                                    pacc[o][g] += x * self._get(wp, 8 * hp + o)
                    else:
                        wp = W[:, 32 * t : 32 * t + 32]
                        # mm32: the product's faces 0 and 3 are B00 A00 + B01 A10 and B10 A01 + B11 A11
                        D += _face(xp, j) @ _face(wp, (0, 2, 1, 3)[j] if self.mm32 else j)
                    for g in range(2):
                        for hp in range(4):
                            acc[g] += self._get(xp, 8 * j + _cell(g, hp >> 1, hp & 1)) ** 2
            s = scratch[core]
            s[:16, :16] = D.float()
            for g in range(2):
                self._set(s, 16 + g, acc[g].float())
                if self.sfpu_proj:
                    mix = torch.stack([pacc[o][g].sum(1) for o in range(8)], dim=1)  # [4 rows, 8 outputs]
                else:
                    mix = self._get(s, _cell(g, 0, 0)) + self._get(s, _cell(g, 1, 1))
                self._set(s, 8 + 2 * g, mix.float())
                self._set(s, 8 + 2 * g + 1, acc[g].float())
        coef = torch.zeros(32, 32)
        for g in range(2):
            m = sum(self._get(scratch[k], 8 + 2 * g) for k in range(K))
            ss = sum(self._get(scratch[k], 8 + 2 * g + 1) for k in range(K)).sum(1, keepdim=True).expand(4, 8)
            rs = torch.rsqrt(ss * self._get(const, layout.CONST_INV_N) + self._get(const, layout.CONST_NORM_EPS))
            mix = (m * rs).float().double()
            z = mix * self._get(const, layout.CONST_SCALE) + self._get(const, layout.CONST_BASE)
            gg = (torch.sigmoid(z) * self._get(const, layout.CONST_GAIN) + self._get(const, layout.CONST_EPS)).float()
            for core in range(K):
                self._set(scratch[core], 24 + g, mix.float())
                self._set(scratch[core], 26 + g, gg)
            if self.root_exchange:
                self._set(coef, 16 + g, mix.float())
                self._set(coef, 18 + g, gg)
            for o in range(8):
                v = (gg.double() * self._get(const, layout.CONST_LANE + o)).sum(1, keepdim=True).expand(4, 8)
                self._set(coef, 2 * o + g, v.float())
        self.coef.store(coef.repeat(1, K))
        Y = torch.zeros(32, self.C // 4)
        for core in range(K):
            accy = [torch.zeros(4, 8, dtype=torch.float64) for _ in range(2)]
            for b in range(core * self.py, (core + 1) * self.py):
                yp = Y[:, 32 * b : 32 * b + 32]
                for k in range(4):
                    t = 4 * b + k
                    xp = X[:, 32 * t : 32 * t + 32]
                    for g in range(2):
                        for hp in range(4):
                            c = _cell(g, hp >> 1, hp & 1)
                            v = sum(self._get(xp, 8 * j + c) * self._get(coef, 2 * j + g) for j in range(4)).float()
                            self._set(yp, 8 * k + c, v)
                            accy[g] += v.double() ** 2
            for g in range(2):
                self._set(scratch[core], 18 + g, accy[g].float())
        y.store(Y)
        stat = torch.zeros(32, 32 * K)
        for g in range(2):
            s = sum(self._get(scratch[k], 18 + g) for k in range(K)).sum(1, keepdim=True).expand(4, 8)
            v = torch.rsqrt(s * self._get(const, layout.CONST_INV_C) + self._get(const, layout.CONST_NORM_EPS))
            self._set(stat[:, :32], 2 * g, v.float())
        self.stat.store(stat)
        if debug:
            self.dbg = HostTensor(torch.cat([s.float() for s in scratch], dim=1), ttnn.float32)

    def pre(self, streams, params, y=None, stop_after=0, debug=False):
        y = y if y is not None else self.empty_rows()
        self._pre(streams.m, params, y, debug)
        return y

    def post(self, streams, f):
        X = streams.m.clone()
        self._post(X, f.m.double(), self.coef.m[:, :32].double())
        streams.m = X
        return streams

    def post_pre(self, streams, f, params, y=None, stop_after=0, debug=False):
        self.post(streams, f)
        return self.pre(streams, params, y, debug=debug)

    def probe(self, page, const):
        out, stat = torch.zeros(32, 32), torch.zeros(32, 32)
        a = layout.unit_get(page, 0, self.lanes)
        self._set(out, 0, a)
        self._set(out, 1, torch.roll(a, -self.rotate, dims=1))
        self._set(out, 2, layout.unit_get(page, 1, self.lanes))
        self._set(out, 3, a.double().sum(1, keepdim=True).expand(4, 8).float())
        self._set(out, 4, 4.0 * a)
        self._set(out, 5, layout.unit_get(const, 0, self.lanes))
        self._set(out, 6, a.double().rsqrt().float())
        self._set(out, 7, torch.sigmoid(a.double()).float())
        for u in (8, 9, 10):
            self._set(out, u, a)
        self._set(stat, 0, a)
        self._set(stat, 1, layout.unit_get(const, 1, self.lanes))
        return out, stat
