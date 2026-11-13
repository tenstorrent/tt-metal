# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Hy4-Preview iHC (hc_mult 4) on the DSV4 mHC kernels, no kernel change: first try of the port plan (identity comb).

Hy4 hyper-connection (oracle reference/oracle.py hc_pre_post / hc_collapse / hc_expand), per sublayer, from the SAME stream x4 [T, 4, 6144]:
    m    = (x_flat * rsqrt(mean(x_flat^2) + eps)) @ hc_fn^T          hc_fn [8, 24576]: rows 0-3 pre, rows 4-7 post (no comb rows)
    pre  = sigmoid(m[:4] * scale[0] + base[:4]) + hc_eps             h = rms_norm(sum_i pre_i x_i) * w
    post = 2 * sigmoid(m[4:] * scale[1] + base[4:])                  x4' = x4 + post * y         (identity residual mixing: comb = I)
DSV4's mHC has 24 mix rows (4 pre, 4 post, 16 comb with a Sinkhorn): here hc_fn is zero-padded to 24 rows and the comb bias is +B on the diagonal / -B off it with
scale[2] = 0, so the Sinkhorn of the existing kernel returns I + O(1e-6) (the kernel's stochastic eps); expand then computes post*y + comb^T x = x + post*y.
Unlike DSV4 (``pre`` of a sub-block comes from the PREVIOUS sub-block's mixes), Hy4 collapses with the pre of the SAME stream: ``collapse_norm(x, pre_of_x, ...)``.
"""
import torch

from models.demos.blackhole.deepseek_v41_flash.tt import mhc_mixes2
from models.demos.blackhole.deepseek_v41_flash.tt.mhc import DSV41MHC


class Hy4ProjPlan(mhc_mixes2.ProjPlan):
    """ProjPlan for D = 6144 (NJ = 192 column tiles): the DSV4 plan picks S in (40, 32, 20) column ranges with S | NJ unchecked (160 = 4 x 40); here the first S that
    divides NJ, with 48 first (TPJ = 4 like DSV4). Identical to the DSV4 plan for D = 5120 (48 does not divide 160)."""

    def __init__(self, T, D=5120, max_cores=100, tg=8, tv=None, shared=False):
        assert T % 4 == 0 and T <= 32
        self.T, self.D = T, D
        self.tv = T if tv is None else tv
        assert 1 <= self.tv <= T
        self.TG = min(T, tg)
        self.G = T // self.TG
        self.NJ = D // 32
        self.S = next(
            s
            for s in (48, 40, 32, 20)
            if self.NJ % s == 0
            and (s if shared else self.G * s) <= max_cores
            and (32 // self.TG)
            and s % (32 // self.TG) == 0
        )
        self.TPJ = self.NJ // self.S
        assert self.TPJ * self.S == self.NJ
        self.PPT = 32 // self.TG
        assert self.S % self.PPT == 0
        self.PGPG = self.S // self.PPT
        self.NPG = self.G * self.PGPG
        self.cores = self.G * self.S


mhc_mixes2.ProjPlan = Hy4ProjPlan  # proj_plan() looks the class up in its module globals at call time

COMB_B = 30.0  # softmax(+-30): diagonal 1 - 1e-13


def pad_hc_params(fn8, base8, scale2, B=COMB_B, n=4):
    """[8, N] / [8] / [2] checkpoint tensors -> DSV4-shaped fn [24, N], base [24], scale [3] whose comb is the identity."""
    fn24 = torch.zeros(24, fn8.shape[1])
    fn24[:8] = fn8.float()
    base24 = torch.zeros(24)
    base24[:8] = base8.float()
    comb = torch.full((n, n), -B)
    comb.fill_diagonal_(B)
    base24[8:] = comb.flatten()
    scale3 = torch.cat([scale2.float(), torch.zeros(1)])
    return fn24, base24, scale3


class Hy4IHC(DSV41MHC):
    def __init__(self, device, fn8, base8, scale2, dim=6144, hc_eps=1e-6, norm_eps=1e-5, **kw):
        fn24, base24, scale3 = pad_hc_params(fn8, base8, scale2)
        super().__init__(device, fn24, base24, scale3, dim=dim, n=4, eps=hc_eps, norm_eps=norm_eps, **kw)
