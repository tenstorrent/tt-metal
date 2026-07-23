# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""mHC-wrapped sublayer (issue #40726): X' = H_res @ X + H_post.T @ F(H_pre @ X).

Splits exactly along the design-doc boundary:
  - Parametrization (X -> H matrices) uses the fused kernel
    ttnn.experimental.deepseek_prefill.mhc_split_sinkhorn, fed by the projection.
  - Computation (apply H around F) reuses the composite reduce / hc_post (tt/mhc_ttnn.py).
F is any callable [1,T,1,C] -> [1,T,1,C] (attention, MLP, ...); it owns its own pre-norm.
"""

from __future__ import annotations

import torch

import ttnn
from models.demos.deepseek_v3_d_p.tt.mhc.mhc_kernel import build_consts
from models.demos.deepseek_v3_d_p.tt.mhc.mhc_ttnn import TtMHC


class TtMHCBlock:
    def __init__(self, device, cfg, fn: torch.Tensor, base: torch.Tensor, scale: torch.Tensor, dtype=ttnn.float32):
        self.cfg = cfg
        self.n = cfg.n
        self.iters = int(cfg.sinkhorn_iters)
        self.eps = float(cfg.eps)
        # TtMHC supplies project() (RMSNorm + fused proj), hc_post(), and _row_to_batch().
        self.mhc = TtMHC(device, cfg, fn, base, scale, dtype)
        self.consts = ttnn.from_torch(
            build_consts(cfg, scale, base), layout=ttnn.TILE_LAYOUT, device=device, dtype=dtype
        )

    def hc_pre(self, x_streams):
        """[1,T,n,C] -> (y [1,T,1,C], post [T,n], comb [1,T,n,n]). Kernel does the H matrices."""
        T = x_streams.shape[1]
        mixes = self.mhc.project(x_streams)  # [1,1,T,mix_hc]
        pre, post, comb = ttnn.experimental.deepseek_prefill.mhc_split_sinkhorn(
            mixes, self.consts, self.n, self.iters, self.eps
        )  # [T,n], [T,n], [T,n*n]
        pre_row = self.mhc._row_to_batch(pre, [1, T, 1, self.n])
        y = ttnn.matmul(pre_row, x_streams, compute_kernel_config=self.mhc.ckc)  # [1,T,1,C]
        comb4 = self.mhc._row_to_batch(comb, [1, T, self.n, self.n])
        return y, post, comb4

    def __call__(self, x_streams, sublayer):
        """x_streams: [1,T,n,C]; sublayer: [1,T,1,C] -> [1,T,1,C]. Returns [1,T,n,C]."""
        y, post, comb = self.hc_pre(x_streams)
        f_out = sublayer(y)
        return self.mhc.hc_post(f_out, x_streams, post, comb)
