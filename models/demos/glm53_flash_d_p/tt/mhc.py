# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""mHC coefficients (attn_hc / ffn_hc) on device, replicated, no CCL.

x [1, 1, S, 4H] (the 4 streams packed along the last dim; the same memory as the reference's token-major [S * 4, H])
-> [1, 1, S, 24] fp32 = [pre 4 | post 4 | comb 16 row-major]. Projection and Sinkhorn reuse DeepSeek's mHC
(deepseek_v3_d_p/tt/mhc/tt_mhc.py: _project, build_consts, ttnn.experimental.deepseek_prefill.mhc_split_sinkhorn),
whose order and forms match glm_ref.hc_weights; the kernel emits comb row-major (entry (i, j) at column 4i + j).
"""

from __future__ import annotations

from types import SimpleNamespace

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.deepseek_v3_d_p.tt.mhc.tt_mhc import W, _project, build_consts
from models.demos.glm53_flash_d_p.tt.common import hifi4_config, replicate


class TtHcWeights(LightweightModule):
    def __init__(self, mesh, fn, base, scale, n=4, iters=20, hc_eps=1e-6, norm_eps=1e-5):
        self.n, self.iters, self.eps, self.norm_eps = n, int(iters), float(hc_eps), float(norm_eps)
        self.ckc = hifi4_config()
        fn = fn.float()  # [24, n*H] -> fn_T [1, 1, n*H, 24]
        self.fn_T = replicate(mesh, fn.t().reshape(1, 1, fn.shape[1], fn.shape[0]), dtype=ttnn.float32)
        consts = build_consts(SimpleNamespace(n=n), scale.float(), base.float())
        self.consts = replicate(mesh, consts.reshape(8, W, W), dtype=ttnn.float32)

    def __call__(self, x):
        """x [1, 1, S, n*H] (bf16 or fp32) -> [1, 1, S, (2 + n) * n] fp32."""
        S = x.shape[-2]
        xf = x if x.dtype == ttnn.float32 else ttnn.typecast(x, ttnn.float32)
        mixes = _project(xf, self.fn_T, self.norm_eps, self.ckc)
        if xf is not x:
            ttnn.deallocate(xf)
        pre, post, comb = ttnn.experimental.deepseek_prefill.mhc_split_sinkhorn(
            mixes, self.consts, self.n, self.iters, self.eps
        )
        ttnn.deallocate(mixes)
        parts = [ttnn.reshape(t, [1, 1, S, t.shape[-1]]) for t in (pre, post, comb)]
        out = ttnn.concat(parts, dim=-1)
        for t in (pre, post, comb):
            ttnn.deallocate(t)
        return out


def build_hc(mesh, loader, cfg, layer: int, which: str) -> TtHcWeights:
    """which: 'attn' or 'ffn' (hc_<which>_fn / _base / _scale of the layer)."""
    get = lambda k: loader.layer(layer, f"hc_{which}_{k}").float()  # noqa: E731
    return TtHcWeights(
        mesh,
        get("fn"),
        get("base"),
        get("scale"),
        n=cfg.hc_mult,
        iters=cfg.hc_sinkhorn_iters,
        hc_eps=cfg.hc_eps,
        norm_eps=cfg.rms_norm_eps,
    )
