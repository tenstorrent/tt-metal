# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""mHC with the fused bring-up ops (GLM_MHC_IMPL=fused, the default; composite = the op chains): ttnn.bringup.mhc_pre / mhc_post.

The composite path (tt/mhc.py, tt/collapse.py, tt/residual.py) runs attn_hc -> [S, 24] (pre | post | comb), then
attn_collapse (sum_n pre_n x_n) and attn_residual (post * y + comb^T x) as separate op chains. Fused, the hc step runs
ttnn.bringup.mhc_pre once and yields (y, post, comb): y is the collapse (attn_in / ffn_in), post and comb feed
ttnn.bringup.mhc_post in the residual step. Same math and order as glm_ref.hc_weights / hc_collapse / hc_residual
(RMS eps = rms_norm_eps, Sinkhorn eps = hc_eps, 20 iterations); the residual keeps post / comb in fp32 where the
reference rounds them to the activation dtype first.

Per row, no collective: in the split layout every chip runs both ops on its own S/4 rows.
"""

from __future__ import annotations

import os

import ttnn
from models.demos.glm53_flash_d_p.tt.common import replicate

IMPLS = ("composite", "fused")


def mhc_impl() -> str:
    mode = os.environ.get("GLM_MHC_IMPL", "fused")
    assert mode in IMPLS, f"GLM_MHC_IMPL={mode!r}, want one of {IMPLS}"
    return mode


class MhcBundle:
    """The fused hc step's output. Its tensors are freed by their consumers (the collapse step takes y, the residual
    step frees post and comb), so the block's freeing wrapper leaves the bundle alone (is_allocated() is False)."""

    __slots__ = ("y", "post", "comb")

    def __init__(self, y, post, comb):
        self.y, self.post, self.comb = y, post, comb

    def is_allocated(self) -> bool:
        return False


class TtMhcPre:
    def __init__(self, mesh, fn, base, scale, n=4, iters=20, hc_eps=1e-6, norm_eps=1e-5):
        # fn [24, n*H] -> W [n*H, 24]; the checkpoint stores fn in bf16, so bf16 on the device is exact
        self.w = replicate(mesh, fn.float().t(), dtype=ttnn.bfloat16)
        self.b = replicate(mesh, base.float().reshape(1, -1), dtype=ttnn.float32)
        self.scale = tuple(float(v) for v in scale.float().flatten())
        self.iters, self.eps, self.norm_eps = int(iters), float(hc_eps), float(norm_eps)

    def __call__(self, x) -> MhcBundle:
        """x [1, 1, S, n*H] bf16 -> (y [1, 1, S, H] bf16, post [1, 1, S, n] fp32, comb [1, 1, S, n*n] fp32)."""
        y, post, comb = ttnn.bringup.mhc_pre(
            x,
            self.w,
            self.b,
            scale=self.scale,
            sinkhorn_iters=self.iters,
            eps=self.eps,
            norm_eps=self.norm_eps,
        )
        return MhcBundle(y, post, comb)


def collapse(bundle: MhcBundle):
    """The collapse step: hand y on (the graph owns it from here)."""
    y, bundle.y = bundle.y, None
    return y


def residual(x, bundle: MhcBundle, out):
    """X' = post * out + comb^T X over the streams; frees post and comb."""
    y = ttnn.bringup.mhc_post(out, x, bundle.post, bundle.comb)
    ttnn.deallocate(bundle.post)
    ttnn.deallocate(bundle.comb)
    bundle.post = bundle.comb = None
    return y


def build_mhc_pre(mesh, loader, cfg, layer: int, which: str) -> TtMhcPre:
    """which: 'attn' or 'ffn' (hc_<which>_fn / _base / _scale of the layer)."""
    get = lambda k: loader.layer(layer, f"hc_{which}_{k}").float()  # noqa: E731
    return TtMhcPre(
        mesh,
        get("fn"),
        get("base"),
        get("scale"),
        n=cfg.hc_mult,
        iters=cfg.hc_sinkhorn_iters,
        hc_eps=cfg.hc_eps,
        norm_eps=cfg.rms_norm_eps,
    )
