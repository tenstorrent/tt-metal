# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Unit test of ``ttnn.experimental.deepseek_prefill.mhc_post_mix`` (the fused exact fp32 mHC residual
update) against a float64 torch reference, on random fp32 inputs and on real layer-4 streams with real
Sinkhorn coefficients (``TtMHCWrap.hc_pre``); both output modes (streams, collapse), T = 32 and 256.

    TT_VISIBLE_DEVICES=1 pytest models/demos/glm53_flash/tests/test_mhc_post_mix.py -s
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.glm53_flash.reference.checkpoint import default_snapshot_dir, load_layer_raw, shard_headers
from models.demos.glm53_flash.tt.config import GLM53FlashConfig
from models.demos.glm53_flash.tt.mhc import TtMHCWrap, mhc_kernel_config

DUMP = Path("bringup/references/glm53_flash/R4097/layer4_prefill.pt")


def post_mix_reference(
    x: torch.Tensor, y: torch.Tensor, post: torch.Tensor, comb: torch.Tensor, n: int, collapse: bool
):
    """x [T, n*C], y [T, C], post [T, n], comb [T, n*n] (column i*n+j = comb[i,j]) -> [T, n*C] or [T, C], float64."""
    T, C = y.shape
    xs = x.double().reshape(T, n, C)
    cm = comb.double().reshape(T, n, n)
    out = torch.einsum("tij,tic->tjc", cm, xs) + post.double()[:, :, None] * y.double()[:, None, :]  # [T, j, C]
    return out.mean(1) if collapse else out.reshape(T, n * C)


def _up(device, t: torch.Tensor) -> ttnn.Tensor:
    return ttnn.from_torch(t.float().contiguous(), layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.float32)


def _err(name: str, ref: torch.Tensor, got: torch.Tensor) -> float:
    ref, got = ref.double().flatten(), got.double().flatten()
    d = ref - got
    rel = (d.norm() / ref.norm()).item()
    logger.info(f"  {name}: rel rms {rel:.3e} max|d|/max|ref| {(d.abs().max() / ref.abs().max()).item():.3e}")
    return rel


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("T", [32, 256])
@pytest.mark.parametrize("collapse", [False, True])
def test_post_mix_random(mesh_device, T, collapse):
    n, C = 4, 4096
    g = torch.Generator().manual_seed(0)
    x = torch.randn(T, n * C, generator=g)
    y = torch.randn(T, C, generator=g)
    post = 2 * torch.rand(T, n, generator=g)
    comb = torch.softmax(torch.randn(T, n, n, generator=g), -1).reshape(T, n * n)
    out = ttnn.experimental.deepseek_prefill.mhc_post_mix(
        _up(mesh_device, x.reshape(1, 1, T, n * C)),
        _up(mesh_device, y.reshape(1, 1, T, C)),
        _up(mesh_device, post),
        _up(mesh_device, comb),
        n,
        collapse,
    )
    assert tuple(out.shape) == (1, 1, T, C if collapse else n * C), out.shape
    ref = post_mix_reference(x, y, post, comb, n, collapse)
    rel = _err(f"random T={T} collapse={collapse}", ref, ttnn.to_torch(out)[0, 0])
    assert rel < 1e-6


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("collapse", [False, True])
def test_post_mix_real(mesh_device, collapse):
    """Real layer-4 streams (dump rows + 1e-3 noise -> fp32, not bf16-exact), real coefficients from the
    Sinkhorn kernel via the composite ``TtMHCWrap``; compared with float64 and with the composite's
    own exact fp32 ``hc_post`` chain."""
    snapshot = default_snapshot_dir()
    cfg = GLM53FlashConfig.from_hf_config_json(snapshot / "config.json")
    kcfg = mhc_kernel_config(cfg)
    raw = load_layer_raw(snapshot, 4, shard_headers(snapshot))
    wrap = TtMHCWrap(
        mesh_device, kcfg, raw["hc_ffn_fn"].float(), raw["hc_ffn_base"].float(), raw["hc_ffn_scale"].float()
    )
    n, C, T = kcfg.n, kcfg.dim, 32
    g = torch.Generator().manual_seed(1)
    x = torch.load(DUMP)["x_in"][0, 300 : 300 + T].reshape(T, n * C).float()
    x = x + torch.randn(x.shape, generator=g) * 1e-3 * x.abs()
    y = (torch.load(DUMP)["x_in"][0, 900 : 900 + T, 0].float() * 0.5).to(torch.bfloat16).float()
    xd, yd = _up(mesh_device, x.reshape(1, 1, T, n * C)), _up(mesh_device, y.reshape(1, 1, T, C))
    _, post, comb = wrap.hc_pre(xd)  # [1,1,T,n], [1,1,T,n*n] views of the kernel outputs
    post2, comb2 = ttnn.reshape(post, [T, n]), ttnn.reshape(comb, [T, n * n])
    out = ttnn.experimental.deepseek_prefill.mhc_post_mix(xd, yd, post2, comb2, n, collapse)
    post_t, comb_t = ttnn.to_torch(post)[0, 0].float(), ttnn.to_torch(comb)[0, 0].float()
    ref = post_mix_reference(x, y, post_t, comb_t, n, collapse)
    got = ttnn.to_torch(out)[0, 0]
    rel = _err(f"real collapse={collapse} vs float64", ref, got)
    assert rel < 1e-6
    comp = ttnn.to_torch(wrap.hc_post(yd, xd, post, comb))[0, 0]
    if collapse:
        comp = comp.reshape(T, n, C).mean(1)
    rel_c = _err(f"real collapse={collapse} vs composite hc_post", comp, got)
    assert rel_c < 1e-6
