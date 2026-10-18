# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Scratch: GLM-5.3-Flash routed experts on real weights against the layer golden, the K3 token-split block
(GLM_EXPERTS_MODE=ts) vs the all-gather path, without the HF checkpoint: the experts' weights from the flat weight
cache (generated/glm53_flash_d_p/tt_cache/flat/<mesh>: a plan whose weight layout is cached, e.g. MIMO_FL_ND=36 for
the 64 gate/up + 36 down one), x / routing / the expected output from the component golden (ffn_norm, router,
experts_out of layer GLM_TSG_LAYER, default 4), tiled to GLM_TSG_CHUNK tokens (default the golden's).

    TT_METAL_PINNED_MEMORY_CACHE_LIMIT_BYTES=0 BRINGUP_FABRIC_PAYLOAD=8704 MIMO_FL_ND=36 \\
    BRINGUP_SPEC=models/demos/glm53_flash_d_p_lb/bringup/spec.yaml \\
    scripts/run_safe_pytest.sh --mesh --no-precompile models/demos/glm53_flash_d_p_lb/tests/zz_scratch_experts_ts_golden.py -s
"""

import os

import torch
from loguru import logger

from models.demos.common.bringup.testing.harness import component_golden, device_timeout, mesh_parametrize, spec

S_ = spec()
pytestmark = device_timeout(S_)
LAYER = int(os.environ.get("GLM_TSG_LAYER", "4"))


def _stats(got, want):
    g, w = got.double(), want.double()
    rel = ((g - w).norm(dim=-1) / w.norm(dim=-1).clamp(min=1e-30)).mean().item()
    a, b = g.flatten() - g.mean(), w.flatten() - w.mean()
    pcc = (a @ b / (a.norm() * b.norm())).item()
    coef = (g.flatten() @ w.flatten() / (w.flatten() @ w.flatten())).item()
    return f"PCC {pcc:.5f}, rel L2 {rel:.4f}, scale {coef:.4f}"


@mesh_parametrize
def test_experts_ts_golden(mesh_device):
    import ttnn
    from models.demos.glm53_flash_d_p.bringup import hooks
    from models.demos.glm53_flash_d_p.tt.common import split_from_host, split_to_host
    from models.demos.glm53_flash_d_p.tt.experts import LazyExpertWeights
    from models.demos.glm53_flash_d_p.tt.experts_ag import TtExpertsAg
    from models.demos.glm53_flash_d_p.tt.experts_ts import TtExpertsTs

    hooks.apply_device_settings(S_)
    loader, cfg = hooks._loader_cfg(S_)
    g, c = component_golden(S_)
    gl = g.layer(c, LAYER)
    x, r, want = gl["ffn_norm"].float(), gl["router"].float(), gl["experts_out"].float()
    T = int(os.environ.get("GLM_TSG_CHUNK", x.shape[0]))
    rep = -(-T // x.shape[0])
    x, r, want = (t.repeat(rep, 1)[:T] for t in (x, r, want))
    K = cfg.num_experts_per_tok
    wts, ids = torch.topk(r, K, dim=-1)
    tw = LazyExpertWeights(loader, LAYER, cfg.n_routed_experts)  # (read only on a flat cache miss: no checkpoint)
    kw = dict(
        num_experts=cfg.n_routed_experts,
        emb_dim=cfg.hidden_size,
        hidden_dim=cfg.moe_intermediate_size,
        top_k=K,
        max_seq_len=T,
        weights_dtype=ttnn.bfloat4_b,
        cache=True,
        num_links=2,
    )
    mods = [("ts", TtExpertsTs(mesh_device, LAYER, tw, **kw))]  # (first: its plan options are process-wide)
    mods.append(("ag", TtExpertsAg(mesh_device, LAYER, tw, **kw)))
    outs = {}
    for name, mod in mods:
        xd = split_from_host(mesh_device, x.view(1, 1, T, -1).to(torch.bfloat16))
        idd = split_from_host(mesh_device, ids.view(1, 1, T, K).to(torch.int32), dtype=ttnn.uint16)
        wd = split_from_host(mesh_device, wts.view(1, 1, T, K).to(torch.bfloat16))
        y = mod(xd, idx=idd, wts=wd, split=True, full_mesh=True)
        y = mod(xd, idx=idd, wts=wd, split=True, full_mesh=True)  # (the ts block's arena check)
        outs[name] = split_to_host(y).reshape(T, -1).float()
        logger.info(f"TSG layer {LAYER} T {T}: {name} vs golden: {_stats(outs[name], want)}")
    logger.info(f"TSG layer {LAYER} T {T}: ts vs ag: {_stats(outs['ts'], outs['ag'])}")
    assert torch.isfinite(outs["ts"]).all()
