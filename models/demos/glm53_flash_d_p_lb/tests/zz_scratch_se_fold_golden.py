# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Scratch: accuracy of the folded shared expert on real weights against the layer golden, without arrival-order noise
first. For each layer in GLM_SFG_LAYERS (default 3,4,20,44), the golden's MoE input (ffn_norm, 2048 tokens):

  1 the shared expert alone (deterministic): the model's TtDenseMLP (8-way split, bf16 weights, HiFi4, fp32
    intermediates) vs the folded slices TtSharedTpRow (SP pairs, bf8 / HiFi2 - the default - and bf16 / HiFi4), each
    slice's partial summed on the host; against the golden shared_out and a host fp32 reference from the HF weights
  2 the MoE block (routed + shared, the golden's routing): the token-split block with the shared expert folded vs
    routed-only + the separate TtDenseMLP on the gathered x (the model's two paths), against the golden mlp_out; each
    twice (the combine sums in arrival order: run-to-run spread)

    TT_METAL_PINNED_MEMORY_CACHE_LIMIT_BYTES=0 BRINGUP_FABRIC_PAYLOAD=8704 \\
    BRINGUP_SPEC=models/demos/glm53_flash_d_p_lb/bringup/spec.yaml \\
    scripts/run_safe_pytest.sh --mesh --no-precompile models/demos/glm53_flash_d_p_lb/tests/zz_scratch_se_fold_golden.py -s
"""

import os

import torch
from loguru import logger

from models.demos.common.bringup.testing.harness import component_golden, device_timeout, mesh_parametrize, spec

S_ = spec()
pytestmark = device_timeout(S_)
LAYERS = [int(v) for v in os.environ.get("GLM_SFG_LAYERS", "3,4,20,44").split(",")]


def _stats(got, want):
    g, w = got.double(), want.double()
    rel = ((g - w).norm(dim=-1) / w.norm(dim=-1).clamp(min=1e-30)).mean().item()
    a, b = g.flatten() - g.mean(), w.flatten() - w.mean()
    pcc = (a @ b / (a.norm() * b.norm())).item()
    coef = (g.flatten() @ w.flatten() / (w.flatten() @ w.flatten())).item()
    return f"rel L2 {rel:.4f}, PCC {pcc:.6f}, scale {coef:.4f}"


@mesh_parametrize
def test_se_fold_golden(mesh_device):
    import ttnn
    from models.demos.glm53_flash_d_p.bringup import hooks
    from models.demos.glm53_flash_d_p.reference.weights import PREFIX
    from models.demos.glm53_flash_d_p.tt.common import split_from_host, split_to_host
    from models.demos.glm53_flash_d_p.tt.experts import LazyExpertWeights
    from models.demos.glm53_flash_d_p.tt.experts_ts import TtExpertsTs
    from models.demos.glm53_flash_d_p.tt.mlp import TtDenseMLP
    from models.demos.glm53_flash_d_p.tt.shared_tp_row import TtSharedTpRow

    hooks.apply_device_settings(S_)
    loader, cfg = hooks._loader_cfg(S_)
    g, c = component_golden(S_)
    R, C = tuple(mesh_device.shape)
    rep = ttnn.ReplicateTensorToMesh(mesh_device)
    for L in LAYERS:
        gl = g.layer(c, L)
        x, r = gl["ffn_norm"].float(), gl["router"].float()
        sh_want, mlp_want = gl["shared_out"].float(), gl["mlp_out"].float()
        T, H = x.shape
        p = f"{PREFIX}layers.{L}.mlp.shared_experts."
        wg, wu, wd = (loader.weight(p + f"{n}_proj.weight") for n in ("gate", "up", "down"))
        # host fp32 reference of the shared expert (the bf16 golden input)
        xf = x.to(torch.bfloat16).float()
        a = torch.nn.functional.silu((xf @ wg.float().T).clamp(max=10.0)) * (xf @ wu.float().T).clamp(-10.0, 10.0)
        sh_ref = a @ wd.float().T
        logger.info(f"SFG L{L} golden shared_out vs host fp32: {_stats(sh_want, sh_ref)}")
        # 1: the shared expert alone
        xd = ttnn.from_torch(
            x.view(1, 1, T, H).to(torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            mesh_mapper=rep,
        )
        mlp = TtDenseMLP(mesh_device, wg, wu, wd, limit=cfg.swiglu_limit)
        o = mlp(xd)
        sep = ttnn.to_torch(ttnn.get_device_tensors(o)[0]).reshape(T, H).float()
        ttnn.deallocate(o)
        logger.info(
            f"SFG L{L} shared, model MLP (8-way, bf16 HiFi4): vs golden {_stats(sep, sh_want)} | vs fp32 {_stats(sep, sh_ref)}"
        )
        F = ttnn.MathFidelity
        variants = {
            "bf8 HiFi2": {},
            "bf16 HiFi4": dict(w_dtype=ttnn.bfloat16, fidelity=F.HiFi4),
            "bf8 HiFi4": dict(fidelity=F.HiFi4),
            "bf8 HiFi3": dict(fidelity=F.HiFi3),
            "bf16 HiFi2": dict(w_dtype=ttnn.bfloat16),
            "bf8 LoFi": dict(fidelity=F.LoFi),
        }
        want_v = os.environ.get("GLM_SFG_VARIANTS", "bf8 HiFi2,bf16 HiFi4").split(",")
        for name, kw in ((f"folded {v}", variants[v]) for v in want_v):
            se = TtSharedTpRow(mesh_device, wg, wu, wd, limit=cfg.swiglu_limit, axis=0, **kw)
            o = se(xd)
            parts = [ttnn.to_torch(t).reshape(T, H).float() for t in ttnn.get_device_tensors(o)]
            got = sum(parts[r_ * C] for r_ in range(R))  # (column 0's chips: the slices of the SP pair)
            ttnn.deallocate(o)
            logger.info(f"SFG L{L} shared, {name}: vs golden {_stats(got, sh_want)} | vs fp32 {_stats(got, sh_ref)}")
        ttnn.deallocate(xd)
        if os.environ.get("GLM_SFG_BLOCK", "1") == "0":
            continue
        # 2: the MoE block, folded vs separate
        K = cfg.num_experts_per_tok
        wts, ids = torch.topk(r, K, dim=-1)
        tw = LazyExpertWeights(loader, L, cfg.n_routed_experts)
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
        folded = TtExpertsTs(mesh_device, L, tw, shared_weights=(wg, wu, wd), shared_axis=0, **kw)
        routed = TtExpertsTs(mesh_device, L, tw, **kw)

        def call(mod):
            xs = split_from_host(mesh_device, x.view(1, 1, T, H).to(torch.bfloat16))
            idd = split_from_host(mesh_device, ids.view(1, 1, T, K).to(torch.int32), dtype=ttnn.uint16)
            wds = split_from_host(mesh_device, wts.view(1, 1, T, K).to(torch.bfloat16))
            y = mod(xs, idx=idd, wts=wds, split=True, full_mesh=True)
            out = split_to_host(y).reshape(T, H).float()
            extra = None
            if mod is routed:  # (the model's separate shared expert on the gathered x)
                ys = mlp(mod.gathered_x, split=True)
                extra = split_to_host(ys).reshape(T, H).float()
                ttnn.deallocate(ys)
            for t in (xs, idd, wds, y):
                if t.is_allocated():
                    ttnn.deallocate(t)
            return out, extra

        call(folded)  # (warm: the block's arena check)
        call(routed)
        for run in range(2):
            yf, _ = call(folded)
            yr, ys = call(routed)
            logger.info(
                f"SFG L{L} run {run} MoE block vs golden mlp_out: folded {_stats(yf, mlp_want)} | separate "
                f"{_stats(yr + ys, mlp_want)} | routed part vs golden experts_out {_stats(yr, gl['experts_out'].float())}"
            )
        del folded, routed, mlp
