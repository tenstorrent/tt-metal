# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Diagnostic: run the SAME inputs through the dense and moe_compute expert flows.

The dense flow is known-good end to end (128 coherent, row-identical demo outputs), so it is
a better reference than the HF golden for localising where moe_compute diverges: comparing
against torch conflates the two paths' shared quantisation error with whatever moe_compute
does differently. This reports overall PCC between the two flows plus the per-token error
distribution, which distinguishes uniform rounding noise from a subset of badly-wrong tokens.
"""

import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc

from ..test_factory import TestFactory, parametrize_mesh_with_fabric


@parametrize_mesh_with_fabric([(4, 8)])
def test_moe_compute_vs_dense(mesh_device, device_params, reset_seeds):
    pass

    from ...tt.experts_throughput import ThroughputExpertConfig, ThroughputExperts
    from ...tt.experts_throughput.moe_compute import create_moe_compute_config

    setup = TestFactory.setup_test(mesh_device, use_real_weights=False)
    hf_config = setup["config"]
    mesh_shape = tuple(mesh_device.shape)
    rows, cols = mesh_shape

    num_tokens = 128
    tokens_per_device = num_tokens // rows
    K = hf_config.hidden_size
    E_global = hf_config.num_local_experts
    k_sel = hf_config.num_experts_per_tok

    cfg = ThroughputExpertConfig(
        intermediate_size=hf_config.intermediate_size,
        num_experts=E_global,
        hidden_size=K,
        num_experts_per_tok=k_sel,
        num_devices=mesh_device.get_num_devices(),
    )

    # Weights: uniform random by default, or the REAL dequantised layer-1 experts when
    # GPT_OSS_AB_REAL_WEIGHTS points at a .pt of them. Real GPT-OSS experts are MXFP4-derived
    # and carry heavy outliers (down_proj absmax 16.0 vs p99.9 0.375, a ~43x ratio) where
    # uniform random weights have ~1x -- and bfloat4_b shares one exponent per block, so how
    # values are GROUPED into blocks matters enormously for real weights and not at all for
    # random ones. The two flows group differently: dense quantises the natural layout, while
    # moe_compute quantises after prepare_* has interleaved/reordered.
    import os

    real_path = os.getenv("GPT_OSS_AB_REAL_WEIGHTS")
    if real_path:
        state_dict = {k: v for k, v in torch.load(real_path).items()}
        logger.info(f"using REAL layer-1 expert weights from {real_path}")
    else:
        torch.manual_seed(1234)
        state_dict = {
            "gate_up_proj": (torch.rand(E_global, K, 2 * cfg.intermediate_size) - 0.5).bfloat16(),
            "gate_up_proj_bias": (torch.rand(E_global, 2 * cfg.intermediate_size) - 0.5).bfloat16(),
            "down_proj": (torch.rand(E_global, cfg.intermediate_size, K) - 0.5).bfloat16(),
            "down_proj_bias": (torch.rand(E_global, K) - 0.5).bfloat16(),
        }
        logger.info("using uniform random expert weights")

    mc_config = create_moe_compute_config(
        mesh_device=mesh_device,
        config=cfg,
        state_dict=state_dict,
        tokens_per_device=tokens_per_device,
        num_links=setup["ccl_manager"].num_links,
    )
    experts = ThroughputExperts(
        mesh_device=mesh_device,
        config=cfg,
        state_dict=state_dict,
        weight_dtype=ttnn.bfloat4_b,
        mesh_config=setup["mesh_config"],
        ccl_manager=setup["ccl_manager"],
        moe_compute_config=mc_config,
    )

    # Identical inputs for both flows.
    # Mirror the demo's structure: 32 distinct prompts cycled across 128 users, so token t and
    # token t+32 (a different mesh row) carry the SAME hidden state and route to the SAME
    # experts. Every row therefore sees an identical routing pattern set -- very different from
    # 128 independent draws, and the regime where the demo fails.
    import os

    correlated = os.getenv("GPT_OSS_AB_CORRELATED", "1") == "1"
    if correlated:
        base_hidden = torch.randn(tokens_per_device, K)
        base_idx = torch.stack([torch.randperm(E_global)[:k_sel] for _ in range(tokens_per_device)])
        bw = torch.rand(tokens_per_device, k_sel)
        base_scores = bw / bw.sum(dim=-1, keepdim=True)
        hidden = base_hidden.repeat(rows, 1).reshape(1, 1, num_tokens, K)
        indices = base_idx.repeat(rows, 1).to(torch.int32)
        scores = base_scores.repeat(rows, 1).bfloat16()
    else:
        hidden = torch.randn(1, 1, num_tokens, K)
        indices = torch.stack([torch.randperm(E_global)[:k_sel] for _ in range(num_tokens)]).to(torch.int32)
        w = torch.rand(num_tokens, k_sel)
        scores = (w / w.sum(dim=-1, keepdim=True)).bfloat16()
    logger.info(f"routing mode: {'correlated (demo-like)' if correlated else 'independent'}")

    row_shard = ttnn.ShardTensor2dMesh(dims=(-2, None), mesh_shape=mesh_shape, mesh_device=mesh_device)
    # The real model hands the MLP bfloat8_b activations (the decoder typecasts before the
    # residual/MLP), not bfloat16. That is the last uncontrolled difference from the demo.
    act_dtype = ttnn.bfloat8_b if os.getenv("GPT_OSS_AB_BF8_ACT", "1") == "1" else ttnn.bfloat16
    logger.info(f"activation dtype: {act_dtype}")
    tt_hidden = ttnn.from_torch(
        hidden, device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=act_dtype, mesh_mapper=row_shard
    )
    shard0 = ttnn.ShardTensor2dMesh(dims=(0, None), mesh_shape=mesh_shape, mesh_device=mesh_device)
    tt_idx = ttnn.from_torch(
        indices, device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.uint16, mesh_mapper=shard0
    )
    tt_scores = ttnn.from_torch(
        scores, device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, mesh_mapper=shard0
    )

    def run(use_moe_compute):
        saved = experts.moe_compute_config
        experts.moe_compute_config = saved if use_moe_compute else None
        try:
            out = experts(
                ttnn.clone(tt_hidden),
                topk_expert_indices=ttnn.clone(tt_idx),
                topk_expert_weights=ttnn.clone(tt_scores),
                is_decode=True,
            )
            composer = ttnn.ConcatMesh2dToTensor(mesh_device, dims=(-2, -1), mesh_shape=mesh_shape)
            return ttnn.to_torch(out, mesh_composer=composer)[..., :num_tokens, :K].float()
        finally:
            experts.moe_compute_config = saved

    out_mc = run(True)
    out_dense = run(False)

    # float32 torch golden for the same routing, so each flow can be scored independently
    # rather than only against each other.
    gu = state_dict["gate_up_proj"].float()
    gub = state_dict["gate_up_proj_bias"].float()
    dpw = state_dict["down_proj"].float()
    dpb = state_dict["down_proj_bias"].float()
    h = hidden.reshape(num_tokens, K).float()
    golden = torch.zeros(num_tokens, K)
    for t in range(num_tokens):
        acc = torch.zeros(K)
        for j in range(k_sel):
            e = int(indices[t, j])
            gu_e = h[t] @ gu[e] + gub[e]
            g, u = gu_e[0::2], gu_e[1::2]
            g = g.clamp(max=cfg.swiglu_limit)
            u = u.clamp(-cfg.swiglu_limit, cfg.swiglu_limit)
            act = (u + 1.0) * (g * torch.sigmoid(g * cfg.alpha))
            acc += float(scores[t, j]) * (act @ dpw[e] + dpb[e])
        golden[t] = acc

    _, pcc = comp_pcc(out_dense, out_mc, 0.99)
    logger.info(f"dense vs moe_compute overall PCC: {pcc}")
    _, p_d = comp_pcc(golden, out_dense.reshape(num_tokens, K), 0.9)
    _, p_m = comp_pcc(golden, out_mc.reshape(num_tokens, K), 0.9)
    logger.info(f"GOLDEN vs dense       : {p_d}")
    logger.info(f"GOLDEN vs moe_compute : {p_m}")

    a = out_dense.reshape(num_tokens, K)
    b = out_mc.reshape(num_tokens, K)
    per_tok = []
    for t in range(num_tokens):
        num = (a[t] * b[t]).sum()
        den = a[t].norm() * b[t].norm()
        per_tok.append(float(num / den) if den > 0 else 1.0)
    per_tok_t = torch.tensor(per_tok)
    bad = (per_tok_t < 0.95).nonzero().flatten().tolist()
    logger.info(
        f"per-token cosine: min {per_tok_t.min():.4f} median {per_tok_t.median():.4f} "
        f"| tokens below 0.95: {len(bad)}/{num_tokens} {bad[:16]}"
    )
    logger.info(f"rows of bad tokens: {sorted({t // tokens_per_device for t in bad})}")
    # The demo's divergent users cluster by within-device token index mod 8 (the Blackhole matmul
    # ring size), so report the per-token agreement bucketed that way -- a ring-position defect
    # shows up here as one or two buckets far below the rest.
    by_mod = {}
    for t in range(num_tokens):
        by_mod.setdefault((t % tokens_per_device) % 8, []).append(per_tok[t])
    logger.info(
        "per-token cosine by (token_idx %% 8): " + " ".join(f"{m}:{min(v):.4f}" for m, v in sorted(by_mod.items()))
    )

    # Row symmetry: with correlated inputs, token t and token t+tokens_per_device are the SAME
    # token on a different mesh row, so a correct flow must return identical vectors for them.
    # Breaking that symmetry at one layer is what makes users sharing a prompt diverge over 36.
    if correlated:
        for name, out in (("dense", out_dense), ("moe_compute", out_mc)):
            m = out.reshape(num_tokens, K)
            sims, maxabs = [], []
            for t in range(tokens_per_device):
                ref = m[t]
                for r in range(1, rows):
                    other = m[t + r * tokens_per_device]
                    den = ref.norm() * other.norm()
                    sims.append(float((ref * other).sum() / den) if den > 0 else 1.0)
                    maxabs.append(float((ref - other).abs().max()))
            st = torch.tensor(sims)
            logger.info(
                f"ROW SYMMETRY {name:<12}: cosine min {st.min():.6f} median {st.median():.6f} "
                f"| exact matches {int((st > 0.999999).sum())}/{len(sims)} | max abs diff {max(maxabs):.4f}"
            )
