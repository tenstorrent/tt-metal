# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU tests for the Hy4-Preview oracle/loader. Real-weight tests skip when the checkpoint is absent.
Run: PYTHONPATH=<worktree> pytest models/demos/blackhole/hy4_preview/tests -q"""
import os
from dataclasses import replace

import pytest
import torch

from models.demos.blackhole.hy4_preview.reference import oracle as O
from models.demos.blackhole.hy4_preview.reference.loader import CKPT, Hy4Checkpoint, split_gate_up

have_ckpt = os.path.exists(os.path.join(CKPT, "model.safetensors.index.json"))
needs_ckpt = pytest.mark.skipif(not have_ckpt, reason="checkpoint not on disk")
torch.set_num_threads(8)


@pytest.fixture(scope="module")
def ck():
    return Hy4Checkpoint()


@needs_ckpt
def test_layer_shapes(ck):
    c = ck.cfg
    sd0 = ck.layer_state(0)
    assert sd0["mlp.gate_proj.weight"].shape == (c.inter_dense, c.hidden)
    assert sd0["self_attn.q_b_proj.weight"].shape == (c.n_heads * (c.nope + c.rope), c.q_lora)
    assert sd0["self_attn.kv_b_proj.weight"].shape == (c.n_heads * (c.nope + c.v_dim), c.kv_lora)
    assert sd0["self_attn.linear_gate.weight"].shape == (c.n_heads * c.v_dim, c.hidden)
    assert sd0["hc_attn_layer.hc_pre.hc_fn"].shape == (2 * c.hc_mult, c.hc_mult * c.hidden)
    assert ck.meta("model.layers.1.mlp.experts.gate_up_proj")[1] == (c.n_experts, 2 * c.inter_moe, c.hidden)
    assert ck.meta("model.layers.1.mlp.experts.down_proj")[1] == (c.n_experts, c.hidden, c.inter_moe)
    assert "mlp.experts.gate_up_proj" not in ck.layer_state(1)  # lazy: fused tensors not materialised


@needs_ckpt
def test_indexer_layers_match_config(ck):
    full = [i for i, t in enumerate(ck.cfg.indexer_types) if t == "full"]
    have = sorted({int(k.split(".")[2]) for k in ck.weight_map if k.startswith("model.layers.") and ".indexer." in k})
    assert full == have and len(full) == 21
    assert any(".indexer." in k for k in ck.layer_keys(-1))  # MTP layer owns an indexer


@needs_ckpt
def test_expert_split_and_down(ck):
    wg, wu, wd = ck.expert_weights(1, 3)
    assert wg.shape == wu.shape == (2048, 6144) and wd.shape == (6144, 2048)
    gu = ck.get("model.layers.1.mlp.experts.gate_up_proj", rows=3).float()
    assert torch.equal(torch.cat([wg, wu]), gu)


def test_split_layouts():
    gu = torch.arange(8.0).view(8, 1).expand(8, 3)
    g, u = split_gate_up(gu, "chunk")
    assert g[:, 0].tolist() == [0, 1, 2, 3]
    g, u = split_gate_up(gu, "interleave")
    assert g[:, 0].tolist() == [0, 2, 4, 6]
    g, u = split_gate_up(gu, "swap")
    assert g[:, 0].tolist() == [4, 5, 6, 7]


@needs_ckpt
def test_router_top8_set_stable_and_weights(ck):
    c = ck.cfg
    sd = {k: v for k, v in ck.layer_state(1).items() if k.startswith("mlp.gate")}
    g = torch.Generator().manual_seed(0)
    x = torch.randn(64, c.hidden, generator=g)
    w, sel = O.router(x, sd, c)
    assert sel.shape == (64, 8) and all(len(set(r.tolist())) == 8 for r in sel)
    assert torch.allclose(w.sum(-1), torch.full((64,), c.routed_scale), atol=1e-4)
    w2, sel2 = O.router(x * (1 + 1e-5), sd, c)  # tiny perturbation: top-8 sets must (almost) not move
    same = sum(set(a.tolist()) == set(b.tolist()) for a, b in zip(sel, sel2))
    assert same >= 62
    w3, sel3 = O.router(x, sd, c)
    assert torch.equal(sel, sel3)  # deterministic


def test_router_bias_selects_but_does_not_scale():
    c = O.Hy4Checkpoint.__new__(O.Hy4Checkpoint)  # unused
    from models.demos.blackhole.hy4_preview.reference.loader import Hy4Config

    cfg = Hy4Config.load() if have_ckpt else None
    if cfg is None:
        pytest.skip("needs config")
    torch.manual_seed(1)
    sd = {"mlp.gate.weight": torch.randn(256, 16), "mlp.gate.e_score_correction_bias": torch.zeros(256)}
    x = torch.randn(4, 16)
    w0, s0 = O.router(x, sd, cfg)
    sd["mlp.gate.e_score_correction_bias"] = torch.zeros(256).index_fill_(0, torch.tensor([5]), 100.0)
    w1, s1 = O.router(x, sd, cfg)
    assert (s1 == 5).any(1).all()  # forced into the selection
    assert torch.allclose(w1.sum(-1), w0.sum(-1))  # but still normalised x scale


def test_hc_identity_when_post_zero():
    x4 = torch.randn(1, 5, 4, 8)
    y = torch.randn(1, 5, 8)
    assert torch.equal(O.hc_expand(x4, y, torch.zeros(1, 5, 4)), x4)
    pre = torch.full((1, 5, 4), 0.25)
    assert torch.allclose(O.hc_collapse(x4.repeat(1, 1, 1, 1), pre), x4.mean(2), atol=1e-6)


def _tiny_attn_sd(cfg, torch_gen):
    H, nope, rope, vd, D = 2, 8, 4, 8, 16
    r = lambda *s: torch.randn(*s, generator=torch_gen) * 0.2
    sd = {
        "self_attn.q_a_proj.weight": r(8, D),
        "self_attn.q_a_layernorm.weight": torch.ones(8),
        "self_attn.q_b_proj.weight": r(H * (nope + rope), 8),
        "self_attn.kv_a_proj_with_mqa.weight": r(6 + rope, D),
        "self_attn.kv_a_layernorm.weight": torch.ones(6),
        "self_attn.kv_b_proj.weight": r(H * (nope + vd), 6),
        "self_attn.learnable_sink_param": torch.zeros(H),
        "self_attn.linear_gate.weight": r(H * vd, D),
        "self_attn.o_proj.weight": r(D, H * vd),
    }
    cfg = replace(cfg, n_heads=H, nope=nope, rope=rope, v_dim=vd, kv_lora=6, hidden=D)
    return sd, cfg


@needs_ckpt
def test_dsa_topk_covering_causal_equals_dense_and_sink_matters():
    cfg0 = Hy4Checkpoint().cfg
    gen = torch.Generator().manual_seed(0)
    sd, cfg = _tiny_attn_sd(cfg0, gen)
    S = 9
    h = torch.randn(1, S, cfg.hidden, generator=gen)
    conv = O.Conv()
    dense = O.attn_block(h, h, sd, cfg, conv, None)
    ix = torch.arange(S).view(1, 1, S).expand(1, S, S).clone()
    ix[ix > torch.arange(S).view(1, S, 1)] = -1  # causal prefix only, rest unused
    assert torch.allclose(O.attn_block(h, h, sd, cfg, conv, ix), dense, atol=1e-5)
    # restricting to one key changes the output; a huge sink suppresses attention output towards 0
    one = torch.arange(S).view(1, S, 1)
    assert not torch.allclose(O.attn_block(h, h, sd, cfg, conv, one), dense, atol=1e-4)
    sd2 = dict(sd, **{"self_attn.learnable_sink_param": torch.full((2,), 50.0)})
    assert O.attn_block(h, h, sd2, cfg, conv, None).abs().max() < 1e-3
    # determinism
    assert torch.equal(O.attn_block(h, h, sd, cfg, conv, None), dense)


@needs_ckpt
def test_indexer_selects_top_scores_and_is_causal():
    cfg0 = Hy4Checkpoint().cfg
    gen = torch.Generator().manual_seed(0)
    D, S = 16, 12
    cfg = replace(cfg0, idx_heads=2, idx_dim=8, rope=4, idx_topk=3, hidden=D)
    r = lambda *s: torch.randn(*s, generator=gen)
    sd = {
        "self_attn.indexer.wq_b.weight": r(16, 6),
        "self_attn.indexer.wk.weight": r(8, D),
        "self_attn.indexer.k_norm.weight": torch.ones(8),
        "self_attn.indexer.k_norm.bias": torch.zeros(8),
        "self_attn.indexer.weights_proj.weight": r(2, D),
    }
    h, qr = r(1, S, D), r(1, S, 6)
    cos, sin = O.rope_cos_sin(S, 4, 1e7)
    idx = O.indexer_topk(h, qr, sd, cfg, O.Conv(), cos, sin)
    assert idx.shape == (1, S, 3)
    for s in range(S):
        valid = idx[0, s][idx[0, s] >= 0]
        assert (valid <= s).all() and len(valid) == min(3, s + 1)


@needs_ckpt
def test_layer0_prefill_deterministic_and_finite(ck):
    o = O.Hy4Oracle()
    ids = torch.tensor([[120000, 1000, 2000, 3000, 4000, 5000]])
    x = o.embed_ids(ids)
    a = O.decoder_layers_multi(o.ck, 0, [x], [O.Conv()], [{}])[0]
    b = O.decoder_layers_multi(o.ck, 0, [x], [O.Conv()], [{}])[0]
    assert a.shape == x.shape and torch.isfinite(a).all() and torch.equal(a, b)
    # causality: changing a later token must not change earlier positions
    ids2 = ids.clone()
    ids2[0, -1] = 7777
    c = O.decoder_layers_multi(o.ck, 0, [o.embed_ids(ids2)], [O.Conv()], [{}])[0]
    assert torch.allclose(a[:, :-1], c[:, :-1], atol=1e-5)
