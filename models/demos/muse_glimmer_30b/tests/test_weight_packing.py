# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""CPU-only check that TP weight packing is mathematically equivalent to the torch reference."""

import json
import os

import pytest
import torch
import torch.nn.functional as F

from models.demos.muse_glimmer_30b.reference.model import (
    MLP,
    Attention,
    MuseGlimmerTextConfig,
    RMSNorm,
    RotaryEmbedding,
    apply_rotary_pos_emb,
    build_attention_mask,
)
from models.demos.muse_glimmer_30b.tt.weight_packing import (
    kv_head_for_rank,
    pack_column_parallel,
    pack_down,
    pack_gate_up,
    pack_lm_head,
    pack_o,
    pack_qkvg,
    unit_offset,
)

HD = 128
S = 40
CKPT = os.environ.get("HF_MODEL", "/mnt/tt-data/Muse-Glimmer-30B")


def small_config():
    return MuseGlimmerTextConfig(
        vocab_size=1000,
        hidden_size=512,
        intermediate_size=1024,
        num_hidden_layers=4,
        num_attention_heads=32,
        num_key_value_heads=2,
        head_dim=HD,
        sliding_window=16,
    )


def rand_init(module, seed=0):
    g = torch.Generator().manual_seed(seed)
    for p in module.parameters():
        p.data = torch.randn(p.shape, generator=g, dtype=torch.float32) * 0.05
    return module


def test_kv_head_for_rank():
    assert [kv_head_for_rank(i, 4) for i in range(4)] == [0, 0, 1, 1]
    assert [kv_head_for_rank(i, 8) for i in range(8)] == [0] * 4 + [1] * 4


@pytest.mark.parametrize("tp", [1, 2, 4, 8])
@pytest.mark.parametrize("layer_idx", [0, 3])
def test_attention_tp_sim(tp, layer_idx):
    cfg = small_config()
    attn = rand_init(Attention(cfg, layer_idx), seed=layer_idx).float().eval()
    g = torch.Generator().manual_seed(1)
    x = torch.randn(1, S, cfg.hidden_size, generator=g)

    sliding = cfg.is_sliding(layer_idx)
    assert sliding == (layer_idx == 0)
    rope = RotaryEmbedding(cfg)
    pos_ids = torch.arange(S)[None]
    cos, sin = rope(x, pos_ids)
    mask = build_attention_mask(S, S, 0, cfg.sliding_window if sliding else None, torch.float32)
    with torch.no_grad():
        ref, _ = attn(x, (cos, sin) if sliding else None, mask)

    packed = pack_qkvg(
        attn.q_proj.weight.data, attn.k_proj.weight.data, attn.v_proj.weight.data, attn.gate_proj.weight.data, tp, HD
    )
    o_packed = pack_o(attn.o_proj.weight.data, tp)
    ql = 32 // tp
    kvl = 2 // tp if tp <= 2 else 1
    w_local = (2 * ql + 2 * kvl) * HD
    assert packed.shape == (cfg.hidden_size, tp * w_local)
    norm = RMSNorm(eps=cfg.rms_norm_eps, with_scale=False)

    total = torch.zeros_like(ref)
    for r in range(tp):
        blk = packed[:, r * w_local : (r + 1) * w_local]
        qkvg = x @ blk
        q, k, v, gt = torch.split(qkvg, [ql * HD, kvl * HD, kvl * HD, ql * HD], dim=-1)
        q = q.view(1, S, ql, HD).transpose(1, 2)
        k = k.view(1, S, kvl, HD).transpose(1, 2)
        v = v.view(1, S, kvl, HD).transpose(1, 2)
        q = norm(q) * cfg.qk_scale_factor
        k = norm(k)
        if sliding:
            q, k = apply_rotary_pos_emb(q, k, cos, sin)
        rep = ql // kvl
        kk = k.repeat_interleave(rep, dim=1)  # local q head h -> local kv head h // rep
        vv = v.repeat_interleave(rep, dim=1)
        w = q @ kk.transpose(2, 3) * HD**-0.5 + mask
        w = F.softmax(w, dim=-1)
        o = (w @ vv).transpose(1, 2).reshape(1, S, ql * HD)
        o = o * torch.sigmoid(gt)
        total += o @ o_packed[r * ql * HD : (r + 1) * ql * HD]  # all_reduce
    assert (total - ref).abs().max().item() < 1e-4


@pytest.mark.parametrize("tp", [1, 2, 4, 8])
def test_mlp_tp_sim(tp):
    cfg = small_config()
    mlp = rand_init(MLP(cfg)).float().eval()
    x = torch.randn(1, S, cfg.hidden_size, generator=torch.Generator().manual_seed(2))
    with torch.no_grad():
        ref = mlp(x)
    gu = pack_gate_up(mlp.gate_proj.weight.data, mlp.up_proj.weight.data, tp)
    down = pack_down(mlp.down_proj.weight.data, tp)
    il = cfg.intermediate_size // tp
    assert gu.shape == (cfg.hidden_size, tp * 2 * il)
    total = torch.zeros_like(ref)
    for r in range(tp):
        h = x @ gu[:, r * 2 * il : (r + 1) * 2 * il]
        gate, up = h[..., :il], h[..., il:]
        total += (F.silu(gate) * up) @ down[r * il : (r + 1) * il]
    assert (total - ref).abs().max().item() < 1e-4


@pytest.mark.parametrize("tp", [1, 2, 4, 8])
def test_mlp_tp_sim_padded(tp):
    cfg = small_config()
    cfg.intermediate_size = 1216  # I_local = 1216/tp: 1216, 608, 304, 152 -> non-trivial pad to 1536/1024/512/512
    mlp = rand_init(MLP(cfg)).float().eval()
    x = torch.randn(1, S, cfg.hidden_size, generator=torch.Generator().manual_seed(2))
    with torch.no_grad():
        ref = mlp(x)
    il = cfg.intermediate_size // tp
    pad = -(-il // 512) * 512
    assert pad > il
    gu = pack_gate_up(mlp.gate_proj.weight.data, mlp.up_proj.weight.data, tp, pad_to=pad)
    down = pack_down(mlp.down_proj.weight.data, tp, pad_to=pad)
    assert gu.shape == (cfg.hidden_size, tp * 2 * pad)
    assert down.shape == (tp * pad, cfg.hidden_size)
    assert (gu[:, pad + il : 2 * pad] == 0).all() and (down[il:pad] == 0).all()
    total = torch.zeros_like(ref)
    for r in range(tp):
        h = x @ gu[:, r * 2 * pad : (r + 1) * 2 * pad]
        gate, up = h[..., :pad], h[..., pad:]
        total += (F.silu(gate) * up) @ down[r * pad : (r + 1) * pad]
    assert (total - ref).abs().max().item() < 1e-4
    # pad_to=None means no padding
    assert torch.equal(pack_down(mlp.down_proj.weight.data, tp), mlp.down_proj.weight.data.T)


@pytest.mark.parametrize("tp", [1, 2, 4, 8])
def test_pack_column_parallel_matches_gate_up(tp):
    g = torch.Generator().manual_seed(5)
    gate_w, up_w = torch.randn(1216, 64, generator=g), torch.randn(1216, 64, generator=g)
    il = 1216 // tp
    pad = -(-il // 512) * 512
    gu = pack_gate_up(gate_w, up_w, tp, pad_to=pad)
    pg, pu = pack_column_parallel(gate_w, tp, pad_to=pad), pack_column_parallel(up_w, tp, pad_to=pad)
    assert pg.shape == (64, tp * pad)
    for r in range(tp):
        assert torch.equal(pg[:, r * pad : (r + 1) * pad], gu[:, r * 2 * pad : r * 2 * pad + pad])
        assert torch.equal(pu[:, r * pad : (r + 1) * pad], gu[:, r * 2 * pad + pad : (r + 1) * 2 * pad])
    assert torch.equal(pack_column_parallel(gate_w, tp), pack_column_parallel(gate_w, tp, pad_to=il))


def test_lm_head_pack():
    g = torch.Generator().manual_seed(3)
    lm_w = torch.randn(1000, 512, generator=g)
    x = torch.randn(S, 512, generator=g)
    p = pack_lm_head(lm_w, 2048)
    assert p.shape == (512, 2048)
    logits = x @ p
    assert torch.allclose(logits[:, :1000], x @ lm_w.T, atol=1e-5)
    assert (logits[:, 1000:] == 0).all()
    assert pack_lm_head(torch.zeros(202048, 4), 202752).shape == (4, 202752)


def test_unit_offset():
    w = torch.randn(16)
    o = unit_offset(w)
    assert o.dtype == torch.bfloat16
    assert torch.equal(o, (1 + w).to(torch.bfloat16))


@pytest.mark.skipif(not os.path.isdir(CKPT), reason="checkpoint not available")
def test_real_checkpoint_shapes():
    from safetensors import safe_open

    with open(os.path.join(CKPT, "model.safetensors.index.json")) as f:
        wm = json.load(f)["weight_map"]
    pre = "model.language_model.layers.3.self_attn."

    def get(name):
        key = pre + name + ".weight"
        with safe_open(os.path.join(CKPT, wm[key]), framework="pt", device="cpu") as sf:
            return sf.get_tensor(key)

    q, k, v, g = (get(n) for n in ["q_proj", "k_proj", "v_proj", "gate_proj"])
    p = pack_qkvg(q, k, v, g, 4)
    assert p.shape == (6656, 4 * 2304)
    w = 2304
    assert torch.equal(p[:, 2 * w + 1024 : 2 * w + 1024 + 128], k[128:256].T)
    assert torch.equal(p[:, 1 * w + 1024 : 1 * w + 1024 + 128], k[0:128].T)
