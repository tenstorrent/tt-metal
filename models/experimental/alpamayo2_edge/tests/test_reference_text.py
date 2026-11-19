# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

# Portions derived from HF transformers (the upstream oracle functions):
# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import replace

import torch
import torch.nn.functional as F

from models.experimental.alpamayo2_edge.reference.config import TextConfig
from models.experimental.alpamayo2_edge.reference.rope_index import get_rope_index
from models.experimental.alpamayo2_edge.reference.text import KVCache, LanguageModel, TextModel, mrope_cos_sin
from models.experimental.alpamayo2_edge.reference.weights import load_weights

SMALL = TextConfig(hidden=64, layers=2, mlp=96, heads=4, kv_heads=2, head_dim=128, vocab=32)


def _random_weights(tcfg: TextConfig, has_embedding: bool, seed: int = 0) -> dict:
    g = torch.Generator().manual_seed(seed)

    def rnd(*shape, scale=1.0):
        return torch.randn(*shape, generator=g) * scale

    q_out, kv_out, h = tcfg.heads * tcfg.head_dim, tcfg.kv_heads * tcfg.head_dim, tcfg.hidden
    w = {"norm.weight": 1 + 0.1 * rnd(h)}
    if has_embedding:
        w["embed_tokens.weight"] = rnd(tcfg.vocab, h)
    for i in range(tcfg.layers):
        p = f"layers.{i}."
        w[p + "input_layernorm.weight"] = 1 + 0.1 * rnd(h)
        w[p + "post_attention_layernorm.weight"] = 1 + 0.1 * rnd(h)
        w[p + "self_attn.q_proj.weight"] = rnd(q_out, h, scale=h**-0.5)
        w[p + "self_attn.k_proj.weight"] = rnd(kv_out, h, scale=h**-0.5)
        w[p + "self_attn.v_proj.weight"] = rnd(kv_out, h, scale=h**-0.5)
        w[p + "self_attn.o_proj.weight"] = rnd(h, q_out, scale=q_out**-0.5)
        w[p + "mlp.fc1.weight"] = rnd(tcfg.mlp, h, scale=h**-0.5)
        w[p + "mlp.fc2.weight"] = rnd(h, tcfg.mlp, scale=tcfg.mlp**-0.5)
    return w


def _mrope_positions(batch: int, seq: int, seed: int = 0) -> torch.Tensor:
    """Distinct, non-monotonic T/H/W positions so any axis mix-up shows."""
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, 4 * seq, (3, batch, seq), generator=g)


# ---------------------------------------------------------------------------------------------------------------------
# Upstream oracle: verbatim copies of the Cosmos3 Edge modeling functions (transformers cosmos3_edge, Apache-2.0).
# ---------------------------------------------------------------------------------------------------------------------


def _upstream_cos_sin(position_ids, inv_freq, mrope_section):
    def recomposition_frequencies(freq):
        freqs_thw = freq[0]  # just overwrite the first dimension T
        for dim, offset in enumerate((1, 2), start=1):  # H, W
            length = mrope_section[dim] * 3
            idx = slice(offset, length, 3)
            freqs_thw[..., idx] = freq[dim, ..., idx]
        return torch.cat((freqs_thw, freqs_thw), dim=-1)

    position_ids = position_ids.expand(3, -1, -1)
    freqs = position_ids[..., None].float() * inv_freq.float()
    cos = freqs.cos()
    sin = freqs.sin()
    return recomposition_frequencies(cos), recomposition_frequencies(sin)


def _upstream_inv_freq(head_dim, base):
    return 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float) / head_dim))


def _rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    return (q * cos) + (_rotate_half(q) * sin), (k * cos) + (_rotate_half(k) * sin)


def _repeat_kv(hidden_states, n_rep):
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def _eager_attention(query, key, value, attention_mask, scaling, n_rep):
    key_states = _repeat_kv(key, n_rep)
    value_states = _repeat_kv(value, n_rep)
    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask
    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    return torch.matmul(attn_weights, value_states).transpose(1, 2).contiguous()


def _rms_norm(x, weight, eps):
    input_dtype = x.dtype
    x = x.to(torch.float32)
    variance = x.pow(2).mean(-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)
    return weight * x.to(input_dtype)


def _upstream_text_model(tcfg, w, embeds, position_ids):
    """Uncached causal Cosmos3EdgeTextModel forward with eager attention."""
    bsz, seq, _ = embeds.shape
    cos, sin = _upstream_cos_sin(position_ids, _upstream_inv_freq(tcfg.head_dim, tcfg.rope_theta), tcfg.mrope_section)
    mask = torch.full((seq, seq), torch.finfo(embeds.dtype).min).triu(1)[None, None]
    x = embeds
    for i in range(tcfg.layers):
        p = f"layers.{i}."
        residual = x
        h = _rms_norm(x, w[p + "input_layernorm.weight"], tcfg.eps)
        shape = (bsz, seq, -1, tcfg.head_dim)
        q = F.linear(h, w[p + "self_attn.q_proj.weight"]).view(shape).transpose(1, 2)
        k = F.linear(h, w[p + "self_attn.k_proj.weight"]).view(shape).transpose(1, 2)
        v = F.linear(h, w[p + "self_attn.v_proj.weight"]).view(shape).transpose(1, 2)
        q, k = _apply_rotary_pos_emb(q, k, cos, sin)
        attn = _eager_attention(q, k, v, mask, tcfg.head_dim**-0.5, tcfg.heads // tcfg.kv_heads)
        x = residual + F.linear(attn.reshape(bsz, seq, -1), w[p + "self_attn.o_proj.weight"])
        residual = x
        h = _rms_norm(x, w[p + "post_attention_layernorm.weight"], tcfg.eps)
        h = F.linear(torch.square(F.relu(F.linear(h, w[p + "mlp.fc1.weight"]))), w[p + "mlp.fc2.weight"])
        x = residual + h
    return _rms_norm(x, w["norm.weight"], tcfg.eps)


# ---------------------------------------------------------------------------------------------------------------------
# M-RoPE
# ---------------------------------------------------------------------------------------------------------------------


def test_mrope_equal_axes_is_1d():
    batch, seq = 2, 37
    pos = torch.arange(seq).expand(3, batch, seq)
    cos, sin = mrope_cos_sin(pos, head_dim=128, theta=1e8, section=(24, 20, 20))
    assert cos.shape == sin.shape == (batch, seq, 128)

    inv_freq = 1.0 / (1e8 ** (torch.arange(0, 128, 2, dtype=torch.float) / 128))
    emb = torch.arange(seq, dtype=torch.float)[:, None] * inv_freq
    emb = torch.cat((emb, emb), dim=-1)
    torch.testing.assert_close(cos, emb.cos().expand(batch, seq, 128))
    torch.testing.assert_close(sin, emb.sin().expand(batch, seq, 128))


def test_mrope_interleaved_matches_upstream():
    pos = _mrope_positions(batch=2, seq=19)
    cos, sin = mrope_cos_sin(pos, head_dim=128, theta=1e8, section=(24, 20, 20))
    ref_cos, ref_sin = _upstream_cos_sin(pos.clone(), _upstream_inv_freq(128, 1e8), (24, 20, 20))
    assert torch.equal(cos, ref_cos)
    assert torch.equal(sin, ref_sin)

    # The layout is interleaved T,H,W,T,H,W,... over the first 60 frequencies with T owning the last 4,
    # not the chunked [T*24 | H*20 | W*20] Qwen2-VL layout.
    t, h, w = (mrope_cos_sin(pos[a].expand(3, -1, -1), section=(24, 20, 20))[0] for a in range(3))
    for freq in range(64):
        source = t if freq >= 60 else (t, h, w)[freq % 3]
        assert torch.equal(cos[..., freq], source[..., freq]), freq
        assert torch.equal(cos[..., 64 + freq], source[..., freq]), freq


# ---------------------------------------------------------------------------------------------------------------------
# Rope index
# ---------------------------------------------------------------------------------------------------------------------


def test_rope_index_text_only():
    batch, seq = 2, 23
    input_ids = torch.randint(100, 1000, (batch, seq))
    position_ids, rope_deltas = get_rope_index(input_ids, torch.zeros_like(input_ids), None, None)
    assert position_ids.shape == (3, batch, seq)
    assert rope_deltas.shape == (batch, 1)
    for a in range(3):
        assert torch.equal(position_ids[a], torch.arange(seq).expand(batch, seq))
    assert torch.equal(rope_deltas, torch.zeros(batch, 1, dtype=rope_deltas.dtype))


def test_rope_index_images_with_left_padding():
    # Row 0: 2 text, image grid (1, 4, 6) -> 2x3 tokens, 3 text.
    # Row 1: 2 left pads, 2 text, image grid (1, 4, 4) -> 2x2 tokens, 3 text.
    types = torch.tensor(
        [
            [0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0],
        ]
    )
    attention_mask = torch.tensor([[1] * 11, [0, 0] + [1] * 9])
    input_ids = torch.randint(100, 1000, types.shape)
    grid = torch.tensor([[1, 4, 6], [1, 4, 4]])
    position_ids, rope_deltas = get_rope_index(input_ids, types, grid, attention_mask, merge=2)

    expected = torch.tensor(
        [
            [
                [0, 1, 2, 2, 2, 2, 2, 2, 5, 6, 7],
                [0, 1, 2, 2, 2, 3, 3, 3, 5, 6, 7],
                [0, 1, 2, 3, 4, 2, 3, 4, 5, 6, 7],
            ],
            [
                [0, 0, 0, 1, 2, 2, 2, 2, 4, 5, 6],
                [0, 0, 0, 1, 2, 2, 3, 3, 4, 5, 6],
                [0, 0, 0, 1, 2, 3, 2, 3, 4, 5, 6],
            ],
        ]
    ).permute(1, 0, 2)
    assert torch.equal(position_ids, expected)
    assert rope_deltas.tolist() == [[-3], [-2]]


def test_rope_index_video_frames_are_split():
    # One video of 2 frames (grid (2, 4, 4)); the processor emits each frame as its own span.
    types = torch.tensor([[0, 2, 2, 2, 2, 0, 0, 2, 2, 2, 2, 0]])
    input_ids = torch.randint(100, 1000, types.shape)
    position_ids, rope_deltas = get_rope_index(
        input_ids, types, None, None, merge=2, video_grid_thw=torch.tensor([[2, 4, 4]])
    )
    expected = torch.tensor(
        [
            [0, 1, 1, 1, 1, 3, 4, 5, 5, 5, 5, 7],
            [0, 1, 1, 2, 2, 3, 4, 5, 5, 6, 6, 7],
            [0, 1, 2, 1, 2, 3, 4, 5, 6, 5, 6, 7],
        ]
    )[:, None]
    assert torch.equal(position_ids, expected)
    assert rope_deltas.tolist() == [[-4]]


# ---------------------------------------------------------------------------------------------------------------------
# KV cache
# ---------------------------------------------------------------------------------------------------------------------


def test_cache_crop():
    prefix, layers = 10, 2
    cache = KVCache(layers)
    assert cache.seq_len == 0
    for layer in range(layers):
        cache.append(layer, torch.randn(2, 8, prefix, 128), torch.randn(2, 8, prefix, 128))
    snapshot = [(cache.k[layer].clone(), cache.v[layer].clone()) for layer in range(layers)]
    assert cache.seq_len == prefix

    for layer in range(layers):
        k, v = cache.append(layer, torch.randn(2, 8, 64, 128), torch.randn(2, 8, 64, 128))
        assert k.shape == v.shape == (2, 8, prefix + 64, 128)
    assert cache.seq_len == prefix + 64

    cache.crop(prefix)
    assert cache.seq_len == prefix
    for layer, (k, v) in enumerate(snapshot):
        assert torch.equal(cache.k[layer], k)
        assert torch.equal(cache.v[layer], v)


def test_cache_repeat_interleave():
    cache = KVCache(1)
    k, v = torch.randn(2, 8, 5, 128), torch.randn(2, 8, 5, 128)
    cache.append(0, k, v)
    cache.repeat_interleave(3)
    assert torch.equal(cache.k[0], k.repeat_interleave(3, dim=0))
    assert torch.equal(cache.v[0], v.repeat_interleave(3, dim=0))


# ---------------------------------------------------------------------------------------------------------------------
# Text model
# ---------------------------------------------------------------------------------------------------------------------


def test_relu2_mlp():
    weights = _random_weights(SMALL, has_embedding=False)
    model = TextModel(SMALL, weights, has_embedding=False)
    x = torch.randn(2, 5, SMALL.hidden)
    fc1, fc2 = weights["layers.0.mlp.fc1.weight"], weights["layers.0.mlp.fc2.weight"]
    expected = F.linear(F.relu(F.linear(x, fc1)) ** 2, fc2)
    with torch.no_grad():
        torch.testing.assert_close(model.layers[0].mlp(x), expected)


def test_text_model_matches_upstream():
    weights = _random_weights(SMALL, has_embedding=True)
    model = TextModel(SMALL, weights, has_embedding=True)
    assert model.embed_tokens is not None and len(model.layers) == SMALL.layers
    embeds = torch.randn(2, 13, SMALL.hidden)
    pos = _mrope_positions(2, 13)
    with torch.no_grad():
        out = model(embeds, pos, cache=None, attn_mask=None, causal=True)
    expected = _upstream_text_model(SMALL, weights, embeds, pos)
    torch.testing.assert_close(out, expected, rtol=1e-5, atol=1e-5)


def test_cached_decode_matches_full_sequence():
    model = TextModel(SMALL, _random_weights(SMALL, has_embedding=False), has_embedding=False)
    embeds = torch.randn(2, 12, SMALL.hidden)
    pos = _mrope_positions(2, 12)
    with torch.no_grad():
        full = model(embeds, pos, causal=True)
        cache = KVCache(SMALL.layers)
        parts = [model(embeds[:, s:e], pos[..., s:e], cache=cache, causal=True) for s, e in ((0, 7), (7, 8), (8, 12))]
    assert cache.seq_len == 12
    torch.testing.assert_close(torch.cat(parts, dim=1), full, rtol=1e-5, atol=1e-5)


def test_noncausal_expert_on_shared_cache():
    lm = TextModel(SMALL, _random_weights(SMALL, has_embedding=True, seed=1), has_embedding=True)
    expert_cfg = replace(SMALL, hidden=32, mlp=48)
    expert = TextModel(expert_cfg, _random_weights(expert_cfg, has_embedding=False, seed=2), has_embedding=False)
    assert expert.embed_tokens is None

    batch, prefix, n_act, valid = 2, 9, 4, 6
    cache = KVCache(SMALL.layers)
    with torch.no_grad():
        lm(lm.embed_tokens(torch.randint(0, SMALL.vocab, (batch, prefix))), _mrope_positions(batch, prefix), cache)
    assert cache.seq_len == prefix

    # Cache columns at or past `valid` are hidden from the expert, as in build_expert_pos_ids_and_attn_mask.
    attn_mask = torch.zeros(batch, 1, n_act, prefix + n_act)
    attn_mask[..., valid:prefix] = torch.finfo(torch.float32).min
    x = torch.randn(batch, n_act, expert_cfg.hidden)
    pos = _mrope_positions(batch, n_act, seed=3)

    def run(inputs):
        with torch.no_grad():
            out = expert(inputs, pos, cache=cache, attn_mask=attn_mask, causal=False)
        assert cache.seq_len == prefix + n_act
        cache.crop(prefix)
        return out

    out = run(x)
    for layer in range(SMALL.layers):
        cache.k[layer][:, :, valid:] += 100.0
        cache.v[layer][:, :, valid:] += 100.0
    torch.testing.assert_close(run(x), out, rtol=0, atol=0)

    # Non-causal: the last action token is visible to the first one.
    x2 = x.clone()
    x2[:, -1] += 1.0
    assert not torch.allclose(run(x2)[:, 0], out[:, 0])


# ---------------------------------------------------------------------------------------------------------------------
# Real checkpoint
# ---------------------------------------------------------------------------------------------------------------------


def _first_layers(weights: dict, layers: int) -> dict:
    return {k: v for k, v in weights.items() if not k.startswith("layers.") or int(k.split(".")[1]) < layers}


def test_checkpoint_lm_and_expert(cfg, model_dir):
    layers = 2
    lm_cfg, expert_cfg = replace(cfg.text, layers=layers), replace(cfg.expert, layers=layers)
    lm = LanguageModel(
        lm_cfg,
        _first_layers(load_weights(model_dir, "vlm.model.language_model."), layers),
        load_weights(model_dir, "vlm.lm_head."),
    )
    expert = TextModel(
        expert_cfg, _first_layers(load_weights(model_dir, "expert.expert."), layers), has_embedding=False
    )
    assert isinstance(lm.text, TextModel) and len(lm.text.layers) == layers

    tok = cfg.tokens
    # 3 text, vision_start, image grid (1, 4, 4) -> 4 image tokens, vision_end, 3 text
    ids = torch.tensor([[1, 2, 3, tok.vision_start] + [tok.image] * 4 + [tok.vision_end, 4, 5, 6]])
    types = (ids == tok.image).long()
    position_ids, rope_deltas = get_rope_index(ids, types, torch.tensor([[1, 4, 4]]), None)
    cache = KVCache(layers)
    with torch.no_grad():
        hidden = lm.text(lm.text.embed_tokens(ids), position_ids, cache=cache, causal=True)
        logits = lm.logits(hidden[:, -1:])
        assert logits.shape == (1, 1, cfg.text.vocab)
        assert torch.isfinite(logits).all()

        prefix, n_act = cache.seq_len, 8
        expert_pos = (torch.arange(n_act) + prefix + rope_deltas).expand(3, 1, n_act)
        out = expert(
            torch.randn(1, n_act, cfg.expert.hidden),
            expert_pos,
            cache=cache,
            attn_mask=torch.zeros(1, 1, n_act, prefix + n_act),
            causal=False,
        )
    assert out.shape == (1, n_act, cfg.expert.hidden)
    assert torch.isfinite(out).all()
    assert cache.seq_len == prefix + n_act
    cache.crop(prefix)
    assert cache.seq_len == prefix
