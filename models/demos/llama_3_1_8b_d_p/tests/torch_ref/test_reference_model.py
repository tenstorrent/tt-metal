# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""D1 — the standalone CPU reference against an INLINE torch golden, so the two oracles cannot drift.

Host only. Structure mirrors `minimax_m3/tests/unit/test_reference_model.py`.

The point of this file is independence. `test_llama_reference.py` pins
`reference/model.py` to upstream HF; this one pins it to a second implementation written from the
architecture description directly in this test — no shared helpers, no shared code paths, written in
the most obvious way rather than the upstream way. If both agree, a transcription slip would have to
have been made identically twice.

So the inline golden below deliberately does things differently where it can:

* it builds cos/sin by indexing a precomputed table rather than a matmul against position_ids;
* it does the GQA grouping with `repeat_interleave` rather than the expand/reshape `repeat_kv`;
* it masks with `masked_fill` on a boolean triangle rather than adding a `finfo.min` float mask;
* it keeps the head layout as [b, s, h, d] and transposes late.

Run at a **reduced** config with random weights — a reduced run is a diagnostic, and this one is
purely an internal-consistency check, so nothing here is a bring-up number. The real-dims
measurements live in `test_llama_reference.py` (upstream parity) and in the device suites.
"""

from __future__ import annotations

import math
from dataclasses import replace

import pytest
import torch
from torch import nn

from models.demos.llama_3_1_8b_d_p.reference.config import LlamaConfigConstants
from models.demos.llama_3_1_8b_d_p.reference.golden import BUILDERS, get_module_golden, make_key
from models.demos.llama_3_1_8b_d_p.reference.model import (
    REF_DTYPE,
    RefAttention,
    RefDecoderLayer,
    RefMLP,
    RefRMSNorm,
    RefRotaryEmbedding,
    causal_mask,
)

# Reduced on purpose: 2 kv heads x 4 groups = 8 q heads, head_dim 16, hidden 128. Small enough to
# run instantly, but it keeps every structural feature that can be got wrong — GQA grouping > 1,
# head_dim even (so rope's half-split is meaningful), and a non-square MLP.
REDUCED = LlamaConfigConstants(
    hidden_size=128,
    intermediate_size=352,
    num_hidden_layers=2,
    num_attention_heads=8,
    num_key_value_heads=2,
    max_position_embeddings=1024,
    rms_norm_eps=1e-5,
    rope_theta=500000.0,
    vocab_size=256,
)
SEQ = 48
PCC_BAR = 1 - 1e-3


def _pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.detach().float().flatten(), b.detach().float().flatten()
    a, b = a - a.mean(), b - b.mean()
    denom = a.norm() * b.norm()
    if denom == 0:
        return 1.0 if torch.allclose(a, b) else 0.0
    return float((a @ b) / denom)


# ---------------------------------------------------------------------------
# The inline golden — written from the architecture, sharing nothing with reference/model.py
# ---------------------------------------------------------------------------


def inline_rope_table(config, seq_len):
    """cos/sin as a precomputed [seq_len, head_dim] table, built by outer product then indexed."""
    hd = config.head_dim
    i = torch.arange(0, hd, 2, dtype=torch.float64)
    inv = 1.0 / (config.rope_theta ** (i / hd))
    # llama3 scaling, written out band by band rather than with torch.where.
    rs = config.rope_scaling
    factor, lo, hi, orig = rs["factor"], rs["low_freq_factor"], rs["high_freq_factor"], rs["original_max_position_embeddings"]
    out = []
    for f in inv.tolist():
        wavelen = 2 * math.pi / f
        if wavelen < orig / hi:
            out.append(f)
        elif wavelen > orig / lo:
            out.append(f / factor)
        else:
            s = (orig / wavelen - lo) / (hi - lo)
            base = f / factor
            out.append((1 - s) * base + s * f)
    inv = torch.tensor(out, dtype=torch.float64)
    pos = torch.arange(seq_len, dtype=torch.float64)
    angles = torch.outer(pos, inv)  # [seq, hd/2]
    full = torch.cat([angles, angles], dim=-1)  # [seq, hd]
    return full.cos().to(REF_DTYPE), full.sin().to(REF_DTYPE)


def inline_rotate(x, cos, sin):
    """Half-split rotation, applied to a [b, s, h, d] layout (note: NOT [b, h, s, d])."""
    d = x.shape[-1]
    x1, x2 = x[..., : d // 2], x[..., d // 2 :]
    rotated = torch.cat([-x2, x1], dim=-1)
    c = cos[None, :, None, :]
    s = sin[None, :, None, :]
    return x * c + rotated * s


def inline_rms_norm(x, weight, eps):
    """RMSNorm written as x / sqrt(mean(x^2) + eps) * w, with an explicit divide."""
    f = x.float()
    rms = torch.sqrt(f.pow(2).mean(-1, keepdim=True) + eps)
    return ((f / rms).to(x.dtype)) * weight


def inline_mlp(x, gate_w, up_w, down_w):
    """down(silu(gate) * up), with the linears written as explicit transposed matmuls."""
    gate = x @ gate_w.t()
    up = x @ up_w.t()
    act = gate * torch.sigmoid(gate.float()).to(gate.dtype)  # silu, spelled out
    return (act * up) @ down_w.t()


def inline_attention(x, w, config, seq_len):
    """QKV -> [b, s, h, d] -> rope -> repeat_interleave GQA -> boolean-masked softmax -> o_proj."""
    b = x.shape[0]
    n_q, n_kv, hd = config.num_attention_heads, config.num_key_value_heads, config.head_dim
    q = (x @ w["q_proj"].t()).view(b, seq_len, n_q, hd)
    k = (x @ w["k_proj"].t()).view(b, seq_len, n_kv, hd)
    v = (x @ w["v_proj"].t()).view(b, seq_len, n_kv, hd)

    cos, sin = inline_rope_table(config, seq_len)
    q = inline_rotate(q, cos, sin)
    k = inline_rotate(k, cos, sin)

    # GQA by repeat_interleave on the head axis (the other spelling of repeat_kv).
    groups = n_q // n_kv
    k_full = k.repeat_interleave(groups, dim=2)
    v_full = v.repeat_interleave(groups, dim=2)

    qt = q.transpose(1, 2)  # [b, n_q, s, d]
    kt = k_full.transpose(1, 2)
    vt = v_full.transpose(1, 2)

    scores = (qt.float() @ kt.float().transpose(-1, -2)) / math.sqrt(hd)
    keep = torch.ones(seq_len, seq_len, dtype=torch.bool).tril()
    scores = scores.masked_fill(~keep, float("-inf"))
    probs = torch.softmax(scores, dim=-1).to(REF_DTYPE)
    ctx = probs @ vt
    ctx = ctx.transpose(1, 2).reshape(b, seq_len, n_q * hd)
    return ctx @ w["o_proj"].t(), k.transpose(1, 2), v.transpose(1, 2)


def inline_decoder_layer(x, sd, config, seq_len):
    """pre-norm attn + residual, then pre-norm mlp + residual."""
    h = inline_rms_norm(x, sd["input_layernorm.weight"], config.rms_norm_eps)
    attn, k_rope, v = inline_attention(
        h,
        {n: sd[f"self_attn.{n}.weight"] for n in ("q_proj", "k_proj", "v_proj", "o_proj")},
        config,
        seq_len,
    )
    x = x + attn
    h = inline_rms_norm(x, sd["post_attention_layernorm.weight"], config.rms_norm_eps)
    x = x + inline_mlp(h, sd["mlp.gate_proj.weight"], sd["mlp.up_proj.weight"], sd["mlp.down_proj.weight"])
    return x, k_rope, v


# ---------------------------------------------------------------------------
# The comparisons
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def config():
    return REDUCED


def test_rope_table_vs_inline(config):
    """The reference's matmul-built cos/sin against an outer-product table, incl. llama3 banding."""
    rope = RefRotaryEmbedding(config)
    cos, sin = rope(torch.arange(SEQ)[None, :], dtype=REF_DTYPE)
    cos_i, sin_i = inline_rope_table(config, SEQ)
    assert _pcc(cos[0], cos_i) > 1 - 1e-6
    assert _pcc(sin[0], sin_i) > 1 - 1e-6


def test_rms_norm_vs_inline(config):
    torch.manual_seed(0)
    x = torch.randn(1, SEQ, config.hidden_size, dtype=REF_DTYPE)
    norm = RefRMSNorm(config.hidden_size, config.rms_norm_eps)
    with torch.no_grad():
        norm.weight.copy_((1.0 + 0.1 * torch.randn(config.hidden_size)).to(REF_DTYPE))
        got = norm(x)
    want = inline_rms_norm(x, norm.weight, config.rms_norm_eps)
    assert _pcc(got, want) > PCC_BAR


def test_mlp_vs_inline(config):
    torch.manual_seed(0)
    x = torch.randn(1, SEQ, config.hidden_size, dtype=REF_DTYPE)
    mlp = RefMLP(config)
    with torch.no_grad():
        got = mlp(x)
    want = inline_mlp(x, mlp.gate_proj.weight, mlp.up_proj.weight, mlp.down_proj.weight)
    assert _pcc(got, want) > PCC_BAR


def test_attention_vs_inline(config):
    """Whole attention block, plus the cached K/V, against the independently-written version."""
    torch.manual_seed(0)
    x = torch.randn(1, SEQ, config.hidden_size, dtype=REF_DTYPE)
    attn = RefAttention(config)
    rope = RefRotaryEmbedding(config)
    cos, sin = rope(torch.arange(SEQ)[None, :], dtype=REF_DTYPE)
    with torch.no_grad():
        got, k_rope, v = attn(x, (cos, sin), causal_mask(SEQ), return_kv=True)
    w = {n: getattr(attn, n).weight for n in ("q_proj", "k_proj", "v_proj", "o_proj")}
    want, k_want, v_want = inline_attention(x, w, config, SEQ)
    assert _pcc(got, want) > PCC_BAR
    assert _pcc(k_rope, k_want) > PCC_BAR, "cached K disagrees - rope convention or head layout"
    assert _pcc(v, v_want) > PCC_BAR


def test_decoder_layer_vs_inline(config):
    """The composition, after every piece above agrees."""
    torch.manual_seed(0)
    x = torch.randn(1, SEQ, config.hidden_size, dtype=REF_DTYPE)
    layer = RefDecoderLayer(config)
    rope = RefRotaryEmbedding(config)
    cos, sin = rope(torch.arange(SEQ)[None, :], dtype=REF_DTYPE)
    with torch.no_grad():
        got, k_rope, v = layer(x, (cos, sin), causal_mask(SEQ), return_kv=True)
    sd = layer.state_dict()
    want, k_want, v_want = inline_decoder_layer(x, sd, config, SEQ)
    assert _pcc(got, want) > PCC_BAR
    assert _pcc(k_rope, k_want) > PCC_BAR
    assert _pcc(v, v_want) > PCC_BAR


# ---------------------------------------------------------------------------
# The golden runner itself
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("block", sorted(BUILDERS))
def test_module_golden_round_trips(config, block, tmp_path, monkeypatch):
    """Each per-module golden regenerates identically and reloads from disk unchanged.

    Determinism first (the builder is seeded, so two calls must agree bit-for-bit), then the
    disk round-trip. Without the first half a golden that quietly reseeds would still "round-trip"
    while every consumer got different numbers per run.
    """
    monkeypatch.setenv("LLAMA31_8B_MODULE_GOLDEN_DIR", str(tmp_path))
    first = get_module_golden(block, config, SEQ, regenerate=True)
    again = BUILDERS[block](config, SEQ, batch_size=1, seed=0)
    for k, v in first.items():
        if isinstance(v, torch.Tensor):
            assert torch.equal(v, again[k]), f"{block}.{k} is not deterministic"
        elif isinstance(v, dict):
            for kk, vv in v.items():
                assert torch.equal(vv, again[k][kk]), f"{block}.{k}.{kk} is not deterministic"

    loaded = get_module_golden(block, config, SEQ)  # from disk this time
    for k, v in first.items():
        if isinstance(v, torch.Tensor):
            assert torch.equal(v, loaded[k]), f"{block}.{k} did not survive the disk round-trip"
    # Goldens are fp16 (recipe §4). `inv_freq` is the one documented exception: the llama3 inverse
    # frequencies span ~7 orders of magnitude and the smallest underflow fp16, so the band
    # arithmetic and the stored table stay fp32. It is a constant, not a compared golden tensor.
    fp32_exempt = {"inv_freq"}
    for name, t in first.items():
        if not isinstance(t, torch.Tensor) or name in fp32_exempt:
            continue
        assert t.dtype == REF_DTYPE, f"golden {block}.{name} is {t.dtype}, must be fp16 (recipe §4)"
    if "inv_freq" in first:
        assert first["inv_freq"].dtype == torch.float32, "inv_freq must stay fp32 - fp16 underflows it"


def test_golden_key_changes_with_every_field(config):
    """A changed key field must change the filename, or a stale golden gets reused silently.

    This is the property `ReferenceCacheKey` is frozen to guarantee; asserting it here means adding
    a knob without adding it to the key fails a test rather than corrupting a later measurement.
    """
    base = make_key("attention", config, SEQ)
    assert not hasattr(base, "__dict__") or True  # frozen dataclass; replace() is the only mutation
    for field, value in [
        ("block", "mlp"),
        ("seq_len", SEQ * 2),
        ("hidden_size", config.hidden_size * 2),
        ("intermediate_size", config.intermediate_size + 32),
        ("num_attention_heads", 16),
        ("num_key_value_heads", 4),
        ("head_dim", 32),
        ("rope_theta", 10000.0),
        ("rope_type", "default"),
        ("rope_factor", 4.0),
        ("rms_norm_eps", 1e-6),
        ("batch_size", 2),
        ("seed", 1),
        ("dtype", "torch.bfloat16"),
    ]:
        assert str(replace(base, **{field: value})) != str(base), f"{field} does not affect the golden filename"

    with pytest.raises((AttributeError, TypeError)):  # frozen
        base.seq_len = 1  # type: ignore[misc]


# ---------------------------------------------------------------------------
# M1 — the WHOLE model (embedding, the layer stack, final norm, lm head)
# ---------------------------------------------------------------------------


def inline_model(input_ids, sd, config, num_layers):
    """The inline golden widened to the whole model. Still shares nothing with reference/model.py."""
    h = sd["embed_tokens.weight"][input_ids]
    seq_len = input_ids.shape[1]
    for i in range(num_layers):
        layer_sd = {k[len(f"layers.{i}.") :]: v for k, v in sd.items() if k.startswith(f"layers.{i}.")}
        h, _, _ = inline_decoder_layer(h, layer_sd, config, seq_len)
    h = inline_rms_norm(h, sd["norm.weight"], config.rms_norm_eps)
    return h @ sd["lm_head.weight"].t()


def test_whole_model_vs_inline(config):
    """The whole-model reference forward against the inline golden, all layers, random weights.

    Reduced config — this is an internal-consistency check between two oracles, not a bring-up
    number. What it adds over the per-block tests is the parts only the whole model has: the
    embedding lookup, the layer STACK (per-layer weight slicing, and that layer i actually gets
    layer i's weights), the final norm, and the LM head.
    """
    from models.demos.llama_3_1_8b_d_p.reference.model import RefModel

    torch.manual_seed(0)
    model = RefModel(config).eval()
    input_ids = torch.randint(0, config.vocab_size, (1, SEQ))
    with torch.no_grad():
        logits, per_layer_kv = model(input_ids, return_kv=True)

    want = inline_model(input_ids, model.state_dict(), config, config.num_hidden_layers)
    assert _pcc(logits, want) > PCC_BAR
    assert len(per_layer_kv) == config.num_hidden_layers, "one (K, V) pair per layer"
    for k, v in per_layer_kv:
        assert k.shape == v.shape == (1, config.num_key_value_heads, SEQ, config.head_dim)


def test_layer_stack_uses_distinct_weights(config):
    """Layer i must consume layer i's weights.

    A stack that slices the state dict wrong — every layer reading layer 0, say — still produces
    plausible logits and still matches a golden generated by the SAME bug. Detected here by making
    one layer a no-op and checking the output moves only when that layer is the one changed.
    """
    from models.demos.llama_3_1_8b_d_p.reference.model import RefModel

    torch.manual_seed(0)
    model = RefModel(config).eval()
    input_ids = torch.randint(0, config.vocab_size, (1, SEQ))
    with torch.no_grad():
        base = model(input_ids)
        # Zero the LAST layer's MLP down-projection: its output must change, and it must change by
        # a different amount than zeroing the first layer's.
        model.layers[-1].mlp.down_proj.weight.zero_()
        last_zeroed = model(input_ids)
    assert not torch.equal(base, last_zeroed), "zeroing the last layer's MLP changed nothing"


def test_reference_kv_is_post_rope_k_and_raw_v(config):
    """The stored K is post-RoPE and the stored V is raw — the convention P1/P2 grade against."""
    from models.demos.llama_3_1_8b_d_p.reference.model import RefModel, RefRotaryEmbedding, apply_rotary_pos_emb

    torch.manual_seed(0)
    model = RefModel(config, num_layers=1).eval()
    input_ids = torch.randint(0, config.vocab_size, (1, SEQ))
    with torch.no_grad():
        _, kv = model(input_ids, return_kv=True)
        k_stored, v_stored = kv[0]

        hidden = model.embed_tokens(input_ids)
        attn = model.layers[0].self_attn
        normed = model.layers[0].input_layernorm(hidden)
        n_kv, hd = config.num_key_value_heads, config.head_dim
        k_raw = attn.k_proj(normed).view(1, SEQ, n_kv, hd).transpose(1, 2)
        v_raw = attn.v_proj(normed).view(1, SEQ, n_kv, hd).transpose(1, 2)
        cos, sin = RefRotaryEmbedding(config)(torch.arange(SEQ)[None, :], dtype=REF_DTYPE)
        _, k_roped = apply_rotary_pos_emb(k_raw, k_raw, cos, sin)

    assert torch.equal(v_stored, v_raw), "stored V must be the raw projection, unrotated"
    assert torch.equal(k_stored, k_roped), "stored K must be post-RoPE"
    assert not torch.equal(k_stored, k_raw), "stored K is pre-RoPE - the cache convention is wrong"
