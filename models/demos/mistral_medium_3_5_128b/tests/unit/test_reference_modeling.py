# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""D1 test 2 — the standalone reference agrees with an inline torch golden.

Host-only; no device, no hardware. The inline goldens below are deliberately written the naive way
(explicit loops, no shared helpers) and import nothing from ``reference.modeling``. If both were
written the same way a shared misconception would pass silently, which is the whole failure mode
this stage exists to catch.

Everything runs on :func:`host_reduced_config` — a diagnostic scale, never an acceptance result.
"""

import math

import pytest
import torch
import torch.nn.functional as F

from models.common.utility_functions import comp_pcc
from models.demos.mistral_medium_3_5_128b.reference.golden import run_reference_layer
from models.demos.mistral_medium_3_5_128b.reference.model_config import MistralMediumConfig, host_reduced_config
from models.demos.mistral_medium_3_5_128b.reference.modeling import (
    REF_DTYPE,
    LayerWeights,
    MistralMLP,
    MistralRMSNorm,
    MistralYarnRotaryEmbedding,
    build_layer,
    causal_mask,
    hf_to_meta,
    hf_to_meta_head_perm,
    repeat_kv,
    yarn_inv_freq,
)
from models.demos.mistral_medium_3_5_128b.reference.regenerate import random_hidden

PCC = 0.9999  # host vs host in the same dtype: anything below this is a real disagreement


# ---------------------------------------------------------------------------------------------
# Inline goldens (independent re-implementations)
# ---------------------------------------------------------------------------------------------
def inline_rmsnorm(x, w, eps):
    x32 = x.to(torch.float32)
    rms = torch.sqrt((x32 * x32).sum(-1, keepdim=True) / x32.shape[-1] + eps)
    return (w.to(torch.float32) * (x32 / rms).to(x.dtype).to(torch.float32)).to(x.dtype)


def inline_swiglu(x, wg, wu, wd):
    g = x @ wg.T
    u = x @ wu.T
    act = g * torch.sigmoid(g.to(torch.float32)).to(g.dtype)  # silu, written out
    return (act * u) @ wd.T


def inline_yarn_inv_freq(cfg: MistralMediumConfig):
    """YaRN inverse frequencies, element by element."""
    dim, base = cfg.head_dim, cfg.rope_theta
    factor = cfg.max_position_embeddings / cfg.rope_original_max_position_embeddings
    orig = cfg.rope_original_max_position_embeddings

    def corr(rot):
        return dim * math.log(orig / (rot * 2 * math.pi)) / (2 * math.log(base))

    low = max(math.floor(corr(cfg.rope_beta_fast)), 0)
    high = min(math.ceil(corr(cfg.rope_beta_slow)), dim - 1)

    out = []
    for j in range(dim // 2):
        pos_freq = base ** (2 * j / dim)
        extrap, interp = 1.0 / pos_freq, 1.0 / (factor * pos_freq)
        ramp = min(max((j - low) / (high - low), 0.0), 1.0)
        extrap_w = 1 - ramp  # 1 at the extrapolated end
        out.append(interp * (1 - extrap_w) + extrap * extrap_w)
    return torch.tensor(out, dtype=torch.float32)


def inline_rope_hf(x, cos, sin):
    """HF half-split rotation, written as an explicit pairing of ``i`` with ``i + half``."""
    half = x.shape[-1] // 2
    out = torch.empty_like(x)
    x1, x2 = x[..., :half], x[..., half:]
    out[..., :half] = x1 * cos[..., :half] - x2 * sin[..., :half]
    out[..., half:] = x2 * cos[..., half:] + x1 * sin[..., half:]
    return out


def inline_rope_meta(x, cos_half, sin_half):
    """Meta interleaved rotation: pairs ``2j`` with ``2j+1``, angle ``theta_j``."""
    out = torch.empty_like(x)
    even, odd = x[..., 0::2], x[..., 1::2]
    out[..., 0::2] = even * cos_half - odd * sin_half
    out[..., 1::2] = odd * cos_half + even * sin_half
    return out


def inline_attention(x, w, cfg, cos, sin):
    """GQA attention with an explicit per-head loop."""
    b, s, _ = x.shape
    nh, nkv, hd = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
    q = (x @ w.q_proj.T).view(b, s, nh, hd).transpose(1, 2)
    k = (x @ w.k_proj.T).view(b, s, nkv, hd).transpose(1, 2)
    v = (x @ w.v_proj.T).view(b, s, nkv, hd).transpose(1, 2)
    q = inline_rope_hf(q, cos[:, None], sin[:, None])
    k = inline_rope_hf(k, cos[:, None], sin[:, None])

    mask = torch.full((s, s), float("-inf")).triu(1)
    heads = []
    for h in range(nh):
        kh, vh = k[:, h // cfg.num_key_value_groups], v[:, h // cfg.num_key_value_groups]
        sc = (q[:, h].float() @ kh.float().transpose(-1, -2)) / math.sqrt(hd) + mask
        heads.append((F.softmax(sc, dim=-1).to(v.dtype) @ vh))
    out = torch.stack(heads, dim=1).transpose(1, 2).reshape(b, s, nh * hd)
    return out @ w.o_proj.T, k, v


# ---------------------------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def cfg():
    return host_reduced_config()


@pytest.fixture(scope="module")
def weights(cfg):
    return LayerWeights.random(cfg, seed=0)


@pytest.fixture(scope="module")
def hidden(cfg):
    return random_hidden(cfg, 256)


def test_rmsnorm(cfg, hidden):
    w = torch.randn(cfg.hidden_size, generator=torch.Generator().manual_seed(3)).to(REF_DTYPE)
    norm = MistralRMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
    with torch.no_grad():
        norm.weight.copy_(w)
        got = norm(hidden)
    passing, pcc = comp_pcc(inline_rmsnorm(hidden, w, cfg.rms_norm_eps).float(), got.float(), PCC)
    assert passing, f"RMSNorm PCC {pcc}"


def test_rmsnorm_is_not_gemma_style(cfg):
    """An all-zero weight must zero the output. Gemma's ``(1 + w)`` fold would pass the input
    through instead — the same one-line difference that silently breaks a ported norm."""
    norm = MistralRMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
    with torch.no_grad():
        norm.weight.zero_()
        out = norm(torch.randn(1, 8, cfg.hidden_size).to(REF_DTYPE))
    assert out.abs().max() == 0


def test_mlp(cfg, weights, hidden):
    mlp = MistralMLP(cfg)
    with torch.no_grad():
        mlp.gate_proj.weight.copy_(weights.gate_proj)
        mlp.up_proj.weight.copy_(weights.up_proj)
        mlp.down_proj.weight.copy_(weights.down_proj)
        got = mlp(hidden)
    ref = inline_swiglu(hidden, weights.gate_proj, weights.up_proj, weights.down_proj)
    passing, pcc = comp_pcc(ref.float(), got.float(), PCC)
    assert passing, f"SwiGLU MLP PCC {pcc}"


def test_yarn_inv_freq(cfg):
    got = yarn_inv_freq(cfg)
    ref = inline_yarn_inv_freq(cfg)
    assert got.shape == (cfg.head_dim // 2,)
    torch.testing.assert_close(got, ref, rtol=1e-6, atol=0)


def test_yarn_against_transformers(tmp_path):
    """Cross-check the full head_dim=128 rope against the library that produced the golden trace.

    The golden was generated by ``transformers`` 5.12.1's own ``Ministral3`` rope init, so this is
    the closest thing to a ground truth for the frequencies and for ``attention_scaling``. Driven
    from the **vendored** config rather than the checkpoint, so it runs on a host with no weights
    mounted.
    """
    pytest.importorskip("transformers")
    import shutil

    from transformers import AutoConfig
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

    from models.demos.mistral_medium_3_5_128b.reference.model_config import VENDORED_CONFIG

    shutil.copy(VENDORED_CONFIG, tmp_path / "config.json")
    hf_text_config = AutoConfig.from_pretrained(tmp_path).text_config
    inv_freq, attention_factor = ROPE_INIT_FUNCTIONS["yarn"](hf_text_config, device="cpu")

    cfg = MistralMediumConfig()
    torch.testing.assert_close(yarn_inv_freq(cfg), inv_freq.float(), rtol=1e-6, atol=0)
    assert attention_factor == pytest.approx(cfg.attention_scaling, abs=1e-15)


def test_rope_against_transformers_full_sequence(tmp_path):
    """cos/sin over a real position range must match transformers element for element.

    ``inv_freq`` agreeing is necessary but not sufficient — the duplication into ``[freqs, freqs]``,
    the fp32 angle and where ``attention_scaling`` is applied all still have to line up.
    """
    pytest.importorskip("transformers")
    import shutil

    from transformers import AutoConfig
    from transformers.models.ministral3.modeling_ministral3 import Ministral3RotaryEmbedding

    from models.demos.mistral_medium_3_5_128b.reference.model_config import VENDORED_CONFIG

    shutil.copy(VENDORED_CONFIG, tmp_path / "config.json")
    hf_text_config = AutoConfig.from_pretrained(tmp_path).text_config

    positions = torch.arange(0, 10240, 337, dtype=torch.int64)[None]  # spans the golden's range
    dummy = torch.zeros(1, positions.shape[1], 1, dtype=torch.float32)
    hf_cos, hf_sin = Ministral3RotaryEmbedding(hf_text_config)(dummy, positions)

    cos, sin = MistralYarnRotaryEmbedding(MistralMediumConfig(), dtype=torch.float32)(positions)
    torch.testing.assert_close(cos, hf_cos.float(), rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(sin, hf_sin.float(), rtol=1e-5, atol=1e-6)


def test_rope_cos_sin_shape_and_scaling(cfg):
    rope = MistralYarnRotaryEmbedding(cfg)
    positions = torch.arange(16)[None]
    cos, sin = rope(positions)
    assert cos.shape == (1, 16, cfg.head_dim)
    # Position 0: every angle is 0, so cos is the bare attention scaling and sin is 0.
    assert cos[0, 0].float().allclose(torch.full((cfg.head_dim,), cfg.attention_scaling), atol=2e-2)
    assert sin[0, 0].abs().max() == 0
    # cos/sin duplicate across the half-split boundary.
    half = cfg.head_dim // 2
    torch.testing.assert_close(cos[..., :half], cos[..., half:])


def test_hf_to_meta_permutation_is_the_layout_change(cfg):
    """Rotating in HF layout then permuting equals permuting then rotating in Meta layout.

    This is the single fact the golden-vs-device K comparison rests on; ``v`` is unrotated so it is
    layout-free.
    """
    hd = cfg.head_dim
    x = torch.randn(2, 3, 5, hd, dtype=torch.float32)
    inv = yarn_inv_freq(cfg)
    angles = torch.arange(5).float()[:, None] * inv  # [5, hd/2]
    cos_half, sin_half = angles.cos(), angles.sin()
    cos_full = torch.cat([cos_half, cos_half], dim=-1)
    sin_full = torch.cat([sin_half, sin_half], dim=-1)

    lhs = hf_to_meta(inline_rope_hf(x, cos_full, sin_full))
    rhs = inline_rope_meta(hf_to_meta(x), cos_half, sin_half)
    torch.testing.assert_close(lhs, rhs, rtol=1e-5, atol=1e-6)


def test_hf_to_meta_permutation_is_a_bijection():
    perm = hf_to_meta_head_perm(128)
    assert sorted(perm) == list(range(128))
    assert perm[:6] == [0, 64, 1, 65, 2, 66]


def test_repeat_kv(cfg):
    x = torch.randn(1, cfg.num_key_value_heads, 4, cfg.head_dim)
    r = repeat_kv(x, cfg.num_key_value_groups)
    assert r.shape[1] == cfg.num_attention_heads
    for h in range(cfg.num_attention_heads):
        torch.testing.assert_close(r[:, h], x[:, h // cfg.num_key_value_groups])


def test_causal_mask_one_shot():
    m = causal_mask(4, 4, dtype=torch.float32)[0, 0]
    assert (m[0, 1:] < -1e30).all() and m[0, 0] == 0
    assert (m[3] == 0).all()


def test_causal_mask_chunked():
    """A 4-query chunk at offset 4 attends to all 4 cached tokens and causally within itself."""
    m = causal_mask(4, 8, dtype=torch.float32)[0, 0]
    assert (m[:, :4] == 0).all(), "a later chunk must see the whole cache"
    assert (m[0, 5:] < -1e30).all()
    assert (m[3, :8] == 0).all()


def test_attention_against_inline(cfg, weights, hidden):
    layer = build_layer(cfg, weights)
    cos, sin = MistralYarnRotaryEmbedding(cfg)(torch.arange(hidden.shape[1])[None])
    mask = causal_mask(hidden.shape[1], hidden.shape[1])
    with torch.no_grad():
        got, gk, gv = layer.self_attn(hidden, cos, sin, mask)
    ref, rk, rv = inline_attention(hidden, weights, cfg, cos, sin)

    for name, a, b in (("out", ref, got), ("k", rk, gk), ("v", rv, gv)):
        passing, pcc = comp_pcc(a.float(), b.float(), PCC)
        assert passing, f"attention {name} PCC {pcc}"


def test_decoder_layer_against_inline(cfg, weights, hidden):
    """The full pre-norm block, composed inline from the pieces above."""
    got, gk, gv = run_reference_layer(cfg, weights, hidden)

    cos, sin = MistralYarnRotaryEmbedding(cfg)(torch.arange(hidden.shape[1])[None])
    h1 = inline_rmsnorm(hidden, weights.input_layernorm, cfg.rms_norm_eps)
    attn, rk, rv = inline_attention(h1, weights, cfg, cos, sin)
    resid = hidden + attn
    h2 = inline_rmsnorm(resid, weights.post_attention_layernorm, cfg.rms_norm_eps)
    ref = resid + inline_swiglu(h2, weights.gate_proj, weights.up_proj, weights.down_proj)

    for name, a, b in (("out", ref, got), ("k", rk, gk), ("v", rv, gv)):
        passing, pcc = comp_pcc(a.float(), b.float(), PCC)
        assert passing, f"decoder layer {name} PCC {pcc}"


def test_decoder_layer_against_transformers(cfg, weights, hidden):
    """The reduced layer against ``Ministral3DecoderLayer`` — the exact class that made the golden.

    Stronger than the inline golden: it pins the reference to the same code path the trace came
    from, including the pieces easy to get wrong by reading alone (where ``attention_scaling``
    lands, the GQA repeat, the residual order). The KV comparison also confirms that what HF caches
    is post-RoPE K and raw V, which is what ``metadata.json`` claims.
    """
    pytest.importorskip("transformers")
    from transformers.cache_utils import DynamicCache
    from transformers.models.ministral3.configuration_ministral3 import Ministral3Config
    from transformers.models.ministral3.modeling_ministral3 import Ministral3DecoderLayer

    hf_cfg = Ministral3Config(
        hidden_size=cfg.hidden_size,
        num_hidden_layers=1,
        num_attention_heads=cfg.num_attention_heads,
        num_key_value_heads=cfg.num_key_value_heads,
        head_dim=cfg.head_dim,
        intermediate_size=cfg.intermediate_size,
        hidden_act=cfg.hidden_act,
        rms_norm_eps=cfg.rms_norm_eps,
        vocab_size=cfg.vocab_size,
        max_position_embeddings=cfg.max_position_embeddings,
        sliding_window=None,
        attention_dropout=0.0,
        rope_parameters={
            "rope_type": "yarn",
            "factor": cfg.rope_factor,
            "beta_fast": cfg.rope_beta_fast,
            "beta_slow": cfg.rope_beta_slow,
            "mscale": cfg.rope_mscale,
            "mscale_all_dim": cfg.rope_mscale_all_dim,
            "original_max_position_embeddings": cfg.rope_original_max_position_embeddings,
            "rope_theta": cfg.rope_theta,
            "llama_4_scaling_beta": cfg.rope_llama_4_scaling_beta,
        },
    )
    hf_cfg._attn_implementation = "eager"

    hf_layer = Ministral3DecoderLayer(hf_cfg, layer_idx=0).to(REF_DTYPE).eval()
    with torch.no_grad():
        hf_layer.input_layernorm.weight.copy_(weights.input_layernorm)
        hf_layer.post_attention_layernorm.weight.copy_(weights.post_attention_layernorm)
        for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            getattr(hf_layer.self_attn, name).weight.copy_(getattr(weights, name))
        for name in ("gate_proj", "up_proj", "down_proj"):
            getattr(hf_layer.mlp, name).weight.copy_(getattr(weights, name))

    seq = hidden.shape[1]
    positions = torch.arange(seq)[None]
    cos, sin = MistralYarnRotaryEmbedding(cfg)(positions)
    cache = DynamicCache(config=hf_cfg)
    with torch.no_grad():
        hf_out = hf_layer(
            hidden_states=hidden,
            attention_mask=causal_mask(seq, seq),
            position_ids=positions,
            past_key_values=cache,
            use_cache=True,
            position_embeddings=(cos, sin),
        )
    hf_k, hf_v = cache.layers[0].keys, cache.layers[0].values

    got, gk, gv = run_reference_layer(cfg, weights, hidden)
    for name, a, b in (("out", hf_out, got), ("k", hf_k, gk), ("v", hf_v, gv)):
        passing, pcc = comp_pcc(a.float(), b.float(), PCC)
        assert passing, f"decoder layer vs transformers {name} PCC {pcc}"


def test_llama4_scaling_is_disabled_for_this_checkpoint():
    """``llama_4_scaling_beta`` must stay 0, or attention needs a position-dependent Q scale."""
    assert MistralMediumConfig().rope_llama_4_scaling_beta == 0.0


def test_chunked_equals_one_shot(cfg, weights):
    """Two 128-token chunks with a carried cache reproduce a 256-token one-shot call.

    This is the reference-side statement of the property P2 has to hold on device; if it did not
    hold here, a multi-chunk device mismatch would be unattributable.
    """
    hidden = random_hidden(cfg, 256)
    full, k_full, v_full = run_reference_layer(cfg, weights, hidden)

    out0, k0, v0 = run_reference_layer(cfg, weights, hidden[:, :128], position_offset=0)
    out1, k1, v1 = run_reference_layer(cfg, weights, hidden[:, 128:], position_offset=128, past_k=k0, past_v=v0)

    chunked = torch.cat([out0, out1], dim=1)
    for name, a, b in (
        ("out", full, chunked),
        ("k", k_full, torch.cat([k0, k1], dim=2)),
        ("v", v_full, torch.cat([v0, v1], dim=2)),
    ):
        passing, pcc = comp_pcc(a.float(), b.float(), PCC)
        assert passing, f"chunked vs one-shot {name} PCC {pcc}"
