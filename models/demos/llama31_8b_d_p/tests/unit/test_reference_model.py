# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""G-REF — the torch reference for Llama-3.1-8B-Instruct: deterministic, causal, and self-consistent.

Two oracles are used by this bring-up (`bringup_log/05_DECISIONS.md` `DEC-005`):

* **hand-written torch math**, transcribed here from
  `transformers.models.llama.modeling_llama`, driving the P5 module gates — no checkpoint, no HF
  model construction, runs on a bare box;
* **HF `LlamaDecoderLayer`** itself, driving the P6/P7 layer- and model-level gates.

This file is what stops the first drifting from the second. It also nails down, by measurement
rather than by trust, the four things that make a torch reference silently wrong for Llama:
the `rope_theta` attribute that does not exist on `transformers` 5.x (recipe P1 trap 1), the
`attention_mask=None` non-causality (trap 3), the GQA repeat order, and whether llama3 RoPE scaling
is actually active.

Host only — no `ttnn`, no `mesh_device` (`DEC-010`). Reference dtype policy: **fp32 everywhere**;
the checkpoint's bf16 tensors are cast to fp32 on load and nothing is rounded back (`DEC-006`).

Anchor: `transformers.models.llama.modeling_llama.LlamaDecoderLayer`.

Run:
    pytest models/demos/llama31_8b_d_p/tests/unit/test_reference_model.py -x -q
"""

import filecmp
import hashlib
import json
import math
import os

import pytest
import torch
import torch.nn.functional as F
from loguru import logger

from models.common.utility_functions import comp_pcc
from models.demos.llama31_8b_d_p.tests.test_factory import (
    bundled_config_path,
    hf_model_path,
    llama_config_dims,
    load_hf_state_dict,
    requires_hf_reference,
)
from models.tt_transformers.tt.common import get_rope_scaling, get_rope_theta, precompute_freqs

CFG = llama_config_dims()
HIDDEN = CFG["hidden_size"]
NQ = CFG["num_attention_heads"]
NKV = CFG["num_key_value_heads"]
HEAD_DIM = HIDDEN // NQ
INTERMEDIATE = CFG["intermediate_size"]
EPS = CFG["rms_norm_eps"]

# Read in exactly ONE place, from the raw config.json dict, and asserted non-None: `cfg.rope_theta`
# raises and `getattr(cfg, "rope_theta", DEFAULT)` silently substitutes the default on
# transformers 5.12.1 (`07_RISKS.md` R-005). `test_rope_theta_is_not_an_attribute` measures that.
ROPE_THETA = get_rope_theta(CFG)
ROPE_SCALING = get_rope_scaling(CFG)
assert ROPE_THETA is not None, "rope_theta resolved to None — see 07_RISKS.md R-005"
assert ROPE_SCALING is not None and ROPE_SCALING["rope_type"] == "llama3"


# ---------------------------------------------------------------------------------------------
# Hand-written torch reference — a transcription of modeling_llama, HF conventions throughout
# (`DEC-011`: `rotate_half` halves, not Meta interleaved; the Meta swizzle is the device path's job).
# ---------------------------------------------------------------------------------------------
def llama3_inv_freq(head_dim=HEAD_DIM, theta=ROPE_THETA, scaling=None):
    """`inv_freq` with llama3 piecewise scaling — transcribed from
    `transformers.modeling_rope_utils._compute_llama3_parameters`. `scaling=None` returns the
    unscaled frequencies (the negative control)."""
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.int64).float() / head_dim))
    if scaling is None:
        return inv_freq
    factor = scaling["factor"]
    low_freq_factor = scaling["low_freq_factor"]
    high_freq_factor = scaling["high_freq_factor"]
    old_context_len = scaling["original_max_position_embeddings"]

    low_freq_wavelen = old_context_len / low_freq_factor
    high_freq_wavelen = old_context_len / high_freq_factor
    wavelen = 2 * math.pi / inv_freq

    inv_freq_llama = torch.where(wavelen > low_freq_wavelen, inv_freq / factor, inv_freq)
    smooth = (old_context_len / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor)
    smoothed = (1 - smooth) * inv_freq_llama / factor + smooth * inv_freq_llama
    is_medium = ~(wavelen < high_freq_wavelen) * ~(wavelen > low_freq_wavelen)
    return torch.where(is_medium, smoothed, inv_freq_llama)


def build_cos_sin(seq_len, inv_freq=None, start_pos=0):
    """HF-convention cos/sin, `[S, head_dim]`, fp32. llama3 RoPE has attention_factor 1.0 (no
    mscale), unlike YaRN — see `_compute_llama3_parameters`, which returns `attention_factor = 1.0`."""
    inv_freq = llama3_inv_freq(scaling=ROPE_SCALING) if inv_freq is None else inv_freq
    pos = torch.arange(start_pos, start_pos + seq_len).float()
    freqs = torch.outer(pos, inv_freq)  # [S, head_dim/2]
    emb = torch.cat([freqs, freqs], dim=-1)  # [S, head_dim]
    return emb.cos(), emb.sin()


def _rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rope(q, k, cos, sin):
    cos, sin = cos.unsqueeze(0).unsqueeze(0), sin.unsqueeze(0).unsqueeze(0)
    return (q * cos) + (_rotate_half(q) * sin), (k * cos) + (_rotate_half(k) * sin)


def rms_norm(x, weight, eps=EPS):
    """Plain RMSNorm — no `+1` weight fold (that is Gemma). Anchor: `LlamaRMSNorm.forward`."""
    variance = x.pow(2).mean(-1, keepdim=True)
    return weight * (x * torch.rsqrt(variance + eps))


def causal_mask(seq_len, dtype=torch.float32):
    """Additive `[1, 1, S, S]` mask. **Mandatory**: `eager_attention_forward` applies only the mask
    it is handed, so `attention_mask=None` is silently non-causal (recipe P1 trap 3)."""
    return torch.triu(torch.full((seq_len, seq_len), float("-inf"), dtype=dtype), diagonal=1)[None, None]


def attention(x, w, cos, sin, mask):
    """GQA attention: q/k/v proj (no bias) -> heads -> full RoPE -> causal SDPA -> o_proj.
    Anchor: `LlamaAttention.forward` + `eager_attention_forward`."""
    b, s, _ = x.shape
    q = F.linear(x, w["q_proj"]).view(b, s, NQ, HEAD_DIM).transpose(1, 2)
    k = F.linear(x, w["k_proj"]).view(b, s, NKV, HEAD_DIM).transpose(1, 2)
    v = F.linear(x, w["v_proj"]).view(b, s, NKV, HEAD_DIM).transpose(1, 2)

    q, k = _apply_rope(q, k, cos, sin)

    # GQA: each KV head is shared by NQ/NKV = 4 Q heads. `repeat_interleave` semantics — HF's
    # `repeat_kv` is expand+reshape, which is the same map. `repeat` is NOT (negative control below).
    n_rep = NQ // NKV
    k = torch.repeat_interleave(k, n_rep, dim=1)
    v = torch.repeat_interleave(v, n_rep, dim=1)

    scaling = HEAD_DIM**-0.5
    attn = torch.matmul(q, k.transpose(2, 3)) * scaling
    # `mask is None` is NOT a shortcut for "causal" — it is the non-causal control (trap 3), and it
    # is what HF's own default gives you. The transcription mirrors `eager_attention_forward`, which
    # applies only the mask it is handed.
    if mask is not None:
        attn = attn + mask
    attn = F.softmax(attn, dim=-1, dtype=torch.float32)
    out = torch.matmul(attn, v).transpose(1, 2).contiguous().reshape(b, s, NQ * HEAD_DIM)
    return F.linear(out, w["o_proj"])


def mlp(x, w):
    """Dense SwiGLU: `down(silu(gate(x)) * up(x))`. Anchor: `LlamaMLP.forward`, `hidden_act: silu`."""
    return F.linear(F.silu(F.linear(x, w["gate_proj"])) * F.linear(x, w["up_proj"]), w["down_proj"])


def decoder_layer(x, w, cos, sin, mask):
    """One decoder layer: norm -> attn -> residual -> norm -> mlp -> residual."""
    x = x + attention(rms_norm(x, w["input_layernorm"]), w, cos, sin, mask)
    return x + mlp(rms_norm(x, w["post_attention_layernorm"]), w)


# ---------------------------------------------------------------------------------------------
# Weights: identical tensors drive both oracles.
# ---------------------------------------------------------------------------------------------
def random_layer_weights(seed=0, scale=0.02):
    """HF `[out, in]` layout, standard-normal (recipe §2.1(b): state the distribution; randn is the
    harder one for a norm)."""
    g = torch.Generator().manual_seed(seed)

    def rn(*shape):
        return torch.randn(*shape, generator=g) * scale

    return {
        "q_proj": rn(NQ * HEAD_DIM, HIDDEN),
        "k_proj": rn(NKV * HEAD_DIM, HIDDEN),
        "v_proj": rn(NKV * HEAD_DIM, HIDDEN),
        "o_proj": rn(HIDDEN, NQ * HEAD_DIM),
        "gate_proj": rn(INTERMEDIATE, HIDDEN),
        "up_proj": rn(INTERMEDIATE, HIDDEN),
        "down_proj": rn(HIDDEN, INTERMEDIATE),
        "input_layernorm": 1.0 + rn(HIDDEN),
        "post_attention_layernorm": 1.0 + rn(HIDDEN),
    }


def real_layer_weights(layer_idx=0):
    """Layer `layer_idx` from the staged checkpoint, cast to **fp32** (`DEC-006`)."""
    p = f"model.layers.{layer_idx}."
    sd = load_hf_state_dict(prefixes=(p,))
    return {
        "q_proj": sd[p + "self_attn.q_proj.weight"].float(),
        "k_proj": sd[p + "self_attn.k_proj.weight"].float(),
        "v_proj": sd[p + "self_attn.v_proj.weight"].float(),
        "o_proj": sd[p + "self_attn.o_proj.weight"].float(),
        "gate_proj": sd[p + "mlp.gate_proj.weight"].float(),
        "up_proj": sd[p + "mlp.up_proj.weight"].float(),
        "down_proj": sd[p + "mlp.down_proj.weight"].float(),
        "input_layernorm": sd[p + "input_layernorm.weight"].float(),
        "post_attention_layernorm": sd[p + "post_attention_layernorm.weight"].float(),
    }


def hf_decoder_layer(w, layer_idx=0):
    """An HF `LlamaDecoderLayer` in fp32 with `_attn_implementation="eager"`, loaded from `w`.

    Built bare from the config rather than through `from_pretrained`, which would load at the
    checkpoint's `torch_dtype` (bf16) and give a reference that shares the device's own rounding
    (recipe §2.1(a); LANDMINES "from_pretrained for a torch reference")."""
    from transformers import AutoConfig
    from transformers.models.llama.modeling_llama import LlamaDecoderLayer

    cfg = AutoConfig.from_pretrained(os.path.dirname(bundled_config_path()))
    cfg._attn_implementation = "eager"
    layer = LlamaDecoderLayer(cfg, layer_idx).float().eval()
    layer.load_state_dict(
        {
            "self_attn.q_proj.weight": w["q_proj"],
            "self_attn.k_proj.weight": w["k_proj"],
            "self_attn.v_proj.weight": w["v_proj"],
            "self_attn.o_proj.weight": w["o_proj"],
            "mlp.gate_proj.weight": w["gate_proj"],
            "mlp.up_proj.weight": w["up_proj"],
            "mlp.down_proj.weight": w["down_proj"],
            "input_layernorm.weight": w["input_layernorm"],
            "post_attention_layernorm.weight": w["post_attention_layernorm"],
        }
    )
    return layer


def run_hf_layer(layer, x, cos, sin, mask):
    with torch.no_grad():
        return layer(
            hidden_states=x,
            attention_mask=mask,
            position_ids=torch.arange(x.shape[1])[None],
            position_embeddings=(cos[None], sin[None]),
        )


def sha256(t):
    return hashlib.sha256(t.detach().contiguous().numpy().tobytes()).hexdigest()


# ---------------------------------------------------------------------------------------------
# G-REF
# ---------------------------------------------------------------------------------------------
@requires_hf_reference
def test_bundled_config_matches_checkpoint():
    """`DEC-009`: the bundled config must be byte-identical to the checkpoint's, or every
    dimension-only test is measuring a different model than the weight-loading tests."""
    checkpoint_config = os.path.join(hf_model_path(), "config.json")
    assert filecmp.cmp(
        bundled_config_path(), checkpoint_config, shallow=False
    ), f"{bundled_config_path()} differs from {checkpoint_config}"
    with open(checkpoint_config) as f:
        assert json.load(f) == CFG


def test_rope_theta_is_not_an_attribute_on_transformers_5(expect_error):
    """Recipe P1 trap 1 / `07_RISKS.md` R-005, measured rather than trusted.

    `getattr(cfg, "rope_theta", DEFAULT)` — the pattern at
    `models/demos/gpt_oss_d_p/tt/model_config.py:76` and
    `models/demos/gpt_oss_d_p/tt/tt_prefill_runtime.py:185` — returns the DEFAULT here, silently. A
    RoPE wrong at every position with no exception anywhere."""
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(os.path.dirname(bundled_config_path()))

    with expect_error(AttributeError, "rope_theta"):
        _ = cfg.rope_theta
    substituted = getattr(cfg, "rope_theta", 10000.0)
    assert substituted == 10000.0, "the getattr trap did not reproduce; re-check this transformers version"
    assert "rope_theta" not in cfg.to_dict()

    # The only correct read: the raw config.json dict, through the repo helper.
    assert get_rope_theta(CFG) == 500000.0
    logger.info(
        f"rope_theta: cfg.rope_theta raises AttributeError | "
        f"getattr(cfg,'rope_theta',10000.0) -> {substituted} (WRONG, silent) | "
        f"get_rope_theta(config.json) -> {get_rope_theta(CFG)}"
    )


@pytest.mark.parametrize("seq_len", [128], ids=["s128"])
def test_handwritten_reference_is_deterministic(seq_len, reset_seeds):
    """Same input twice -> bit-identical output. Both hashes are recorded in the gate ledger."""
    w = random_layer_weights()
    x = torch.randn(1, seq_len, HIDDEN, generator=torch.Generator().manual_seed(0))
    cos, sin = build_cos_sin(seq_len)
    mask = causal_mask(seq_len)

    out1 = decoder_layer(x, w, cos, sin, mask)
    out2 = decoder_layer(x, w, cos, sin, mask)
    h1, h2 = sha256(out1), sha256(out2)
    logger.info(f"hand-written reference sha256: {h1} / {h2}")
    assert torch.equal(out1, out2)
    assert h1 == h2


@pytest.mark.parametrize("seq_len", [128], ids=["s128"])
def test_hf_reference_is_deterministic(seq_len, reset_seeds):
    """Same for the HF oracle — the one the P6/P7 gates score against."""
    w = random_layer_weights()
    x = torch.randn(1, seq_len, HIDDEN, generator=torch.Generator().manual_seed(0))
    cos, sin = build_cos_sin(seq_len)
    mask = causal_mask(seq_len)
    layer = hf_decoder_layer(w)

    out1 = run_hf_layer(layer, x, cos, sin, mask)
    out2 = run_hf_layer(layer, x, cos, sin, mask)
    h1, h2 = sha256(out1), sha256(out2)
    logger.info(f"HF LlamaDecoderLayer sha256: {h1} / {h2}")
    assert torch.equal(out1, out2)
    assert h1 == h2


@pytest.mark.parametrize("seq_len", [128], ids=["s128"])
def test_handwritten_matches_hf_decoder_layer(seq_len, reset_seeds):
    """The cross-reference gate: identical random weights, both oracles, one decoder layer.

    Threshold PCC >= 0.9999; a faithful transcription is expected to be **bit-exact**. Read that
    honestly: it proves the transcription is faithful, not that either side is right about Llama
    (`07_RISKS.md` R-006)."""
    w = random_layer_weights()
    x = torch.randn(1, seq_len, HIDDEN, generator=torch.Generator().manual_seed(0))
    cos, sin = build_cos_sin(seq_len)
    mask = causal_mask(seq_len)

    ours = decoder_layer(x, w, cos, sin, mask)
    theirs = run_hf_layer(hf_decoder_layer(w), x, cos, sin, mask)

    max_delta = (ours - theirs).abs().max().item()
    passing, pcc = comp_pcc(theirs, ours, 0.9999)
    logger.info(f"hand-written vs HF (random weights, s{seq_len}): PCC {pcc}, max|delta| {max_delta:.3e}")
    assert passing, f"cross-reference PCC fail: {pcc}"
    assert max_delta == 0.0, f"expected bit-exact, got max|delta| = {max_delta:.3e}"


@requires_hf_reference
@pytest.mark.parametrize("seq_len", [64], ids=["s64"])
def test_handwritten_matches_hf_decoder_layer_real_weights(seq_len, reset_seeds):
    """Same, with the checkpoint's real layer-0 weights (cast to fp32, `DEC-006`)."""
    w = real_layer_weights(0)
    x = torch.randn(1, seq_len, HIDDEN, generator=torch.Generator().manual_seed(0))
    cos, sin = build_cos_sin(seq_len)
    mask = causal_mask(seq_len)

    ours = decoder_layer(x, w, cos, sin, mask)
    theirs = run_hf_layer(hf_decoder_layer(w), x, cos, sin, mask)

    max_delta = (ours - theirs).abs().max().item()
    passing, pcc = comp_pcc(theirs, ours, 0.9999)
    logger.info(f"hand-written vs HF (real layer-0 weights, s{seq_len}): PCC {pcc}, max|delta| {max_delta:.3e}")
    assert passing, f"cross-reference PCC fail on real weights: {pcc}"
    assert max_delta == 0.0, f"expected bit-exact, got max|delta| = {max_delta:.3e}"


@pytest.mark.parametrize("seq_len", [64], ids=["s64"])
def test_reference_attention_is_causal(seq_len, reset_seeds):
    """Perturb the last token: rows [:-1] must be unchanged at max|delta| = 0 (recipe P1 trap 3).

    **Negative control in the same test:** the identical probe with `attention_mask=None` — HF's own
    default — must show a non-zero delta, i.e. the reference is silently non-causal without an
    explicit mask."""
    w = random_layer_weights()
    x = torch.randn(1, seq_len, HIDDEN, generator=torch.Generator().manual_seed(0))
    x_perturbed = x.clone()
    x_perturbed[:, -1, :] += 1.0
    cos, sin = build_cos_sin(seq_len)
    mask = causal_mask(seq_len)
    layer = hf_decoder_layer(w)

    for name, fwd in (
        ("hand-written", lambda inp, m: decoder_layer(inp, w, cos, sin, m)),
        ("HF", lambda inp, m: run_hf_layer(layer, inp, cos, sin, m)),
    ):
        delta_masked = (fwd(x, mask) - fwd(x_perturbed, mask))[:, :-1].abs().max().item()
        delta_unmasked = (fwd(x, None) - fwd(x_perturbed, None))[:, :-1].abs().max().item()
        logger.info(f"causality ({name}): with mask max|delta| {delta_masked:.3e}, mask=None {delta_unmasked:.3e}")
        assert delta_masked == 0.0, f"{name} reference is not causal with an explicit mask"
        assert delta_unmasked > 0.0, f"{name} control failed: mask=None should be non-causal"


def test_llama3_rope_scaling_is_active(reset_seeds):
    """llama3 scaling must be applied, and must be piecewise.

    Positive: the scaled `inv_freq` matches the repo helper P5.3 will use
    (`models/tt_transformers/tt/common.py:489` `precompute_freqs`, `:437` `apply_scaling`).
    Control: unscaled frequencies must produce a *different* embedding beyond
    `original_max_position_embeddings`."""
    scaled = llama3_inv_freq(scaling=ROPE_SCALING)
    unscaled = llama3_inv_freq(scaling=None)
    orig_max = ROPE_SCALING["original_max_position_embeddings"]

    # The repo helper, fed the same theta/scaling, must agree with this transcription.
    cos_repo, sin_repo = precompute_freqs(
        HEAD_DIM, orig_max + 64, ROPE_THETA, ROPE_SCALING["factor"], orig_max, rope_type="llama3"
    )
    pos = torch.arange(orig_max + 64).float()
    cos_ours = torch.outer(pos, scaled).cos()
    delta_repo = (cos_ours - cos_repo).abs().max().item()
    logger.info(f"llama3 inv_freq vs tt_transformers precompute_freqs: max|delta| {delta_repo:.3e}")
    assert delta_repo < 1e-6

    # Control: without scaling the tables must differ, and only the low-frequency lanes move.
    delta_freq = (scaled - unscaled).abs().max().item()
    cos_s, _ = build_cos_sin(1, inv_freq=scaled, start_pos=orig_max)
    cos_u, _ = build_cos_sin(1, inv_freq=unscaled, start_pos=orig_max)
    delta_pos = (cos_s - cos_u).abs().max().item()
    logger.info(
        f"llama3 scaling control: max|delta inv_freq| {delta_freq:.3e}, max|delta cos@{orig_max}| {delta_pos:.3e}"
    )
    assert delta_freq > 0.0, "scaled and unscaled inv_freq are identical — llama3 scaling is not applied"
    assert delta_pos > 0.0


@pytest.mark.parametrize("seq_len", [64], ids=["s64"])
def test_negative_controls_are_rejected(seq_len, reset_seeds):
    """Three deliberately-wrong references the PCC >= 0.9999 assertion must reject.

    Without these, a positive PCC cannot distinguish a correct reference from a symmetric bug.
    1. **wrong theta** — the `getattr` trap's 10000.0 instead of 500000.0;
    2. **`repeat` instead of `repeat_interleave`** — the classic GQA head-mapping error;
    3. **no RoPE at all**."""
    w = random_layer_weights()
    x = torch.randn(1, seq_len, HIDDEN, generator=torch.Generator().manual_seed(0))
    mask = causal_mask(seq_len)
    cos, sin = build_cos_sin(seq_len)
    good = decoder_layer(x, w, cos, sin, mask)

    cos_bad, sin_bad = build_cos_sin(seq_len, inv_freq=llama3_inv_freq(theta=10000.0, scaling=ROPE_SCALING))
    _, pcc_theta = comp_pcc(good, decoder_layer(x, w, cos_bad, sin_bad, mask), 0.9999)

    def attention_repeat(x_in, w_in, cos_in, sin_in, mask_in):
        b, s, _ = x_in.shape
        q = F.linear(x_in, w_in["q_proj"]).view(b, s, NQ, HEAD_DIM).transpose(1, 2)
        k = F.linear(x_in, w_in["k_proj"]).view(b, s, NKV, HEAD_DIM).transpose(1, 2)
        v = F.linear(x_in, w_in["v_proj"]).view(b, s, NKV, HEAD_DIM).transpose(1, 2)
        q, k = _apply_rope(q, k, cos_in, sin_in)
        n_rep = NQ // NKV
        k, v = k.repeat(1, n_rep, 1, 1), v.repeat(1, n_rep, 1, 1)  # WRONG head map
        attn = torch.matmul(q, k.transpose(2, 3)) * HEAD_DIM**-0.5 + mask_in
        attn = F.softmax(attn, dim=-1, dtype=torch.float32)
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().reshape(b, s, NQ * HEAD_DIM)
        return F.linear(out, w_in["o_proj"])

    h = rms_norm(x, w["input_layernorm"])
    bad_gqa = x + attention_repeat(h, w, cos, sin, mask)
    bad_gqa = bad_gqa + mlp(rms_norm(bad_gqa, w["post_attention_layernorm"]), w)
    _, pcc_gqa = comp_pcc(good, bad_gqa, 0.9999)

    ones = torch.ones_like(cos)
    zeros = torch.zeros_like(sin)
    _, pcc_norope = comp_pcc(good, decoder_layer(x, w, ones, zeros, mask), 0.9999)

    logger.info(f"controls: wrong-theta PCC {pcc_theta}, repeat-vs-interleave PCC {pcc_gqa}, no-RoPE PCC {pcc_norope}")
    for name, pcc in (("wrong theta", pcc_theta), ("GQA repeat", pcc_gqa), ("no RoPE", pcc_norope)):
        assert float(pcc) < 0.9999, f"control '{name}' was NOT rejected (PCC {pcc}) — the gate cannot discriminate"


def test_hf_reference_wrapper_signature_branch():
    """Recipe P1 trap 5: the `models/tt_transformers` HF wrappers branch on whether
    `position_embeddings` is in the layer's forward signature
    (`models/tt_transformers/tt/model_config.py:4393`, `:4410`). A wrapper that feeds RoPE twice, or
    not at all, looks exactly like a model bug — so confirm which way the branch resolves for
    `LlamaAttention` / `LlamaDecoderLayer` *before* trusting a low PCC."""
    import inspect

    from transformers.models.llama.modeling_llama import LlamaAttention, LlamaDecoderLayer

    for cls in (LlamaAttention, LlamaDecoderLayer):
        params = inspect.signature(cls.forward).parameters
        logger.info(f"{cls.__name__}.forward params: {list(params)}")
        assert "position_embeddings" in params, (
            f"{cls.__name__}.forward has no `position_embeddings` — the tt_transformers wrappers "
            "would take the legacy branch and apply RoPE differently than this reference does"
        )
