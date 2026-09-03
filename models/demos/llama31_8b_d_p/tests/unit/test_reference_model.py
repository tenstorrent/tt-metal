# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Gate `G-REF` — the torch oracle is deterministic and self-consistent.

Pattern: `models/demos/minimax_m3/tests/unit/test_reference_model.py` (a self-checking reference
test: an inline, hand-written torch reference is proved equal to the reference the rest of the
bring-up will call).

**Host only.** No device, no `mesh_device` fixture, no checkpoint, no network. Everything runs from
a fixed seed in fp32.

What this file proves, and why each part is here:

1. **Determinism** — the oracle, rebuilt from the same seed, produces a *bit-identical* hidden
   state. Asserted with `torch.equal` and recorded as a sha256 of the raw bytes. This is the
   precondition for every later PCC number meaning anything: a non-deterministic oracle turns every
   downstream gate into a coin flip.
2. **Self-consistency** — an independent, hand-written torch decoder layer agrees with HF's
   `LlamaDecoderLayer` to PCC >= 0.9999 from *identical* weights. This is what licenses the P5/P6
   unit tests to use cheap in-test torch math (BRINGUP_RECIPE.md:305-308) instead of paying the HF
   construction cost in every inner loop: if the two agree here, the cheap one is a valid oracle.
3. **The llama3 RoPE scaling is actually active** — a RoPE test that passes with scaling silently
   disabled is worthless (BRINGUP_RECIPE.md:650-652), so the fact is pinned here rather than only
   at `G-ROPE`.
4. **`models/tt_transformers/tt/common.py:precompute_freqs` agrees with HF's llama3 RoPE** — the
   repo helper P5.3 will reuse is checked against HF *now*, on the frequency tables, where a
   mismatch is unambiguous. It also pins the Meta-vs-HF duplication difference that Appendix B
   names as the classic RoPE bug.
5. **The bundled config has not drifted** from the `tt_transformers` original.

HF anchors: `transformers.models.llama.modeling_llama.LlamaDecoderLayer`, `.LlamaRMSNorm`,
`.LlamaMLP`, `.LlamaAttention`, `.LlamaRotaryEmbedding`.

Run:
    pytest models/demos/llama31_8b_d_p/tests/unit/test_reference_model.py -x -q
"""

from __future__ import annotations

import hashlib
import json
import math

import pytest
import torch
from loguru import logger
from transformers import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaRotaryEmbedding

from models.common.utility_functions import comp_pcc
from models.demos.llama31_8b_d_p.tests.test_factory import (
    CONFIG_JSON,
    UPSTREAM_CONFIG_JSON,
    llama_config_dims,
    rope_scaling,
    rope_theta,
)

# fp32 everywhere. The reference is computed in fp32 and only cast at the comparison boundary
# with a device tensor -- see bringup_log/01_REFERENCE.md section 3.
REF_DTYPE = torch.float32

# The gate's threshold (BRINGUP_RECIPE.md:339).
CROSS_REF_PCC = 0.9999

SEED = 0


# --------------------------------------------------------------------------------------
# Dimension sets
# --------------------------------------------------------------------------------------
def _full_dims() -> dict:
    """The real Llama-3.1-8B per-layer dims, from the bundled config."""
    return llama_config_dims()


def _tiny_dims() -> dict:
    """A shrunken but structurally identical shape: same GQA group (4), same head_dim parity.

    Exists so the structural assertions run in milliseconds and so a failure can be inspected by
    hand. It is a *scaled* Llama, not a different model: `hidden/heads = head_dim`, `heads/kv = 4`,
    plain RMSNorm, no biases, llama3 rope scaling with the same factors.
    """
    cfg = llama_config_dims()
    cfg = dict(cfg)
    cfg["hidden_size"] = 256
    cfg["num_attention_heads"] = 8
    cfg["num_key_value_heads"] = 2
    cfg["head_dim"] = 32
    cfg["intermediate_size"] = 512
    cfg["num_hidden_layers"] = 2
    cfg["gqa_group_size"] = 4
    return cfg


DIM_SETS = {"full": _full_dims, "tiny": _tiny_dims}


def _hf_config(dims: dict) -> LlamaConfig:
    """Build a `LlamaConfig` from a raw dims dict, forcing the eager attention path.

    `_attn_implementation` is pinned to `"eager"` so the reference is the explicit
    `eager_attention_forward` math (`modeling_llama.eager_attention_forward`) rather than a fused
    SDPA kernel whose reduction order is unspecified. That is what makes assertion (1)
    -- bit-identical reruns -- achievable at all.
    """
    cfg = LlamaConfig(
        architectures=dims["architectures"],
        hidden_size=dims["hidden_size"],
        intermediate_size=dims["intermediate_size"],
        num_hidden_layers=dims["num_hidden_layers"],
        num_attention_heads=dims["num_attention_heads"],
        num_key_value_heads=dims["num_key_value_heads"],
        head_dim=dims["head_dim"],
        hidden_act=dims["hidden_act"],
        rms_norm_eps=dims["rms_norm_eps"],
        max_position_embeddings=dims["max_position_embeddings"],
        vocab_size=dims["vocab_size"],
        attention_bias=dims["attention_bias"],
        mlp_bias=dims["mlp_bias"],
        tie_word_embeddings=dims["tie_word_embeddings"],
        attention_dropout=dims["attention_dropout"],
        rope_theta=rope_theta(dims),
        rope_scaling=dims["rope_scaling"],
        dtype=REF_DTYPE,
    )
    cfg._attn_implementation = "eager"
    return cfg


# --------------------------------------------------------------------------------------
# The hand-written torch reference -- an independent implementation
# --------------------------------------------------------------------------------------
def _rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Plain RMSNorm. NO Gemma `+1` weight fold -- Llama's norm is `rms(x) * w`.

    Anchor: `transformers.models.llama.modeling_llama.LlamaRMSNorm.forward`.
    """
    var = x.to(torch.float32).pow(2).mean(-1, keepdim=True)
    return weight * (x.to(torch.float32) * torch.rsqrt(var + eps)).to(x.dtype)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """HF convention: split the head in halves and rotate. `modeling_llama.rotate_half`."""
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def _apply_rope_hf(t: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """`modeling_llama.apply_rotary_pos_emb` with `unsqueeze_dim=1`, for one tensor."""
    return t * cos.unsqueeze(1) + _rotate_half(t) * sin.unsqueeze(1)


def _causal_mask(seq_len: int, dtype: torch.dtype) -> torch.Tensor:
    """Additive causal mask, `[1, 1, S, S]`, `-inf` strictly above the diagonal.

    Built explicitly rather than relying on an implicit `is_causal`: `eager_attention_forward`
    applies whatever mask it is handed and nothing more, so an absent mask means *no* causality.
    Same construction the recipe requires of the attention reference (BRINGUP_RECIPE.md:691-692).
    """
    mask = torch.full((seq_len, seq_len), float("-inf"), dtype=dtype)
    return torch.triu(mask, diagonal=1)[None, None, :, :]


def _torch_decoder_layer(x: torch.Tensor, w: dict, cos: torch.Tensor, sin: torch.Tensor, dims: dict) -> torch.Tensor:
    """One Llama decoder layer, written from the architecture, in fp32.

    `norm -> attn -> residual -> norm -> SwiGLU MLP -> residual`.
    Independent of `LlamaDecoderLayer`; only the weight tensors are shared.
    """
    b, s, _ = x.shape
    nq = dims["num_attention_heads"]
    nkv = dims["num_key_value_heads"]
    hd = dims["head_dim"]
    group = nq // nkv
    eps = dims["rms_norm_eps"]
    scaling = hd**-0.5

    # ---- attention sublayer
    residual = x
    h = _rms_norm(x, w["input_layernorm.weight"], eps)

    # HF stores Linear weights as [out, in]; `x @ W.T` is the projection.
    q = (h @ w["self_attn.q_proj.weight"].T).view(b, s, nq, hd).transpose(1, 2)
    k = (h @ w["self_attn.k_proj.weight"].T).view(b, s, nkv, hd).transpose(1, 2)
    v = (h @ w["self_attn.v_proj.weight"].T).view(b, s, nkv, hd).transpose(1, 2)

    q = _apply_rope_hf(q, cos, sin)
    k = _apply_rope_hf(k, cos, sin)

    # GQA: expand the KV heads to the Q head count. On device this expansion does NOT happen --
    # ttnn.transformer.scaled_dot_product_attention handles the group natively
    # (sdpa_device_operation.cpp:97-101 requires only nqh % nkv == 0). Here it is the plain,
    # obviously-correct thing to do.
    k = k.repeat_interleave(group, dim=1)
    v = v.repeat_interleave(group, dim=1)

    scores = (q @ k.transpose(-1, -2)) * scaling + _causal_mask(s, q.dtype)
    probs = torch.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
    attn = (probs @ v).transpose(1, 2).reshape(b, s, nq * hd)
    x = residual + attn @ w["self_attn.o_proj.weight"].T

    # ---- MLP sublayer: down(silu(gate(x)) * up(x)), no biases
    residual = x
    h = _rms_norm(x, w["post_attention_layernorm.weight"], eps)
    gate = h @ w["mlp.gate_proj.weight"].T
    up = h @ w["mlp.up_proj.weight"].T
    x = residual + (torch.nn.functional.silu(gate) * up) @ w["mlp.down_proj.weight"].T

    return x


# --------------------------------------------------------------------------------------
# Fixtures / builders
# --------------------------------------------------------------------------------------
def _build_layer(dims: dict, seed: int) -> tuple[LlamaDecoderLayer, LlamaConfig]:
    """Build a randomly-initialised `LlamaDecoderLayer` deterministically from `seed`.

    Every parameter -- including the two RMSNorm weights, which HF initialises to *ones* -- is
    filled from the seeded generator. Leaving the norm weights at 1.0 would make the norm's weight
    multiply a no-op and hide a whole class of bug.
    """
    torch.manual_seed(seed)
    cfg = _hf_config(dims)
    layer = LlamaDecoderLayer(cfg, layer_idx=0).to(REF_DTYPE).eval()

    gen = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, p in sorted(layer.named_parameters()):
            if "layernorm" in name:
                # Norm gains: centred on 1 so the layer stays numerically sane, but not constant.
                p.copy_(1.0 + 0.1 * torch.randn(p.shape, generator=gen, dtype=REF_DTYPE))
            else:
                p.copy_(torch.randn(p.shape, generator=gen, dtype=REF_DTYPE) * (p.shape[-1] ** -0.5))
    return layer, cfg


def _rope_tables(cfg: LlamaConfig, seq_len: int, batch: int) -> tuple[torch.Tensor, torch.Tensor]:
    """HF cos/sin for positions `[0, seq_len)`, shape `[batch, seq_len, head_dim]`."""
    rotary = LlamaRotaryEmbedding(cfg)
    pos = torch.arange(seq_len, dtype=torch.long)[None, :].expand(batch, seq_len)
    dummy = torch.zeros(batch, seq_len, cfg.hidden_size, dtype=REF_DTYPE)
    return rotary(dummy, pos)


def _run_hf(layer: LlamaDecoderLayer, x: torch.Tensor, cos, sin) -> torch.Tensor:
    with torch.no_grad():
        return layer(
            hidden_states=x,
            attention_mask=_causal_mask(x.shape[1], x.dtype),
            position_ids=torch.arange(x.shape[1], dtype=torch.long)[None, :],
            position_embeddings=(cos, sin),
        )


def _sha256(t: torch.Tensor) -> str:
    return hashlib.sha256(t.detach().contiguous().numpy().tobytes()).hexdigest()


def _seq_len_for(name: str) -> int:
    return 128 if name == "full" else 64


# --------------------------------------------------------------------------------------
# G-REF (1): determinism
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("dim_set", list(DIM_SETS), ids=list(DIM_SETS))
def test_reference_is_deterministic(dim_set):
    """The oracle, rebuilt from the same seed, produces a bit-identical hidden state.

    Rebuilding (rather than re-running one object) is deliberate: it proves the whole
    seed -> weights -> forward chain is reproducible, not merely that a cached tensor equals
    itself.
    """
    dims = DIM_SETS[dim_set]()
    seq_len = _seq_len_for(dim_set)

    outs, hashes = [], []
    for _ in range(2):
        layer, cfg = _build_layer(dims, SEED)
        cos, sin = _rope_tables(cfg, seq_len, batch=1)
        torch.manual_seed(SEED + 1)
        x = torch.randn(1, seq_len, dims["hidden_size"], dtype=REF_DTYPE)
        out = _run_hf(layer, x, cos, sin)
        outs.append(out)
        hashes.append(_sha256(out))

    max_abs = (outs[0] - outs[1]).abs().max().item()
    logger.info(f"[G-REF/{dim_set}] HF LlamaDecoderLayer determinism: sha256 run0 = {hashes[0]}")
    logger.info(f"[G-REF/{dim_set}] HF LlamaDecoderLayer determinism: sha256 run1 = {hashes[1]}")
    logger.info(f"[G-REF/{dim_set}] torch.equal = {torch.equal(outs[0], outs[1])}, max|delta| = {max_abs}")

    assert hashes[0] == hashes[1], f"HF reference is not deterministic: {hashes[0]} != {hashes[1]}"
    assert torch.equal(outs[0], outs[1]), "HF reference outputs differ bitwise"


@pytest.mark.parametrize("dim_set", list(DIM_SETS), ids=list(DIM_SETS))
def test_handwritten_reference_is_deterministic(dim_set):
    """The hand-written reference is bit-identical across rebuilds too.

    It is the oracle the P5/P6 module tests will actually use, so its determinism is not implied by
    the HF one's.
    """
    dims = DIM_SETS[dim_set]()
    seq_len = _seq_len_for(dim_set)

    outs, hashes = [], []
    for _ in range(2):
        layer, cfg = _build_layer(dims, SEED)
        w = {k: v.detach() for k, v in layer.state_dict().items()}
        cos, sin = _rope_tables(cfg, seq_len, batch=1)
        torch.manual_seed(SEED + 1)
        x = torch.randn(1, seq_len, dims["hidden_size"], dtype=REF_DTYPE)
        with torch.no_grad():
            out = _torch_decoder_layer(x, w, cos, sin, dims)
        outs.append(out)
        hashes.append(_sha256(out))

    logger.info(f"[G-REF/{dim_set}] hand-written determinism: sha256 = {hashes[0]} / {hashes[1]}")
    assert hashes[0] == hashes[1], "hand-written reference is not deterministic"
    assert torch.equal(outs[0], outs[1])


# --------------------------------------------------------------------------------------
# G-REF (2): the two references agree
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("dim_set", list(DIM_SETS), ids=list(DIM_SETS))
def test_handwritten_matches_hf_decoder_layer(dim_set):
    """Hand-written torch layer vs HF `LlamaDecoderLayer`, identical weights. PCC >= 0.9999.

    This is the assertion that lets the rest of the bring-up use cheap in-test torch math as its
    oracle. Any residual is fp32 reassociation noise from a different matmul order -- the two
    implementations are algebraically the same function.
    """
    dims = DIM_SETS[dim_set]()
    seq_len = _seq_len_for(dim_set)

    layer, cfg = _build_layer(dims, SEED)
    w = {k: v.detach() for k, v in layer.state_dict().items()}

    # Prove the weight set is exactly what the model card says: 9 tensors, no biases.
    expected_keys = {
        "input_layernorm.weight",
        "post_attention_layernorm.weight",
        "self_attn.q_proj.weight",
        "self_attn.k_proj.weight",
        "self_attn.v_proj.weight",
        "self_attn.o_proj.weight",
        "mlp.gate_proj.weight",
        "mlp.up_proj.weight",
        "mlp.down_proj.weight",
    }
    assert set(w) == expected_keys, f"unexpected per-layer keys: {set(w) ^ expected_keys}"
    assert not any(k.endswith(".bias") for k in w), "Llama-3.1 has no biases (config attention_bias/mlp_bias = false)"

    cos, sin = _rope_tables(cfg, seq_len, batch=1)
    torch.manual_seed(SEED + 1)
    x = torch.randn(1, seq_len, dims["hidden_size"], dtype=REF_DTYPE)

    hf_out = _run_hf(layer, x, cos, sin)
    with torch.no_grad():
        our_out = _torch_decoder_layer(x, w, cos, sin, dims)

    assert hf_out.shape == our_out.shape == (1, seq_len, dims["hidden_size"])

    passing, pcc = comp_pcc(hf_out, our_out, CROSS_REF_PCC)
    max_abs = (hf_out - our_out).abs().max().item()
    rel_l2 = ((hf_out - our_out).norm() / hf_out.norm()).item()
    logger.info(
        f"[G-REF/{dim_set}] hand-written vs HF LlamaDecoderLayer: {pcc} "
        f"(threshold {CROSS_REF_PCC}), max|delta| = {max_abs:.3e}, rel-L2 = {rel_l2:.3e}"
    )
    assert passing, f"cross-reference PCC below {CROSS_REF_PCC}: {pcc}"


# --------------------------------------------------------------------------------------
# G-REF (3): the llama3 RoPE scaling is actually active
# --------------------------------------------------------------------------------------
def test_llama3_rope_scaling_is_active():
    """Assert scaled `inv_freq` differs from unscaled, and that the delta has the analytic shape.

    A RoPE test that passes with scaling silently disabled is worthless
    (BRINGUP_RECIPE.md:650-652), so the fact is pinned before any device code exists.

    The llama3 schedule (`transformers.modeling_rope_utils._compute_llama3_parameters`):
      * wavelength <  `orig/high_freq_factor` -> untouched
      * wavelength >  `orig/low_freq_factor`  -> divided by `factor`
      * in between                            -> smoothly interpolated
    So the maximum relative deviation must be exactly `1 - 1/factor`, and it must be attained.
    """
    dims = _full_dims()
    rs = rope_scaling(dims)
    theta = rope_theta(dims)
    factor = float(rs["factor"])
    orig = int(rs["original_max_position_embeddings"])
    hd = dims["head_dim"]

    cfg_scaled = _hf_config(dims)
    scaled = LlamaRotaryEmbedding(cfg_scaled).inv_freq.to(torch.float64)

    unscaled_dims = dict(dims)
    unscaled_dims["rope_scaling"] = None
    cfg_plain = _hf_config(unscaled_dims)
    unscaled = LlamaRotaryEmbedding(cfg_plain).inv_freq.to(torch.float64)

    assert scaled.shape == unscaled.shape == (hd // 2,)

    rel = (scaled - unscaled).abs() / unscaled
    n_diff = int((rel > 1e-12).sum())
    low_wl = orig / float(rs["low_freq_factor"])
    high_wl = orig / float(rs["high_freq_factor"])
    wavelen = 2 * math.pi / unscaled

    logger.info(
        f"[G-REF] llama3 rope scaling: factor={factor}, orig_max_pos={orig}, theta={theta}, "
        f"low_freq_wavelen={low_wl}, high_freq_wavelen={high_wl}"
    )
    logger.info(
        f"[G-REF] inv_freq slots differing scaled-vs-unscaled: {n_diff}/{hd // 2}; "
        f"max relative diff = {rel.max().item():.6f} (expected 1 - 1/factor = {1 - 1 / factor:.6f})"
    )

    assert n_diff > 0, "llama3 rope scaling had NO effect -- it is silently disabled"
    assert not torch.equal(scaled, unscaled)

    # The low-frequency limb: every frequency whose wavelength exceeds low_freq_wavelen is
    # divided by exactly `factor`.
    low_limb = wavelen > low_wl
    assert low_limb.any(), "no frequency falls in the low-frequency limb; the test proves nothing"
    torch.testing.assert_close(scaled[low_limb], unscaled[low_limb] / factor, rtol=1e-12, atol=0.0)

    # The high-frequency limb is untouched.
    high_limb = wavelen < high_wl
    assert high_limb.any()
    torch.testing.assert_close(scaled[high_limb], unscaled[high_limb], rtol=0.0, atol=0.0)

    torch.testing.assert_close(rel.max().item(), 1.0 - 1.0 / factor, rtol=1e-9, atol=1e-12)


# --------------------------------------------------------------------------------------
# G-REF (4): the repo's own llama3 RoPE helper agrees with HF
# --------------------------------------------------------------------------------------
def test_tt_transformers_precompute_freqs_matches_hf():
    """`models/tt_transformers/tt/common.py:489 precompute_freqs` == HF's llama3 RoPE.

    P5.3 reuses that helper (BRINGUP_RECIPE.md:620-624), so its agreement with HF is checked here,
    on the frequency tables, where a mismatch is unambiguous and cheap to localise.

    It also pins the **convention difference** that Appendix B calls the classic RoPE bug:
      * HF builds `cos/sin` of shape `[S, head_dim]` as `cat(freqs, freqs)`  -> halves concatenated
      * `precompute_freqs` returns `[S, head_dim/2]`; `gather_cos_sin` (:525) then duplicates with
        `stack([c, c], -1).flatten(-2)`                                       -> pairs interleaved
    Both encode the same `[S, head_dim/2]` frequency table; only the expansion differs. Mixing the
    expansions is what produces "attention PCC ~0.5-0.9, norms fine".
    """
    from models.tt_transformers.tt.common import precompute_freqs

    dims = _full_dims()
    rs = rope_scaling(dims)
    hd = dims["head_dim"]
    seq_len = 256

    tt_cos, tt_sin = precompute_freqs(
        dim=hd,
        end=seq_len,
        theta=rope_theta(dims),
        scale_factor=float(rs["factor"]),
        orig_context_len=int(rs["original_max_position_embeddings"]),
        rope_type="llama3",
    )
    assert tt_cos.shape == (seq_len, hd // 2), f"unexpected precompute_freqs shape {tuple(tt_cos.shape)}"

    cfg = _hf_config(dims)
    hf_cos, hf_sin = _rope_tables(cfg, seq_len, batch=1)
    assert hf_cos.shape == (1, seq_len, hd), f"unexpected HF cos shape {tuple(hf_cos.shape)}"

    # HF's first half is the same frequency table precompute_freqs returns whole.
    hf_cos_half = hf_cos[0, :, : hd // 2]
    hf_sin_half = hf_sin[0, :, : hd // 2]

    cos_err = (tt_cos - hf_cos_half).abs().max().item()
    sin_err = (tt_sin - hf_sin_half).abs().max().item()
    logger.info(
        f"[G-REF] precompute_freqs vs HF llama3 (S={seq_len}, head_dim={hd}): "
        f"max|cos delta| = {cos_err:.3e}, max|sin delta| = {sin_err:.3e}"
    )

    # HF duplicates by concatenation; tt_transformers by interleaving. Confirm HF's duplication is
    # the concatenated kind, so P5.3 cannot mistake one for the other.
    torch.testing.assert_close(hf_cos[0, :, hd // 2 :], hf_cos_half, rtol=0.0, atol=0.0)

    assert cos_err < 1e-5, f"precompute_freqs cos disagrees with HF by {cos_err}"
    assert sin_err < 1e-5, f"precompute_freqs sin disagrees with HF by {sin_err}"


# --------------------------------------------------------------------------------------
# G-REF (5): the bundled config has not drifted
# --------------------------------------------------------------------------------------
def test_bundled_config_matches_upstream():
    """`configs/Llama-3.1-8B-Instruct/config.json` is byte-identical to the tt_transformers copy.

    DEC-005: the config is bundled verbatim so dimension-only tests need neither network nor
    checkpoint. Asserting byte-identity is what stops the copy silently drifting from the source
    the model card cites.
    """
    ours = CONFIG_JSON.read_bytes()
    theirs = UPSTREAM_CONFIG_JSON.read_bytes()
    logger.info(f"[G-REF] bundled config sha256   = {hashlib.sha256(ours).hexdigest()}")
    logger.info(f"[G-REF] upstream config sha256  = {hashlib.sha256(theirs).hexdigest()}")
    assert ours == theirs, f"{CONFIG_JSON} has drifted from {UPSTREAM_CONFIG_JSON}"

    # And that the dims helper's derivations are the ones the model card claims.
    dims = llama_config_dims()
    raw = json.loads(theirs)
    assert "head_dim" not in raw, "config now ships head_dim; the model card's derivation note is stale"
    assert dims["head_dim"] == 128 == raw["hidden_size"] // raw["num_attention_heads"]
    assert dims["gqa_group_size"] == 4 == raw["num_attention_heads"] // raw["num_key_value_heads"]
    assert rope_theta(dims) == 500000.0
