# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""D1 — the config class against the vendored config.json, and the vendored reference against
upstream HF llama math.

Host only. No TTNN, no device, no checkpoint. Structure mirrors
`deepseek_v3_d_p/tests/torch/test_kimi_k3_mla_reference.py`.

Two jobs:

1. **Constants.** Every field of `LlamaConfigConstants` is asserted against
   `configs/Llama-3.1-8B-Instruct/config.json`, so a checkpoint swap that moves a dim fails here
   rather than as a mystery PCC drop on the mesh.
2. **Upstream parity.** Each vendored block is run against the live
   `transformers.models.llama.modeling_llama` class, same weights on both sides, at the model's real
   dims. This is what stops the vendored copy drifting from upstream — vendoring is a purity and
   version-boundary decision (see `reference/model.py`), not a licence to diverge.

The bar is bit-exactness where the math is genuinely identical, and `1 - 1e-3` relative agreement
where fp16 rounding order can differ (a matmul reassociated by a different reshape). The reference
computes in fp16 (recipe §4), which has ~3 decimal digits, so a tighter bar here would be measuring
noise rather than agreement.

NOTE the directory name: `tests/torch_ref/`, not `tests/torch/`. Running any script that lives in a
`tests/torch/` package puts that directory on `sys.path[0]`, so `import torch` resolves to the test
package and `import ttnn` then dies with "module 'torch' has no attribute 'nn'".
"""

from __future__ import annotations

import pytest
import torch

from models.demos.llama_3_1_8b_d_p.reference.config import CONFIG_JSON, LlamaConfigConstants, load_config_json
from models.demos.llama_3_1_8b_d_p.reference.model import (
    REF_DTYPE,
    RefAttention,
    RefDecoderLayer,
    RefMLP,
    RefRMSNorm,
    RefRotaryEmbedding,
    apply_rotary_pos_emb,
    causal_mask,
    compute_llama3_inv_freq,
)

SEQ = 256  # enough tokens to exercise the causal mask and all three rope frequency bands' plumbing
ATOL_PCC = 1 - 1e-3


def _pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    """Pearson correlation over flattened fp32 copies. The same measure the device tests use."""
    a, b = a.detach().float().flatten(), b.detach().float().flatten()
    a, b = a - a.mean(), b - b.mean()
    denom = a.norm() * b.norm()
    if denom == 0:
        return 1.0 if torch.allclose(a, b) else 0.0
    return float((a @ b) / denom)


@pytest.fixture(scope="module")
def config() -> LlamaConfigConstants:
    return LlamaConfigConstants.from_json()


@pytest.fixture(scope="module")
def hf_config(config):
    return config.to_hf_config()


# ---------------------------------------------------------------------------
# 1. constants vs the vendored config.json
# ---------------------------------------------------------------------------


def test_config_constants_match_vendored_json(config):
    """Every constant in the config class is what the checkpoint's own config.json says."""
    raw = load_config_json()
    assert raw["architectures"] == ["LlamaForCausalLM"], "not a plain LlamaForCausalLM checkpoint"
    assert raw["model_type"] == "llama"

    assert config.hidden_size == raw["hidden_size"] == 4096
    assert config.intermediate_size == raw["intermediate_size"] == 14336
    assert config.num_hidden_layers == raw["num_hidden_layers"] == 32
    assert config.num_attention_heads == raw["num_attention_heads"] == 32
    assert config.num_key_value_heads == raw["num_key_value_heads"] == 8
    assert config.max_position_embeddings == raw["max_position_embeddings"] == 131072
    assert config.rms_norm_eps == raw["rms_norm_eps"] == 1e-5
    assert config.rope_theta == raw["rope_theta"] == 500000.0
    assert config.vocab_size == raw["vocab_size"] == 128256
    assert config.hidden_act == raw["hidden_act"] == "silu"
    assert config.attention_bias is raw["attention_bias"] is False
    assert config.mlp_bias is raw["mlp_bias"] is False
    assert config.tie_word_embeddings is raw["tie_word_embeddings"] is False
    assert config.torch_dtype == raw["torch_dtype"] == "bfloat16"
    assert config.rope_scaling == raw["rope_scaling"]
    assert config.rope_scaling["rope_type"] == "llama3"


def test_derived_dims(config):
    """head_dim is absent from this config.json — assert the derivation, and the GQA grouping."""
    assert config.head_dim == 128
    assert config.num_key_value_groups == 4
    assert config.attn_scale == pytest.approx(128**-0.5)
    assert config.hidden_size == config.num_attention_heads * config.head_dim


def test_checkpoint_is_unquantized(config):
    """No quantization block in the config: there is no dequant step in this bring-up.

    This is what justifies skipping the `test_mxfp4_loader.py` row of P1's Testing table.
    """
    raw = load_config_json()
    for k in ("quantization_config", "quantization", "quant_method"):
        assert k not in raw, f"config.json carries {k}: the P1 loader needs a dequant path after all"


@pytest.mark.parametrize("tp", [4, 8])
def test_kv_heads_per_chip(config, tp):
    """The shape trap: at the spec's TP=4 a chip holds TWO KV heads, not one.

    Pinned as a test because every donor package on this engine has exactly one and says so, and
    getting it wrong raises nothing — it silently mis-sizes the per-chip cache row.
    """
    expected = {4: 2, 8: 1}[tp]
    assert config.kv_heads_per_chip(tp) == expected
    assert config.q_heads_per_chip(4) == 8


def test_kv_heads_per_chip_rejects_indivisible_tp(config):
    with pytest.raises(AssertionError, match="cannot straddle"):
        config.kv_heads_per_chip(3)


def test_spec_alignment_constraints():
    """The spec's two shapes against the block-cyclic alignment rule `% (32 * sp) == 0`.

    A misaligned chunk_size corrupts KV addresses silently rather than failing, so the constraint is
    asserted in a host test that runs in a second instead of being discovered on the mesh.
    """
    import json
    from pathlib import Path

    # CONFIG_JSON is <pkg>/configs/Llama-3.1-8B-Instruct/config.json -> three levels up is <pkg>.
    spec = json.loads((Path(CONFIG_JSON).parents[2] / "llama_3_1_8b.spec.json").read_text())
    sp = spec["parallelism"]["sp"]
    tp = spec["parallelism"]["tp"]
    period = 32 * sp
    for name in ("max_seq_len", "chunk_size"):
        value = spec["shapes"][name]
        assert value % period == 0, f"{name}={value} is not a multiple of 32*sp={period}"
    assert sp * tp == 32, "spec parallelism does not fill one Blackhole Galaxy"
    assert spec["hf_repo"] == "meta-llama/Llama-3.1-8B-Instruct"
    assert spec["acceptance"]["pcc_lower_bound"] <= spec["acceptance"]["pcc_target"]


# ---------------------------------------------------------------------------
# 2. vendored reference vs upstream HF math
# ---------------------------------------------------------------------------


def test_llama3_inv_freq_matches_upstream(config, hf_config):
    """Our transcribed llama3 rope frequencies vs `_compute_llama3_parameters`, bit-exact.

    This is the single most load-bearing parity check in D1: llama3 scaling only bites past the
    original 8192-token context, so a wrong transcription is invisible in every short-ISL test and
    then destroys accuracy at the 131072 the spec asks for.
    """
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

    upstream, upstream_scaling = ROPE_INIT_FUNCTIONS["llama3"](hf_config, device=None)
    ours = compute_llama3_inv_freq(config.head_dim, config.rope_theta, config.rope_scaling)

    assert ours.shape == upstream.shape == (config.head_dim // 2,)
    assert torch.equal(ours, upstream), f"max abs diff {(ours - upstream).abs().max()}"
    assert upstream_scaling == 1.0, "llama3 rope returns attention_scaling 1.0; the reference hardcodes that"


def test_llama3_inv_freq_actually_scales(config):
    """Sanity on the transcription itself: the three bands must differ from unscaled rope.

    Guards against a transcription that happens to type-check but drops the scaling — which
    `test_llama3_inv_freq_matches_upstream` alone would not catch if upstream were also broken.
    """
    scaled = compute_llama3_inv_freq(config.head_dim, config.rope_theta, config.rope_scaling)
    unscaled = 1.0 / (config.rope_theta ** (torch.arange(0, config.head_dim, 2).float() / config.head_dim))
    factor = config.rope_scaling["factor"]
    assert not torch.allclose(scaled, unscaled), "llama3 scaling had no effect"
    # inv_freq DECREASES with index: index 0 is the highest frequency / shortest wavelength, the
    # last index the lowest frequency / longest wavelength. So index 0 is the untouched short-
    # wavelength end, and the last index is the long-wavelength end that gets divided by `factor`.
    assert scaled[0] == pytest.approx(float(unscaled[0]), rel=1e-6), "high-frequency end must be unscaled"
    assert scaled[-1] == pytest.approx(float(unscaled[-1]) / factor, rel=1e-6), "low-frequency end must be /factor"
    # And the middle band must be neither: strictly between the two treatments somewhere.
    ratio = (unscaled / scaled)[1:-1]
    assert ((ratio > 1.0 + 1e-6) & (ratio < factor - 1e-6)).any(), "no smoothly-interpolated middle band"


def test_rope_tables_match_upstream(config, hf_config):
    """cos/sin tables, and a rotated Q/K pair, vs upstream `LlamaRotaryEmbedding` + `apply_rotary_pos_emb`."""
    from transformers.models.llama import modeling_llama as hf

    ours = RefRotaryEmbedding(config)
    theirs = hf.LlamaRotaryEmbedding(hf_config)
    assert torch.equal(ours.inv_freq, theirs.inv_freq)

    position_ids = torch.arange(SEQ)[None, :]
    cos, sin = ours(position_ids, dtype=REF_DTYPE)
    dummy = torch.zeros(1, SEQ, config.hidden_size, dtype=REF_DTYPE)
    cos_hf, sin_hf = theirs(dummy, position_ids)
    assert torch.equal(cos, cos_hf) and torch.equal(sin, sin_hf)

    n_q, n_kv, hd = config.num_attention_heads, config.num_key_value_heads, config.head_dim
    torch.manual_seed(0)
    q = torch.randn(1, n_q, SEQ, hd, dtype=REF_DTYPE)
    k = torch.randn(1, n_kv, SEQ, hd, dtype=REF_DTYPE)
    q_ours, k_ours = apply_rotary_pos_emb(q, k, cos, sin)
    q_hf, k_hf = hf.apply_rotary_pos_emb(q, k, cos, sin)
    assert torch.equal(q_ours, q_hf) and torch.equal(k_ours, k_hf)


def test_rope_convention_is_half_split_not_interleaved(config):
    """The rotation convention is HF half-split, not the Meta interleaved one.

    Both conventions are self-consistent and both produce plausible outputs from the same weights;
    they simply pair different head columns. The device side uses `rotary_embedding_llama`, whose
    tables must match THIS convention — so pin which one the oracle is, rather than leaving it to
    be discovered as a PCC drop in D3.
    """
    hd = config.head_dim
    ours = RefRotaryEmbedding(config)
    cos, sin = ours(torch.arange(SEQ)[None, :], dtype=torch.float32)
    # Half-split: cos duplicates its first half into its second (emb = cat(freqs, freqs)).
    assert torch.equal(cos[..., : hd // 2], cos[..., hd // 2 :])
    # Interleaved tables would instead repeat each element pairwise.
    assert not torch.allclose(cos[..., 0::2], cos[..., 1::2])


def test_rms_norm_matches_upstream(config):
    """Plain RMSNorm, bit-exact vs `LlamaRMSNorm` — and NOT the Gemma `1 + weight` form."""
    from transformers.models.llama import modeling_llama as hf

    torch.manual_seed(0)
    x = torch.randn(1, SEQ, config.hidden_size, dtype=REF_DTYPE)
    w = (1.0 + 0.1 * torch.randn(config.hidden_size)).to(REF_DTYPE)

    ours = RefRMSNorm(config.hidden_size, config.rms_norm_eps)
    theirs = hf.LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps).to(REF_DTYPE)
    with torch.no_grad():
        ours.weight.copy_(w)
        theirs.weight.copy_(w)
        assert torch.equal(ours(x), theirs(x))
        # The Gemma fold would multiply by (1 + w); assert we are measurably not that.
        gemma = ours(x) / w * (1.0 + w)
        assert not torch.allclose(ours(x).float(), gemma.float(), atol=1e-2)


def test_mlp_matches_upstream(config, hf_config):
    """Dense SwiGLU MLP at real dims (4096 -> 14336 -> 4096) vs `LlamaMLP`."""
    from transformers.models.llama import modeling_llama as hf

    torch.manual_seed(0)
    x = torch.randn(1, SEQ, config.hidden_size, dtype=REF_DTYPE)
    ours = RefMLP(config)
    theirs = hf.LlamaMLP(hf_config).to(REF_DTYPE)
    theirs.load_state_dict({k: v.detach().clone() for k, v in ours.state_dict().items()})
    with torch.no_grad():
        assert _pcc(ours(x), theirs(x)) > ATOL_PCC


def test_mlp_activation_is_plain_silu_swiglu(config):
    """Not the clamped swigluoai the donor MLPs use.

    minimax_m3 and gpt_oss_d_p both apply α=1.702 / clamp-limit-7.0 swigluoai. Their MLP structure
    was borrowed; this asserts the activation math was not.
    """
    torch.manual_seed(0)
    x = torch.randn(1, 32, config.hidden_size, dtype=REF_DTYPE)
    mlp = RefMLP(config)
    with torch.no_grad():
        gate, up = mlp.gate_proj(x), mlp.up_proj(x)
        expected = mlp.down_proj(torch.nn.functional.silu(gate) * up)
        assert torch.equal(mlp(x), expected)
        # swigluoai: gate is clamped to +/-7 and scaled by alpha before the sigmoid. With a clamp
        # this tight on real-scale activations the two activations must differ.
        clamped = gate.clamp(max=7.0)
        oai = mlp.down_proj((clamped * torch.sigmoid(1.702 * clamped)) * (up.clamp(-7.0, 7.0) + 1))
        assert _pcc(mlp(x), oai) < 0.999, "activation is indistinguishable from swigluoai - check the port"


def test_attention_matches_upstream(config, hf_config):
    """Whole attention block vs `LlamaAttention` — QKV proj, head split, RoPE, causal GQA, o_proj."""
    from transformers.models.llama import modeling_llama as hf

    torch.manual_seed(0)
    x = torch.randn(1, SEQ, config.hidden_size, dtype=REF_DTYPE)
    ours = RefAttention(config)
    theirs = hf.LlamaAttention(hf_config, layer_idx=0).to(REF_DTYPE)
    theirs.load_state_dict({k: v.detach().clone() for k, v in ours.state_dict().items()})

    rope = RefRotaryEmbedding(config)
    cos, sin = rope(torch.arange(SEQ)[None, :], dtype=REF_DTYPE)
    mask = causal_mask(SEQ)

    with torch.no_grad():
        out_ours, k_rope, v = ours(x, (cos, sin), mask, return_kv=True)
        out_theirs, _ = theirs(x, position_embeddings=(cos, sin), attention_mask=mask)
    assert _pcc(out_ours, out_theirs) > ATOL_PCC

    # The KV the cache stores: K after RoPE, V raw, both at n_kv heads (not inflated).
    assert k_rope.shape == v.shape == (1, config.num_key_value_heads, SEQ, config.head_dim)
    q_hf = theirs.q_proj(x).view(1, SEQ, -1, config.head_dim).transpose(1, 2)
    k_hf = theirs.k_proj(x).view(1, SEQ, -1, config.head_dim).transpose(1, 2)
    _, k_hf_rope = hf.apply_rotary_pos_emb(q_hf, k_hf, cos, sin)
    assert torch.equal(k_rope, k_hf_rope), "cached K must be post-RoPE, matching upstream"
    assert torch.equal(v, theirs.v_proj(x).view(1, SEQ, -1, config.head_dim).transpose(1, 2))


def test_attention_is_causal(config):
    """A token's output must not move when a strictly later token changes.

    Cheap, and it catches a mask that is transposed or off by one — which a PCC-vs-upstream check
    cannot, since it would compare two identically-wrong masks.
    """
    torch.manual_seed(0)
    x = torch.randn(1, 64, config.hidden_size, dtype=REF_DTYPE)
    attn = RefAttention(config)
    rope = RefRotaryEmbedding(config)
    cos, sin = rope(torch.arange(64)[None, :], dtype=REF_DTYPE)
    with torch.no_grad():
        base = attn(x, (cos, sin), causal_mask(64))
        perturbed_x = x.clone()
        perturbed_x[:, 40:, :] += 1.0
        perturbed = attn(perturbed_x, (cos, sin), causal_mask(64))
    assert torch.equal(base[:, :40, :], perturbed[:, :40, :]), "future tokens leaked into earlier positions"
    assert not torch.equal(base[:, 40:, :], perturbed[:, 40:, :])


def test_decoder_layer_matches_upstream(config, hf_config):
    """One complete decoder layer, residuals included, vs `LlamaDecoderLayer`."""
    from transformers.models.llama import modeling_llama as hf

    torch.manual_seed(0)
    x = torch.randn(1, SEQ, config.hidden_size, dtype=REF_DTYPE)
    ours = RefDecoderLayer(config)
    theirs = hf.LlamaDecoderLayer(hf_config, layer_idx=0).to(REF_DTYPE)
    theirs.load_state_dict({k: v.detach().clone() for k, v in ours.state_dict().items()})

    rope = RefRotaryEmbedding(config)
    cos, sin = rope(torch.arange(SEQ)[None, :], dtype=REF_DTYPE)
    mask = causal_mask(SEQ)
    with torch.no_grad():
        out_ours = ours(x, (cos, sin), mask)
        out_theirs = theirs(x, attention_mask=mask, position_embeddings=(cos, sin))
    if isinstance(out_theirs, (tuple, list)):
        out_theirs = out_theirs[0]
    assert _pcc(out_ours, out_theirs) > ATOL_PCC


def test_state_dict_keys_match_upstream(config, hf_config):
    """Identical parameter names, so the P1 checkpoint loader's key mapping is the identity.

    A silent rename here would show up at P1 as weights that load but sit in the wrong slot.
    """
    from transformers.models.llama import modeling_llama as hf

    ours = set(RefDecoderLayer(config).state_dict())
    theirs = set(hf.LlamaDecoderLayer(hf_config, layer_idx=0).state_dict())
    assert ours == theirs, f"only ours: {ours - theirs}; only upstream: {theirs - ours}"
