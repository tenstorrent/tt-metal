# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""D1 host-only reference tests. No TTNN, no device, no checkpoint, no network.

Two things are pinned here:

1. **Config constants vs the vendored ``config.json``.** ``Llama31_8BConfig`` is what every TT
   module shapes its weights from; if a checkpoint bump moves a dim, this fails instead of a
   matmul silently mis-shaping.
2. **The vendored torch reference vs upstream ``transformers``.** The reference is the oracle every
   downstream PCC test measures against, so it has to be the same math HF runs — not merely
   self-consistent. Verified for RMSNorm, SwiGLU MLP, llama3 RoPE frequencies, GQA attention,
   the decoder layer, and the whole model.
"""

import json

import pytest
import torch

from models.demos.llama3_1_8b_d_p.reference.config import CONFIG_JSON, Llama31_8BConfig, LlamaConfig
from models.demos.llama3_1_8b_d_p.reference import model as ref

# Upstream HF is a TEST-ONLY dependency: the reference module itself must stay torch-pure.
transformers = pytest.importorskip("transformers", reason="upstream HF needed to pin the reference")


@pytest.fixture(scope="module")
def raw_config():
    with open(CONFIG_JSON) as f:
        return json.load(f)


@pytest.fixture(scope="module")
def cfg():
    """Reduced config: same head geometry, small FFN/vocab so the HF comparison is seconds."""
    return LlamaConfig.from_json().reduced(num_hidden_layers=2, intermediate_size=256, vocab_size=512)


def _hf_config(cfg: LlamaConfig):
    from transformers import LlamaConfig as HFLlamaConfig

    return HFLlamaConfig(
        hidden_size=cfg.hidden_size,
        num_hidden_layers=cfg.num_hidden_layers,
        num_attention_heads=cfg.num_attention_heads,
        num_key_value_heads=cfg.num_key_value_heads,
        head_dim=cfg.head_dim,
        intermediate_size=cfg.intermediate_size,
        vocab_size=cfg.vocab_size,
        rms_norm_eps=cfg.rms_norm_eps,
        attention_bias=cfg.attention_bias,
        mlp_bias=cfg.mlp_bias,
        hidden_act=cfg.hidden_act,
        rope_theta=cfg.rope_theta,
        max_position_embeddings=cfg.max_position_embeddings,
        tie_word_embeddings=cfg.tie_word_embeddings,
        rope_scaling=dict(cfg.rope_scaling),
        attn_implementation="eager",
    )


def _assert_close(got, want, *, atol=1e-5, rtol=1e-5, what=""):
    """fp32 host math on both sides — this is an equality check, not a PCC gate."""
    torch.testing.assert_close(got.float(), want.float(), atol=atol, rtol=rtol, msg=lambda m: f"{what}: {m}")


# ------------------------------------------------------------------------------------------
# 1. Config constants vs the vendored config.json
# ------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "attr, json_key",
    [
        ("HIDDEN_SIZE", "hidden_size"),
        ("NUM_LAYERS", "num_hidden_layers"),
        ("VOCAB_SIZE", "vocab_size"),
        ("RMS_NORM_EPS", "rms_norm_eps"),
        ("NUM_ATTENTION_HEADS", "num_attention_heads"),
        ("NUM_KEY_VALUE_HEADS", "num_key_value_heads"),
        ("INTERMEDIATE_SIZE", "intermediate_size"),
        ("ATTENTION_BIAS", "attention_bias"),
        ("MLP_BIAS", "mlp_bias"),
        ("HIDDEN_ACT", "hidden_act"),
        ("ROPE_THETA", "rope_theta"),
        ("MAX_POSITION_EMBEDDINGS", "max_position_embeddings"),
        ("TIE_WORD_EMBEDDINGS", "tie_word_embeddings"),
        ("TORCH_DTYPE", "torch_dtype"),
    ],
)
def test_config_constant_matches_json(raw_config, attr, json_key):
    assert getattr(Llama31_8BConfig, attr) == raw_config[json_key], f"{attr} != config.json[{json_key}]"


@pytest.mark.parametrize(
    "attr, json_key",
    [
        ("ROPE_TYPE", "rope_type"),
        ("ROPE_FACTOR", "factor"),
        ("ROPE_LOW_FREQ_FACTOR", "low_freq_factor"),
        ("ROPE_HIGH_FREQ_FACTOR", "high_freq_factor"),
        ("ROPE_ORIGINAL_MAX_POSITION", "original_max_position_embeddings"),
    ],
)
def test_rope_constant_matches_json(raw_config, attr, json_key):
    assert getattr(Llama31_8BConfig, attr) == raw_config["rope_scaling"][json_key]


def test_derived_constants_are_consistent(raw_config):
    """head_dim and the GQA group are DERIVED — config.json for 3.1-8B does not state head_dim."""
    assert Llama31_8BConfig.HEAD_DIM == raw_config["hidden_size"] // raw_config["num_attention_heads"]
    assert Llama31_8BConfig.ROTARY_DIM == Llama31_8BConfig.HEAD_DIM, "Llama is full-rotary"
    assert Llama31_8BConfig.NUM_ATTENTION_HEADS % Llama31_8BConfig.NUM_KEY_VALUE_HEADS == 0
    assert Llama31_8BConfig().gqa_group_size == 4


def test_spec_topology_divisibility():
    """The spec's shapes.divisibility block, asserted rather than trusted (spec §shapes)."""
    sp, tp = 8, 4
    chunk_size, max_seq_len = 4096, 131072
    assert chunk_size % (32 * sp) == 0
    assert max_seq_len % (32 * sp) == 0
    assert max_seq_len % chunk_size == 0
    assert Llama31_8BConfig.NUM_KEY_VALUE_HEADS % tp == 0, "TP must divide n_kv_heads"


# ------------------------------------------------------------------------------------------
# 2. The vendored reference vs upstream transformers
# ------------------------------------------------------------------------------------------


def test_rms_norm_matches_hf(cfg):
    from transformers.models.llama.modeling_llama import LlamaRMSNorm as HFRMSNorm

    torch.manual_seed(0)
    x = torch.randn(1, 16, cfg.hidden_size)
    ours = ref.LlamaRMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
    theirs = HFRMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
    w = torch.randn(cfg.hidden_size)
    ours.weight.data, theirs.weight.data = w.clone(), w.clone()
    _assert_close(ours(x), theirs(x), what="rms_norm")


def test_mlp_matches_hf(cfg):
    from transformers.models.llama.modeling_llama import LlamaMLP as HFMLP

    torch.manual_seed(0)
    x = torch.randn(1, 16, cfg.hidden_size)
    ours = ref.LlamaMLP(cfg)
    theirs = HFMLP(_hf_config(cfg))
    theirs.load_state_dict(ours.state_dict())
    _assert_close(ours(x), theirs(x), what="swiglu_mlp")


def test_llama3_rope_frequencies_match_hf(cfg):
    """The rope block is the one place the donor (YaRN) is actively wrong for this model."""
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

    hf_cfg = _hf_config(cfg)
    hf_inv_freq, hf_attn_factor = ROPE_INIT_FUNCTIONS["llama3"](hf_cfg, device="cpu")
    our_inv_freq, our_attn_factor = ref.llama3_inv_freq(cfg.head_dim, cfg)
    _assert_close(our_inv_freq, hf_inv_freq, atol=1e-9, rtol=1e-7, what="llama3 inv_freq")
    assert our_attn_factor == pytest.approx(hf_attn_factor), "llama3 scaling applies no mscale"


def test_rope_cos_sin_matches_hf_rotary_embedding(cfg):
    from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding

    seq_len = 128
    hf_rot = LlamaRotaryEmbedding(_hf_config(cfg))
    x = torch.zeros(1, seq_len, cfg.hidden_size)
    pos = torch.arange(seq_len)[None]
    hf_cos, hf_sin = hf_rot(x, pos)
    cos, sin = ref.build_cos_sin_hf(seq_len, cfg)
    _assert_close(cos, hf_cos, what="cos")
    _assert_close(sin, hf_sin, what="sin")


def test_meta_and_hf_rope_conventions_agree_under_head_permutation(cfg):
    """The Meta interleaved cos/sin + swizzled q is the same rotation as HF half-split on plain q.

    This is the invariant that lets the device use ``rotary_embedding_indexed`` while the oracle
    stays HF-shaped, and it is what ``meta_to_hf_head_perm`` undoes for KV comparisons.
    """
    torch.manual_seed(0)
    seq_len, n_heads, hd = 64, 4, cfg.head_dim
    q_hf = torch.randn(1, n_heads, seq_len, hd)

    cos_hf, sin_hf = ref.build_cos_sin_hf(seq_len, cfg)
    out_hf, _ = ref.apply_rotary_pos_emb(q_hf, q_hf, cos_hf, sin_hf)

    # Meta layout: interleave the two HF halves, i.e. the inverse of meta_to_hf_head_perm.
    a, b = q_hf[..., : hd // 2], q_hf[..., hd // 2 :]
    q_meta = torch.stack([a, b], dim=-1).flatten(-2)
    cos_m, sin_m = ref.build_cos_sin_meta(seq_len, cfg)

    # Meta rotation: pairs (x0,x1) -> (x0*c - x1*s, x1*c + x0*s).
    x0, x1 = q_meta[..., 0::2], q_meta[..., 1::2]
    c, s = cos_m[..., 0::2], sin_m[..., 0::2]
    r0, r1 = x0 * c - x1 * s, x1 * c + x0 * s
    out_meta = torch.stack([r0, r1], dim=-1).flatten(-2)

    _assert_close(ref.meta_to_hf_head_perm(out_meta, hd), out_hf, what="meta vs hf rope")


def test_attention_matches_hf(cfg):
    from transformers.models.llama.modeling_llama import LlamaAttention as HFAttention

    torch.manual_seed(0)
    seq_len = 64
    x = torch.randn(1, seq_len, cfg.hidden_size)
    hf_cfg = _hf_config(cfg)
    ours = ref.LlamaAttention(cfg, layer_idx=0)
    theirs = HFAttention(hf_cfg, layer_idx=0)
    theirs.load_state_dict(ours.state_dict())

    cos, sin = ref.build_cos_sin_hf(seq_len, cfg)
    mask = torch.full((seq_len, seq_len), float("-inf")).triu(1)[None, None]
    our_out, _, _ = ours(x, (cos, sin))
    their_out = theirs(x, position_embeddings=(cos, sin), attention_mask=mask)[0]
    _assert_close(our_out, their_out, atol=2e-5, rtol=2e-5, what="attention")


def test_decoder_layer_matches_hf(cfg):
    from transformers.models.llama.modeling_llama import LlamaDecoderLayer as HFLayer

    torch.manual_seed(0)
    seq_len = 64
    x = torch.randn(1, seq_len, cfg.hidden_size)
    ours = ref.LlamaDecoderLayer(cfg, layer_idx=0)
    theirs = HFLayer(_hf_config(cfg), layer_idx=0)
    theirs.load_state_dict(ours.state_dict())

    cos, sin = ref.build_cos_sin_hf(seq_len, cfg)
    mask = torch.full((seq_len, seq_len), float("-inf")).triu(1)[None, None]
    our_out, _, _ = ours(x, (cos, sin))
    their_out = theirs(x, position_embeddings=(cos, sin), attention_mask=mask)
    their_out = their_out[0] if isinstance(their_out, tuple) else their_out
    _assert_close(our_out, their_out, atol=3e-5, rtol=3e-5, what="decoder layer")


def test_whole_model_matches_hf(cfg):
    """Ground truth for the full stack: embedding, N layers, final norm, lm head."""
    from transformers.models.llama.modeling_llama import LlamaForCausalLM

    torch.manual_seed(0)
    ours = ref.LlamaModel(cfg)
    theirs = LlamaForCausalLM(_hf_config(cfg))
    sd = ours.state_dict()
    mapped = {("lm_head.weight" if k == "lm_head.weight" else f"model.{k}"): v for k, v in sd.items()}
    missing, unexpected = theirs.load_state_dict(mapped, strict=False)
    assert not [k for k in missing if "rotary" not in k], f"unmapped HF params: {missing}"
    assert not unexpected, f"unexpected params: {unexpected}"

    ids = torch.randint(0, cfg.vocab_size, (1, 64))
    our_logits, our_kvs, _ = ours(ids)
    their_logits = theirs(ids).logits
    _assert_close(our_logits, their_logits, atol=1e-4, rtol=1e-4, what="logits")
    assert len(our_kvs) == cfg.num_hidden_layers
    for k, v in our_kvs:
        assert k.shape == (1, cfg.num_key_value_heads, 64, cfg.head_dim)
        assert v.shape == k.shape


def test_chunked_prefill_equals_one_shot(cfg):
    """Two half-chunks with a carried KV prefix == one full-length pass.

    This is the reference-side statement of the invariant P2 has to reproduce on device, and it
    validates ``kv_offset`` in the oracle before any hardware is involved.
    """
    torch.manual_seed(0)
    ours = ref.LlamaModel(cfg)
    seq_len = 64
    ids = torch.randint(0, cfg.vocab_size, (1, seq_len))

    full_logits, full_kvs, _ = ours(ids)

    half = seq_len // 2
    _, kvs0, _ = ours(ids[:, :half], kv_offset=0)
    logits1, kvs1, _ = ours(ids[:, half:], past_kvs=kvs0, kv_offset=half)

    _assert_close(logits1, full_logits[:, half:], atol=2e-4, rtol=2e-4, what="chunked logits")
    for i, ((k0, v0), (k1, v1)) in enumerate(zip(kvs0, kvs1)):
        _assert_close(torch.cat([k0, k1], dim=2), full_kvs[i][0], atol=2e-4, rtol=2e-4, what=f"L{i} K")
        _assert_close(torch.cat([v0, v1], dim=2), full_kvs[i][1], atol=2e-4, rtol=2e-4, what=f"L{i} V")
