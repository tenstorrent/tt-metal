# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""D1: every constant in the config class against the vendored ``config.json``, and the imported
torch reference against the upstream HF math. No TTNN, no device, no checkpoint.

Pattern: ``deepseek_v3_d_p/tests/torch/test_kimi_k3_mla_reference.py``.

Two halves:
  1. ``MistralMedium35Config`` is a transcription of ``configs/.../config.json``; a drift between
     the two must fail here rather than surface as a wrong shape 88 layers deep.
  2. The reference's own RoPE / attention assumptions are checked against what transformers
     actually computes for THIS config — in particular the YaRN ``truncate`` default and the
     ``llama_4_scaling_beta`` query scale, both of which are silent-wrongness risks.
"""

from __future__ import annotations

import math

import pytest
import torch

from models.demos.mistral_3_5_d_p.reference import model as reference
from models.demos.mistral_3_5_d_p.reference.mistral_config import (
    MistralMedium35Config as C,
)
from models.demos.mistral_3_5_d_p.reference.mistral_config import (
    load_text_config,
    raw_config,
)
from models.demos.mistral_3_5_d_p.spec import SPEC


def test_config_constants_match_vendored_json():
    """Each transcribed dimension against ``config.json``'s ``text_config``."""
    tc = raw_config()["text_config"]
    assert C.NUM_LAYERS == tc["num_hidden_layers"]
    assert C.HIDDEN_SIZE == tc["hidden_size"]
    assert C.INTERMEDIATE_SIZE == tc["intermediate_size"]
    assert C.NUM_ATTENTION_HEADS == tc["num_attention_heads"]
    assert C.NUM_KEY_VALUE_HEADS == tc["num_key_value_heads"]
    assert C.HEAD_DIM == tc["head_dim"]
    assert C.VOCAB_SIZE == tc["vocab_size"]
    assert C.RMS_NORM_EPS == tc["rms_norm_eps"]
    assert C.HIDDEN_ACT == tc["hidden_act"]
    assert C.MAX_POSITION_EMBEDDINGS == tc["max_position_embeddings"]
    assert C.SLIDING_WINDOW == tc["sliding_window"]


def test_rope_constants_match_vendored_json():
    rp = raw_config()["text_config"]["rope_parameters"]
    assert C.ROPE_TYPE == rp["rope_type"] == rp["type"]
    assert C.ROPE_THETA == rp["rope_theta"]
    assert C.YARN_FACTOR == rp["factor"]
    assert C.YARN_ORIG_MAX_POS == rp["original_max_position_embeddings"]
    assert C.YARN_BETA_FAST == rp["beta_fast"]
    assert C.YARN_BETA_SLOW == rp["beta_slow"]
    assert C.YARN_MSCALE == rp["mscale"]
    assert C.YARN_MSCALE_ALL_DIM == rp["mscale_all_dim"]
    assert C.LLAMA4_SCALING_BETA == rp["llama_4_scaling_beta"]
    # HF's own default when the key is absent — the reason YARN_TRUNCATE is True.
    assert "truncate" not in rp
    assert C.YARN_TRUNCATE is True


def test_quantization_constants_match_vendored_json():
    """Per-tensor fp8, and the modules the checkpoint leaves unquantized."""
    q = raw_config()["quantization_config"]
    assert C.QUANT_METHOD == q["quant_method"]
    assert C.ACTIVATION_SCHEME == q["activation_scheme"]
    assert C.WEIGHT_BLOCK_SIZE == q["weight_block_size"] is None, "null weight_block_size => PER-TENSOR scale"
    assert tuple(C.MODULES_TO_NOT_CONVERT) == tuple(q["modules_to_not_convert"])


def test_architecture_absences():
    """The features this model does NOT have — each one a block the TT side deliberately omits."""
    tc = raw_config()["text_config"]
    assert tc["sliding_window"] is None, "every layer is full-causal"
    assert "layer_types" not in tc, "no sliding/full alternation schedule"
    assert not any(
        k.startswith("num_experts") or k in ("n_routed_experts", "moe_layer_freq") for k in tc
    ), "dense model: no MoE keys expected"
    assert C.NUM_EXPERTS == 0
    assert C.ATTENTION_BIAS is False and "attention_bias" not in tc, "Ministral3 q/k/v/o are bias-free"
    assert C.USE_QK_NORM is False and "use_qk_norm" not in tc
    assert C.HAS_ATTENTION_SINKS is False


def test_hf_config_loads_and_unwraps():
    """``AutoConfig`` on the vendored dir yields the Mistral3 wrapper; the text backbone unwraps."""
    tc = load_text_config()
    assert type(tc).__name__ == "Ministral3Config"
    assert (tc.num_hidden_layers, tc.hidden_size, tc.head_dim) == (C.NUM_LAYERS, C.HIDDEN_SIZE, C.HEAD_DIM)
    assert (tc.num_attention_heads, tc.num_key_value_heads) == (C.NUM_ATTENTION_HEADS, C.NUM_KEY_VALUE_HEADS)


def test_spec_and_config_agree_on_geometry():
    """The BINDING spec's parallelism must divide the config's head/feature counts."""
    assert C.NUM_ATTENTION_HEADS % SPEC.tp == 0, "Q heads must split across TP"
    assert C.NUM_KEY_VALUE_HEADS == SPEC.tp, "one KV head per TP column is what the KV layout assumes"
    assert C.HIDDEN_SIZE % SPEC.tp == 0
    assert C.INTERMEDIATE_SIZE % SPEC.tp == 0
    assert C.VOCAB_SIZE % SPEC.tp == 0
    assert SPEC.max_seq_len <= C.MAX_POSITION_EMBEDDINGS


def test_llama4_query_scale_is_identity():
    """``get_llama_4_attn_scale`` degenerates to 1.0 at every position, so the TT attention's
    omission of it is exact. This is the assert that catches a config which turns it on."""
    tc = load_text_config()
    reference.assert_llama4_scale_is_identity(tc)  # must not raise
    m = reference.hf_modules()
    pos = torch.tensor([[0, 1, 4095, 4096, 262143]])
    scale = m.get_llama_4_attn_scale(pos, C.LLAMA4_SCALING_BETA, C.YARN_ORIG_MAX_POS)
    assert torch.allclose(scale, torch.ones_like(scale)), f"expected identity query scale, got {scale.flatten()}"


def test_reference_yarn_matches_transformers():
    """The inline golden's YaRN table against ``ROPE_INIT_FUNCTIONS['yarn']`` on the real config.

    This is the check that pins ``truncate=True``: with float correction dims the inv_freq drifts by
    ~3.4e-4, which is invisible at short sequence and collapses long-context K PCC.
    """
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

    hf_inv_freq, hf_attention_factor = ROPE_INIT_FUNCTIONS["yarn"](load_text_config(), None)
    inv_freq, mscale = reference.golden_yarn_inv_freq(
        C.HEAD_DIM,
        C.ROPE_THETA,
        C.YARN_FACTOR,
        C.YARN_ORIG_MAX_POS,
        C.YARN_BETA_FAST,
        C.YARN_BETA_SLOW,
        truncate=C.YARN_TRUNCATE,
    )
    assert torch.equal(inv_freq, hf_inv_freq), "inline YaRN inv_freq differs from transformers"
    assert mscale == pytest.approx(hf_attention_factor), f"mscale {mscale} != HF {hf_attention_factor}"
    # mscale_all_dim is 0.0 (falsy), so HF takes the plain get_mscale(factor) branch.
    assert mscale == pytest.approx(0.1 * math.log(C.YARN_FACTOR) + 1.0)

    wrong, _ = reference.golden_yarn_inv_freq(
        C.HEAD_DIM,
        C.ROPE_THETA,
        C.YARN_FACTOR,
        C.YARN_ORIG_MAX_POS,
        C.YARN_BETA_FAST,
        C.YARN_BETA_SLOW,
        truncate=False,
    )
    assert not torch.equal(wrong, hf_inv_freq), "truncate=False must NOT match HF (guards the donor's wrong note)"


def test_reference_rope_table_matches_hf_module():
    """The inline golden's cos/sin against ``Ministral3RotaryEmbedding``, at an offset."""
    tc = load_text_config()
    seq_len, offset = 64, 128
    hf_cos, hf_sin = reference.hf_rope_cos_sin(tc, seq_len, offset=offset)
    cos, sin = reference.golden_cos_sin_hf(
        seq_len,
        C.HEAD_DIM,
        offset=offset,
        theta=C.ROPE_THETA,
        factor=C.YARN_FACTOR,
        orig_max_pos=C.YARN_ORIG_MAX_POS,
        beta_fast=C.YARN_BETA_FAST,
        beta_slow=C.YARN_BETA_SLOW,
        truncate=C.YARN_TRUNCATE,
    )
    assert torch.allclose(cos, hf_cos[0], atol=1e-6), "cos table differs from the HF rotary module"
    assert torch.allclose(sin, hf_sin[0], atol=1e-6), "sin table differs from the HF rotary module"
