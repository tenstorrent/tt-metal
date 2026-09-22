# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""D1 test 1 — the config constants agree with the checkpoint, the spec and the golden trace.

Host-only; no device, no hardware. Every number the reference hard-codes is checked against the
file it was copied from, so a config drift shows up here rather than as an unexplained PCC drop
eight stages later.
"""

import json
from pathlib import Path

import pytest

from models.demos.mistral_medium_3_5_128b.reference.golden import DEFAULT_GOLDEN_TRACE, GoldenTrace
from models.demos.mistral_medium_3_5_128b.reference.model_config import (
    VENDORED_CONFIG,
    MistralMediumConfig,
    PrefillSpec,
    host_reduced_config,
)

UPSTREAM_CONFIG = Path("/mnt/models/mistralai/Mistral-Medium-3.5-128B/config.json")


def test_defaults_match_vendored_config():
    """The dataclass defaults are exactly what ``reference/config.json`` says."""
    assert MistralMediumConfig() == MistralMediumConfig.from_json(VENDORED_CONFIG)


def test_vendored_config_matches_checkpoint():
    """The vendored copy has not drifted from the checkpoint it was taken from."""
    if not UPSTREAM_CONFIG.exists():
        pytest.skip(f"checkpoint config not present at {UPSTREAM_CONFIG}")
    vendored = json.loads(VENDORED_CONFIG.read_text())
    upstream = json.loads(UPSTREAM_CONFIG.read_text())
    assert vendored["text_config"] == upstream["text_config"]
    assert vendored.get("quantization_config") == upstream.get("quantization_config")


def test_shape_constants():
    """The shapes the whole bring-up is built around, stated once and asserted."""
    cfg = MistralMediumConfig()
    assert (cfg.num_hidden_layers, cfg.hidden_size) == (88, 12288)
    assert (cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim) == (96, 8, 128)
    assert cfg.num_key_value_groups == 12
    assert cfg.intermediate_size == 28672
    assert cfg.hidden_act == "silu"
    assert cfg.sliding_window is None, "a sliding window would change the KV cache and the mask"
    assert cfg.tie_word_embeddings is False, "lm_head is a separate tensor in this checkpoint"
    assert cfg.quant_weight_block_size is None, "null block size => per-tensor scalar fp8 scales"


def test_attention_scaling_matches_golden_trace():
    """``0.1*ln(factor)+1`` is the branch transformers actually took for this config.

    ``mscale: 1.0`` with ``mscale_all_dim: 0.0`` looks like it might select the DeepSeek two-mscale
    form; the golden's recorded value settles it. This is the single most consequential constant in
    the reference — a wrong attention scaling gives a uniformly scaled K that still reaches ~0.99
    PCC on some comparisons.
    """
    cfg = MistralMediumConfig()
    if not (DEFAULT_GOLDEN_TRACE / "metadata.json").exists():
        pytest.skip(f"golden trace not present at {DEFAULT_GOLDEN_TRACE}")
    trace = GoldenTrace()
    assert cfg.attention_scaling == pytest.approx(trace.attention_scaling, rel=0, abs=1e-15)
    assert cfg.attention_scaling == pytest.approx(1.4158883083359672, abs=1e-15)


def test_rope_parameters_match_golden_trace():
    if not (DEFAULT_GOLDEN_TRACE / "metadata.json").exists():
        pytest.skip(f"golden trace not present at {DEFAULT_GOLDEN_TRACE}")
    cfg = MistralMediumConfig()
    rope = GoldenTrace().metadata["rope"]["rope_parameters"]
    assert rope["rope_type"] == cfg.rope_type
    assert rope["rope_theta"] == cfg.rope_theta
    assert rope["factor"] == cfg.rope_factor
    assert rope["beta_fast"] == cfg.rope_beta_fast
    assert rope["beta_slow"] == cfg.rope_beta_slow
    assert rope["original_max_position_embeddings"] == cfg.rope_original_max_position_embeddings


def test_golden_trace_dims_match_config():
    if not (DEFAULT_GOLDEN_TRACE / "metadata.json").exists():
        pytest.skip(f"golden trace not present at {DEFAULT_GOLDEN_TRACE}")
    cfg, trace = MistralMediumConfig(), GoldenTrace()
    assert trace.num_layers == cfg.num_hidden_layers
    assert trace.num_kv_heads == cfg.num_key_value_heads
    assert trace.head_dim == cfg.head_dim
    assert trace.n_tokens == 10240
    assert len(trace.metadata["token_ids"]) == trace.n_tokens


def test_spec_defaults():
    """The spec-side values: parallelism, chunking and the two PCC thresholds."""
    spec = PrefillSpec()
    assert (spec.sp, spec.tp) == (8, 4)
    assert spec.mesh_shape == (8, 4)
    assert spec.target_hw == "bh_galaxy"
    assert spec.chunk_size == 5120
    assert spec.max_seq_len == 262144
    assert (spec.pcc_target, spec.pcc_lower_bound) == (0.99, 0.85)
    assert (spec.activations_dtype, spec.kv_cache_dtype, spec.weights_dtype) == (
        "bfloat16",
        "bfloat8_b",
        "bfloat8_b",
    )


def test_spec_shape_divisibility():
    """The spec's own rule: both sequence values are multiples of ``32 * sp``."""
    spec = PrefillSpec()
    period = 32 * spec.sp
    assert spec.max_seq_len % period == 0
    assert spec.chunk_size % period == 0
    # And the head counts have to survive the TP split.
    cfg = MistralMediumConfig()
    assert cfg.num_attention_heads % spec.tp == 0, "Q heads must divide over TP"
    assert cfg.num_key_value_heads % spec.tp == 0, "KV heads must divide over TP"
    assert cfg.hidden_size % spec.tp == 0
    assert cfg.intermediate_size % spec.tp == 0


def test_host_reduced_config_keeps_the_structure():
    """The reduced host config must still exercise GQA and the real rope parameters."""
    full, red = MistralMediumConfig(), host_reduced_config()
    assert red.num_key_value_groups > 1, "a reduced config with group=1 would not be GQA"
    assert red.hidden_size == red.num_attention_heads * red.head_dim
    for field in ("rope_type", "rope_theta", "rope_factor", "rope_beta_fast", "rope_beta_slow"):
        assert getattr(red, field) == getattr(full, field)
    assert red.attention_scaling == full.attention_scaling
