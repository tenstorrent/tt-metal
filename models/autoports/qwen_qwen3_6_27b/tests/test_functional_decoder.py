# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import inspect
import json
from pathlib import Path

from transformers import AutoConfig

from models.autoports.qwen_qwen3_6_27b.tt.functional_decoder import (
    ADVERTISED_CONTEXT,
    MODEL_ID,
    MODEL_REVISION,
    REPRESENTATIVE_LAYERS,
    FunctionalDecoder,
    _candidate_keys,
)

ROOT = Path("models/autoports/qwen_qwen3_6_27b")


def _text_config():
    return AutoConfig.from_pretrained(MODEL_ID, revision=MODEL_REVISION).text_config


def test_real_config_and_layer_kind_contract():
    config = _text_config()
    assert config.hidden_size == 5120
    assert config.intermediate_size == 17408
    assert config.head_dim == 256
    assert config.max_position_embeddings == ADVERTISED_CONTEXT
    assert config.layer_types.count("linear_attention") == 48
    assert config.layer_types.count("full_attention") == 16
    for kind, layer_idx in REPRESENTATIVE_LAYERS.items():
        assert config.layer_types[layer_idx] == kind


def test_context_contract_matches_hf_advertisement():
    contract = json.loads((ROOT / "doc/context_contract.json").read_text())
    assert contract["model_id"] == MODEL_ID
    assert contract["model_revision"] == MODEL_REVISION
    assert contract["hf_advertised_context"] == ADVERTISED_CONTEXT
    assert contract["functional_decoder"]["serving_decode_batch"] == 32
    functional = contract["functional_decoder"]
    assert functional["supported_context"] == ADVERTISED_CONTEXT
    assert functional["largest_decode_position_tested"] == ADVERTISED_CONTEXT - 1
    assert functional["watcher_interval_seconds"] == 10
    assert functional["numerical_tests"]["full_attention"]["numerical_traced_decode_batches"] == [1, 32]
    assert functional["numerical_tests"]["linear_attention"]["numerical_traced_decode_batches"] == [1, 32]
    reduction = functional["capability_reduction"]
    assert reduction["reason"] == "hard device DRAM contiguous-allocation limit"
    assert reduction["largest_feasible_prefill_tested"] == 192511
    assert reduction["smallest_prefill_failure_tested"] == 194559
    assert reduction["failure"]["requested_bytes_per_bank"] > reduction["failure"]["largest_free_block_bytes_per_bank"]


def test_canonical_checkpoint_prefix_is_first():
    keys = _candidate_keys(3, "self_attn.q_proj.weight")
    assert keys[0] == "model.language_model.layers.3.self_attn.q_proj.weight"


def test_decode_runtime_has_no_host_fallback_or_placeholder():
    runtime_methods = (
        FunctionalDecoder._linear_attention_prefill,
        FunctionalDecoder._linear_attention_prefill_chunk,
        FunctionalDecoder._linear_attention_decode,
        FunctionalDecoder._full_attention_prefill,
        FunctionalDecoder._full_attention_decode,
        FunctionalDecoder._per_head_norm,
        FunctionalDecoder._per_head_norm_prefill,
        FunctionalDecoder._partial_rope_decode,
        FunctionalDecoder._partial_rope_prefill,
        FunctionalDecoder._mlp,
        FunctionalDecoder.prefill_forward,
        FunctionalDecoder.decode_forward,
    )
    forbidden = (
        "torch",
        "from_torch",
        "to_torch",
        ".cpu(",
        "NotImplementedError",
    )
    for method in runtime_methods:
        source = inspect.getsource(method)
        assert all(token not in source for token in forbidden), method.__name__
