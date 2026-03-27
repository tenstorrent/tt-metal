# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os

import pytest

from models.experimental.glm4_moe_lite.tt.weights import (
    find_missing_shards,
    load_glm_lazy_state_dict,
    resolve_best_effort_snapshot_dir,
)


@pytest.mark.skipif(
    os.environ.get("TT_REQUIRE_FULL_GLM47_SNAPSHOT") != "1",
    reason="Enable with TT_REQUIRE_FULL_GLM47_SNAPSHOT=1 (requires all 48 safetensors shards).",
)
def test_glm47_snapshot_is_complete() -> None:
    snap = resolve_best_effort_snapshot_dir("zai-org/GLM-4.7-Flash")
    missing = find_missing_shards(snap)
    assert not missing, f"Missing {len(missing)} shard files (example: {missing[0]})"


@pytest.mark.skipif(
    os.environ.get("TT_ENABLE_LARGE_MODEL_TESTS") != "1",
    reason="Enable with TT_ENABLE_LARGE_MODEL_TESTS=1 (touches real GLM weights).",
)
def test_layer0_required_keys_exist_in_snapshot() -> None:
    snap = resolve_best_effort_snapshot_dir("zai-org/GLM-4.7-Flash")
    state = load_glm_lazy_state_dict(snap, num_layers=47)

    # Layer-0 keys (dense; no experts)
    required = [
        "model.embed_tokens.weight",
        "model.layers.0.input_layernorm.weight",
        "model.layers.0.post_attention_layernorm.weight",
        "model.layers.0.self_attn.q_a_proj.weight",
        "model.layers.0.self_attn.q_a_layernorm.weight",
        "model.layers.0.self_attn.q_b_proj.weight",
        "model.layers.0.self_attn.kv_a_proj_with_mqa.weight",
        "model.layers.0.self_attn.kv_a_layernorm.weight",
        "model.layers.0.self_attn.kv_b_proj.weight",
        "model.layers.0.self_attn.o_proj.weight",
        "model.layers.0.mlp.gate_proj.weight",
        "model.layers.0.mlp.up_proj.weight",
        "model.layers.0.mlp.down_proj.weight",
    ]
    for k in required:
        assert k in state, f"Missing key in snapshot: {k}"
