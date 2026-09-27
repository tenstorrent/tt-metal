# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-only guards against silently ignored precision requests."""
import json

import pytest

from models.autoports.google_gemma_4_26b_a4b_it.tt import precision_policy as policy


def test_selected_default_and_baseline_override(tmp_path, monkeypatch):
    path = tmp_path / "selected.json"
    selected = policy.baseline_precision_config()
    selected["model"]["head_weight_dtype"] = "bfloat8_b"
    path.write_text(json.dumps(selected))
    monkeypatch.setattr(policy, "SELECTED_CONFIG", path)
    assert policy.resolve_precision_config()["model"]["head_weight_dtype"] == "bfloat8_b"
    assert policy.resolve_precision_config({})["model"]["head_weight_dtype"] == "bfloat16"


def test_layer_exception_precedence():
    config = policy.resolve_precision_config(
        {
            "layer_types": {"sliding_attention": {"expert_gate_dtype": "bfloat4_b"}},
            "layer_overrides": {"0": {"expert_gate_dtype": "bfloat8_b"}},
        }
    )
    assert policy.layer_precision_config(config, 0, "sliding_attention")["expert_gate_dtype"] == "bfloat8_b"
    assert policy.layer_precision_config(config, 1, "sliding_attention")["expert_gate_dtype"] == "bfloat4_b"
    assert policy.layer_precision_config(config, 5, "full_attention")["expert_gate_dtype"] == "bfloat4_b"


@pytest.mark.parametrize(
    "override",
    [
        {"ignored_field": "LoFi"},
        {"model": {"fixed": {"residual_dtype": "bfloat8_b"}}},
        {"layer_types": {"sliding_attention": {"kv_cache_dtype": "bfloat4_b"}}},
        {"layer_types": {"full_attention": {"qkv_fidelity": "Unknown"}}},
        {"layer_overrides": {"00": {"expert_gate_dtype": "bfloat4_b"}}},
    ],
)
def test_invalid_policy_fails_closed(override, expect_error):
    with expect_error(ValueError, "Unknown|Unsupported|Layer overrides"):
        policy.resolve_precision_config(override)


def test_runtime_mismatch_rejected(expect_error):
    with expect_error(ValueError, "Precision mismatch"):
        policy.assert_precision_matches({"group": {"dtype": "bfloat8_b"}}, {"group": {"dtype": "bfloat4_b"}})
