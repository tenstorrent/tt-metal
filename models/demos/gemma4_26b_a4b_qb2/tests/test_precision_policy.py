# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import copy
import json

from models.demos.gemma4_26b_a4b_qb2.tt import precision_policy


def test_committed_precision_policy_is_complete():
    selected = precision_policy.resolve_precision_config()

    assert selected["schema_version"] == 1
    assert selected["config_id"]
    assert set(selected["layer_types"]) == {"sliding_attention", "full_attention"}
    for kind in selected["layer_types"]:
        assert precision_policy.layer_precision_config(selected, 0, kind)


def test_precision_policy_returns_independent_values():
    first = precision_policy.resolve_precision_config()
    second = precision_policy.resolve_precision_config()

    first["model"]["head_weight_dtype"] = "mutated"
    assert second["model"]["head_weight_dtype"] != first["model"]["head_weight_dtype"]


def test_unknown_precision_field_is_rejected(expect_error):
    with expect_error(ValueError, "Unknown precision field"):
        precision_policy.resolve_precision_config({"model": {"unknown": "value"}})


def test_fixed_precision_field_is_rejected(expect_error):
    override = copy.deepcopy(precision_policy.baseline_precision_config())
    override["model"]["fixed"]["page_size"] = 64

    with expect_error(ValueError, "Unsupported fixed precision fields"):
        precision_policy.resolve_precision_config(override)


def test_committed_policy_is_json_round_trip_stable():
    selected = precision_policy.resolve_precision_config()
    assert json.loads(json.dumps(selected)) == selected
