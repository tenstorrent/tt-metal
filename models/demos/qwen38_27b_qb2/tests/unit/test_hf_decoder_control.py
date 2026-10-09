# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Keep precision diagnostics from inheriting qualification or changing state math."""

import copy
from pathlib import Path

import pytest

from models.demos.qwen38_27b_qb2.demo.probe_hf_layers import validate_decoder_control
from models.demos.qwen38_27b_qb2.tt.precision import load_precision

CONFIG = Path(__file__).resolve().parents[2] / "config"


def policies():
    return (
        load_precision(CONFIG / "precision_accurate_decode_bfp8_head.json"),
        load_precision(CONFIG / "precision_accurate_decode_hifi2.json"),
    )


@pytest.mark.parametrize("name", ["precision_accurate_decode_hifi2.json", "precision_accurate_decode_bfp8_all.json"])
def test_explicit_decoder_controls_are_accepted(name):
    baseline, _ = policies()
    before = copy.deepcopy(baseline)
    candidate = load_precision(CONFIG / name)
    assert validate_decoder_control(baseline, candidate) == candidate
    assert baseline == before


@pytest.mark.parametrize(
    "key,value", [("kv_cache_dtype", "bfloat4_b"), ("decode_recurrence", "single_step"), ("fp32_dest_acc_en", False)]
)
def test_control_cannot_change_nonprojection_settings(key, value, expect_error):
    baseline, candidate = policies()
    candidate[key] = value
    with expect_error(ValueError, "retain baseline state"):
        validate_decoder_control(baseline, candidate)


def test_control_cannot_change_head(expect_error):
    baseline, candidate = policies()
    candidate["weight_groups"]["head"] = "bfloat4_b"
    with expect_error(ValueError, "retain the baseline head"):
        validate_decoder_control(baseline, candidate)


def test_control_cannot_silently_leave_one_projection_lofi(expect_error):
    baseline, candidate = policies()
    candidate["compute_fidelities"]["down"] = "LoFi"
    with expect_error(ValueError, "uniform BFP4 or BFP8"):
        validate_decoder_control(baseline, candidate)
