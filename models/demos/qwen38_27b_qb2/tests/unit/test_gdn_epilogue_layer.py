# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest

from models.demos.qwen38_27b_qb2.tests.gdn_epilogue_layer import VARIANTS, compare
from models.demos.qwen38_27b_qb2.tt.precision import load_precision


def cases():
    return [
        dict(
            variant=variant,
            batch=16,
            passed=True,
            input_sha256=["input"] * 20,
            state_sha256_per_rank=["state"] * 4,
            output_sha256_per_rank=["raw"] * 4,
            projected_output_sha256_per_rank=["projected"] * 4,
            traced_call_us=[100 if variant == "native" else 90] * 5,
        )
        for variant in VARIANTS
    ]


@pytest.mark.parametrize(
    "key", ["input_sha256", "state_sha256_per_rank", "output_sha256_per_rank", "projected_output_sha256_per_rank"]
)
def test_changed_rank_rejects_claim_even_with_passing_tolerance(key, expect_error):
    rows = cases()
    rows[1][key][-1] = "changed"
    with expect_error(ValueError, "changed"):
        compare(rows)


def test_real_layer_gate_requires_fp32_reference_and_stable_controls(expect_error):
    rows = cases()
    assert compare(rows)["qualified_speedup"] == 100 / 90
    rows[-1]["traced_call_us"] = [110] * 5
    assert compare(rows)["qualified_speedup"] is None
    rows[1]["passed"] = False
    with expect_error(ValueError, "FP32 reference"):
        compare(rows)


def test_opt_in_policy_only_changes_recurrence_and_identifier():
    config = Path(__file__).resolve().parents[2] / "config"
    control = load_precision(config / "precision_single_step_shared_qk_bfp8_all.json")
    candidate = load_precision(config / "precision_single_step_shared_qk_epilogue_bfp8_all.json")
    assert {key for key in control if candidate[key] != control[key]} == {"config_id", "decode_recurrence"}
    assert set(candidate["weight_groups"].values()) == {"bfloat8_b"}
    assert candidate["kv_cache_dtype"] == "bfloat8_b"
    assert candidate["recurrent_dtype"] == "float32"
