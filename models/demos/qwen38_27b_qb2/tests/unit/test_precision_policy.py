# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Keep experimental attention explicit and retain historical config behavior."""

from pathlib import Path

from models.demos.qwen38_27b_qb2.tt.precision import decoder_policy, load_precision


def test_existing_artifact_still_uses_native_decode():
    path = Path(__file__).resolve().parents[2] / "config/precision.json"
    policy = load_precision(path)
    assert policy["decode_attention"] == "native"
    assert decoder_policy(policy, 3)["decode_attention"] == "native"


def test_attention_override_preserves_weights_and_prefill():
    baseline = load_precision("baseline")
    candidate = dict(baseline, decode_attention="accurate_full_tile")
    loaded = load_precision(candidate)
    assert baseline["decode_attention"] == "native"
    assert loaded == candidate
    assert decoder_policy(loaded, 3)["decode_attention"] == "accurate_full_tile"


def test_unsupported_attention_is_rejected(expect_error):
    with expect_error(ValueError, "Unsupported decode attention"):
        load_precision(dict(load_precision("baseline"), decode_attention="unqualified"))
