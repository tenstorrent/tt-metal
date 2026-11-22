# SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
# SPDX-License-Identifier: Apache-2.0
"""demo/perf.py gate logic (host only, no ttnn): the absolute app-PCC floor holds with and without a baseline."""

from models.demos.inkling2_mini.demo import perf


def _gate(app_pcc):
    run = {"T": 10, "app_pcc": [app_pcc] * 48, "logits_pcc": 0.999, "top1_agree": 0.95, "top5_has_ref_top1": 1.0}
    tok = {"T": 10, "logits_pcc": 0.999, "top1_agree": 0.9, "kl_mean": 0.01}
    return {n: {"forced": run, "tokens": tok} for n in perf.SHORT}


def test_gate_passes():
    _, fails = perf.summarize_gate(_gate(0.995), None)
    assert fails == []


def test_app_pcc_floor_without_baseline():
    _, fails = perf.summarize_gate(_gate(0.95), None)
    assert any("forced app pcc" in f for f in fails), fails


def test_app_pcc_floor_with_low_baseline():
    base = {"app_pcc": [0.9] * 48, "tokens_top1_hits": 0}
    _, fails = perf.summarize_gate(_gate(0.95), base)
    assert any("forced app pcc" in f for f in fails), fails
