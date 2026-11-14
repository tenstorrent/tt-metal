# SPDX-License-Identifier: Apache-2.0
"""Host regressions for isolated precision-labelled benchmark reporting."""

import importlib
import json
import sys
import unittest
from pathlib import Path

TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))
r = importlib.import_module("remeasure_agentic_rates")


def test_population_rejects_short_generation():
    row = {"input_tokens": 64, "output_tokens": 128}
    raw = {"failed": 0, "completed": 1, "input_lens": [64], "output_lens": [127]}
    with unittest.TestCase().assertRaises(ValueError):
        r.validate_raw(raw, row, 1)


def test_rate_math_and_precision_identity(tmp_path):
    row = {"id": "x-gpqa", "input_tokens": 64, "output_tokens": 128, "requests": 1, "sampling": "gpqa"}
    raw = {
        "failed": 0,
        "completed": 1,
        "input_lens": [64],
        "output_lens": [128],
        "mean_ttft_ms": 2000,
        "mean_tpot_ms": 25,
        "duration": 6,
    }
    policy = {"config_id": "candidate", "weight_groups": {"lm_head": "bfp8"}}
    result = r.measured_row(row, raw, tmp_path, policy)
    assert result["prefill_tok_s"] == 32 and result["decode_tok_s_user"] == 40
    assert result["precision_sha256"] == r.policy_hash(policy)
    assert r.policy_hash(policy) != r.policy_hash(dict(policy, weight_groups={"lm_head": "bfp4"}))


def test_report_separates_configurations_and_policies(tmp_path):
    original = {
        "server_config": {"precision": {"config_id": "old"}},
        "sampling_policies": {"gpqa": {}, "swe": {}},
        "rows": [{"id": "original"}],
    }
    policy = {"config_id": "candidate"}
    rows = [{"id": "x-gpqa", "sampling": "gpqa"}, {"id": "x-swe", "sampling": "swe"}]
    before = json.dumps(original, sort_keys=True)
    r.report(tmp_path, original, rows, {"precision": policy}, policy, ["x-gpqa", "x-swe", "missing-swe"])
    merged = json.loads((tmp_path / "serving_rates-by-config.json").read_text())
    assert "rows" not in merged
    assert merged["configurations"]["old"] == original
    assert json.dumps(original, sort_keys=True) == before
    gpqa = json.loads((tmp_path / "serving_rates-gpqa.json").read_text())
    assert gpqa["rows"] == rows[:1] and gpqa["missing_rows"] == []
    assert gpqa["status"] == "incomplete"


def test_collector_rejects_wrong_precision_before_model_call(monkeypatch, tmp_path):
    c = importlib.import_module("collect_benchmark_roofline")
    monkeypatch.setattr(
        c,
        "exchange",
        lambda *args: dict(
            clock="perf_counter_ns",
            identity={"layers": list(range(60)), "precision": {"config_id": "wrong"}, "device_ids": [0, 1, 2, 3]},
        ),
    )
    with unittest.TestCase().assertRaisesRegex(ValueError, "identity"):
        c.check(
            type("Args", (), {"run_dir": tmp_path, "concurrency": 32})(),
            {"phase_log": str(tmp_path / "phase"), "precision": {"config_id": "candidate"}},
        )


def test_head_precision_changes_only_head_dram_estimate():
    w = importlib.import_module("benchmark_work")
    policy = {
        "weight_groups": {"attention": "bfp4", "mlp": "bfp4", "down": "bfp4", "lm_head": "bfp8_lofi"},
        "layer_exceptions": {},
        "kv_cache_dtype": "bfloat8_b",
    }
    old = w.decode_work([100], 32)
    new = w.decode_work([100], 32, policy)
    assert new["weight_bytes"] - old["weight_bytes"] == (w.BFP8_BYTES - w.BFP4_BYTES) * w.H * w.V
    assert new["kv_read_bytes"] == old["kv_read_bytes"]
    policy["weight_groups"]["attention"] = "bf16"
    with unittest.TestCase().assertRaises(ValueError):
        w.decode_work([100], 32, policy)


def test_timeout_stops_only_client_and_reports_pending_drain(tmp_path):
    import os

    with unittest.TestCase().assertRaisesRegex(RuntimeError, "server may still drain"):
        r.invoke([sys.executable, "-c", "import time; time.sleep(10)"], dict(os.environ), tmp_path / "client.log", 0.02)


def test_rate_prediction_preserves_prior_and_source_identities():
    prior = {"id": "row", "decode_tok_s_user": 40, "prefill_tok_s": 1000}
    original = {"source_commits": {"tt-metal": "original"}, "configuration_evidence_sha256": "old-hash"}
    candidate = {"source_commits": {"tt-metal": "candidate"}, "configuration_evidence_sha256": "new-hash"}
    prediction = r.rate_prediction(prior, original, candidate)
    assert prediction["original_row"] == prior
    assert prediction["candidate_source_commits"]["tt-metal"] == "candidate"
    assert prediction["ranges"]["decode_tok_s_user"]["low"] == 36
    assert prediction["ranges"]["prefill_tok_s"]["high"] == 1050
    hit = r.compare_prediction({"decode_tok_s_user": 39.416, "prefill_tok_s": 900}, prediction)
    assert not hit["explanation_required"]
    miss = r.compare_prediction({"decode_tok_s_user": 35, "prefill_tok_s": 1100}, prediction)
    assert miss["explanation_required"] and miss["explanation"] is None
    assert miss["rates"]["decode_tok_s_user"]["ratio_to_prior"] == 0.875
    assert not miss["rates"]["prefill_tok_s"]["range_hit"]
