# SPDX-License-Identifier: Apache-2.0
"""Read-only Stage8 evidence checks; no device initialization or reference changes."""

import hashlib
import json
from pathlib import Path


def main():
    model = Path(__file__).resolve().parents[1]
    root = model / "doc/datatype_sweep"

    def read(path):
        return json.loads((root / path).read_text())

    report = read("sweep_results.json")
    selected = read("selected_precision_config.json")
    final = read("selected_default/result.json")
    assert report["final_selection"]
    assert report["thresholds"] == dict(top1=0.9, top5=0.98, top100=1.0)
    assert set(read("configs/matrix.json")) == {row["config_id"] for row in report["results"]}
    assert all(row["status"] in ("pass", "accuracy-fail", "capacity-rejected") for row in report["results"])
    for row in report["results"]:
        if row["decode_t_s_u"] is not None:
            assert row["trace_verified"] and row["token_count"] == 100
            assert row["environment"].get("TT_METAL_TRACE_ALLOC_TRACKING") == "0"
    passing = [row for row in report["results"] if row["status"] == "pass" and row["trace_verified"]]
    fastest = max(passing, key=lambda row: row["decode_t_s_u"])
    assert fastest["config_id"] == report["selected_config_id"] == selected["config_id"]
    assert final["precision_config"] == selected and final["status"] == "pass"
    assert final["runtime_summary"]["config"] == selected
    assert len(final["runtime_summary"]["layers"]) == 50
    assert final["runtime_summary"]["capacity"] == 1048576
    command = read("selected_default/run.command.json")
    assert "--config" not in command["command"]
    assert not command["environment"].get("KOLIBRI_PRECISION_CONFIG")
    assert command["exit_code"] == 0
    assert command["environment"].get("TT_METAL_TRACE_ALLOC_TRACKING") == "0"
    for name, digest in final["provenance"]["source_sha256"].items():
        assert hashlib.sha256((model / name).read_bytes()).hexdigest() == digest
        snapshot = root / "selected_default/source_snapshots" / (digest + ".py.txt")
        assert hashlib.sha256(snapshot.read_bytes()).hexdigest() == digest
    for sample in final["teacher_forcing_samples"]:
        assert sample["counters"]["decode_replays"] >= 99
        assert sample["rows"][0]["total"] == 100
        for key, threshold in report["thresholds"].items():
            assert sample["rows"][0][key] >= threshold
    for row in final["prefill"]:
        for key, threshold in report["thresholds"].items():
            assert row[key] >= threshold
    token_out = final["token_out"]
    assert len(final["token_out_repetitions"]) == token_out["repetitions"] == 5
    assert token_out["decode_t_s_u"] == sorted(token_out["decode_t_s_u_samples"])[2]
    assert token_out["prompt_tokens"] == token_out["generated_tokens"] == 128
    assert token_out["counters"]["decode_replays"] == token_out["counters"]["split_replays"] == 128
    for key in (
        "token_refreshes",
        "position_refreshes",
        "rope_refreshes",
        "page_table_refreshes",
        "token_readbacks",
        "logit_readbacks",
        "synchronizations",
    ):
        assert token_out["counters"].get(key, 0) == 0
    checks = final["capability"]["non_aligned_checks"]
    assert {31, 32, 33, 34, 127, 128, 129, 130, 8191, 8192, 8193, 8194, 8209} <= {r["length"] for r in checks}
    assert all(r["eager_traced_equal"] and r["exact_logits_equal"] for r in checks)
    assert final["long_prefix"]["initialized_full_prefix"]
    assert final["long_prefix"]["last_consumed_position"] == 1048575
    assert final["long_prefix"]["native_next_positions"] == {"kv": [1048576], "rope": [1048576]}
    assert final["batch32"]["layer_count"] == 50 and final["batch32"]["capacity"] == 8192
    assert final["batch32"]["b1_token_match"] and final["batch32"]["equal_slot_logits"]
    assert final["batch32"]["feedback_without_host_refresh"]
    assert final["b1_generator_released"]
    assert final["batch32"]["repeated_all_slot_logits_exact"] and final["batch32"]["all_slot_logits_finite"]
    assert final["batch32_attempt"]["changed_slots"] == []
    hf = read("selected_default/qualitative_hf.json")
    tt = read("selected_default/qualitative_tt.json")
    assert len(hf) == len(tt) == 6
    for control, generated in zip(hf, tt):
        assert control["id"] == generated["id"]
        assert control["prompt_ids"] == generated["prompt_ids"]
        assert control["revision"] == generated["revision"] == "7a8f290e7858825c3cf5e4c447ba68345de9f1d3"
        assert generated["generated_ids"] and generated["completion"]
    qualitative_control = read("qualitative_control.json")
    assert qualitative_control["status"] == "complete"
    assert qualitative_control["prompt_ids"] == tt[0]["prompt_ids"]
    prefix = qualitative_control["same_draft_prefix"]["prefix_generated_ids"]
    assert tt[0]["generated_ids"][: len(prefix)] == prefix
    assert qualitative_control["unforced_256"]["generated_ids"][:128] == hf[0]["generated_ids"]
    context = json.loads((model / "doc/context_contract.json").read_text())
    assert context["supported_context"] == context["datatype_sweep"]["supported_context"] == 1048576
    assert context["datatype_sweep"]["selected_config_id"] == selected["config_id"]
    assert context["datatype_sweep"]["kv_cache_dtype"] == selected["runtime"]["kv_cache_dtype"]
    assert context["datatype_sweep"]["capability_reduction"] is None
    for name in ("top1_perf_pareto.png", "top5_perf_pareto.png", "sweep_results.csv"):
        assert (root / name).stat().st_size > 1000
    handoff = read("perf_summary.json")
    assert handoff["primary_later_comparison"] == "post_selection_token_out"
    assert handoff["config_id"] == selected["config_id"]
    assert handoff["post_selection_token_out"]["decode_t_s_u"] == token_out["decode_t_s_u"]
    construction = read("construction_contract.json")
    assert construction["status"] == "pass" and construction["selected_config_id"] == selected["config_id"]
    assert construction["default_equals_selected"] and construction["missing_default_raises"]
    ranking = read("token_out_ranking.json")
    assert ranking["status"] == "measured"
    assert ranking["fresh_results"][0]["config_id"] == selected["config_id"]
    assert ranking["fresh_results"][0]["median"] == token_out["decode_t_s_u"]
    tracked = read("selected_tracked/result.json")
    assert tracked["status"] == "smoke-pass" and tracked["precision_config"] == selected
    assert tracked["provenance"]["environment"]["TT_METAL_TRACE_ALLOC_TRACKING"] == "1"
    assert tracked["provenance"]["environment"]["TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE"] == "0"
    assert len(tracked["capability"]["non_aligned_checks"]) == 21
    assert tracked["batch32"]["b1_token_match"] and tracked["batch32"]["feedback_without_host_refresh"]
    assert tracked["batch32"]["repeated_all_slot_logits_exact"] and tracked["batch32"]["all_slot_logits_finite"]
    for batch in (1, 2, 31, 32):
        contract = read(f"selected_tracked/trace_contract_b{batch}.json")
        for key in ("tracking", "program_cache_tracking", "exact_feedback", "page_change", "cross_request_reuse"):
            assert contract[key]
    print(
        json.dumps(
            dict(
                status="pass",
                selected=selected["config_id"],
                candidates=len(report["results"]),
                teacher_forcing_t_s_u=fastest["decode_t_s_u"],
                post_selection_token_out_t_s_u=token_out["decode_t_s_u"],
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
