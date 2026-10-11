# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-only closure checks for the stage8 evidence contract."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "doc/datatype_sweep"


def read(path):
    return json.loads(path.read_text())


selected = read(DOC / "selected_precision_config.json")
results = read(DOC / "sweep_results.json")
rows = results["configs"]
passing = [r for r in rows if r["selection_eligible"]]
winner = max(passing, key=lambda r: r["teacher_forcing_decode_tps"])
assert selected == winner["dtype_policy"]
assert winner["capability_status"] == "pass"
assert winner["context_status"] == "pass" and winner["batch_status"] == "pass"
assert results["selected_config_id"] == winner["config_id"]
for row in rows:
    assert row["trace_verified"] and row["token_count"] == 100
    assert row["mesh"] == [1, 4] and row["command"] and row["runtime_policy_evidence"]
    assert row["environment"]["TT_METAL_TRACE_ALLOC_TRACKING"] in ("unset", "0")
    raw = read(ROOT / row["evidence_path"])
    for trial in raw["teacher_trials"]:
        assert trial["trace_verified"] and trial["counters"]["replay_decode"] == 99
        assert trial["counters"].get("captures", 0) == 0
    assert len(raw["runtime_summary"]["layers"]) == 52
    gate = all(
        stats[key] >= threshold for stats in [row, *row["prefill"]] for key, threshold in results["thresholds"].items()
    )
    assert row["accuracy_pass"] == gate
    assert row["selection_eligible"] == (gate and row["capability_status"] != "fail")
for kind in ("sliding_attention", "full_attention"):
    for group in ("qkv", "self_attn.o_proj", "mlp.gate_up", "mlp.down_proj"):
        policies = [row["dtype_policy"]["decoder_kinds"][kind]["weight_groups"][group] for row in rows]
        assert {(dtype, fidelity) for dtype in ("bfloat4_b", "bfloat8_b") for fidelity in ("LoFi", "HiFi2")} <= {
            (p["dtype"], p["fidelity"]) for p in policies
        }
assert {(dtype, fidelity) for dtype in ("bfloat4_b", "bfloat8_b") for fidelity in ("LoFi", "HiFi2")} <= {
    (row["dtype_policy"]["terminal"]["weight_dtype"], row["dtype_policy"]["terminal"]["fidelity"]) for row in rows
}
for name in ("top1_perf_pareto.png", "top5_perf_pareto.png"):
    assert (DOC / name).read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
assert (DOC / "sweep_results.csv").stat().st_size > 0
post = read(DOC / "post_selection.json")
assert post["precision_config"] == selected and post["accuracy_gate_pass"]
assert post["precision_config_environment_override"] is None
assert post["runtime_summary"]["context"] == 131072
assert {r["logical_length"] for r in post["prefill_lengths"]} == {33, 127, 128, 129, 4095, 4096, 4097}
for path, digest in post["implementation_sha256"].items():
    assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest, path
assert len(post["qualitative"]) == 6
sample = [r for r in post["token_out_no_readback"] if r["mode"] == "sample"]
assert len(sample) == 4 and all(r["steps"] == 127 and r["equals_output_returning_path"] for r in sample)
for row in sample:
    assert row["counters"]["replay_decode"] == 127
    assert row["counters"]["replay_sample"] == 127
    for key in (
        "captures",
        "token_refreshes",
        "token_readbacks",
        "async_token_readbacks",
        "position_refreshes",
        "rope_refreshes",
        "page_table_refreshes",
        "synchronizations",
        "full_logits_readbacks",
    ):
        assert row["counters"].get(key, 0) == 0, (key, row)
tracked = read(DOC / "selected_tracking.json")
assert tracked["precision_config"] == selected and tracked["status"] == "pass"
assert tracked["environment"]["TT_METAL_TRACE_ALLOC_TRACKING"] == "1"
assert tracked["environment"]["TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE"] == "0"
contract = read(ROOT / "doc/context_contract.json")["datatype_sweep_stage_context_preservation"]
assert contract["config_id"] == selected["config_id"] and contract["kv_cache_dtype"] == selected["kv_cache_dtype"]
assert contract["supported_context"] == 131072 and contract["largest_feasible_per_request_context"] == 131072
assert contract["capability_reduction"] is None
context = read(DOC / "selected_context.json")
assert context["layers"] == 52 and context["context"] == 131072
assert context["nonaligned_prefill"] == 131071 and context["decode_position"] == 131071
assert context["prefill_decode_cosine"] > 0.99
assert context["precision_config"] == selected and context["precision_config_environment_override"] is None
assert context["finite_logits"] and context["decode_token_in_prefill_top5"]
assert context["counters"]["captures"] == 25
for path, digest in context["implementation_sha256"].items():
    assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest, path
batch = read(DOC / "selected_batch32.json")
assert batch["layers"] == 52 and batch["batch"] == 32 and batch["captures"] == 25
assert batch["inactive_cache_preserved"] and batch["reordered_slots_preserved"]
assert batch["precision_config"] == selected and batch["precision_config_environment_override"] is None
for path, digest in batch["implementation_sha256"].items():
    assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest, path
probe = read(DOC / "selected_probe.json")
assert probe["runtime_fallback_audit"] and probe["external_cache"]
assert probe["precision_config"] == selected and probe["precision_config_environment_override"] is None
for path, digest in probe["implementation_sha256"].items():
    assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest, path
assert probe["eager_control_provenance"]["policy"] == selected
for stats in probe["accuracy"].values():
    assert stats["trace_eager_pcc"] > 0.999 and stats["pcc"] >= stats["eager_hf_pcc"] - 0.001
assert read(DOC / "norm_guard_host_check.json")["status"] == "pass"
eager_commands = read(DOC / "prefill_guard_commands.json")
assert len(eager_commands) == 4 and all(r["returncode"] == 0 for r in eager_commands)
for row in eager_commands:
    eager = read(Path(row["log"]).with_suffix(".json"))
    assert {r["length"] for r in eager} == {321, 353, 769, 961, 993, 1025}
    assert all(r["prefill_pcc"] >= 0.995 and r["continuation"]["pcc"] >= 0.995 for r in eager)
tail_commands = read(DOC / "prefill_guard_commands_tails.json")
assert len(tail_commands) == 4 and all(r["returncode"] == 0 for r in tail_commands)
for row in tail_commands:
    eager = read(Path(row["log"]).with_suffix(".json"))
    assert {r["length"] for r in eager} == {513, 897}
    assert all(r["prefill_pcc"] >= 0.995 and r["continuation"]["pcc"] >= 0.995 for r in eager)
for name in (
    "selected_probe.json",
    "selected_context.json",
    "selected_batch32.json",
    "qualitative_review.md",
    "README.md",
    "work_log.md",
    "POLICY_PROPAGATION.md",
):
    assert (DOC / name).stat().st_size > 0, name
report = dict(
    status="pass",
    selected_config_id=selected["config_id"],
    evaluated_configs=len(rows),
    fastest_passing_teacher_forcing_decode_tps=winner["teacher_forcing_decode_tps"],
    scope="artifact consistency, traced ranking, normal-default precision, post-selection no-readback counters, context and required evidence; independent review is separate",
)
(DOC / "stage08_checks.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
