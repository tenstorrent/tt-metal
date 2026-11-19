# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Assess focused tail-two watcher evidence without changing raw probe results."""
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "doc/optimized_decoder"


def read(name):
    return json.loads((DOC / (name + ".json")).read_text())


def ple_contract(result):
    assert result["seq_len"] == 34 and result["decode_steps"] == 3
    for metrics in [result["prefill"], *result["decode"]]:
        assert metrics["finite"]
        for kind in ("whole", "contribution"):
            assert metrics[kind]["global_pcc"] >= 0.995
            assert metrics[kind]["minimum_row_pcc"] >= 0.99
            assert metrics[kind]["nonfinite_row_count"] == 0
    assert all(row["passed"] for row in result["decode"])
    assert result["captures"] == 1 and result["replays"] == 6
    assert result["prefill_reset_equal"] and result["reset_replay_equal"]
    assert all(result["reset_state_equal"].values()) and all(result["runtime_audit"].values())


def main():
    rows = []
    for layer in (0, 1, 3):
        name = f"review_watcher_tail2_layer{layer}"
        result, provenance = read(name), read(name + "_provenance")
        env = provenance["environment"]
        assert env["TT_METAL_WATCHER"] == "10"
        assert env["TT_METAL_TRACE_ALLOC_TRACKING"] == "1"
        assert env["TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE"] == "0"
        assert env["TT_METAL_DEVICE_PROFILER"] is None
        assert env["TT_METAL_STREAMING_PROFILER"] is None
        assert provenance["status"] == "finished"
        assert (
            provenance["source_sha256"]["tt/optimized_decoder.py"]
            == hashlib.sha256((ROOT / "tt/optimized_decoder.py").read_bytes()).hexdigest()
        )
        watcher = DOC / "watcher" / name / "generated/watcher/watcher.log"
        stream = watcher.read_text()
        assert "detach device 0" in stream and "Dump #" in stream
        assert not re.search(r"assertion|watcher detected|ERROR|FATAL|exception", stream, re.IGNORECASE)
        log = Path(provenance["log"])
        assert hashlib.sha256(log.read_bytes()).hexdigest() == provenance["log_sha256"]
        if layer == 1:
            ple_contract(result)
            assert provenance["exit_code"] == 1 and result["passed"] is False
        else:
            assert result["factory"] == "OptimizedDecoder" and result["passed"]
            assert provenance["exit_code"] == 0
            assert [row["sequence"] for row in result["rows"]] == [34]
        rows.append(
            dict(
                layer=layer,
                artifact=name + ".json",
                raw_passed=result["passed"],
                raw_exit_code=provenance["exit_code"],
                watcher_log=str(watcher.relative_to(ROOT)),
                watcher_sha256=hashlib.sha256(watcher.read_bytes()).hexdigest(),
                watcher_clean=True,
                user_contract_passed=True,
            )
        )
    optimized = read("review_watcher_tail2_layer1")
    fused = read("review_tail2_fused_ple_control")
    ple_contract(fused)
    for key in ("snapshot", "cache_dir", "seq_len", "decode_steps", "expert_dtype"):
        assert optimized[key] == fused[key]
    for kind in ("whole", "contribution"):
        for result in (optimized, fused):
            assert result["prefill"][kind]["failed_row_count"] == 1
            assert result["prefill"][kind]["worst_rows"][0]["row"] == 5
        assert optimized["prefill"][kind]["minimum_row_pcc"] > fused["prefill"][kind]["minimum_row_pcc"]
    report = dict(
        passed=True,
        scope="Final default public prefill34 (32+2), changing traced decode, all three layer kinds",
        rows=rows,
        ple_control="review_tail2_fused_ple_control.json",
        ple_prefill={"optimized": optimized["prefill"], "fused": fused["prefill"]},
        interpretation="Both raw PLE probes retain false for row5; whole-output>=.995, diagnostic floor>=.99 and strict decode pass. The predecessor reproduces the same row diagnostic. This is not a claim of literal-HF failure. Row5 precedes the new physical two-row tail.",
        token_top1="Not applicable to isolated decoder hidden streams",
    )
    (DOC / "review_watcher_tail2_summary.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
