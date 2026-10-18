# SPDX-License-Identifier: Apache-2.0
"""Check current-source gates and assemble a compact stage evidence manifest."""

import collections
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "doc/optimized_decoder"


def main():
    source_hash = hashlib.sha256((ROOT / "tt/optimized_decoder.py").read_bytes()).hexdigest()
    tags = [
        "final_boundaries_02_final4",
        "default_chunk_final_final2",
        "final_batch32_final2",
        "qsa_positions_final_final2",
        "gdn_long_final_final2",
        "qsa_slot_context_final2",
        "decoder1_long_final_final2",
        "watcher_final_final2",
    ]
    groups = collections.defaultdict(list)
    gates = []
    for tag in tags:
        path = DEST / (tag + "_command.json")
        record = json.loads(path.read_text())
        assert record["returncode"] == 0, tag
        assert record["runtime_sha256"] == source_hash, tag
        assert record["environment"]["TT_METAL_TRACE_ALLOC_TRACKING"] == "1", tag
        assert record["environment"]["TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE"] == "0", tag
        gates.append(dict(tag=tag, manifest=path.name, elapsed_seconds=record["elapsed_seconds"]))
        for line in (DEST / (tag + ".log")).read_text().splitlines():
            if not line.startswith("{"):
                continue
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if "layer_idx" in row and "pcc" in row:
                assert row["pcc"] >= row["threshold"], (tag, row)
                groups[row["layer_idx"], row["batch"], row["mode"]].append(row)
    checks = [
        dict(
            layer=key[0],
            batch=key[1],
            mode=key[2],
            records=len(rows),
            minimum_pcc=min(r["pcc"] for r in rows),
            lengths=sorted({r["length"] for r in rows}),
            max_position=max(r["position"] for r in rows),
            thresholds=sorted({r["threshold"] for r in rows}),
        )
        for key, rows in sorted(groups.items())
    ]
    boundary_rows = {}
    for label, path in (
        ("fused", ROOT / "doc/fused_decoder/final_boundaries_02_native.log"),
        ("optimized", DEST / "final_boundaries_02_final4.log"),
    ):
        rows = []
        for line in path.read_text().splitlines():
            if line.startswith("{"):
                row = json.loads(line)
                if "layer_idx" in row and "pcc" in row:
                    rows.append(row)
        boundary_rows[label] = rows
    case_keys = ("layer_idx", "batch", "mode", "position", "length")
    assert [tuple(r[k] for k in case_keys) for r in boundary_rows["fused"]] == [
        tuple(r[k] for k in case_keys) for r in boundary_rows["optimized"]
    ]
    boundary_comparison = []
    for layer in (0, 1, 3):
        for decode in (False, True):
            row = dict(layer=layer, phase="decode" if decode else "prefill")
            for label, rows in boundary_rows.items():
                selected = [r for r in rows if r["layer_idx"] == layer and ("decode" in r["mode"]) == decode]
                row[label + "_minimum_pcc"] = min(r["pcc"] for r in selected)
                row[label + "_checks"] = len(selected)
            boundary_comparison.append(row)
    gdn = json.loads((DEST / "gdn_long_final_layer0_B1.json").read_text())
    ple = json.loads((DEST / "decoder1_long_context.json").read_text())
    qsa = [json.loads((DEST / f"qsa_context_{n}_B1_slots.json").read_text()) for n in (262143, 262144)]
    assert gdn["complete"] and gdn["largest_decode_context"] == 262144
    assert ple["complete"] and ple["tested_context"] == 262144 and ple["validated_coupled_prefix"] == 262143
    ple_min = min(r["pcc"] for r in ple["records"])
    assert min(gdn["minimum_pcc"], ple_min, *(r[k] for r in qsa for k in ("prefill_pcc", "traced_decode_pcc"))) >= 0.995
    assert all(r["determinism_bitwise"] and r["request_slots"] and r["page_permutation"] for r in qsa)
    watcher_path = DEST / "watcher_final2/generated/watcher/watcher.log"
    watcher_text = watcher_path.read_text()
    errors = [
        line for line in watcher_text.splitlines() if re.search(r"(?i)error|assert.*fail|corrupt|invalid.*noc", line)
    ]
    assert not errors, errors[:10]
    env = json.loads((DEST / "watcher_final_final2_command.json").read_text())["environment"]
    assert env["TT_METAL_WATCHER"] == "10"
    assert not any(k.startswith("TT_METAL_WATCHER_DISABLE") for k in env)
    watcher = dict(
        correctness_log="watcher_final_final2.log",
        command="watcher_final_final2_command.json",
        watcher_log=str(watcher_path.relative_to(DEST)),
        watcher_log_sha256=hashlib.sha256(watcher_path.read_bytes()).hexdigest(),
        watcher_error_matches=errors,
        watcher_features_disabled=None,
        passed_cases=3,
        trace_alloc_tracking=1,
        trace_alloc_skip_program_cache=0,
    )
    (DEST / "watcher_final_summary.json").write_text(json.dumps(watcher, indent=2) + "\n")
    performances = []
    for layer in (0, 1, 3):
        for tag in (
            f"delivered_l{layer}_final4",
            f"steady_optimized_l{layer}_final4",
            f"final4_profile_optimized_l{layer}",
        ):
            timing = json.loads((DEST / tag / "timing.json").read_text())
            assert timing["source_sha256"] == source_hash, tag
            assert "optimized_decoder.Optimized" in timing["runtime_classes"]["attention"], tag
            performances.append(
                dict(
                    tag=tag,
                    prefill_ms=timing["median_prefill_host_ms"],
                    decode_ms=timing["median_traced_decode_ms"],
                    profiler=timing["profiler"],
                )
            )
    report = dict(
        status="ready_for_independent_review",
        runtime_sha256=source_hash,
        baseline_sha256=hashlib.sha256((ROOT / "tt/fused_decoder.py").read_bytes()).hexdigest(),
        gates=gates,
        correctness=checks,
        boundary_comparison=boundary_comparison,
        performance=performances,
        context=dict(
            supported=262144,
            full_context_batch=1,
            batch32_tested_context=130,
            capability_reduction=None,
            gdn_minimum_pcc=gdn["minimum_pcc"],
            ple_minimum_pcc=ple_min,
            ple_complete_checks=len(ple["records"]),
            qsa_prefill_pcc=min(r["prefill_pcc"] for r in qsa),
            qsa_decode_pcc=min(r["traced_decode_pcc"] for r in qsa),
        ),
        watcher="watcher_final_summary.json",
        selection="final_selection.json",
        profiler_accounting="final_performance_accounting.json",
    )
    (DEST / "final_evidence.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            dict(source=source_hash, gate_count=len(gates), groups=len(checks), context=report["context"]), indent=2
        )
    )


if __name__ == "__main__":
    main()
