# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Check the fused-stage contract against raw correctness/trace/watcher artifacts."""

import gzip
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "doc/fused_decoder"
ACCEPTANCE = 0.995


def main():
    sha = hashlib.sha256((ROOT / "tt/fused_decoder.py").read_bytes()).hexdigest()
    result = {"decoder_sha256": sha, "acceptance": ACCEPTANCE, "layers": {}}
    geometry = json.loads((DOC / "batch_geometry.json").read_text())
    assert geometry["status"] == "pass" and geometry["decoder_sha256"] == sha
    assert geometry["evidence_type"] == "primitive_isolation_not_hf_accuracy"
    assert {(r["batch"], r["kind"]) for r in geometry["cases"]} == {
        (b, k) for b in range(1, 33) for k in ("rope", "nope")
    }
    assert all(len(r["passes"]) == 2 and all(p["status"] == "pass" for p in r["passes"]) for r in geometry["cases"])
    result["primitive_geometry"] = {"file": "batch_geometry.json", "cases": 64, "passes_per_case": 2}
    for layer, kind in [(0, "sliding_attention"), (3, "full_attention")]:
        accepted = {}
        scores = []
        for name in ["boundaries", "continuation", "control", "long_context"] + [
            f"partial_batch_{layer}_{b}" for b in (9, 17, 25, 31)
        ]:
            path = DOC / (f"{name}.json" if name.startswith("partial_batch_") else f"{name}_{layer}.json")
            rows = json.loads(path.read_text())
            provenance = json.loads(path.with_suffix(".provenance.json").read_text())
            if name == "long_context" and provenance["source_sha256"]["tt/fused_decoder.py"] != sha:
                archive_sha = hashlib.sha256(
                    gzip.decompress((DOC / "pre_join_decoder_source.py.gz").read_bytes())
                ).hexdigest()
                assert provenance["source_sha256"]["tt/fused_decoder.py"] == archive_sha
                join = json.loads((DOC / f"long_join_{layer}.json").read_text())
                assert join["source_sha256"]["tt/fused_decoder.py"] == sha
                assert join["control_source_sha256"] == archive_sha
                assert join["logical_join_control_ast_matches_archive"]
                assert len(join["unchanged_prefill_math_methods_ast"]) == 7
                assert [r["length"] for r in join["rows"]] == [131071, 131072]
                assert all(
                    r["output_bitwise_equal"] and r["kv_bitwise_equal"] and r["decode_bitwise_equal"]
                    for r in join["rows"]
                )
            else:
                assert provenance["source_sha256"]["tt/fused_decoder.py"] == sha, path
            assert provenance["trace_alloc_tracking"] == "1"
            assert provenance["trace_alloc_skip_program_cache"] == "0"
            assert "ALL_PASS" in path.with_suffix(".log").read_text()
            for row in rows:
                assert row["real_weights"] and row["kind"] == kind
                for field in ["prefill_pcc", "decode_pcc"]:
                    assert row[field] >= ACCEPTANCE, (path, field, row[field])
                    scores.append(row[field])
                assert row["repeated_replay_equal"]
                assert row["prefill_repeated_equal"] if row["length"] <= 4096 else row["prefill_repeated_equal"] is None
                assert row["random_positions"]["pcc"] >= ACCEPTANCE
                assert len(row["refreshed_replays"]) == 2
                assert all(
                    r["pcc"] >= ACCEPTANCE and r["trace_id"] == row["trace_id"] for r in row["refreshed_replays"]
                )
                if name == "continuation":
                    assert (
                        row["batch"] == 2
                        and row["continuation"]["start"] == 33
                        and row["continuation"]["length"] == 1057
                    )
                    assert row["continuation"]["pcc"] >= ACCEPTANCE
                if name == "control":
                    assert row["unchunked_control_pcc"] >= ACCEPTANCE
                if name.startswith("partial_batch_"):
                    assert row["batch"] == int(name.rsplit("_", 1)[1])
                    assert min(row["prefill_user_pcc"] + row["decode_user_pcc"]) >= ACCEPTANCE
                    assert min(row["random_positions"]["per_user_pcc"]) >= ACCEPTANCE
                    assert all(min(r["per_user_pcc"]) >= ACCEPTANCE for r in row["refreshed_replays"])
            if name == "boundaries":
                assert [r["length"] for r in rows] == [1, 31, 32, 33, 1023, 1024, 1025, 2047, 2048, 2049, 3073, 17]
            if name == "long_context":
                assert [r["length"] for r in rows] == [4097, 131071, 131072]
                assert rows[-1]["decode_position"] == 131071
            accepted[name] = {
                "file": path.name,
                "cases": len(rows),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "source_evidence": (
                    "direct_current_source_hf"
                    if provenance["source_sha256"]["tt/fused_decoder.py"] == sha
                    else "archived_hf_with_final_exact_bridge"
                ),
            }
        reuse_path = DOC / f"trace_reuse_{layer}.json"
        reuse = json.loads(reuse_path.read_text())
        assert reuse["decoder_sha256"] == sha
        assert reuse["tracking_enabled"] and reuse["program_cache_included"]
        assert reuse["capture_count"] == 1 and reuse["request_release_count"] == reuse["request_recapture_count"] == 0
        assert len(reuse["rows"]) == 13 and len({r["trace_id"] for r in reuse["rows"]}) == 1
        assert min(min(r["prefill_pcc"], r["decode_pcc"]) for r in reuse["rows"]) >= ACCEPTANCE
        for batch in (9, 32):
            watcher = DOC / f"watcher_{layer}_b{batch}"
            console = (watcher / "console.log").read_text()
            log = (watcher / "watcher.log").read_text()
            assert "ALL_PASS" in console
            assert "Dump" in log or "dump" in log
            assert not re.search(r"TT_FATAL|AssertionError|invalid NoC|corrupt|out.of.bounds|stack overflow", log, re.I)
            wp = json.loads((watcher / "results.provenance.json").read_text())
            assert wp["source_sha256"]["tt/fused_decoder.py"] == sha
            assert wp["watcher"] and wp["device_profiler"] is None
            assert wp["trace_alloc_tracking"] == "1" and wp["trace_alloc_skip_program_cache"] == "0"
            wr = json.loads((watcher / "results.json").read_text())[0]
            assert wr["batch"] == batch and wr["random_positions"]["pcc"] >= ACCEPTANCE
            assert (
                min(wr["prefill_user_pcc"] + wr["decode_user_pcc"] + wr["random_positions"]["per_user_pcc"])
                >= ACCEPTANCE
            )
            accepted[f"watcher_b{batch}"] = {
                "directory": watcher.name,
                "sha256": hashlib.sha256((watcher / "watcher.log").read_bytes()).hexdigest(),
                "random_positions_pcc": wr["random_positions"]["pcc"],
            }
        accepted["trace_reuse"] = {"file": reuse_path.name, "requests": len(reuse["rows"])}
        result["layers"][kind] = {"minimum_prefill_decode_pcc": min(scores), "accepted": accepted}
    (DOC / "correctness_summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
