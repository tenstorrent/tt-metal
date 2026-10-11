# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Check completed runner evidence and summarize the functional stage."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "doc/functional_decoder"


def main():
    report = {"acceptance_pcc": 0.995, "layers": {}}
    for layer, kind in [(0, "sliding_attention"), (3, "full_attention")]:
        evidence = {}
        all_scores = []
        for stem in ("final_boundaries", "continuation_batch2", "batch32", "control", "long_context"):
            path = DOC / f"{stem}_{layer}.json"
            rows = json.loads(path.read_text())
            assert rows and all(x["real_weights"] for x in rows)
            scores = []
            for row in rows:
                scores.extend([row["prefill_pcc"], row["decode_pcc"]])
                scores.extend(x["pcc"] for x in row.get("refreshed_replays", []))
                if "continuation" in row:
                    scores.append(row["continuation"]["pcc"])
                if "random_positions" in row:
                    scores.append(row["random_positions"]["pcc"])
                assert row["repeated_replay_equal"]
            assert all(0.995 <= s <= 1.000000000001 for s in scores), (path, scores)
            all_scores.extend(scores)
            evidence[stem] = {"path": path.name, "rows": len(rows), "min_pcc": min(scores)}
            if stem == "long_context":
                assert {131071, 131072}.issubset(x["length"] for x in rows)
                assert any(x["decode_position"] == 131071 for x in rows)
            if stem == "batch32":
                assert all(x["batch"] == 32 for x in rows)
        reuse = json.loads((DOC / f"trace_reuse_{layer}.json").read_text())
        assert (
            reuse["capture_count"] == 1
            and reuse["request_release_count"] == 0
            and reuse["request_recapture_count"] == 0
        )
        assert reuse["tracking_enabled"] and reuse["program_cache_included"]
        assert len({x["trace_id"] for x in reuse["rows"]}) == 1
        assert any(x["prefix_length"] % 32 for x in reuse["rows"])
        assert any(x["prefix_length"] > 0 and x["prefix_length"] % 32 == 0 for x in reuse["rows"])
        assert {tuple(x["slots"]) for x in reuse["rows"]} == {(0, 1), (1, 0)}
        for row in reuse["rows"]:
            assert 0.995 <= row["prefill_pcc"] <= 1.000000000001
            assert 0.995 <= row["decode_pcc"] <= 1.000000000001
        report["layers"][kind] = {
            "layer_index": layer,
            "evidence": evidence,
            "min_pcc": min(all_scores),
            "trace_reuse_requests": len(reuse["rows"]),
        }
    audit = json.loads((DOC / "runtime_audit.json").read_text())
    assert audit["status"] == "clean" and not audit["violations"]
    report["runtime_audit"] = "clean"
    (DOC / "correctness_summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
