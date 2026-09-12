# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-only source-linked validation summary; fail if a required artifact is absent."""

import gzip
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "doc/multichip_decoder"
source_hash = hashlib.sha256((ROOT / "tt/multichip_decoder.py").read_bytes()).hexdigest()
regressions = sorted([*DOC.glob("final_l*_c*.json"), *DOC.glob("final_stack_b*.json")])
regressions = [p for p in regressions if not p.stem.endswith("_baseline")]
assert len(regressions) == 31, len(regressions)
summary = {"source_sha256": source_hash, "regressions": [], "watcher": [], "capacity": [], "stress": []}
for category, paths in [
    ("regressions", regressions),
    ("stress", [DOC / "stress_stack_batch32.json"]),
    (
        "watcher",
        [
            DOC / f"{name}.json"
            for name in [
                "watcher_linear_tail",
                "watcher_linear_batch32",
                "watcher_full_batch32",
                "watcher_stack_batch3",
            ]
        ],
    ),
    (
        "capacity",
        [
            DOC / f"{name}.json"
            for name in [
                "capacity_l0_s262143",
                "capacity_l0_s262144",
                "capacity_l3_s262143",
                "capacity_l3_s262144",
                "capacity_stack_s262143",
            ]
        ],
    ),
]:
    for path in paths:
        data = json.loads(path.read_text())
        assert data["source_sha256"] == source_hash, path
        assert min(data["pcc"].values()) >= 0.995, path
        if data["length"] != 262144:
            assert (
                data["trace_bitwise_equal"]
                and data["changed_input_replay_bitwise_equal"]
                and data["post_decode_state_bitwise_equal"]
            ), path
        row = {
            "artifact": path.name,
            "layer": data["layer"],
            "batch": data["batch"],
            "length": data["length"],
            "minimum_pcc": min(data["pcc"].values()),
            "stack": data.get("stack", False),
        }
        if category == "watcher":
            log = DOC / path.stem / "generated/watcher/watcher.log"
            if not log.exists():
                log = log.with_suffix(".log.gz")
                text = gzip.open(log, "rt").read()
            else:
                text = log.read_text()
            console = path.with_suffix(".log").read_text()
            assert "disabled features: ETH" in console
            assert all(f"Watcher checking device {rank}" in console for rank in range(4))
            assert "TT_FATAL" not in console and "Watcher detected" not in console
            assert log.stat().st_size > 0
            row.update(
                watcher_log=str(log.relative_to(DOC)),
                disabled_features=["ETH"],
                limitation="ACTIVE_ETH instrumented fabric exceeds native config buffer; worker/NoC watcher enabled.",
            )
        if category == "stress":
            assert data["queued_stress_bitwise_equal"] and data["queued_stress_iterations"] >= 100
            assert data["stack_shared_collective_workspace"]
            row["queued_stress_iterations"] = data["queued_stress_iterations"]
        summary[category].append(row)
summary["status"] = "passed"
summary["minimum_pcc"] = min(
    row["minimum_pcc"] for category in ("regressions", "watcher", "capacity", "stress") for row in summary[category]
)
(DOC / "validation_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(summary["status"], "minimum PCC", summary["minimum_pcc"])
