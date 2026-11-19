# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Summarize retained successful and failed stage-five whole-layer experiments."""
import ast
import csv
import json
from pathlib import Path
import re
import statistics

D = Path(__file__).resolve().parents[1] / "doc/optimized_multichip_decoder"
rows = []
for sidecar in sorted(D.glob("*_command.json")):
    provenance = json.loads(sidecar.read_text())
    command = provenance.get("command", [])
    if not any("multichip_probe" in str(arg) for arg in command) or "--layer" not in command:
        continue
    label = sidecar.name.removesuffix("_command.json")
    result = D / (label + ".json")
    # Some historical commands used a different output filename.
    if "--output" in command:
        result = Path(command[command.index("--output") + 1])
    row = dict(label=label, layer=command[command.index("--layer") + 1],
               layout=command[command.index("--layout") + 1] if "--layout" in command else "sharded",
               passed=False, prefill_ms=None, decode_ms=None, minimum_pcc=None, failure="",
               command_sidecar=str(sidecar.relative_to(D)))
    if result.exists():
        data = json.loads(result.read_text())
        records = data.get("records", [])
        scores = [value for record in records for key, value in record.items() if "pcc" in key]
        timing = next((record for record in records if record.get("label") == "timing_1024"), {})
        row.update(passed=True, minimum_pcc=min(scores) if scores else None,
                   prefill_ms=timing.get("prefill_ms"), decode_ms=timing.get("decode_ms"))
    else:
        log_path = D / (label + ".log")
        log = log_path.read_text(errors="replace") if log_path.exists() else ""
        timings = re.findall(r"^CANDIDATE_TIMING 1024 (\[.*?\]) (\[.*?\])$", log, re.M)
        if timings:
            pre, dec = map(ast.literal_eval, timings[-1])
            row.update(prefill_ms=statistics.median(pre) if pre else None,
                       decode_ms=statistics.median(dec) if dec else None)
        errors = [line for line in log.splitlines() if line.startswith(("AssertionError", "RuntimeError", "TypeError"))]
        row["failure"] = errors[-1][:400] if errors else "No successful report; inspect retained log."
    rows.append(row)
if rows:
    with (D / "candidate_results.csv").open("w") as output:
        writer = csv.DictWriter(output, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
(D / "candidate_results.json").write_text(json.dumps(rows, indent=2) + "\n")
print("CANDIDATE_TABLE", len(rows), "rows", sum(row["passed"] for row in rows), "passed")
