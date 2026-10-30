"""Build a compact candidate ledger from measured results, never estimates."""

import csv
import json
import re
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "doc/optimized_decoder"


def option(command, name, default=None):
    return command[command.index(name) + 1] if name in command else default


def main():
    rows = []
    for manifest in sorted(ROOT.glob("*/manifest.json")):
        directory = manifest.parent
        for record in json.loads(manifest.read_text()):
            if record.get("phase") != "correctness":
                continue
            command = record["command"]
            name = record.get("candidate", record.get("key"))
            config = record.get("configuration", {})
            geometry = config.get("geometry", record.get("geometry", json.loads(option(command, "--geometry", "{}"))))
            correctness = directory / f"{name}_correctness.json"
            timing = directory / f"{name}_timing.json"
            c = json.loads(correctness.read_text()) if correctness.exists() else {}
            t = json.loads(timing.read_text()) if timing.exists() else {}
            log = directory / f"{name}_correctness.log"
            failure = ""
            text = log.read_text(errors="replace") if log.exists() else ""
            if not c:
                matches = [
                    line
                    for line in text.splitlines()
                    if "AssertionError" in line
                    or "RuntimeError:" in line
                    or "TT_FATAL @" in line
                    or "Error Message" in line
                ]
                failure = " | ".join(matches[:2])[:400]

            def measured(key, pattern):
                if key in c:
                    return c[key]
                match = re.search("^" + pattern + r"\s+([0-9.eE+-]+)", text, re.M)
                return float(match.group(1)) if match else None

            note = ""
            if directory.name == "prefill_refined" and name.startswith("full_qk24"):
                note = "No-op Q/K overrides: full attention uses qk_proj. Excluded as geometry evidence; corrected in full_refined."
            if "--inputs" not in command:
                note += " Random activation diagnostic; precision failure cannot veto a real-input win."
            rows.append(
                dict(
                    matrix=directory.name,
                    candidate=name,
                    layer=int(option(command, "--layer", 0)),
                    batch=int(option(command, "--batch", 1)),
                    input_source=option(command, "--inputs", "random activation diagnostic"),
                    precision=option(command, "--precision", "runtime default"),
                    attention_precision=option(command, "--attention-precision", "runtime default"),
                    prefill_pcc=measured("prefill_pcc", "PREFILL_PCC"),
                    decode_pcc=measured("decode_pcc", "EAGER_DECODE_PCC"),
                    changed_pcc=measured("updated_inputs_pcc", "UPDATED_POSITIONS_AND_PAGE_ROWS_PCC"),
                    prefill_host_ms=t.get("prefill_host_ms"),
                    decode_host_us=statistics.median(t["traced_decode_host_us"]) if t else None,
                    status="correctness_pass" if record["exit_code"] == 0 and c else "failed",
                    failure=failure,
                    note=note,
                    configuration=json.dumps(geometry, sort_keys=True),
                    evidence=str(directory.relative_to(ROOT) / name),
                )
            )
    with (ROOT / "candidate_ledger.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (ROOT / "candidate_ledger.json").write_text(
        json.dumps(
            dict(
                scope="Host fixed-position4096 candidate ranking only. Device telemetry uses advancing128-step profiles. CPU reference preparation overlapped part of early tuning; final phase comparisons are remeasured without that load.",
                rows=rows,
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
