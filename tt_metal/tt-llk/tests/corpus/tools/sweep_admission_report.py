#!/usr/bin/env python3
"""Join a completed LLK sweep with a source-bound admission exception ledger.

This is a claim filter, not a validator or a replacement for raw sweep results.
Missing ledger rows are unreviewed, not admitted. Partial numerical coverage
and causal-only comparisons never establish a full-domain semantic replacement.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import re
from collections import Counter
from pathlib import Path

DECISIONS = {
    "CAUSAL_OFF_ON_ONLY",
    "BOUNDED_PARTIAL_ADMITTED",
    "CONDITIONAL_UNFULFILLED",
    "DOMAIN_SCOPED_UNFULFILLED",
    "REFUSED_STALE_SEMANTICS",
    "REFUSED_FULL_VALIDATION_FAILURE",
    "REFUSED_ACCURACY_WIN",
}
LEDGER_FIELDS = {
    "row",
    "raw_disposition",
    "comparison_admission",
    "replacement_scope",
    "validation_status",
    "actual_counts",
    "unfulfilled_requirements",
    "reason",
    "evidence_ref",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(data: bytes) -> dict:
    value = json.loads(data)
    if not isinstance(value, dict):
        raise ValueError("expected a JSON object")
    return value


def read_ledger(data: bytes) -> tuple[dict, dict]:
    metadata, lines = {}, []
    for line in data.decode().splitlines():
        if line.startswith("# "):
            key, separator, value = line[2:].partition("\t")
            if not separator or not value or key in metadata:
                raise ValueError("invalid or duplicate ledger metadata")
            metadata[key] = value
        elif line.strip():
            lines.append(line)
    reader = csv.DictReader(io.StringIO("\n".join(lines)), delimiter="\t")
    if set(reader.fieldnames or []) != LEDGER_FIELDS or len(reader.fieldnames) != len(
        LEDGER_FIELDS
    ):
        raise ValueError("ledger columns do not match schema")
    rows = {}
    for row in reader:
        if None in row or any(not value for value in row.values()):
            raise ValueError("empty or malformed ledger field")
        if row["row"] in rows:
            raise ValueError("duplicate ledger operation")
        if (
            row["raw_disposition"] != "RETAIN"
            or row["comparison_admission"] not in DECISIONS
        ):
            raise ValueError("unsupported ledger disposition or decision")
        if row["comparison_admission"] == "BOUNDED_PARTIAL_ADMITTED":
            passing = row["validation_status"] in {
                "ACTUAL_PARTIAL_PASS",
                "ACTUAL_NINE_STRATA_PARTIAL_PASS",
            }
            counts = re.fullmatch(
                r"OFF:([1-9][0-9]*)_PASS/0_FAIL;ON:([1-9][0-9]*)_PASS/0_FAIL;"
                r"sem_n_out=0;hand_n_out=0;sem_max_bf16_ulp=([0-9]+);hand_max_bf16_ulp=([0-9]+)",
                row["actual_counts"],
            )
            if not passing or not counts or int(counts[3]) > int(counts[4]):
                raise ValueError(
                    "bounded admission requires coherent passing numerical counts"
                )
            if row[
                "validation_status"
            ] == "ACTUAL_NINE_STRATA_PARTIAL_PASS" and counts.group(1, 2) != ("9", "9"):
                raise ValueError(
                    "nine-strata admission requires nine passing rows per profile"
                )
        rows[row["row"]] = row
    for key, width in (
        ("campaign_source", 40),
        ("compiler_driver_sha256", 64),
        ("compiler_cc1plus_sha256", 64),
        ("campaign_manifest_sha256", 64),
        ("terminal_results_sha256", 64),
        ("terminal_summary_sha256", 64),
    ):
        if not re.fullmatch(rf"[0-9a-f]{{{width}}}", metadata.get(key, "")):
            raise ValueError(f"invalid ledger pin: {key}")
    if metadata.get("campaign_fully_admitted") != "FALSE":
        raise ValueError("exception ledger cannot assert full campaign admission")
    return metadata, rows


def positive(row: dict, field: str) -> float:
    try:
        value = float(row[field])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"{row.get('op')}: invalid {field}") from error
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{row.get('op')}: invalid {field}")
    return value


def load_campaign(
    artifacts: dict[str, bytes], metadata: dict
) -> tuple[dict, dict, dict]:
    if (
        hashlib.sha256(artifacts["manifest.json"]).hexdigest()
        != metadata["campaign_manifest_sha256"]
    ):
        raise ValueError("campaign manifest does not match ledger")
    for name, key in (
        ("results.tsv", "terminal_results_sha256"),
        ("summary.json", "terminal_summary_sha256"),
    ):
        if hashlib.sha256(artifacts[name]).hexdigest() != metadata[key]:
            raise ValueError(f"terminal {name} does not match ledger")
    manifest = read_json(artifacts["manifest.json"])
    summary = read_json(artifacts["summary.json"])
    head = manifest.get("tt_metal_head", {})
    if (
        head.get("returncode") != 0
        or head.get("stdout", "").strip() != metadata["campaign_source"]
    ):
        raise ValueError("campaign source does not match ledger")
    for key in ("tt_metal_status", "tt_metal_diff"):
        capture = manifest.get(key, {})
        if capture.get("returncode") != 0 or capture.get("stdout") != "":
            raise ValueError("source pin requires clean recorded tracked source")
    for name in ("driver", "cc1plus"):
        if (
            manifest.get("compiler", {}).get(name, {}).get("sha256")
            != metadata[f"compiler_{name}_sha256"]
        ):
            raise ValueError(f"campaign {name} does not match ledger")
    with io.StringIO(artifacts["results.tsv"].decode()) as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        fields = {"op", "kind", "status", "sem_off", "sem_on", "hand_off", "hand_on"}
        if not fields.issubset(reader.fieldnames or []):
            raise ValueError("missing raw result columns")
        if len(set(reader.fieldnames)) != len(reader.fieldnames):
            raise ValueError("duplicate raw result columns")
        records = list(reader)
    expected = manifest.get("rows")
    count = summary.get("selected_rows")
    if (
        type(count) is not int
        or count <= 0
        or type(summary.get("completed_rows")) is not int
        or summary.get("completed_rows") != count
        or len(records) != count
        or not isinstance(expected, list)
        or len(expected) != count
    ):
        raise ValueError("campaign is incomplete")
    rows = {}
    for row in records:
        if None in row or not row.get("op") or row["op"] in rows:
            raise ValueError("invalid or duplicate raw operation")
        if row["status"] not in {"PASS", "FAIL", "SKIP"}:
            raise ValueError("unsupported raw status")
        if row["status"] == "PASS":
            if row["kind"] not in {"semantic", "full2x2"}:
                raise ValueError("invalid passing row kind")
            for field in ("sem_off", "sem_on") + (
                ("hand_off", "hand_on") if row["kind"] == "full2x2" else ()
            ):
                positive(row, field)
        rows[row["op"]] = row
    expected_pairs = [
        (row.get("op"), row.get("kind")) for row in expected if isinstance(row, dict)
    ]
    if sorted(expected_pairs) != sorted((row["op"], row["kind"]) for row in records):
        raise ValueError("raw population differs from manifest")
    counts = dict(Counter(row["status"] for row in records))
    if summary.get("counts") != counts or summary.get("status") != (
        "FAIL" if counts.get("FAIL") else "PASS"
    ):
        raise ValueError("campaign is incomplete or has incoherent terminal counts")
    return manifest, summary, rows


def build_report(root: Path, ledger: Path, band_pct: float = 0.5) -> dict:
    if not math.isfinite(band_pct) or band_pct <= 0:
        raise ValueError("band must be finite and positive")
    ledger_bytes = ledger.read_bytes()
    artifacts = {
        name: (root / name).read_bytes()
        for name in ("manifest.json", "summary.json", "results.tsv")
    }
    metadata, decisions = read_ledger(ledger_bytes)
    manifest, summary, rows = load_campaign(artifacts, metadata)
    if set(decisions) - set(rows):
        raise ValueError("ledger includes operations absent from campaign")
    comparisons = {"raw_causal_on_vs_off": {}, "bounded_partial_on_vs_hand_on": {}}
    joined = {}
    for op, raw in sorted(rows.items()):
        decision = decisions.get(op)
        effective = (
            "UNREVIEWED" if decision is None else decision["comparison_admission"]
        )
        if effective == "CAUSAL_OFF_ON_ONLY":
            effective = "RAW_CAUSAL_ONLY"
        if raw["status"] != "PASS":
            effective = "RAW_NOT_PASS"
        joined[op] = {
            "raw_status": raw["status"],
            "raw_kind": raw["kind"],
            "ledger_decision": decision,
            "effective_admission": effective,
        }
        if raw["status"] != "PASS" or decision is None:
            continue
        status = decision["comparison_admission"]
        selected = []
        if status in {"CAUSAL_OFF_ON_ONLY", "BOUNDED_PARTIAL_ADMITTED"}:
            selected.append(("raw_causal_on_vs_off", "sem_off"))
        if status == "BOUNDED_PARTIAL_ADMITTED":
            if raw["kind"] != "full2x2":
                raise ValueError(
                    f"{op}: bounded paired admission needs four measured arms"
                )
            selected.append(("bounded_partial_on_vs_hand_on", "hand_on"))
        for name, denominator in selected:
            delta = 100 * (positive(raw, "sem_on") / positive(raw, denominator) - 1)
            comparisons[name][op] = {
                "delta_pct": delta,
                "class": (
                    "faster"
                    if delta <= -band_pct
                    else "slower" if delta >= band_pct else "parity"
                ),
                "scope": decision["replacement_scope"],
                "limitations": decision["unfulfilled_requirements"],
            }
    return {
        "schema": 1,
        "campaign_fully_admitted": False,
        "note": "Claim filter only: causal timings remain raw evidence, not numerical admission. No full-domain replacement or full-win conclusion. Raw evidence is unchanged.",
        "campaign_source": metadata["campaign_source"],
        "raw_counts": summary["counts"],
        "completion": (
            "COMPLETE" if summary["counts"].get("PASS") else "COMPLETE_NO_PASS"
        ),
        "band_pct": band_pct,
        "verification": {
            key: summary.get(key, "NOT_RUN")
            for key in ("formal", "exhaustive", "ulp_admission")
        },
        "context": {
            key: manifest.get(key)
            for key in ("compiler", "flags", "seed", "repeats", "device_index")
        },
        "artifacts_sha256": {
            **{
                name: hashlib.sha256(data).hexdigest()
                for name, data in artifacts.items()
            },
            "admission_ledger": hashlib.sha256(ledger_bytes).hexdigest(),
        },
        "ledger_metadata": metadata,
        "rows": joined,
        "comparisons": comparisons,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--current", required=True, type=Path)
    parser.add_argument("--ledger", required=True, type=Path)
    parser.add_argument("--band-pct", type=float, default=0.5)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.out and (
            args.out.resolve().is_relative_to(args.current.resolve())
            or args.out.resolve() == args.ledger.resolve()
        ):
            raise ValueError("output must be separate from raw campaign and ledger")
        report = build_report(args.current, args.ledger, args.band_pct)
        text = json.dumps(report, indent=2, sort_keys=True) + "\n"
        if args.out:
            with args.out.open("x") as stream:
                stream.write(text)
        else:
            print(text, end="")
    except (OSError, ValueError, TypeError, KeyError) as error:
        parser.exit(2, f"admission report refused: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
