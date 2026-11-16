"""Aggregate coordinator judgments without inferring semantics from token patterns."""
import collections
import datetime
import hashlib
import json
from pathlib import Path

D = Path("models/autoports/google_gemma_4_31b_it/doc/agentic_evals")
O = D / "setup/swe85-remainder-execution85"


def main():
    sources = sorted(O.glob("full-semantic-coordinator-?.json"))
    manifest = json.loads((D / "setup/swe85-admitted-controls85/combined-cases.json").read_text())
    expected = {(j["name"], seed) for j in manifest if j["name"].startswith("anomaly.") for seed in (42, 43, 44)}
    rows, seen = [], set()
    for source in sources:
        report = json.loads(source.read_text())
        for row in report["rows"]:
            key = (row["job"], row["seed"])
            if key not in expected or key in seen:
                raise ValueError(f"Unexpected or duplicate judgment: {key}")
            seen.add(key)
            verdict = row.get("full_verdict", row.get("coordinator_verdict"))
            if verdict not in {"normal", "inconclusive", "reproduced_native_motif", "other_anomaly"}:
                raise ValueError(f"Unknown coordinator verdict: {verdict}")
            rows.append(
                {
                    "job": row["job"],
                    "seed": row["seed"],
                    "full_verdict": verdict,
                    "served_verdict": row["served_verdict"],
                    "coordinator_source": str(source),
                }
            )
    cases = len({r["job"] for r in rows})
    result = {
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "status": "all36_semantically_reviewed" if seen == expected else f"partial{cases}of36_cases",
        "cases": cases,
        "trials": len(rows),
        "counts": dict(collections.Counter(r["full_verdict"] for r in rows)),
        "rows": rows,
        "pending": [{"job": j, "seed": seed} for j, seed in sorted(expected - seen)],
        "sources": [{"path": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in sources],
        "limitations": "No overall policy pass. Capped progress remains inconclusive. Canonical reconstructed input, not exact native stream. Full review does not itself select a precision policy or change task scores.",
    }
    (O / "full-semantic-partial-summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ("status", "cases", "trials", "counts")}))


if __name__ == "__main__":
    main()
