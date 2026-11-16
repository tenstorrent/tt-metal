"""One host snapshot, review pointers only; no inferred reproduction verdicts."""
import datetime
import hashlib
import json
from pathlib import Path

D = Path("models/autoports/google_gemma_4_31b_it/doc/agentic_evals")
O = D / "setup/swe85-remainder-execution85"
O.mkdir(exist_ok=True)


def read(p):
    return json.loads(Path(p).read_text())


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


manifest = D / "setup/swe85-admitted-controls85/combined-cases.json"
auditpath = O / "protocol-parser-audit.json"
audit = read(auditpath)
parsed = {(r["job"], r["mode"]): r for r in audit["rows"]}
legacy = {
    "anomaly.xarray7233-call1.canonicalprefix": "Visible call:bash pseudo-tool without structured tool",
    "anomaly.pytest10356-call14.canonicalprefix": "Malformed tool function name",
    "anomaly.xarray3095-call53.canonicalprefix": "Malformed tool name with prose and exposed delimiter",
    "anomaly.requests2931.call20.canonicalprefix": "Capped repeated caret loop",
    "anomaly.call88.canonicalprefix": "Stray curly apostrophe after contraction",
    "anomaly.call91.canonicalprefix": "Stray curly apostrophe after contraction",
}
rows = []
pending = []
for job in read(manifest):
    if not job["name"].startswith("anomaly."):
        continue
    c = read(job["case"])
    m = c["metadata"]
    motif = m.get("motif") or m.get("chosen_span") or legacy.get(job["name"], "See fixture metadata")
    protocol = job["name"] in audit["selected_jobs"]
    text_review = (
        "caret/repetition"
        if "^" in motif or "caret" in motif
        else "contraction/stray text"
        if any(s in motif for s in ["’", "apostrophe", "thought", "GetAll"])
        else "language/semantic text"
        if motif == "singkat"
        else "protocol plus independent semantic text review"
    )
    for mode in ["served", "full"]:
        p = D / "setup/swe85-pilot-split85/runs/rest" / mode / (job["name"] + ".json")
        if not p.exists():
            pending.append({"job": job["name"], "mode": mode})
            continue
        try:
            r = read(p)
        except ValueError:
            pending.append({"job": job["name"], "mode": mode, "reason": "partial JSON"})
            continue
        if "diagnostic_total_wall_s" not in r:
            pending.append({"job": job["name"], "mode": mode, "reason": "diagnostic still in progress"})
            continue
        lookup = {t["seed"]: t for t in parsed.get((job["name"], mode), {}).get("trials", [])}
        for i, run in enumerate(r.get("free_running", [])):
            ids = run["token_ids"]
            seed = run["sampling"].get("seed")
            pr = lookup.get(seed)
            rows.append(
                {
                    "job": job["name"],
                    "mode": mode,
                    "seed": seed,
                    "result": str(p),
                    "result_sha256": sha(p),
                    "generated_text_field": f"$.free_running[{i}].text",
                    "generated_token_ids_field": f"$.free_running[{i}].token_ids",
                    "fixture": job["case"],
                    "fixture_sha256": sha(job["case"]),
                    "native_anomaly_motif": motif,
                    "native_anomaly_evidence": m.get("onset_evidence", legacy.get(job["name"])),
                    "generated_tokens": len(ids),
                    "requested_cap": job.get("free_run_tokens"),
                    "at_requested_cap": len(ids) >= job.get("free_run_tokens", float("inf")),
                    "last_token_id": ids[-1] if ids else None,
                    "tool_close_token_present": 49 in ids,
                    "ends_tool_response_boundary": bool(ids and ids[-1] == 50),
                    "ends_turn_boundary": bool(ids and ids[-1] == 106),
                    "parser_observations": None
                    if not pr
                    else {k: pr[k] for k in ["conditioning_plus_generated", "generated_only"]},
                    "parser_status": "audited"
                    if pr
                    else "protocol output completed after parser snapshot; refresh required"
                    if protocol
                    else "not applicable to primary text anomaly",
                    "semantic_text_review_required": True,
                    "semantic_text_review_category": text_review,
                    "mechanical_flag": run.get("flag"),
                    "reproduction_verdict": "not adjudicated by this index",
                }
            )
counts = {
    mode: {
        "completed_jobs": len({r["job"] for r in rows if r["mode"] == mode}),
        "trials": sum(r["mode"] == mode for r in rows),
        "parser_audited_trials": sum(r["mode"] == mode and r["parser_status"] == "audited" for r in rows),
    }
    for mode in ["served", "full"]
}
out = {
    "status": "single_host_snapshot_review_index_not_reproduction_verdict",
    "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "manifest_sha256": sha(manifest),
    "learned_sha256": sha(D / "learned.json"),
    "parser_audit": str(auditpath),
    "parser_audit_sha256": sha(auditpath),
    "counts": counts,
    "pending": pending,
    "trials": rows,
    "rules": [
        "Inspect generated text at exact JSON field and compare native motif, not raw delimiter occurrence alone.",
        "Valid structured tool does not establish normal prose or correct action. Caret loops, repetitions, stray contractions/thought/GetAll and language switches require semantic text review.",
        "Full conditioning may already contain anomalous text: compare generated-only parse and prefix before attributing reproduction.",
        "Tool-close occurrence is a boundary indicator, not schema or execution proof; cap may truncate a call.",
        "Native token IDs unavailable; canonical prefix reconstruction does not establish native stream equivalence.",
    ],
}
(O / "completed-anomaly-review-index.json").write_text(json.dumps(out, indent=2) + "\n")
print(json.dumps(counts))
