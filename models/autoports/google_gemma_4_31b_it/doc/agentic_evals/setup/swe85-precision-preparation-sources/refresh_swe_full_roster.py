import datetime
import json
import pathlib
import subprocess

D = pathlib.Path("models/autoports/google_gemma_4_31b_it/doc/agentic_evals")
B = D / "anomalies/swe85-below-band"
parts = sorted(B.glob("full-replies*/cases.json")) + [
    B / "requests2931-semantic-full-reply/cases.json",
    B / "correct-actions/semantic-full-replies/cases.json",
]
jobs = []
for p in parts:
    if p.exists():
        jobs += json.loads(p.read_text())
assert len({j["name"] for j in jobs}) == len(jobs)
out = B / "all-prepared-full-reply-cases.json"
out.write_text(json.dumps(jobs, indent=2) + "\n")
validation = []
for mode, cfg in [("served", "head_bfp8_build_kwargs.json"), ("full", "full_precision_build_kwargs.json")]:
    cmd = [
        "python_env/bin/python",
        "-m",
        "models.autoports.google_gemma_4_31b_it.tools.precision_repro_batch",
        "--cases",
        str(out),
        "--build-kwargs-file",
        str(D / "setup" / cfg),
        "--output-dir",
        "/tmp/swe85-fulls-schema-unused",
        "--validate-only",
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    assert r.returncode == 0
    validation.append({"mode": mode, "exit_code": r.returncode, "stdout": r.stdout.strip()})
rows = []
for job in jobs:
    c = json.loads(pathlib.Path(job["case"]).read_text())
    m = c["metadata"]
    rows.append(
        {
            "task": m["task"],
            "call": m["call"],
            "response_id": m["response_id"],
            "case": job["case"],
            "chosen_tokens": len(c["continuations"]["chosen"]),
            "native_reply_tokens": m["native_completion_tokens"],
        }
    )
failed = []
for p in (D / "current/swe-extended85-head-bfp8/live-tasks").glob("*/coordinator.json"):
    j = json.loads(p.read_text())
    if j.get("counted_reward") == 0:
        failed.append(j["task"])
summary = {
    "status": "CPU prepared only, native token identity unproven; no inference or completed fix claim",
    "snapshot_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "task_count": len({r["task"] for r in rows}),
    "case_count": len(rows),
    "cases": rows,
    "CPU_validation": validation,
    "definitive_failures_without_full_case": sorted(set(failed) - {r["task"] for r in rows}),
    "excluded_loop": "Requests2931 capped32768 anomaly reply excluded; semantic29 separate. Pytest5787 anomalyfull1/65 archival only; onset jobs separate.",
    "correct_actions_status": "Pending candidates and fetch/action/protocol controls have separate semantic validation requirements.",
}
(D / "setup/swe85-full-reply-control-roster.json").write_text(json.dumps(summary, indent=2) + "\n")
print({k: summary[k] for k in ["task_count", "case_count", "definitive_failures_without_full_case"]})
