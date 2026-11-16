import json
import pathlib
import subprocess

D = pathlib.Path("models/autoports/google_gemma_4_31b_it/doc/agentic_evals")
mp = D / "anomalies/swe85-sympy13878-controls/manifest.json"
m = json.loads(mp.read_text())
p = D / "anomalies/swe85-precision/pending.json"
j = json.loads(p.read_text())
for r in m["cases"]:
    assert r["native_message_indices"] == [], r
    j["additional_cases"] = [
        x for x in j["additional_cases"] if (x.get("task"), x.get("call")) != (r["task"], r["call"])
    ]
    j["additional_cases"].append(
        {
            "task": r["task"],
            "call": r["call"],
            "response_id": r["response_id"],
            "scope": r["evidence"],
            "status": "prepared_not_executed_pending_sole_device_release",
            "preferred_manifest": str(mp),
            "private_case": r["private_case"],
            "reconstruction_limits": {
                "native_token_ids_absent": True,
                "native_message_absent": True,
                "native_reply_tokens": r["native_reply_tokens"],
                "reconstructed_reply_tokens": r["reconstructed_reply_tokens"],
            },
        }
    )
    print(r["task"], r["native_reply_tokens"], r["reconstructed_reply_tokens"], r["onset_positions"])
p.write_text(json.dumps(j, indent=2) + "\n")
O = D / "anomalies/swe85-below-band/fetch-controls"
jobs = []
rows = []
for p in sorted(O.glob("*/manifest.json")):
    m = json.loads(p.read_text())
    jobs.append({"name": m["task"] + ".fetch-next-action", "case": m["action_case"], "branches": ["chosen", "correct"]})
    rows.append(
        {
            "task": m["task"],
            "call": m["call"],
            "manifest": str(p),
            "status": "candidate prepared; semantic results in separate validation artifacts",
            "scope": m["scope"],
        }
    )
(O / "candidate-action-cases.json").write_text(json.dumps(jobs, indent=2) + "\n")
(O / "manifest.json").write_text(
    json.dumps(
        {
            "schema": "swe85-fetch-next-action-v1",
            "status": "Candidate roster; runtime validation outcomes maintained separately",
            "tasks": rows,
            "n": len(rows),
        },
        indent=2,
    )
    + "\n"
)
for cfg in ["head_bfp8_build_kwargs.json", "full_precision_build_kwargs.json"]:
    r = subprocess.run(
        [
            "python_env/bin/python",
            "-m",
            "models.autoports.google_gemma_4_31b_it.tools.precision_repro_batch",
            "--cases",
            str(O / "candidate-action-cases.json"),
            "--build-kwargs-file",
            str(D / "setup" / cfg),
            "--output-dir",
            "/tmp/swe85-action-schema-unused",
            "--validate-only",
        ],
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, r.stderr
    print(r.stdout.strip())
r = subprocess.run(
    ["python", str(D / "setup/validate_fetch_next_action.py"), "--task", "sympy__sympy-13878"],
    capture_output=True,
    text=True,
)
assert r.returncode == 0, r.stderr
print(json.loads(r.stdout)["status"])
p = D / "setup/prepare_swe85_tt_execution_plan.py"
s = p.read_text()
s = s.replace(
    "'swe85-18763-7432-controls/cases.json']",
    "'swe85-18763-7432-controls/cases.json','swe85-sympy13878-controls/cases.json']",
)
p.write_text(s)
