import hashlib
import json
import pathlib
import subprocess

from transformers import AutoTokenizer

D = pathlib.Path("models/autoports/google_gemma_4_31b_it/doc/agentic_evals")
P = D / "anomalies/swe85-below-band/correct-actions/psf__requests-5414"
F = D / "anomalies/swe85-below-band/full-replies-requests5414"
V = P / "validated-pure-urls"
V.mkdir(exist_ok=False)
V.chmod(0o700)
rev = "842da3794eaa0b77d5f08bae87a17459d91ff475"
tok = AutoTokenizer.from_pretrained(
    "/home/mvasiljevic/.cache/huggingface/hub/models--google--gemma-4-31B-it/snapshots/" + rev,
    revision=rev,
    local_files_only=True,
)
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()


def write(p, j):
    p.write_text(json.dumps(j, indent=2) + "\n")
    p.chmod(0o600)


summary = P / "semantic-pure-urls/summary.json"
s = json.loads(summary.read_text())
assert [r["counts"] for r in s["states"]] == [
    {"pass": 31, "fail": 1, "error": 0, "skip": 0},
    {"pass": 24, "fail": 8, "error": 0, "skip": 0},
    {"pass": 32, "fail": 0, "error": 0, "skip": 0},
]
evidence = {
    "narrow_summary": str(summary),
    "narrow_summary_sha256": sha(summary),
    "before_counts": s["states"][0]["counts"],
    "chosen30_counts": s["states"][1]["counts"],
    "correct_counts": s["states"][2]["counts"],
    "full_verifier": {},
}
for state in ["before", "after"]:
    p = P / "semantic-offline-wheels" / state
    prov = p / "semantic-validation-provenance.json"
    j = json.loads(prov.read_text())
    assert j["source_and_test_hashes_preserved"]
    r = p / "verifier/report.json"
    evidence["full_verifier"][state] = {
        "report": str(r),
        "report_sha256": sha(r),
        "provenance": str(prov),
        "provenance_sha256": sha(prov),
        "resolved": json.loads(r.read_text())["psf__requests-5414"]["resolved"],
    }
jobs = []
cases = []
for n in [23, 30]:
    name = f"psf__requests-5414.call{n}"
    full = F / (name + ".case.private.json")
    packet = F / (name + ".packet.private.json")
    fc = json.loads(full.read_text())
    pk = json.loads(packet.read_text())
    raw = pk["reconstructed_full_reply"]
    at = raw.index("<|tool_call>")
    prefix = raw[:at]
    chosen = tok.encode(raw[at:], add_special_tokens=False) + [pk["native_stop_id"]]
    command = (P / "candidate-action.private.sh").read_text()
    correct = tok.encode(
        '<|tool_call>call:bash{command:<|"|>' + command + '<|"|>}<tool_call|>', add_special_tokens=False
    ) + [pk["native_stop_id"]]
    out = V / (name + ".action.private.json")
    meta = {
        **fc["metadata"],
        "scope": "Action counterfactual conditioned on entire original request and complete original reasoning before first tool marker. Not a corrected reasoning chain.",
        "correct_semantic_validation": "Original pure URL test subset: before31pass1fail; chosen final source24pass8fail; correct32pass0fail. Full unchanged suite remains fixture/network limited, not full-suite validated.",
        "semantic_validation_evidence": evidence,
        "reference_informed_target_not_model_answer": True,
        "source_scope": "Exact cached image models.py matches native nl source outputs at messages13/17. Before23 no source mutation; native23 script fails TypeError before write. Before30 original models.py restored at native55. Does not replay complete native filesystem/dependencies.",
        "native_source_checks": str(P / "native-source-read-checks.json"),
        "correct_branch": "original-source ASCII-host guard broadened only from wildcard to wildcard or leading dot; non-ASCII IDNA conditional retained.",
        "prior_actions": "raw23 TypeError before file write; raw25 broad sed later reverted by raw29 restoring original backup; raw30 applies all-host IDNA and causes8 pure URL regressions.",
        "executed": False,
    }
    write(
        out,
        {
            "prompt_token_ids": fc["prompt_token_ids"] + tok.encode(prefix, add_special_tokens=False),
            "continuations": {"chosen": chosen, "correct": correct},
            "metadata": meta,
        },
    )
    jobs.append({"name": name + ".validated-pure-url-action", "case": str(out), "branches": ["chosen", "correct"]})
    cases.append(
        {
            "call": n,
            "case": str(out),
            "sha256": sha(out),
            "chosen_action_tokens": len(chosen),
            "correct_action_tokens": len(correct),
        }
    )
write(V / "cases.json", jobs)
commands = {}
checks = []
for mode, config in [("served", "head_bfp8_build_kwargs.json"), ("full", "full_precision_build_kwargs.json")]:
    a = [
        "python_env/bin/python",
        "-m",
        "models.autoports.google_gemma_4_31b_it.tools.precision_repro_batch",
        "--cases",
        str(V / "cases.json"),
        "--build-kwargs-file",
        str(D / "setup" / config),
        "--output-dir",
        str(V / mode),
    ]
    if mode == "full":
        a += ["--compare-dir", str(V / "served")]
    r = subprocess.run(a + ["--validate-only"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    commands[mode] = a
    checks.append({"mode": mode, "exit_code": r.returncode, "stdout": r.stdout.strip()})
write(
    V / "manifest.json",
    {
        "task": "psf__requests-5414",
        "status": "narrow_semantic_targets_validated_full_suite_fixture_limited_TT_not_executed",
        "cases": cases,
        "semantic_evidence": evidence,
        "original_score_unchanged": True,
        "private_reference_diagnostic_only": True,
        "source_and_test_hashes_preserved": True,
        "cpu_validation": checks,
        "commands_staged_only": commands,
        "full_reply_roster": str(F / "cases.json"),
        "onset_roster": str(D / "anomalies/swe85-requests5414-caret-controls/cases.json"),
        "full_verifier_limit": "158 recursive missing-httpbin fixture errors and4 offline connect-timeout test failures persist before/after. No cached genuine pytest-httpbin distribution located. No dependency substitution or fixture modification.",
        "request_revision": rev,
    },
)
print(json.dumps({"manifest": str(V / "manifest.json"), "cases": cases, "checks": checks}))
