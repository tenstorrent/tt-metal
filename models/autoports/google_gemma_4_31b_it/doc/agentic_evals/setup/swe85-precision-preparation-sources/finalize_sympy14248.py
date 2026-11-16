import hashlib
import json
import pathlib
import subprocess

from transformers import AutoTokenizer

D = pathlib.Path("models/autoports/google_gemma_4_31b_it/doc/agentic_evals")
P = D / "anomalies/swe85-below-band/correct-actions/sympy__sympy-14248"
F = D / "anomalies/swe85-below-band/full-replies-sympy14248"
V = P / "validated"
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


sp = P / "semantic-original-tests/summary.json"
s = json.loads(sp.read_text())
assert [r["result"]["resolved"] for r in s["states"]] == [False, False, True]
sm = P / "state-manifest.json"
state = json.loads(sm.read_text())
evidence = {"summary": str(sp), "sha256": sha(sp), "states": []}
for r in s["states"]:
    evidence["states"].append(
        {
            "state": r["state"],
            "report": r["report"],
            "report_sha256": r["report_sha256"],
            "resolved": r["result"]["resolved"],
            "counts": {k: {kk: len(vv) for kk, vv in v.items()} for k, v in r["result"]["tests_status"].items()},
        }
    )
name = "sympy__sympy-14248.call50"
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
limits = "Original official test script unchanged. Before/chosen/correct use same pinned image and native pre50 production changes. Four unchanged legacy parser AST exceptions and one XPASS remain in raw correct test log despite official report resolved=true; no full-clean-suite claim. Tests are original official files/test patch,11 unrelated native test-file mutations excluded and inventoried. No scored rerun."
meta = {
    **fc["metadata"],
    "scope": "Action counterfactual conditioned on entire original request and original reasoning before first tool marker. Full chosen reply separately retained; no corrected reasoning claim.",
    "correct_semantic_validation": "Official expected tests: before F2P0/2 P2P433/433; chosen50 F2P0/2 P2P431/433; correct F2P2/2 P2P433/433. "
    + limits,
    "semantic_validation_evidence": evidence,
    "source_state_manifest": str(sm),
    "source_state_manifest_sha256": sha(sm),
    "source_scope": state["source_scope"],
    "native_source_checks": str(P / "additional-native-source-checks.json"),
    "reference_informed_target_not_model_answer": True,
    "correct_branch": "State-relative patch fixes sign handling in string, pretty, latex matrix printing, preserving audited pre50 production compatibility edits.",
    "selected_action": "raw50/native100 is first actual recursion-introducing printer change; raw46/native92 sed no-op verified. Raw79/native157 final inert print_method assignment retained as separate full-reply evidence; later source history not claimed reconstructed.",
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
jobs = [{"name": name + ".validated-action", "case": str(out), "branches": ["chosen", "correct"]}]
write(V / "cases.json", jobs)
checks = []
commands = {}
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
        "task": "sympy__sympy-14248",
        "status": "official_expected_tests_validated_baseline_AST_exceptions_retained_no_inference",
        "call": 50,
        "native_message": 100,
        "action_case": str(out),
        "action_case_sha256": sha(out),
        "chosen_action_tokens": len(chosen),
        "correct_action_tokens": len(correct),
        "cases": jobs,
        "semantic_evidence": evidence,
        "source_state_manifest": str(sm),
        "source_state_manifest_sha256": sha(sm),
        "scope_limits": limits,
        "no_inference": True,
        "no_scored_rerun": True,
        "cpu_validation": checks,
        "commands_staged_only": commands,
        "full_reply_roster": str(F / "cases.json"),
        "onset_roster": str(D / "anomalies/swe85-sympy14248-caret-controls/cases.json"),
    },
)
print(
    json.dumps({"manifest": str(V / "manifest.json"), "checks": checks, "chosen": len(chosen), "correct": len(correct)})
)
