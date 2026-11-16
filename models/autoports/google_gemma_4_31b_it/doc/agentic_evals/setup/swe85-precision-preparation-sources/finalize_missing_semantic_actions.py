import hashlib
import json
import pathlib
import subprocess
import sys

from transformers import AutoTokenizer

task = sys.argv[1]
assert task in ["psf__requests-2931", "matplotlib__matplotlib-20488"]
D = pathlib.Path("models/autoports/google_gemma_4_31b_it/doc/agentic_evals")
P = D / "anomalies/swe85-below-band/correct-actions" / task
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
sm = P / "state-manifest.json"
state = json.loads(sm.read_text())
assert [r["state"] for r in s["states"]] == ["before", "chosen29", "correct"]
evidence = {"summary": str(sp), "sha256": sha(sp), "states": []}
bindingrows = []
for r in s["states"]:
    assert r["status"] == "verifier_finished"
    assert r["source_and_test_hashes_preserved"]
    evidence["states"].append(
        {
            "state": r["state"],
            "report": r["report"],
            "report_sha256": r["report_sha256"],
            "resolved": r["result"]["resolved"],
            "counts": {k: {kk: len(vv) for kk, vv in v.items()} for k, v in r["result"]["tests_status"].items()},
        }
    )
    root = P / "semantic-original-tests" / r["state"]
    cmds = json.loads((root / "commands.json").read_text())
    testchecks = []
    sourcechecks = []
    for i, c in enumerate(cmds):
        if "sha256sum" in c["argv"] and "/tests/config.json" in c["argv"]:
            actual = {
                pathlib.PurePosixPath(l.split()[1]).name: l.split()[0]
                for l in (root / f"{i:02d}.stdout.txt").read_text().splitlines()
            }
            assert actual == s["test_hashes"]
            testchecks.append({"command_index": i, "hashes": actual})
        if any("all_source_hashes_match" in a for a in c["argv"]):
            assert c["exit_code"] == 0
            sourcechecks.append(i)
    assert len(testchecks) == 2 and len(sourcechecks) == 3
    bindingrows.append(
        {
            "state": r["state"],
            "commands": str(root / "commands.json"),
            "commands_sha256": sha(root / "commands.json"),
            "original_test_hash_checks": testchecks,
            "source_hash_check_command_indices": sourcechecks,
        }
    )
if task == "psf__requests-2931":
    b, c, a = [r["result"]["tests_status"] for r in s["states"]]
    ftp = "test_requests.py::TestRequests::test_binary_put"
    reg = "test_requests.py::TestRequests::test_params_bytes_are_encoded"
    assert (
        ftp in b["FAIL_TO_PASS"]["failure"]
        and ftp in c["FAIL_TO_PASS"]["success"]
        and ftp in a["FAIL_TO_PASS"]["success"]
    )
    assert (
        reg in b["PASS_TO_PASS"]["success"]
        and reg in c["PASS_TO_PASS"]["failure"]
        and reg in a["PASS_TO_PASS"]["success"]
    )
    limits = "Original full suite retains81 missing-httpbin fixture errors and2 offline connect-timeout failures in all3 states. Correct binary body and byte query-parameter tests both pass; no full-suite pass claim. Other source compatibility edits are preserved identically in all3 states."
    meaning = "raw29/native57 returns bytes unchanged in shared _encode_params and regresses original byte-parameter encoding test. Correct action also converts URL params with to_native_string in prepare_url."
else:
    b, c, a = [r["result"]["tests_status"] for r in s["states"]]
    target = "lib/matplotlib/tests/test_image.py::test_huge_range_log[png--1]"
    assert (
        target in b["FAIL_TO_PASS"]["failure"]
        and target in c["FAIL_TO_PASS"]["failure"]
        and target in a["FAIL_TO_PASS"]["success"]
    )
    reg = "lib/matplotlib/tests/test_image.py::test_huge_range_log[png-1]"
    assert (
        reg in b["PASS_TO_PASS"]["success"]
        and reg in c["PASS_TO_PASS"]["failure"]
        and reg in a["PASS_TO_PASS"]["success"]
    )
    limits = "Original full suite has10 inherited failures and19 skips after correct action, including offline HTTPS test (the only remaining expected P2P failure). Before11failed110passed; chosen13failed108passed; correct10failed111passed. Chosen introduces UnboundLocalError in image_shift[pdf] and huge_range_log[png-1], both passing before/correct. No full-suite pass claim. Prior native test_colors.py np.Inf edit excluded identically so original tests stay unchanged."
    meaning = "raw29/native58 adds DEBUG print of t_vmin/t_vmax before assignment, introducing UnboundLocalError. Correct action fixes LogNorm nonpositive scaled bounds in image.py without erroneous debugging insertion."
full = pathlib.Path(state["full_reply_case"])
packet = pathlib.Path(state["full_reply_packet"])
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
out = V / (task + ".call29.action.private.json")
meta = {
    **fc["metadata"],
    "scope": "Original full request plus original reasoning before first tool marker; action-only counterfactual, not corrected reasoning.",
    "correct_semantic_validation": meaning + " " + limits,
    "semantic_validation_evidence": evidence,
    "source_state_manifest": str(sm),
    "source_state_manifest_sha256": sha(sm),
    "source_scope": state["source_scope"],
    "native_source_checks": str(P / "native-source-read-checks.json"),
    "reference_informed_target_not_model_answer": True,
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
jobs = [{"name": task + ".call29.validated-action", "case": str(out), "branches": ["chosen", "correct"]}]
write(V / "cases.json", jobs)
binding = {
    "action_case": str(out),
    "action_case_sha256": sha(out),
    "state_manifest": str(sm),
    "state_manifest_sha256": sha(sm),
    "semantic_summary": str(sp),
    "semantic_summary_sha256": sha(sp),
    "image_id": state["extraction"]["image_id"],
    "image_head": state["image_head"],
    "source_file_count": len(state["files"]),
    "original_test_hashes": s["test_hashes"],
    "all_states": bindingrows,
    "source_and_test_hashes_preserved": True,
}
write(V / "validation-binding.json", binding)
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
        "task": task,
        "status": "supplemental_semantic_action_validated_not_executed",
        "call": 29,
        "native_message": state["native_message"],
        "action_case": str(out),
        "action_case_sha256": sha(out),
        "cases": jobs,
        "semantic_evidence": evidence,
        "source_state_manifest": str(sm),
        "source_state_manifest_sha256": sha(sm),
        "scope_limits": limits,
        "semantic_action": meaning,
        "validation_binding": str(V / "validation-binding.json"),
        "validation_binding_sha256": sha(V / "validation-binding.json"),
        "no_inference": True,
        "no_scored_rerun": True,
        "cpu_validation": checks,
        "commands_staged_only": commands,
        "supplemental_only_frozen161_unchanged": True,
    },
)
print(
    json.dumps(
        {
            "manifest": str(V / "manifest.json"),
            "chosen_tokens": len(chosen),
            "correct_tokens": len(correct),
            "checks": checks,
        }
    )
)
for p in P.rglob("*"):
    if p.is_file() and not p.is_symlink():
        p.chmod(0o600)
