import ast
import difflib
import hashlib
import json
import pathlib
import shutil
import subprocess

from reconstruct_gap_states import reconstruct
from transformers import AutoTokenizer

D = pathlib.Path("models/autoports/google_gemma_4_31b_it/doc/agentic_evals")
O = D / "anomalies/swe85-below-band/correct-actions"
F = D / "anomalies/swe85-below-band/full-replies"
rev = "842da3794eaa0b77d5f08bae87a17459d91ff475"
tok = AutoTokenizer.from_pretrained(
    "/home/mvasiljevic/.cache/huggingface/hub/models--google--gemma-4-31B-it/snapshots/" + rev,
    local_files_only=True,
    revision=rev,
)
calls = {"mwaskom__seaborn-3069": 50, "psf__requests-1142": 31}
jobs = []
summ = []


def sha(b):
    return hashlib.sha256(b).hexdigest()


def write(p, j):
    p.write_text(json.dumps(j, indent=2) + "\n")
    p.chmod(0o600)


for task, n in calls.items():
    d = O / task
    meta = json.loads((d / "extraction.json").read_text())
    base = d / "base"
    pre = d / "precall"
    post = d / "candidate"
    shutil.copytree(base, pre, dirs_exist_ok=True)
    edits = []
    if task.endswith("6197"):
        p = pre / "src/_pytest/config/__init__.py"
        s = p.read_text()
        old = "__import__(importspec)"
        new = 'print(f"DEBUG: importing {importspec}"); __import__(importspec)'
        assert old in s
        p.write_text(s.replace(old, new))
        edits.append(
            {
                "native_message": 64,
                "effect": "Audited sed replacement adds DEBUG before __import__; reproduced by literal text replacement, not shell execution.",
                "file": str(p.relative_to(pre)),
            }
        )
    if task.endswith("8898"):
        p = pre / "pylint/utils/utils.py"
        s = p.read_text()
        assert "import os" in s
        p.write_text(s.replace("import os", "import os\nimport csv\nimport io"))
        edits.append(
            {
                "native_message": 80,
                "effect": "Audited sed insertion imports csv/io before bad _splitstrip rewrite; literal text reproduction.",
                "file": str(p.relative_to(pre)),
            }
        )
    edits, source_proof = reconstruct(task, base, pre)
    shutil.copytree(pre, post, dirs_exist_ok=True)
    patch = d / "reference.patch.private"
    dry = subprocess.run(
        ["patch", "--batch", "--dry-run", "-p1", "-i", str(patch.resolve())], cwd=post, capture_output=True, text=True
    )
    applied = None
    syntax = []
    if dry.returncode == 0:
        applied = subprocess.run(
            ["patch", "--batch", "-p1", "-i", str(patch.resolve())], cwd=post, capture_output=True, text=True
        )
        if applied.returncode == 0:
            for p in post.rglob("*.py"):
                try:
                    ast.parse(p.read_text())
                    syntax.append({"file": str(p.relative_to(post)), "syntax": True})
                except SyntaxError as e:
                    syntax.append({"file": str(p.relative_to(post)), "syntax": False, "error": str(e)})
    dif = ""
    hashes = []
    for p in pre.rglob("*.py"):
        rel = p.relative_to(pre)
        a = p.read_text()
        b = (post / rel).read_text()
        hashes.append(
            {"file": str(rel), "before_sha256": sha(p.read_bytes()), "after_sha256": sha((post / rel).read_bytes())}
        )
        if a != b:
            dif += "".join(
                difflib.unified_diff(
                    a.splitlines(True), b.splitlines(True), fromfile="a/" + str(rel), tofile="b/" + str(rel)
                )
            )
    (d / "state-relative.patch.private").write_text(dif)
    full = F / f"{task}.call{n}.case.private.json"
    packet = F / f"{task}.call{n}.packet.private.json"
    if task.endswith("10356"):
        full = O / "semantic-full-replies" / full.name
        packet = O / "semantic-full-replies" / packet.name
    fc = json.loads(full.read_text())
    pk = json.loads(packet.read_text())
    tr = json.loads(pathlib.Path(fc["metadata"]["native_trajectory"]).read_text())
    mi = fc["metadata"]["native_message_indices"][0]
    audit = []
    matches = []
    for i, m in enumerate(tr["messages"][:mi]):
        for tc in m.get("tool_calls") or []:
            try:
                cmd = json.loads(tc["function"]["arguments"]).get("command", "")
            except Exception:
                continue
            audit.append({"message": i, "command": cmd})
            if cmd.startswith("cat ") and "\n" not in cmd and "|" not in cmd and ">" not in cmd:
                rel = cmd[4:].strip()
                p = base / rel
                if p.is_file() and i + 1 < len(tr["messages"]):
                    try:
                        output = json.loads(tr["messages"][i + 1]["content"])["output"]
                        matches.append(
                            {"file": rel, "message": i, "exact_native_output_match": p.read_text() == output}
                        )
                    except (ValueError, KeyError, TypeError):
                        try:
                            out = json.loads(tr["messages"][i + 1]["content"])
                            head = out.get("output_head")
                            tail = out.get("output_tail")
                            matches.append(
                                {
                                    "file": rel,
                                    "message": i,
                                    "native_output_truncated": True,
                                    "head_matches": p.read_text().startswith(head) if head else None,
                                    "tail_matches": p.read_text().endswith(tail) if tail else None,
                                    "full_source_equivalence_proven": False,
                                }
                            )
                        except (ValueError, KeyError, TypeError):
                            pass
    write(d / "prior-actions.audit.private.json", audit)
    status = {
        "task": task,
        "call": n,
        "response_id": fc["metadata"]["response_id"],
        "full_chosen_case": str(full),
        "reference_source": str(D / "setup/swe85-candidate/tasks85-offline-verifier" / task / "tests/config.json"),
        "reference_informed_target_not_model_answer": True,
        "extraction": meta,
        "image_head": (base / ".git/refs/heads/main").read_text().strip()
        if (base / ".git/refs/heads/main").exists()
        else "refs/heads/master: immutable image ID recorded",
        "source_scope": "Only reference-touched source files plus known prior tracked edit. Does not reconstruct complete filesystem/dependency state. Image HEAD differs from dataset base; exact native cat matches and patch applicability provide file-level checks, not full-tree equivalence.",
        "native_source_read_checks": matches,
        "audited_prior_tracked_edits": edits,
        "reconstruction_proof": source_proof,
        "other_prior_actions": "Audited transcript has reproducer/config/temp script writes and dependency installs; preserved in private audit, not replayed. No earlier edits to candidate-touched production files beyond listed changes.",
        "precall_files": hashes,
        "patch_application_dry_run": {"returncode": dry.returncode, "stdout": dry.stdout, "stderr": dry.stderr},
        "patch_application": None if applied is None else {"returncode": applied.returncode, "stdout": applied.stdout},
        "syntax_checks": syntax,
        "semantic_execution_validation": "pending isolated test slot; not passed",
        "correct_branch_status": "reference-informed candidate; applicability/syntax only, semantic correctness not yet validated",
        "test_plan": "Run unchanged reference FAIL_TO_PASS and PASS_TO_PASS in isolated offline validation only, never scored task source; compare baseline/candidate and preserve native scores.",
    }
    if dif and applied and applied.returncode == 0 and all(s["syntax"] for s in syntax):
        raw = pk["reconstructed_full_reply"]
        at = raw.index("<|tool_call>")
        prefix = raw[:at]
        suffix = raw[at:]
        stop = pk["native_stop_id"]
        command = "patch -p1 <<'SWE_DIAGNOSTIC_PATCH'\n" + dif + "SWE_DIAGNOSTIC_PATCH\n"
        alternative = '<|tool_call>call:bash{command:<|"|>' + command + '<|"|>}<tool_call|>'
        prompt = fc["prompt_token_ids"] + tok.encode(prefix, add_special_tokens=False)
        chosen = tok.encode(suffix, add_special_tokens=False) + [stop]
        correct = tok.encode(alternative, add_special_tokens=False) + [stop]
        case = d / "action_counterfactual.candidate.private.json"
        write(
            case,
            {
                "prompt_token_ids": prompt,
                "continuations": {"chosen": chosen, "correct": correct},
                "metadata": {
                    "task": task,
                    "call": n,
                    "response_id": status["response_id"],
                    "full_reply_case": str(full),
                    "correct_semantic_validation": "pending; candidate only",
                    "conditioning": "Original entire request plus original reasoning/visible content before first tool marker. Correct action is conditional on potentially faulty original reasoning; not corrected model reasoning or a success claim. Full chosen-reply agreement remains separate.",
                    "hf_revision": rev,
                    "source_status": status["source_scope"],
                    "reference_informed_target_not_model_answer": True,
                },
            },
        )
        (d / "candidate-action.private.sh").write_text(command)
        status["action_case"] = str(case)
        status["chosen_action_tokens"] = len(chosen)
        status["candidate_correct_action_tokens"] = len(correct)
        jobs.append({"name": task + ".candidate-action", "case": str(case), "branches": ["chosen", "correct"]})
    write(d / "manifest.json", status)
    summ.append(
        {
            "task": task,
            "call": n,
            "manifest": str(d / "manifest.json"),
            "apply_status": dry.returncode,
            "syntax_ok": all(s["syntax"] for s in syntax),
            "native_cat_checks": matches,
            "semantic_validation": "pending",
            "case": status.get("action_case"),
        }
    )
write(O / "gap-candidate-action-cases.json", jobs)
write(
    O / "gap-candidate-actions-manifest.json",
    {"status": "CPU application/syntax only; semantic validation pending; not completed attempt", "tasks": summ},
)
print(json.dumps(summ))
for p in O.rglob("*"):
    if p.is_file():
        p.chmod(0o600)
