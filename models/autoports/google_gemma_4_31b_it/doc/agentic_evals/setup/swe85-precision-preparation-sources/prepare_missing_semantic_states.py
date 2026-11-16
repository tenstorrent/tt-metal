import ast
import difflib
import hashlib
import json
import pathlib
import re
import shutil
import subprocess

D = pathlib.Path("models/autoports/google_gemma_4_31b_it/doc/agentic_evals")
O = D / "anomalies/swe85-below-band/correct-actions"
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
for task, job, folder, extra, prior, chosen_i in [
    (
        "matplotlib__matplotlib-20488",
        "matplotlib__matplotlib-20488__cAgyfNT",
        "full-replies",
        ["lib/matplotlib/colors.py", "lib/matplotlib/axes/_base.py"],
        [42],
        58,
    ),
    (
        "psf__requests-2931",
        "psf__requests-2931__rrZvSY9",
        "requests2931-semantic-full-reply",
        [
            "requests/cookies.py",
            "requests/structures.py",
            "requests/packages/urllib3/_collections.py",
            "requests/sessions.py",
            "requests/utils.py",
        ],
        [8, 14, 30, 34],
        57,
    ),
]:
    P = O / task
    P.mkdir(exist_ok=False)
    P.chmod(0o700)
    T = D / "setup/swe85-candidate/tasks85-offline-verifier" / task
    cfg = json.loads((T / "tests/config.json").read_text())
    tag = re.search(r'docker_image = "([^"]+)"', (T / "task.toml").read_text()).group(1)
    im = json.loads(subprocess.check_output(["docker", "image", "inspect", tag]))[0]
    cid = subprocess.check_output(["docker", "create", im["Id"]]).decode().strip()
    files = [l.split(" b/", 1)[1] for l in cfg["patch"].splitlines() if l.startswith("diff --git ")] + extra
    base = P / "base"
    base.mkdir()
    ex = []
    try:
        for rel in files + [".git/HEAD", ".git/refs/heads/main", ".git/refs/heads/master", ".git/packed-refs"]:
            dst = base / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            r = subprocess.run(["docker", "cp", cid + ":/testbed/" + rel, str(dst)], capture_output=True, text=True)
            ex.append({"path": rel, "returncode": r.returncode, "stderr": r.stderr})
    finally:
        subprocess.run(["docker", "rm", cid], stdout=subprocess.DEVNULL)
    extraction = {
        "image_tag": tag,
        "image_id": im["Id"],
        "base_commit": cfg["base_commit"],
        "container_never_started": True,
        "container_removed": True,
        "files": ex,
    }
    (P / "extraction.json").write_text(json.dumps(extraction, indent=2) + "\n")
    (P / "reference.patch.private").write_text(cfg["patch"])
    (P / "reference.tests.private.patch").write_text(cfg["test_patch"])
    pre = P / "precall"
    shutil.copytree(base, pre)
    J = D / "setup/swe-jobs/swe-extended85-head-bfp8" / job / "agent/mini-swe-agent.trajectory.json"
    j = json.loads(J.read_text())
    audits = []
    for i in prior:
        cmd = j["messages"][i]["extra"]["actions"][0]["command"]
        mapped = cmd.replace("/testbed/", str(pre.resolve()) + "/")
        if task.startswith("matplotlib"):
            mapped = mapped.replace(" lib/matplotlib/tests/test_colors.py", "")
        r = subprocess.run(["/bin/bash", "-c", mapped], cwd=pre, capture_output=True, text=True)
        assert r.returncode == 0, (i, r.stderr)
        audits.append(
            {
                "message": i,
                "command": cmd,
                "materialized_command": mapped,
                "returncode": r.returncode,
                "exclusion": "Native test_colors.py np.Inf edit excluded; original tests retained"
                if task.startswith("matplotlib")
                else None,
            }
        )
    chosen = P / "chosen29"
    correct = P / "candidate"
    shutil.copytree(pre, chosen)
    shutil.copytree(pre, correct)
    cmd = j["messages"][chosen_i]["extra"]["actions"][0]["command"]
    r = subprocess.run(
        ["/bin/bash", "-c", cmd.replace("/testbed/", str(chosen.resolve()) + "/")],
        cwd=chosen,
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, r.stderr
    r = subprocess.run(
        ["patch", "--batch", "-p1", "-i", str((P / "reference.patch.private").resolve())],
        cwd=correct,
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, (r.stdout, r.stderr)
    rows = []
    dif = ""
    syntax = []
    for rel in files:
        a = (pre / rel).read_text()
        b = (correct / rel).read_text()
        rows.append(
            {
                "file": rel,
                "base_sha256": sha(base / rel),
                "before_sha256": sha(pre / rel),
                "chosen29_sha256": sha(chosen / rel),
                "correct_sha256": sha(correct / rel),
            }
        )
        if a != b:
            dif += "".join(
                difflib.unified_diff(a.splitlines(True), b.splitlines(True), fromfile="a/" + rel, tofile="b/" + rel)
            )
        try:
            ast.parse((chosen / rel).read_text())
            syntax.append({"file": rel, "valid": True})
        except SyntaxError as e:
            syntax.append({"file": rel, "valid": False, "error": str(e)})
    (P / "state-relative.patch.private").write_text(dif)
    (P / "candidate-action.private.sh").write_text(
        "patch -p1 <<'SWE_DIAGNOSTIC_PATCH'\n" + dif + "SWE_DIAGNOSTIC_PATCH\n"
    )
    (P / "prior-actions.audit.private.json").write_text(json.dumps(audits, indent=2) + "\n")
    head = (base / ".git/HEAD").read_text().strip()
    head = (base / ".git" / head[5:]).read_text().strip() if head.startswith("ref: ") else head
    meta = {
        "task": task,
        "call": 29,
        "native_message": chosen_i,
        "image_head": head,
        "extraction": extraction,
        "files": rows,
        "source_scope": "Exact cached image relevant production files plus audited prior native compatibility edits. Original tests unchanged; prior native test-only modifications excluded where recorded. Not whole native environment replay.",
        "prior_actions": audits,
        "chosen_syntax": syntax,
        "chosen_command": cmd,
        "chosen_materialization": "Exact native extra.actions command executed on isolated local copied sources with /testbed mapped; no target code inference.",
        "full_reply_case": str(D / "anomalies/swe85-below-band" / folder / (task + ".call29.case.private.json")),
        "full_reply_packet": str(D / "anomalies/swe85-below-band" / folder / (task + ".call29.packet.private.json")),
    }
    (P / "state-manifest.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(task, "files", len(rows), "chosen syntax", syntax, flush=True)
