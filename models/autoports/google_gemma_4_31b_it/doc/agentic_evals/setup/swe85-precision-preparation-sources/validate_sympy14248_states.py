import fcntl
import hashlib
import json
import pathlib
import subprocess
import tarfile
import time

D = pathlib.Path("models/autoports/google_gemma_4_31b_it/doc/agentic_evals")
P = D / "anomalies/swe85-below-band/correct-actions/sympy__sympy-14248"
O = P / "semantic-original-tests"
O.mkdir(exist_ok=False)
O.chmod(0o700)
meta = json.loads((P / "state-manifest.json").read_text())
T = D / "setup/swe85-candidate/tasks85-offline-verifier/sympy__sympy-14248/tests"
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
testhashes = {p.name: sha(p) for p in [T / "config.json", T / "test.sh"]}
lock = (D / "setup/.semantic-verifier-worker.lock").open("a")
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
rows = []
for state, folder, key in [
    ("before", "precall", "before_sha256"),
    ("chosen50", "chosen50", "chosen50_sha256"),
    ("correct", "candidate", "correct_sha256"),
]:
    out = O / state
    out.mkdir()
    (out / "verifier").mkdir()
    name = "v5-sympy14248-" + state + "-" + str(time.time_ns())
    calls = []
    created = False
    archive = out / "source-state.tar"
    with tarfile.open(archive, "w") as tar:
        for f in meta["files"]:
            p = P / folder / f["file"]
            assert sha(p) == f[key]
            tar.add(p, arcname=f["file"])
    before = {f["file"]: f["base_sha256"] for f in meta["files"]}
    after = {f["file"]: f[key] for f in meta["files"]}
    checkscript = "import json,hashlib; d=json.loads(__import__('sys').argv[1]); a={p:hashlib.sha256(open('/testbed/'+p,'rb').read()).hexdigest() for p in d}; assert a==d, [(p,a[p],d[p]) for p in d if a[p]!=d[p]]; print('all_source_hashes_match',len(d))"

    def run(argv, check=True, timeout=1800):
        r = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
        idx = len(calls)
        (out / f"{idx:02d}.stdout.txt").write_text(r.stdout)
        (out / f"{idx:02d}.stderr.txt").write_text(r.stderr)
        calls.append({"argv": argv, "exit_code": r.returncode})
        if check and r.returncode:
            raise RuntimeError(r.stderr[-3000:])
        return r

    try:
        run(
            [
                "docker",
                "create",
                "--name",
                name,
                "--network",
                "none",
                "--cpus",
                "1",
                "--memory",
                "4g",
                "--memory-swap",
                "4g",
                "--pids-limit",
                "256",
                "-e",
                "PIP_NO_INDEX=1",
                "-e",
                "UV_OFFLINE=1",
                "-e",
                "UV_PYTHON_DOWNLOADS=never",
                "--entrypoint",
                "/bin/bash",
                meta["extraction"]["image_id"],
                "-lc",
                "sleep 1800",
            ]
        )
        created = True
        run(["docker", "start", name])
        assert (
            run(["docker", "exec", name, "git", "-C", "/testbed", "rev-parse", "HEAD"]).stdout.strip()
            == meta["image_head"]
        )
        run(["docker", "exec", name, "/opt/miniconda3/bin/python", "-c", checkscript, json.dumps(before)])
        run(["docker", "cp", str(archive.resolve()), name + ":/tmp/source-state.tar"])
        run(["docker", "exec", name, "tar", "-xf", "/tmp/source-state.tar", "-C", "/testbed"])
        run(["docker", "exec", name, "/opt/miniconda3/bin/python", "-c", checkscript, json.dumps(after)])
        run(["docker", "exec", name, "mkdir", "-p", "/tests", "/logs/verifier"])
        run(["docker", "cp", str(T.resolve()) + "/.", name + ":/tests"])
        run(["docker", "exec", name, "chmod", "-R", "a-w", "/tests"])
        testbefore = run(["docker", "exec", name, "sha256sum", "/tests/config.json", "/tests/test.sh"]).stdout
        run(
            [
                "docker",
                "exec",
                name,
                "/bin/bash",
                "-lc",
                "source /opt/miniconda3/bin/activate && conda activate testbed && python --version && python -m pip freeze",
            ]
        )
        wrapper = 'mkdir -p /logs/verifier\ntrap \'status=$?; if [ -n "${LOG_FILE:-}" ] && [ -f "$LOG_FILE" ]; then cp "$LOG_FILE" /logs/verifier/test-output.txt; fi; if [ -f /parser.py ]; then cp /parser.py /logs/verifier/parser.py; fi; exit "$status"\' EXIT\ncd /testbed\nsource /tests/test.sh\n'
        result = run(["docker", "exec", name, "/bin/bash", "-lc", wrapper], check=False)
        run(["docker", "exec", name, "/opt/miniconda3/bin/python", "-c", checkscript, json.dumps(after)])
        assert testbefore == run(["docker", "exec", name, "sha256sum", "/tests/config.json", "/tests/test.sh"]).stdout
        run(["docker", "cp", name + ":/logs/verifier/.", str((out / "verifier").resolve())])
        report = out / "verifier/report.json"
        r = json.loads(report.read_text())["sympy__sympy-14248"] if report.exists() else None
        row = {
            "state": state,
            "source_folder": folder,
            "status": "verifier_finished" if r else "incomplete_no_report",
            "verifier_exit_code": result.returncode,
            "source_and_test_hashes_preserved": True,
            "report": str(report) if r else None,
            "report_sha256": sha(report) if r else None,
            "result": r,
        }
        rows.append(row)
    finally:
        if created:
            subprocess.run(
                ["docker", "cp", name + ":/logs/verifier/.", str((out / "verifier").resolve())], capture_output=True
            )
            run(["docker", "rm", "-f", name], check=False)
        (out / "commands.json").write_text(json.dumps(calls, indent=2) + "\n")
    assert {p.name: sha(p) for p in [T / "config.json", T / "test.sh"]} == testhashes
    summary = {
        "task": "sympy__sympy-14248",
        "states": rows,
        "image_id": meta["extraction"]["image_id"],
        "state_manifest": str(P / "state-manifest.json"),
        "state_manifest_sha256": sha(P / "state-manifest.json"),
        "test_hashes": testhashes,
        "network": "none",
        "limits": {"cpus": 1, "memory_gib": 4},
        "scope": meta["source_scope"],
        "official_test_script_unchanged": True,
        "no_scored_rerun": True,
    }
    (O / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(
        json.dumps(
            {
                "state": state,
                "status": row["status"],
                "resolved": r["resolved"] if r else None,
                "tests_status": r["tests_status"] if r else None,
            }
        ),
        flush=True,
    )
for p in O.rglob("*"):
    if p.is_file():
        p.chmod(0o600)
