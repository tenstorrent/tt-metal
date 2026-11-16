import datetime
import hashlib
import json
import pathlib
import shutil
import subprocess

root = pathlib.Path.cwd()
d = root / "models/autoports/google_gemma_4_31b_it/doc/agentic_evals"
jobs = d / "setup/swe-jobs/swe-phase2-ready"
out = d / "current/swe-phase2-ready/verifier-recovery"
out.mkdir(exist_ok=True)


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


shutil.copy("/tmp/swe-verifier-recovery-install.log", out / "dependency-install.log")
env = root / "bringup/artifacts/swe-verifier-recovery-env/bin/python"
freeze = subprocess.run(["uv", "pip", "freeze", "--python", str(env)], capture_output=True, text=True, check=True)
(out / "requirements-resolved.txt").write_text(freeze.stdout)
rows = []
for trial in sorted(jobs.iterdir()):
    raw = trial / "verifier/test-stdout.txt"
    if not raw.exists() or not (trial / "result.json").exists():
        continue
    task = trial.name.rsplit("__", 1)[0]
    row = {"task": task, "trial": trial.name, "raw_result": str(trial / "result.json"), "raw_log_sha256": sha(raw)}
    text = raw.read_text()
    cause = "Because swebench was not found in the cache and you require swebench==4.0.3"
    if cause not in text:
        row["status"] = "excluded_different_failure"
        rows.append(row)
        continue
    dest = out / task
    dest.mkdir(exist_ok=True)
    tests = d / "setup/swe-tasks" / task / "tests"
    script = tests / "test.sh"
    source = script.read_text()
    heredoc = "cat > parser.py <<EOF\n" + source.split("cat > parser.py <<EOF\n", 1)[1].split("\nEOF", 1)[0] + "\nEOF\n"
    subprocess.run(["bash", "-c", heredoc], cwd=dest, check=True)
    lines = text.splitlines(keepends=True)
    starts = [i for i, line in enumerate(lines) if line.rstrip() == "+ set +x"]
    start = starts[-1]
    assert any(x.rstrip() == "+ exec" for x in lines[max(0, start - 4) : start])
    err = next(
        i
        for i, line in enumerate(lines)
        if line.startswith("error: No solution found when resolving script dependencies")
    )
    end = err
    while end > start and lines[end - 1].startswith("Updated "):
        end -= 1
    assert end < err
    (dest / "test-output.original.txt").write_text("".join(lines[start:end]))
    shutil.copy(dest / "test-output.original.txt", dest / "test-output.parser.txt")
    cmd = [
        "docker",
        "run",
        "--rm",
        "--network",
        "none",
        "--user",
        "6002:6002",
        "--entrypoint",
        str(env),
        "-v",
        f"{root}:{root}:ro",
        "-v",
        f"{tests}:/tests:ro",
        "-v",
        f"{dest}:/recovery:rw",
        "-v",
        f"{dest}:/logs/verifier:rw",
        "-e",
        "LOG_FILE=/recovery/test-output.parser.txt",
        "-e",
        "HF_HUB_OFFLINE=1",
        "-e",
        "HF_DATASETS_OFFLINE=1",
        "v5-swe-prepared:pylint-dev__pylint-6903",
        "/recovery/parser.py",
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    (dest / "parser.stdout.txt").write_text(r.stdout)
    (dest / "parser.stderr.txt").write_text(r.stderr)
    row.update(
        status="parsed" if (dest / "report.json").exists() else "parser_error",
        command=cmd,
        returncode=r.returncode,
        source_script_sha256=sha(script),
        parser_sha256=sha(dest / "parser.py"),
        config_sha256=sha(tests / "config.json"),
        raw_test_output_lines_inclusive=[start + 1, end],
        test_output_sha256=sha(dest / "test-output.original.txt"),
        raw_reward=json.loads((trial / "result.json").read_text()).get("verifier_result"),
        network="none",
        time_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    )
    if (dest / "report.json").exists():
        report = json.loads((dest / "report.json").read_text())
        row["recovered_reward"] = int(report[task]["resolved"])
        (dest / "recovered_reward.txt").write_text(str(row["recovered_reward"]) + "\n")
    (dest / "provenance.json").write_text(json.dumps(row, indent=2) + "\n")
    rows.append(row)
    print(task, row["status"], row.get("recovered_reward"), r.returncode, flush=True)
(out / "recovery-summary.json").write_text(
    json.dumps(
        {
            "status": "partial_until_all_trials_final",
            "rows": rows,
            "method": "Unchanged shell-expanded parser from each original test.sh; exact config. Reconstructed test-output slice from preserved verifier stdout bounded by traced exec/set+x and final checkout output. Original files untouched. Parser mutates only copied test output. No agent or repository tests rerun.",
        },
        indent=2,
    )
    + "\n"
)
