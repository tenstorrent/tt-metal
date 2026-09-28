#!/usr/bin/env python3
"""Run ONE workflow script to completion in headless Claude sessions that cannot lose finished work (use in tmux).

  run_workflow_headless.py --script SCRIPT.js --args ARGS.json --out OUTPUT.json --workdir DIR
                           [--add-dir PATH ...] [--max-attempts 12] [--then 'shell command']

For any engine or mining workflow (recheck-wave.js, dedup-wave.js, judge-wave.js, ...). run_headless.py drives whole
audits wave after wave; this drives a single workflow. Each attempt resumes the SAME session with resumeFromRunId, so
agents that already finished replay from cache. stdin stays open, so the session waits for the workflow instead of
exiting after about 10 minutes (a plain `claude -p` kills its workflow when it exits). Once OUTPUT.json exists, it runs
--then (for example, the matching persist step). State and logs are kept in --workdir, so re-running the same command
resumes.
"""
import argparse
import glob
import json
import os
import re
import subprocess
import time

p = argparse.ArgumentParser()
p.add_argument("--script", required=True)
p.add_argument("--args", required=True)
p.add_argument("--out", required=True)
p.add_argument("--workdir", required=True)
p.add_argument("--add-dir", action="append", default=[])
p.add_argument("--max-attempts", type=int, default=12)
p.add_argument("--then")
a = p.parse_args()
wd = os.path.abspath(a.workdir)
os.makedirs(wd, exist_ok=True)
tag = os.path.splitext(os.path.basename(a.out))[0]
logf = open(os.path.join(wd, f"{tag}.driver.log"), "a")


def note(m):
    logf.write(f"{time.strftime('%F %T')} {m}\n")
    logf.flush()
    print(m, flush=True)


def scan():
    sid = rid = tid = None
    for f in sorted(
        glob.glob(os.path.join(wd, f"{tag}.claude-*.jsonl")), key=os.path.getmtime
    ):
        t = open(f, errors="replace").read()
        m = re.findall(r'"session_id":"([0-9a-f-]{36})"', t)
        sid = m[-1] if m else sid
        m = re.findall(r"Run ID: (wf_[0-9a-z-]+)", t)
        rid = m[-1] if m else rid
        m = re.findall(r"Task ID: ([0-9a-z]+)", t)
        tid = m[-1] if m else tid
    return sid, rid, tid


script = os.path.abspath(a.script)
attempt = 0
while not os.path.exists(a.out) and attempt < a.max_attempts:
    attempt += 1
    sid, rid, _ = scan()
    msg = (
        f"I explicitly ask you to run a workflow. Call the Workflow tool with scriptPath {script}"
        + (f", resumeFromRunId {rid}" if rid else "")
        + f", and args set to the JSON object in the file {os.path.abspath(a.args)} (read it; pass it as an object, "
        "not a string). Then wait for the completion notification; do not end your turn while it is running. When it "
        f"completes, copy the workflow task's output file to {os.path.abspath(a.out)} with Bash cp and reply with the path."
    )
    cmd = [
        "claude",
        "-p",
        "--input-format",
        "stream-json",
        "--output-format",
        "stream-json",
        "--verbose",
        "--permission-mode",
        "auto",
        "--add-dir",
        os.path.dirname(script),
        "--add-dir",
        wd,
        "--add-dir",
        os.path.dirname(os.path.abspath(a.args)),
        "--add-dir",
        os.path.dirname(os.path.abspath(a.out)),
    ]
    for d in a.add_dir:
        cmd += ["--add-dir", d]
    if sid:
        cmd += ["--resume", sid]
    note(f"attempt {attempt}: session {sid} resume-run {rid}")
    with (
        open(os.path.join(wd, f"{tag}.claude-{attempt:02d}.jsonl"), "w") as fo,
        open(os.path.join(wd, f"{tag}.claude.err"), "a") as fe,
    ):
        proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=fo, stderr=fe, text=True, cwd=wd
        )
        proc.stdin.write(
            json.dumps({"type": "user", "message": {"role": "user", "content": msg}})
            + "\n"
        )
        proc.stdin.flush()
        while proc.poll() is None:
            if os.path.exists(a.out):
                time.sleep(20)
                break
            time.sleep(30)
            s2, _, t2 = scan()
            hits = (
                glob.glob(f"/tmp/claude-*/*/{s2}/tasks/{t2}.output")
                if (s2 and t2)
                else []
            )
            if hits and os.path.getsize(hits[0]) > 50 and not os.path.exists(a.out):
                try:
                    json.load(open(hits[0]))  # complete JSON = the workflow finished
                    subprocess.run(["cp", hits[0], a.out])
                    note(f"copied workflow output from {hits[0]}")
                except ValueError:
                    pass
        if proc.poll() is None:
            proc.stdin.close()
            try:
                proc.wait(timeout=120)
            except subprocess.TimeoutExpired:
                proc.kill()
    note(f"attempt {attempt} ended; output present: {os.path.exists(a.out)}")
if not os.path.exists(a.out):
    note("GAVE UP; rerun the same command to resume")
    raise SystemExit(2)
if a.then:
    r = subprocess.run(a.then, shell=True, capture_output=True, text=True)
    note(f"then: rc={r.returncode} {r.stdout.strip()[-800:]} {r.stderr.strip()[-400:]}")
