#!/usr/bin/env python3
"""Run a whole audit, wave after wave, in headless Claude sessions that cannot lose finished work. Launch it in
tmux or screen, so it survives the interactive session and a closed terminal.

  run_headless.py --run DIR [--wave-size 45] [--prios AB] [--second-pass] [--max-waves 0] [--max-attempts 12]
                   [--bench-cases HOLDOUT.jsonl]

Loop: next_wave.py picks the batches -> a headless `claude -p` session runs engine/audit-wave.js on them ->
persist_wave.py -> consolidate.py -> next wave. It stops when nothing is pending, or after --max-waves.

Why this exists: a plain `claude -p` exits about 10 minutes in and KILLS its background workflow. Here each attempt
keeps stdin open (stream-json input) so the session waits for the workflow. If it dies anyway, the next attempt
resumes the SAME session with resumeFromRunId, and every agent that already finished replays from cache.
All driver state is in <run>/headless.json. To resume after a crash or reboot, run the same command again.
--bench-cases scores the run (bench.py score) after each wave, for recall-benchmark runs.
"""
import argparse
import glob
import json
import os
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
SKILL = os.path.dirname(HERE)
p = argparse.ArgumentParser()
p.add_argument("--run", required=True)
p.add_argument(
    "--wave-size",
    type=int,
    default=45,
    help="batches per wave; keep <= 50 (1000-agent cap at ~18/batch)",
)
p.add_argument("--prios", default="")
p.add_argument("--second-pass", action="store_true")
p.add_argument("--max-waves", type=int, default=0)
p.add_argument("--max-attempts", type=int, default=12)
p.add_argument("--bench-cases")
a = p.parse_args()
run = os.path.abspath(a.run)
raw_dir = os.path.join(run, "raw_wave_outputs")
os.makedirs(raw_dir, exist_ok=True)
state_path = os.path.join(run, "headless.json")
logf = open(os.path.join(run, "headless.log"), "a")


def note(msg):
    logf.write(f"{time.strftime('%F %T')} {msg}\n")
    logf.flush()
    print(msg, flush=True)


def load_state():
    return json.load(open(state_path)) if os.path.exists(state_path) else {"waves": []}


def save_state(s):
    with open(state_path + ".tmp", "w") as fh:
        json.dump(s, fh, indent=1)
    os.replace(state_path + ".tmp", state_path)


def py(*args, **kw):
    return subprocess.run(["python3", *args], capture_output=True, text=True, **kw)


def scan(wave):
    """Session id, workflow run id and task id from this wave's claude logs."""
    sid = rid = tid = None
    for f in sorted(
        glob.glob(os.path.join(raw_dir, f"wave{wave:03d}.claude-*.jsonl")),
        key=os.path.getmtime,
    ):
        txt = open(f, errors="replace").read()
        m = re.findall(r'"session_id":"([0-9a-f-]{36})"', txt)
        sid = m[-1] if m else sid
        m = re.findall(r"Run ID: (wf_[0-9a-z-]+)", txt)
        rid = m[-1] if m else rid
        m = re.findall(r"Task ID: ([0-9a-z]+)", txt)
        tid = m[-1] if m else tid
    return sid, rid, tid


def run_wave(wave, args_path, out_path):
    attempt = 0
    while not os.path.exists(out_path) and attempt < a.max_attempts:
        attempt += 1
        sid, rid, _ = scan(wave)
        args_json = open(args_path).read().strip()
        msg = (
            f"I explicitly ask you to run a workflow. Call the Workflow tool with scriptPath {HERE}/audit-wave.js"
            + (f", resumeFromRunId {rid}" if rid else "")
            + f", and args set to this JSON value (pass it as an object, not a string): {args_json}\n"
            "Then wait for the completion notification; do not end your turn while the workflow is running. When it "
            f"completes, copy the workflow task's output file to {out_path} with Bash cp and reply with the path."
        )
        st = json.load(open(os.path.join(run, "state.json")))
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
            SKILL,
            "--add-dir",
            run,
            "--add-dir",
            st["root"],
        ]
        roots = json.loads(args_json).get("roots_dir")
        if roots:
            cmd += ["--add-dir", roots]
        if sid:
            cmd += ["--resume", sid]
        note(f"wave {wave} attempt {attempt}: session {sid} resume-run {rid}")
        logp = os.path.join(raw_dir, f"wave{wave:03d}.claude-{attempt:02d}.jsonl")
        with (
            open(logp, "w") as fo,
            open(os.path.join(raw_dir, f"wave{wave:03d}.claude.err"), "a") as fe,
        ):
            proc = subprocess.Popen(
                cmd, stdin=subprocess.PIPE, stdout=fo, stderr=fe, text=True, cwd=run
            )
            proc.stdin.write(
                json.dumps(
                    {"type": "user", "message": {"role": "user", "content": msg}}
                )
                + "\n"
            )
            proc.stdin.flush()  # stdin stays open, so the session waits for the workflow's completion notification
            while proc.poll() is None:
                if os.path.exists(out_path):
                    time.sleep(20)
                    break
                time.sleep(30)
                s2, _, t2 = scan(wave)
                hits = (
                    glob.glob(f"/tmp/claude-*/*/{s2}/tasks/{t2}.output")
                    if (s2 and t2)
                    else []
                )
                if (
                    hits
                    and os.path.getsize(hits[0]) > 200
                    and not os.path.exists(out_path)
                ):
                    try:  # fallback copy: the workflow finished but the model has not copied it yet
                        r = json.load(open(hits[0]))
                        if "results" in json.dumps(r)[:4000]:
                            subprocess.run(["cp", hits[0], out_path])
                            note(f"wave {wave}: copied workflow output from {hits[0]}")
                    except ValueError:
                        pass
            if proc.poll() is None:
                proc.stdin.close()
                try:
                    proc.wait(timeout=120)
                except subprocess.TimeoutExpired:
                    proc.kill()
        note(
            f"wave {wave} attempt {attempt} ended; output present: {os.path.exists(out_path)}"
        )
    return os.path.exists(out_path)


S = load_state()
while True:
    cur = next((w for w in S["waves"] if not w.get("persisted")), None)
    if cur is None:
        if a.max_waves and len(S["waves"]) >= a.max_waves:
            note(f"stopping: --max-waves {a.max_waves} reached")
            break
        nw = (
            [f"{HERE}/next_wave.py", "--run", run, str(a.wave_size)]
            + ([a.prios] if a.prios else [])
            + (["--second-pass"] if a.second_pass else [])
        )
        r = py(*nw)
        if r.returncode != 0:
            note(f"next_wave failed: {r.stderr[-400:]}")
            sys.exit(1)
        args = json.loads(r.stdout.strip().splitlines()[0])
        if not args["batches"]:
            note(
                "nothing pending: the hunt is complete. Next: recheck.py queue / recheck-wave.js, then consolidate."
            )
            break
        k = len(S["waves"]) + 1
        ap = os.path.join(raw_dir, f"wave{k:03d}.args.json")
        json.dump(args, open(ap, "w"))
        cur = {
            "wave": k,
            "args": ap,
            "out": os.path.join(raw_dir, f"wave{k:03d}.output.json"),
            "batches": len(args["batches"]),
        }
        S["waves"].append(cur)
        save_state(S)
        note(f"wave {k}: {cur['batches']} batches")
    if not run_wave(cur["wave"], cur["args"], cur["out"]):
        note(
            f"wave {cur['wave']}: GAVE UP after {a.max_attempts} attempts; rerun this command to resume it"
        )
        sys.exit(2)
    r = py(f"{HERE}/persist_wave.py", "--run", run, cur["out"])
    note(r.stdout.strip()[-1500:] or r.stderr[-800:])
    if r.returncode != 0:
        note(
            "persist failed: fix it and rerun this command; the wave output is safe on disk"
        )
        sys.exit(3)
    py(f"{HERE}/consolidate.py", "--run", run)
    cur["persisted"] = True
    save_state(S)
    if a.bench_cases:
        r = py(f"{HERE}/bench.py", "score", "--run", run, "--cases", a.bench_cases)
        note(" | ".join(ln for ln in r.stdout.splitlines() if "recall" in ln))
note(py(f"{HERE}/status.py", "--run", run).stdout)
