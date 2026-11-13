#!/usr/bin/env python3
"""Execute the evidence-preserving handoff after the owned monitor pauses the grid."""
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

root = Path.cwd()
model = root / "models/autoports/google_gemma_4_31b_it"
base = model / "doc/benchmark"
sys.path.insert(0, str(model / "tools"))
from benchmark_server import idle, process, processes

python = str(root / "python_env/bin/python")


def group_members(group):
    found = []
    for entry in processes():
        try:
            if os.getpgid(entry["pid"]) == group:
                found.append(entry)
        except ProcessLookupError:
            pass
    return found


clock = json.loads((base / "setup/runner_clock.json").read_text())
original_end = clock["process_started_monotonic"] + clock["budget_seconds"]
marker = base / "setup/budget_handoff.json"
while not marker.exists():
    if time.monotonic() > original_end - 900:
        raise RuntimeError("No safe handoff before reserve")
    time.sleep(1)
handoff = json.loads(marker.read_text())
current = process(handoff["pid"])
assert current and current["start_ticks"] == handoff["start_ticks"]
assert os.getpgid(handoff["pid"]) == handoff["process_group"]
for _ in range(3):
    assert idle("http://127.0.0.1:8000"), "Outstanding work: leave group paused for coordinator"
    time.sleep(1)
members = group_members(handoff["process_group"])
assert all(
    "/benchmark_grid.py" in " ".join(p["command"])
    or "/benchmark_eval_client.py" in " ".join(p["command"])
    or "vllm_bench_cache_off.py" in " ".join(p["command"])
    for p in members
), members
handoff["idle_verified_unix"] = time.time()
handoff["terminated_members"] = members
os.killpg(handoff["process_group"], signal.SIGKILL)
for _ in range(60):
    if process(4818) is None:
        break
    time.sleep(1)
else:
    raise RuntimeError("Original native controller did not finish")
assert not group_members(handoff["process_group"])
assert json.loads((base / "run/summary.json").read_text())["status"] == "failed"
(base / "setup/budget_handoff.json").write_text(json.dumps(handoff, indent=2) + "\n")
subprocess.run(
    [
        python,
        str(model / "tools/benchmark_grid.py"),
        "--run-dir",
        str(base / "run"),
        "--concurrency",
        "32",
        "--action",
        "collect",
        "--grid-only",
        "--release-only",
        "--deadline-seconds",
        "6000",
    ],
    check=True,
)
subprocess.run([python, str(model / "tools/benchmark_server.py"), "--stop"], check=True)
archive = base / "attempts/budget-handoff"
assert not archive.exists()
shutil.move(str(base / "run"), archive)
shutil.copy2(base / "runner.log", archive / "outer-runner.log")
config = json.loads((base / "run_config.json").read_text())
config["roofline_command"] = [python, str(model / "tools/collect_benchmark_roofline.py")]
config["budget_seconds"] = int(
    min(
        original_end - time.monotonic(),
        json.loads((base / "stage_timing.json").read_text())["deadline_monotonic"] - time.monotonic(),
    )
    - 60
)
assert config["budget_seconds"] >= 900, config
(base / "run_config.json").write_text(json.dumps(config, indent=2) + "\n")
ledger = dict(
    original_native_end_monotonic=original_end,
    fallback_launch_monotonic=time.monotonic(),
    budget_seconds=config["budget_seconds"],
    closure_reserve_seconds=60,
    archived_attempt=str(archive),
    preserved_rows=[],
)
with (base / "runner.log").open("w") as log:
    runner = subprocess.Popen(
        [
            python,
            "-m",
            "benchmark_stage",
            "run",
            "--config",
            str(base / "run_config.json"),
            "--output",
            str(base / "run"),
        ],
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    ledger["fallback_pid"] = runner.pid
    while not (base / "run").exists():
        assert runner.poll() is None
        time.sleep(0.1)
    for source in sorted((archive / "grid").glob("*")):
        if not (source / "complete.json").exists():
            continue
        complete = json.loads((source / "complete.json").read_text())
        assert complete["row"]["id"] == source.name
        dest = base / "run/grid" / source.name
        shutil.copytree(source, dest)
        hashes = {
            str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in source.rglob("*")
            if p.is_file()
        }
        assert all(hashlib.sha256((dest / name).read_bytes()).hexdigest() == sha for name, sha in hashes.items())
        ledger["preserved_rows"].append(
            dict(
                row=source.name,
                original_bundle=str(source),
                discovery_copy=str(dest),
                sha256=hashes,
                absolute_path_note="Embedded old canonical paths resolve to the archived attempt, not the new server. Original identities and raw bytes retained.",
            )
        )
    (base / "setup/native_fallback.json").write_text(json.dumps(ledger, indent=2) + "\n")
    code = runner.wait()
ledger.update(
    exit_code=code,
    finished_monotonic=time.monotonic(),
    within_original_native_boundary=time.monotonic() <= original_end,
)
(base / "setup/native_fallback.json").write_text(json.dumps(ledger, indent=2) + "\n")
print(json.dumps({k: v for k, v in ledger.items() if k != "preserved_rows"}), flush=True)
sys.exit(code)
