"""Run a reviewable candidate matrix with commands, hashes, and bounded recovery."""

import argparse
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


@contextmanager
def pause_reference(pid):
    active = pid and Path(f"/proc/{pid}/cmdline").exists()
    if active:
        assert (
            "record_decoder_activations" in Path(f"/proc/{pid}/cmdline").read_text()
        ), "Only this stage's CPU reference helper may be paused"
        os.kill(pid, signal.SIGSTOP)
    try:
        yield bool(active)
    finally:
        if active:
            try:
                os.kill(pid, signal.SIGCONT)
            except ProcessLookupError:
                pass


def main():
    p = argparse.ArgumentParser()
    p.add_argument("matrix")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--pause-cpu-helper", type=int)
    a = p.parse_args()
    matrix = Path(a.matrix)
    out = matrix.parent
    records = json.loads((out / "manifest.json").read_text()) if a.resume and (out / "manifest.json").exists() else []
    done = {(r["candidate"], r["phase"]) for r in records}
    for row in json.loads(matrix.read_text()):
        name = row["name"]
        layer = row.get("layer", 0)
        for phase in row.get("phases", ["correctness", "timing"]):
            if (name, phase) in done:
                continue
            module = "run_optimized_decoder" if phase == "correctness" else "time_optimized_candidate"
            cmd = [
                sys.executable,
                "-m",
                "models.autoports.google_gemma_4_31b_it.tests." + module,
                "--layer",
                str(layer),
                "--batch",
                str(row.get("batch", 1)),
                "--result",
                str(out / f"{name}_{phase}.json"),
            ]
            if "precision" in row:
                cmd += ["--precision", row["precision"]]
            if "attention_precision" in row:
                cmd += ["--attention-precision", row["attention_precision"]]
            if row.get("fused_control"):
                cmd += ["--fused-control"]
            if "geometry" in row:
                cmd += ["--layout", "--geometry", json.dumps(row["geometry"])]
            if phase == "correctness":
                cmd += ["--seq", str(row.get("seq", 33)), "--replays", str(row.get("replays", 3))]
                if row.get("inputs", True):
                    cmd += [
                        "--inputs",
                        row.get("inputs", f"bringup/artifacts/optimized_decoder_layer{layer}_inputs.pt"),
                    ]
                if row.get("extended"):
                    cmd += ["--extended", "--new-bucket-case"]
            env = dict(
                os.environ,
                TT_METAL_TRACE_ALLOC_TRACKING="1",
                TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE="0",
                **row.get("environment", {}),
            )
            hashes = {
                str(f): hashlib.sha256(f.read_bytes()).hexdigest()
                for f in [ROOT / "tt/optimized_decoder.py", *ROOT.glob("tests/*optimized*.py")]
            }
            print("START", name, phase, flush=True)
            with (
                pause_reference(a.pause_cpu_helper if phase == "timing" else None) as cpu_paused,
                (out / f"{name}_{phase}.log").open("w") as log,
            ):
                proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)
                try:
                    code = proc.wait(timeout=row.get("timeout", 180))
                except subprocess.TimeoutExpired:
                    try:
                        subprocess.run(
                            [
                                sys.executable,
                                "tools/tt-triage.py",
                                "--llm-output",
                                "--llm-output-path",
                                str(out / f"{name}_triage.txt"),
                                "--triage-summary-path",
                                str(out / f"{name}_triage_summary.txt"),
                            ],
                            stdout=log,
                            stderr=subprocess.STDOUT,
                            timeout=60,
                        )
                    except subprocess.TimeoutExpired:
                        pass
                    proc.kill()
                    proc.wait()
                    (out / f"{name}_timeout.json").write_text(
                        json.dumps(dict(command=cmd, status="timeout_after_triage", requires_recovery=True))
                    )
                    raise RuntimeError("Candidate stalled; queue stopped for device recovery")
            if env.get("TT_METAL_WATCHER"):
                watcher = Path("generated/watcher/watcher.log")
                if watcher.exists():
                    shutil.copy2(watcher, out / f"{name}_watcher.log")
            records.append(
                dict(
                    candidate=name,
                    phase=phase,
                    command=cmd,
                    exit_code=code,
                    configuration=row,
                    source_hashes=hashes,
                    cpu_reference_paused_for_timing=cpu_paused,
                    environment={k: v for k, v in env.items() if k.startswith("TT_METAL_")},
                )
            )
            (out / "manifest.json").write_text(json.dumps(records, indent=2) + "\n")
            print("END", name, phase, code, flush=True)
            if code:
                break


if __name__ == "__main__":
    main()
