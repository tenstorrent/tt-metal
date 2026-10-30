"""Serialize warmed layer profiles and retain exact report/calculation provenance."""

import argparse
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "doc/optimized_decoder"


def command(args, log, env=None):
    with log.open("w") as stream:
        proc = subprocess.Popen(args, stdout=stream, stderr=subprocess.STDOUT, env=env, start_new_session=True)
        try:
            code = proc.wait(timeout=600)
        except subprocess.TimeoutExpired:
            try:
                if "tracy" in args:
                    subprocess.run(
                        [
                            sys.executable,
                            "tools/tt-triage.py",
                            "--llm-output",
                            "--llm-output-path",
                            str(log.with_suffix(".triage.txt")),
                            "--triage-summary-path",
                            str(log.with_suffix(".triage_summary.txt")),
                        ],
                        stdout=stream,
                        stderr=subprocess.STDOUT,
                        timeout=60,
                    )
            finally:
                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait()
            raise RuntimeError(f"Timed out after triage; hardware ownership/health must be checked: {log}")
    if code:
        raise RuntimeError(f"Command exited {code}; inspect {log}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", default="tracy_final")
    parser.add_argument("--batches", nargs="+", type=int, default=[1, 32, 16, 8])
    parser.add_argument("--retain-batch1", action="store_true")
    parser.add_argument("--precision", default=None)
    parser.add_argument("--attention-precision", default=None)
    parser.add_argument("--geometry")
    parser.add_argument("--layers", nargs="+", type=int, default=[0, 5])
    options = parser.parse_args()
    provenance = OUT / options.artifact_root / "profile_provenance.json"
    records = json.loads(provenance.read_text()) if provenance.exists() else []
    for layer, kind in [(0, "sliding_attention"), (5, "full_attention")]:
        if layer not in options.layers:
            continue
        for batch in options.batches:
            path = OUT / options.artifact_root / kind / f"batch{batch}"
            path.mkdir(parents=True, exist_ok=True)
            env = dict(os.environ, TT_METAL_DEVICE_PROFILER="1", TT_METAL_TRACE_ALLOC_TRACKING="0")
            env.pop("TT_METAL_WATCHER", None)
            args = [
                sys.executable,
                "-m",
                "tracy",
                "-r",
                "-p",
                "-v",
                "--no-web-server",
                "--op-support-count",
                "16384",
                "-o",
                str(path),
                "-m",
                "models.autoports.google_gemma_4_31b_it.tests.profile_optimized_decoder",
                "--layer",
                str(layer),
                "--batch",
                str(batch),
            ]
            args += ["--result", str(path / "host_timing.json")]
            if options.precision:
                args += ["--precision", options.precision]
            if options.attention_precision:
                args += ["--attention-precision", options.attention_precision]
            if options.geometry:
                geometry_file = path / "geometry.json"
                geometry_file.write_text(json.dumps(json.loads(options.geometry), indent=2) + "\n")
                args += ["--geometry-file", str(geometry_file)]
            print("PROFILE_START", kind, batch, flush=True)
            source_hashes = {
                str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in [
                    ROOT / "tt/optimized_decoder.py",
                    ROOT / "tests/profile_optimized_decoder.py",
                    *ROOT.glob("tests/optimized_*candidate.py"),
                ]
            }
            manifest = path / "collection_manifest.json"
            if manifest.exists():
                saved = json.loads(manifest.read_text())
                if saved["source_hashes"] != source_hashes or saved["command"] != args:
                    raise RuntimeError(
                        "Existing profile is from different source; preserve it and choose a new artifact root"
                    )
                print("REUSE_COLLECTION", kind, batch, flush=True)
            else:
                command(args, path / "collection.log", env)
                collection_text = (path / "collection.log").read_text()
                if f"PROFILE_COMPLETE {kind} batch={batch} input=4096 decode_steps=128" not in collection_text:
                    raise RuntimeError(
                        "Tracy child did not complete; inspect collection.log even if launcher returned zero"
                    )
                if "markers were dropped" in collection_text:
                    raise RuntimeError("Profiler dropped markers; reject capture")
                manifest.write_text(
                    json.dumps(dict(command=args, source_hashes=source_hashes, exit_code=0), indent=2) + "\n"
                )
            csvs = list(path.rglob("ops_perf_results*.csv"))
            if not csvs:
                raise RuntimeError(f"No processed ops CSV under {path}")
            original = max(csvs, key=lambda p: p.stat().st_mtime_ns)
            ops = path / "ops.csv"
            shutil.copy2(original, ops)
            representative = path / "decode_representative_ops.csv"
            select_command = [
                sys.executable,
                "-m",
                "models.autoports.google_gemma_4_31b_it.tests.select_optimized_replay",
                str(ops),
                str(representative),
            ]
            command(select_command, path / "replay_selection.log")
            report_commands = [select_command]
            for phase in ["prefill", "decode"]:
                base = [
                    "tt-perf-report",
                    str(ops if phase == "prefill" else representative),
                    "--start-signpost",
                    "PERF_" + phase.upper(),
                    "--end-signpost",
                    "PERF_" + phase.upper() + "_END",
                    "--no-summary",
                    "--no-color",
                ]
                csv_command = base + ["--csv", str(path / (phase + "_perf_report.csv"))]
                command(csv_command, path / (phase + "_perf_report.console.log"))
                command(base, path / (phase + "_perf_report.txt"))
                report_commands.extend([csv_command, base])
            result = path / "performance.json"
            summarize = [
                sys.executable,
                "-m",
                "models.autoports.google_gemma_4_31b_it.tests.summarize_optimized_profile",
                str(ops),
                "--trace-host-data",
                str(path / ".logs/tracy_ops_data.csv"),
                "--layer-type",
                kind,
                "--batch",
                str(batch),
                "--output",
                str(result),
            ]
            if not options.geometry and options.precision is None and options.attention_precision is None:
                summarize += ["--expected-weight-dtype", "BFLOAT4_B", "--expected-fidelity", "LoFi"]
            command(summarize, path / "whole_layer_calculation.log")
            records.append(
                dict(
                    layer_type=kind,
                    batch=batch,
                    collection_command=args,
                    report_commands=report_commands,
                    calculation_command=summarize,
                    raw_csv=str(original),
                    copied_csv=str(ops),
                    sha256=hashlib.sha256(ops.read_bytes()).hexdigest(),
                    result=str(result),
                    environment={
                        "TT_METAL_DEVICE_PROFILER": "1",
                        "TT_METAL_TRACE_ALLOC_TRACKING": "0",
                        "TT_METAL_WATCHER": None,
                    },
                )
            )
            provenance.write_text(json.dumps(records, indent=2) + "\n")
            print("PROFILE_END", kind, batch, flush=True)


if __name__ == "__main__":
    main()
