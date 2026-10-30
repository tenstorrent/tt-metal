"""Serialize warmed layer profiles and retain exact report/calculation provenance."""

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "doc/functional_decoder"


def command(args, log, env=None):
    with log.open("w") as stream:
        result = subprocess.run(args, stdout=stream, stderr=subprocess.STDOUT, env=env)
    if result.returncode:
        raise RuntimeError(f"Command exited {result.returncode}; inspect {log}")


def main():
    records = []
    for layer, kind in [(0, "sliding_attention"), (5, "full_attention")]:
        for batch in [1, 32, 16, 8]:
            path = OUT / "tracy" / kind / f"batch{batch}"
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
                "-o",
                str(path),
                "-m",
                "models.autoports.google_gemma_4_31b_it.tests.profile_decoder",
                "--layer",
                str(layer),
                "--batch",
                str(batch),
            ]
            print("PROFILE_START", kind, batch, flush=True)
            source_hashes = {
                str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in [ROOT / "tt/functional_decoder.py", ROOT / "tests/profile_decoder.py"]
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
                manifest.write_text(
                    json.dumps(dict(command=args, source_hashes=source_hashes, exit_code=0), indent=2) + "\n"
                )
            csvs = list(path.rglob("ops_perf_results*.csv"))
            if not csvs:
                raise RuntimeError(f"No processed ops CSV under {path}")
            original = max(csvs, key=lambda p: p.stat().st_mtime_ns)
            ops = path / "ops.csv"
            shutil.copy2(original, ops)
            report_commands = []
            for phase in ["prefill", "decode"]:
                base = [
                    "tt-perf-report",
                    str(ops),
                    "--start-signpost",
                    "PERF_" + phase.upper(),
                    "--end-signpost",
                    "PERF_" + phase.upper() + "_END",
                    "--no-advice",
                    "--no-summary",
                    "--no-color",
                ]
                csv_command = base + ["--csv", str(path / (phase + "_perf_report.csv"))]
                command(csv_command, path / (phase + "_perf_report.console.log"))
                command(base, path / (phase + "_perf_report.txt"))
                report_commands.extend([csv_command, base])
            result = OUT / f"performance_{kind}_batch{batch}.json"
            summarize = [
                sys.executable,
                "-m",
                "models.autoports.google_gemma_4_31b_it.tests.summarize_profile",
                str(ops),
                "--layer-type",
                kind,
                "--batch",
                str(batch),
                "--output",
                str(result),
            ]
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
            (OUT / "profile_provenance.json").write_text(json.dumps(records, indent=2) + "\n")
            print("PROFILE_END", kind, batch, flush=True)


if __name__ == "__main__":
    main()
