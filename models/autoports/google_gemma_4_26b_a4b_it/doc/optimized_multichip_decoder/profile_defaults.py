# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Serialize final-default captures and retain commands, tables and provenance."""

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

root = Path(__file__).resolve().parent
for layer, kind in ((0, "sliding"), (5, "full")):
    out = root / ("profile_final_" + kind)
    out.mkdir(exist_ok=True)
    commands = []
    native_hash = hashlib.sha256()
    with Path("build/lib/_ttnncpp.so").open("rb") as native:
        for chunk in iter(lambda: native.read(1024 * 1024), b""):
            native_hash.update(chunk)

    def run(command, log):
        commands.append(command)
        with Path(log).open("w") as stream:
            result = subprocess.run(
                command,
                stdout=stream,
                stderr=subprocess.STDOUT,
                env={**os.environ, "HF_HUB_OFFLINE": "1"},
                timeout=1200,
            )
        (out / "commands.json").write_text(json.dumps(commands, indent=2) + "\n")
        if result.returncode:
            raise RuntimeError(f"{command} exited {result.returncode}; see {log}")

    command = [
        sys.executable,
        "-m",
        "tracy",
        "-r",
        "-p",
        "-v",
        "--op-support-count",
        "100000",
        "--no-op-info-cache",
        "--disable-device-data-dump-to-files",
        "--disable-device-data-push-to-tracy",
        "-o",
        str(out),
        "-n",
        "tp4",
        "-m",
        "models.autoports.google_gemma_4_26b_a4b_it.tests.run_multichip_decoder",
        "--tp",
        "4",
        "--layer",
        str(layer),
        "--length",
        "4096",
        "--steps",
        "128",
        "--trace",
        "--profile",
        "--prefill-timing-samples",
        "1",
        "--output",
        str(out / "run.json"),
    ]
    print("CAPTURE", kind, flush=True)
    run(command, root / ("profile_final_" + kind + ".log"))
    (out / "capture_command.json").write_text(
        json.dumps({"command": command, "exit_code": 0, "watcher": False}, indent=2) + "\n"
    )
    source = next(out.glob("reports/tp4/*/ops_perf_results*.csv"))
    for phase in ("prefill", "decode"):
        base = [
            str(Path(sys.executable).parent / "tt-perf-report"),
            str(source),
            "--start-signpost",
            "PERF_" + phase.upper(),
            "--end-signpost",
            "PERF_" + phase.upper() + "_END",
        ]
        if phase == "decode":
            base += ["--tracing-mode", "--active-experts", "8"]
        run(base + ["--csv", str(out / (phase + "_perf_report.csv"))], out / (phase + "_csv.log"))
        run(base + ["--no-summary"], out / (phase + "_table.txt"))
    run(
        [
            sys.executable,
            "-m",
            "models.autoports.google_gemma_4_26b_a4b_it.tests.summarize_multichip_perf",
            str(source),
            "--layer-type",
            kind + "_attention",
            "--steps",
            "128",
            "--precision-policy",
            "Stage05 final default; native dtype and fidelity rows authoritative",
            "--output",
            str(out / "whole_layer.json"),
        ],
        out / "accounting.log",
    )
    run([sys.executable, str(root / "audit_profile.py"), str(out)], out / "capture_audit.log")
    files = [source] + [p for p in out.iterdir() if p.is_file() and p.name != "provenance.json"]
    (out / "provenance.json").write_text(
        json.dumps(
            {
                "native_binary_sha256": native_hash.hexdigest(),
                "native_binary": "build/lib/_ttnncpp.so",
                "native_kernel_sha256": hashlib.sha256(
                    Path(
                        "ttnn/cpp/ttnn/operations/experimental/ccl/all_gather_async/device/kernels/minimal_default_writer.cpp"
                    ).read_bytes()
                ).hexdigest(),
                "native_kernel": "ttnn/cpp/ttnn/operations/experimental/ccl/all_gather_async/device/kernels/minimal_default_writer.cpp",
                "files": [
                    dict(path=str(p), bytes=p.stat().st_size, sha256=hashlib.sha256(p.read_bytes()).hexdigest())
                    for p in files
                ],
            },
            indent=2,
        )
        + "\n"
    )
    print("DONE", kind, flush=True)
