# SPDX-License-Identifier: Apache-2.0
"""Serialize final stage gates with exact commands, environments and source hashes."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("gates", help="Comma-separated gate names; hardware work is strictly serial")
    parser.add_argument("--suffix", default="")
    parser.add_argument("--layer-filter", default="")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    evidence = root / "doc/optimized_decoder"
    module = "models.autoports.qwen_qwen3_8_flash_next.tests."
    test = str(root / "tests/test_optimized_decoder.py")
    pytest = [sys.executable, "-m", "pytest", "-s", "-q", "--timeout=0"]
    base = dict(os.environ)
    for key in list(base):
        if key.startswith("TT_METAL_WATCHER"):
            del base[key]
    base.update(
        QWEN38_WEIGHT_SOURCE="real",
        QWEN38_WEIGHT_CACHE="/home/vkovacevic/qwen38-flash/tt-metal-cache/qwen_qwen3_8_flash_next",
        TT_METAL_TRACE_ALLOC_TRACKING="1",
        TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE="0",
        OMP_NUM_THREADS="8",
    )
    gates = {
        "boundary": [
            (
                "final_boundaries_02",
                pytest + [test],
                dict(QWEN38_DECODER_BATCHES="1,2", QWEN38_DECODER_LENGTHS="1,2,3,4,5,31,32,33,127,128,129,259"),
            )
        ],
        "batch32": [
            (
                "final_batch32",
                pytest + [test],
                dict(
                    QWEN38_DECODER_BATCHES="32",
                    QWEN38_DECODER_LENGTHS="1,2,3,4,5,33,129",
                    QWEN38_DECODER_CHUNK_SIZE="8",
                    QWEN38_QSA_BATCH32_SHORT="1",
                ),
            )
        ],
        "ple_context": [
            (
                "decoder1_long_final",
                [sys.executable, "-m", module + "optimized_decoder1_long_context", "--warm-reference-pages"],
                {},
            )
        ],
        "qsa_context": [
            (
                "qsa_long_final",
                pytest + [str(root / "tests/test_optimized_qsa.py"), "-k", "hf_parity"],
                dict(
                    TT_QSA_LENGTHS="262143,262144",
                    TT_QSA_BATCHES="1",
                    TT_QSA_WEIGHTS="real",
                    TT_QSA_REFERENCE_CACHE="/home/vkovacevic/qwen38-flash/tt-metal-cache/qwen_qwen3_8_flash_next/qsa_reference/real_seed10_b1_s262144.pt",
                ),
            )
        ],
        "qsa_positions": [
            (
                "qsa_positions_final",
                pytest
                + [
                    str(root / "tests/test_optimized_qsa_positions.py"),
                    str(root / "tests/test_optimized_qsa.py"),
                    "-k",
                    "heterogeneous_positions or trace_request_slots",
                ],
                {},
            )
        ],
        "watcher": [
            (
                "watcher_final",
                pytest + [test],
                dict(
                    QWEN38_DECODER_BATCHES="1",
                    QWEN38_DECODER_LENGTHS="1,2,3,4,5,33",
                    TT_METAL_WATCHER="10",
                    TT_METAL_WATCHER_APPEND="1",
                    TT_METAL_LOGS_PATH=str(evidence / "watcher_final"),
                ),
            )
        ],
        "benchmarks": [
            (
                f"delivered_l{layer}",
                [
                    sys.executable,
                    "-m",
                    module + "benchmark_optimized_decoder",
                    "--impl",
                    "optimized",
                    "--layer",
                    str(layer),
                    "--tag",
                    f"delivered_l{layer}",
                ],
                dict(TT_METAL_TRACE_ALLOC_TRACKING="0"),
            )
            for layer in (0, 1, 3)
        ],
        "profiles": [
            (
                f"profiles_{impl}",
                [sys.executable, "-m", module + "profile_optimized_decoder", "--impl", impl],
                dict(TT_METAL_TRACE_ALLOC_TRACKING="0"),
            )
            for impl in ("fused", "optimized")
        ],
    }
    gates["steady_benchmarks"] = [
        (f"steady_{impl}_l{layer}", [sys.executable, "-m", module + "benchmark_optimized_decoder",
          "--impl", impl, "--layer", str(layer), "--tag", f"steady_{impl}_l{layer}",
          "--decode-position", "128", "--advance-position"], dict(TT_METAL_TRACE_ALLOC_TRACKING="0"))
        for impl in ("fused", "optimized") for layer in (0,1,3)
    ]
    gates["optimized_steady_benchmarks"] = [row for row in gates["steady_benchmarks"] if row[0].startswith("steady_optimized")]
    gates["default_chunk"] = [
        ("default_chunk_final", pytest + [test],
         dict(QWEN38_DECODER_BATCHES="1,2", QWEN38_DECODER_LENGTHS="511,512,513",
              QWEN38_DECODER_CHUNK_SIZE="512"))
    ]
    gates["gdn_context"] = [
        (
            "gdn_long_final",
            pytest + [str(root / "tests/optimized_gdn_long_context.py"), "-k", "layer0"],
            dict(QWEN38_GDN_LONG_BATCHES="1"),
        )
    ]
    gates["optimized_profiles"] = [
        (
            "profiles_optimized",
            [sys.executable, "-m", module + "profile_optimized_decoder", "--impl", "optimized"],
            dict(TT_METAL_TRACE_ALLOC_TRACKING="0"),
        )
    ]
    gates["qsa_slot_context"] = [
        (
            "qsa_slot_context",
            pytest + [str(root / "tests/test_optimized_qsa.py"), "-k", "hf_parity"],
            dict(
                TT_QSA_LENGTHS="262143,262144",
                TT_QSA_BATCHES="1",
                TT_QSA_WEIGHTS="real",
                TT_QSA_REQUEST_SLOTS="1",
                TT_QSA_REFERENCE_CACHE="/home/vkovacevic/qwen38-flash/tt-metal-cache/qwen_qwen3_8_flash_next/qsa_reference/real_seed10_b1_s262144.pt",
            ),
        )
    ]
    gates["watcher_qsa_slots"] = [
        (
            "watcher_qsa_slots",
            pytest + [str(root / "tests/test_optimized_qsa.py"), "-k", "trace_request_slots"],
            dict(
                TT_METAL_WATCHER="10",
                TT_METAL_WATCHER_APPEND="1",
                TT_METAL_LOGS_PATH=str(evidence / "watcher_qsa_slots"),
            ),
        )
    ]
    for gate in args.gates.split(","):
        for tag, command, overrides in gates[gate]:
            if args.suffix:
                tag += "_" + args.suffix
            if args.layer_filter and gate in ("boundary", "batch32", "watcher"):
                command = command + ["-k", args.layer_filter]
            if args.suffix and gate in ("benchmarks", "steady_benchmarks", "optimized_steady_benchmarks"):
                command = list(command)
                command[command.index("--tag") + 1] += "_" + args.suffix
            if args.suffix and gate in ("profiles", "optimized_profiles"):
                command = command + ["--prefix", args.suffix + "_"]
            if args.suffix and gate == "watcher":
                overrides = overrides | {"TT_METAL_LOGS_PATH": str(evidence / ("watcher_" + args.suffix))}
            if args.suffix and gate == "watcher_qsa_slots":
                overrides = overrides | {"TT_METAL_LOGS_PATH": str(evidence / ("watcher_qsa_slots_" + args.suffix))}
            env = base | overrides
            snapshot = evidence / (tag + "_source.py.txt")
            source = (root / "tt/optimized_decoder.py").read_bytes()
            snapshot.write_bytes(source)
            record = dict(
                command=command,
                environment={
                    k: v
                    for k, v in env.items()
                    if k.startswith(("QWEN38_", "TT_QSA_", "TT_METAL_")) or k == "OMP_NUM_THREADS"
                },
                runtime_sha256=hashlib.sha256(source).hexdigest(),
                source_snapshot=str(snapshot),
                started_unix=time.time(),
            )
            manifest = evidence / (tag + "_command.json")
            manifest.write_text(json.dumps(record, indent=2) + "\n")
            print("START", tag, flush=True)
            with (evidence / (tag + ".log")).open("w") as handle:
                result = subprocess.run(command, env=env, stdout=handle, stderr=subprocess.STDOUT)
            record.update(returncode=result.returncode, elapsed_seconds=time.time() - record["started_unix"])
            manifest.write_text(json.dumps(record, indent=2) + "\n")
            print("END", tag, result.returncode, record["elapsed_seconds"], flush=True)
            if result.returncode:
                raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
