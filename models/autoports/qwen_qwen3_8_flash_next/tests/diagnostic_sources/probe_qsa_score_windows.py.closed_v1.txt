# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Exact-HF score/selection controls for failed full-context QSA windows.

Builds literal HF projection/norm/RoPE prefix fixtures without prefix attention,
then compares native and SFPU score arithmetic on identical HF operands. CPU
HF attention on the selected IDs measures selection impact after device readback.
These controls isolate score arithmetic; they are not full-device acceptance.
"""

import argparse
import json
import sys
from pathlib import Path
from unittest import mock

import torch

from models.autoports.qwen_qwen3_8_flash_next.tests.probe_qsa import load_reference
from models.autoports.qwen_qwen3_8_flash_next.tests.probe_qsa_context_window import prepare
from models.autoports.qwen_qwen3_8_flash_next.tests.probe_qsa_sfpu_scores import main as score_main


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path("/home/ttuser/qwen38-flash"))
    parser.add_argument(
        "--source-cache", type=Path, default=Path("/home/ttuser/qwen38-flash/qsa_cpu_prefill_262144.pt")
    )
    parser.add_argument("--windows", default="205504,165856,215424,222272")
    parser.add_argument("--key-block", type=int, default=8192)
    parser.add_argument("--cpu-only", action="store_true")
    args = parser.parse_args()
    windows = [int(value) for value in args.windows.split(",")]
    if not windows or any(prefix % 32 or not 2048 <= prefix <= 262112 for prefix in windows):
        parser.error("Windows must be aligned 32-row ranges within positions2048..262143")
    torch.set_num_threads(8)
    _, model, rope, _, provenance = load_reference(args.snapshot)
    source = torch.load(args.source_cache, weights_only=True, mmap=True, map_location="cpu")
    source_key = source["key"]
    report = {
        "scope": __doc__,
        "provenance": provenance,
        "source_key": source_key,
        "device_executed": not args.cpu_only,
        "windows": {},
    }
    args.root.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Prepare every window before any TTNN import, keeping the CPU-only contract.
    for prefix in windows:
        path = args.root / f"qsa_score_window_{prefix}.pt"
        if path.exists():
            fixture = torch.load(path, weights_only=True, mmap=True, map_location="cpu")
            assert fixture["source_key"] == source_key and fixture["provenance"] == provenance
            assert fixture["prefix"] == prefix and fixture["rows"] == 32 and fixture["capacity"] == 262144
        else:
            fixture = prepare(
                argparse.Namespace(
                    source_cache=args.source_cache,
                    snapshot=args.snapshot,
                    capacity=262144,
                    prefix=prefix,
                    rows=32,
                    fixture=path,
                ),
                model,
                rope,
                provenance,
            )
        report["windows"][str(prefix)] = {
            "fixture": str(path),
            "reference_control": fixture["rebuilt_reference_vs_saved"],
        }
        del fixture
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    assert "ttnn" not in sys.modules
    del model, rope, source
    if args.cpu_only:
        return
    for prefix in windows:
        output = args.output.with_name(f"{args.output.stem}_{prefix}.json")
        capture = args.root / f"{args.output.stem}_{prefix}.pt"
        command = [
            "probe_qsa_sfpu_scores",
            "--fixture",
            report["windows"][str(prefix)]["fixture"],
            "--snapshot",
            str(args.snapshot),
            "--key-block",
            str(args.key_block),
            "--output",
            str(output),
            "--capture",
            str(capture),
        ]
        with mock.patch.object(sys, "argv", command):
            score_main()
        report["windows"][str(prefix)].update(
            output=str(output), capture=str(capture), scores=json.loads(output.read_text())
        )
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
