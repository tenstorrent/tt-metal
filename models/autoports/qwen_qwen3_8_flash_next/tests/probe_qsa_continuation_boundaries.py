# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""QSA pool/page/tile continuation cuts using the saved full-context HF oracle.

Only the first seq_len + 3 causal rows are consumed (68 by default). No HF cache
or projected tensor is injected into device state: each case runs real prefill
from empty state, continuation, traced decode, repeat, and dropped-history
negative controls. The source full-context fixture is opened read-only and
never overwritten. Use --seq-len 2053 --cuts 2047,2048,2049,2051,2052 to cover
the sparse-selection onset.
"""

import argparse
import json
from pathlib import Path

import torch

from models.autoports.qwen_qwen3_8_flash_next.tests.probe_qsa import load_reference
from models.autoports.qwen_qwen3_8_flash_next.tests.probe_qsa_prefill import run_device


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--source-cache", type=Path, default=Path("/home/ttuser/qwen38-flash/qsa_cpu_prefill_262144.pt")
    )
    parser.add_argument("--seq-len", type=int, default=65)
    parser.add_argument("--cuts", default="3,4,5,31,32,33")
    args = parser.parse_args()
    torch.set_num_threads(8)
    if not 2 <= args.seq_len <= 262144 - 3:
        parser.error("Sequence length must leave three decode tokens inside the 262144-row context")
    cuts = [int(value) for value in args.cuts.split(",")]
    if not cuts or any(cut < 1 or cut >= args.seq_len for cut in cuts):
        parser.error(f"Cuts must be inside the {args.seq_len}-row prompt")
    rows = args.seq_len + 3
    config, _, _, weights, provenance = load_reference(args.snapshot)
    fixture = torch.load(args.source_cache, weights_only=True, mmap=True, map_location="cpu")
    key = fixture["key"]
    for field in ("config_sha256", "reference_sha256", "transformers"):
        assert key[field] == provenance[field], field
    assert key["snapshot"] == str(args.snapshot.resolve())
    assert key["length"] == config.max_position_embeddings == 262144
    assert key["decode_steps"] == 0 and key["capacity"] == 262144 and key["seed"] == 1979
    for field in ("hidden", "expected"):
        assert fixture[field].shape == (1, 262144, 2560) and fixture[field].dtype == torch.bfloat16
    report = {
        "scope": "Real-weight QSA continuation cuts, empty initial device state; full physical context capacity",
        "provenance": provenance,
        "source_cache": str(args.source_cache),
        "source_key": key,
        "reference_semantics": f"First {rows} causal outputs from the unchanged all262144 HF reference",
        "seq_len": args.seq_len,
        "reference_rows": rows,
        "decode_steps": 3,
        "physical_capacity": 262144,
        "threshold": 0.995,
        "cuts": {},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        for cut in cuts:
            case = {}
            report["cuts"][str(cut)] = case
            run_device(
                config,
                weights,
                fixture["hidden"][:, :rows],
                fixture["expected"][:, :rows],
                length=args.seq_len,
                decode_steps=3,
                capacity=262144,
                continuation_cut=cut,
                report=case,
            )
            args.output.write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps({"cut": cut, "passed": case["passed"], "continuation": case["continuation"]}), flush=True)
        report["passed"] = all(case["passed"] for case in report["cuts"].values())
    except Exception as exc:
        report["passed"] = False
        report["error"] = f"{type(exc).__name__}: {str(exc).split('backtrace:')[0]}"
        raise
    finally:
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
