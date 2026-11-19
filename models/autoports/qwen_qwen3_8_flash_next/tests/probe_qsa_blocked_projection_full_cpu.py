# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Full-context CPU projection equality for a single fixed accumulation policy.

Real BF16 products are FP32-exact for the observed normal operands. Accumulate
high then low member of each adjacent K pair with separate FP32 additions,
restart at K=1024 and2048, then add the three FP32 partials in order. Final
rounding is BF16. Every literal projection retains its original32-row HF call.
No selection or acceptance oracle changes are made, and no device is opened.
"""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import torch

from .probe_qsa import load_reference
from .probe_qsa_fp64_indexer_control import digest, save_tensors, write_json
from .probe_qsa_projection_orders_cpu import COORDINATES, blocked_high_low


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--source-cache", type=Path, required=True)
    parser.add_argument("--tensor-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(8)
    assert "ttnn" not in sys.modules
    started = time.monotonic()
    source = torch.load(args.source_cache, weights_only=True, mmap=True, map_location="cpu")
    _, model, _, _, provenance = load_reference(args.snapshot)
    assert source["key"]["reference_sha256"] == provenance["reference_sha256"]
    assert source["key"]["config_sha256"] == provenance["config_sha256"]
    assert source["key"]["snapshot"] == str(args.snapshot.resolve())
    length = source["key"]["length"]
    assert length == 262144 and source["hidden"].shape == (1, length, 2560)
    assert source["key"]["decode_steps"] == 0
    weight = model.indexer.index_qk_proj.weight
    assert weight.dtype == torch.bfloat16 and tuple(weight.shape) == (640, 2560)
    weight_t = weight.T.float()[:, None, :]
    literal_all = torch.empty(1, length, 640, dtype=torch.bfloat16)
    candidate_all = torch.empty_like(literal_all)
    mismatches = torch.zeros(length, dtype=torch.int32)
    report = {
        "scope": __doc__,
        "accepted": False,
        "device_executed": False,
        "status": "running",
        "identity": {
            "source_key": source["key"],
            "source_sha256": digest(args.source_cache),
            "harness_sha256": digest(Path(__file__)),
            "policy_source_sha256": digest(Path(__file__).with_name("probe_qsa_projection_orders_cpu.py")),
            "reference": provenance,
            "weight_bf16_sha256": hashlib.sha256(weight.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest(),
        },
        "literal_call_shape": [1, 32, 2560],
        "projection_shape": [1, 32, 640],
        "policy": {
            "K_blocks": [1024, 1024, 512],
            "within_block": "FP32 add high member then low member per adjacent pair",
            "partial_combine": "(partial0 + partial1) + partial2, FP32 each",
            "output": "BF16 RNE",
        },
        "rows_processed": 0,
        "mismatch_elements": 0,
        "mismatch_rows": 0,
        "first_mismatch_coordinates": [],
        "named_coordinates": [],
        "original_reference_selected_ids_sha256": source["reference_report"]["selected_ids_sha256"],
    }
    write_json(args.output, report)
    for start in range(0, length, 32):
        hidden = source["hidden"][:, start : start + 32]
        literal = model.indexer.index_qk_proj(hidden)
        products = (hidden[0].T.float()[:, :, None] * weight_t).contiguous()
        candidate_fp32 = blocked_high_low(products)
        candidate = candidate_fp32.bfloat16().unsqueeze(0)
        assert bool(torch.isfinite(candidate).all() and torch.isfinite(literal).all())
        different = candidate != literal
        mismatches[start : start + 32] = different.sum(-1)[0].int()
        literal_all[:, start : start + 32] = literal
        candidate_all[:, start : start + 32] = candidate
        report["mismatch_elements"] += int(different.sum())
        report["mismatch_rows"] += int(different.any(-1).sum())
        if len(report["first_mismatch_coordinates"]) < 32 and bool(different.any()):
            for _, row, column in different.nonzero().tolist():
                if len(report["first_mismatch_coordinates"]) == 32:
                    break
                report["first_mismatch_coordinates"].append(
                    {
                        "position": start + row,
                        "column": column,
                        "literal": float(literal[0, row, column]),
                        "candidate": float(candidate[0, row, column]),
                        "candidate_fp32": float(candidate_fp32[row, column]),
                    }
                )
        for position, column in COORDINATES:
            if start <= position < start + 32:
                row = position - start
                report["named_coordinates"].append(
                    {
                        "position": position,
                        "column": column,
                        "literal": float(literal[0, row, column]),
                        "candidate": float(candidate[0, row, column]),
                        "candidate_fp32": float(candidate_fp32[row, column]),
                    }
                )
        report["rows_processed"] = start + 32
        del products
        if (start + 32) % 16384 == 0:
            report["elapsed_seconds"] = time.monotonic() - started
            write_json(args.output, report)
            print(
                json.dumps(
                    {
                        key: report[key]
                        for key in ["rows_processed", "mismatch_elements", "mismatch_rows", "elapsed_seconds"]
                    }
                ),
                flush=True,
            )
    report["all_projection_outputs_bitexact"] = bool(torch.equal(literal_all, candidate_all))
    report["projection_mismatch_fraction"] = report["mismatch_elements"] / literal_all.numel()
    report["status"] = "completed_diagnostic"
    report["elapsed_seconds"] = time.monotonic() - started
    report["tensor_output"] = str(args.tensor_output)
    report["selection_recompute_scope"] = (
        "Unnecessary for this CPU policy: every indexer projection is bitexact literal; all downstream HF operations unchanged."
        if report["all_projection_outputs_bitexact"]
        else "Required for changed projections; not performed here."
    )
    save_tensors(
        args.tensor_output,
        {
            "identity": report["identity"],
            "report": report,
            "literal_projection": literal_all,
            "candidate_projection": candidate_all,
            "per_row_mismatch_count": mismatches,
        },
    )
    report["tensor_output_sha256"] = digest(args.tensor_output)
    assert "ttnn" not in sys.modules
    write_json(args.output, report)
    print(
        json.dumps(
            {
                key: report[key]
                for key in [
                    "status",
                    "mismatch_elements",
                    "mismatch_rows",
                    "all_projection_outputs_bitexact",
                    "elapsed_seconds",
                ]
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
