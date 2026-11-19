# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU-only re-evaluation of saved decoder outputs under the owner wrap-up policy."""

import argparse
import json
import sys
from pathlib import Path

import torch


def correlation(expected, actual):
    expected = expected.double().flatten()
    actual = actual.double().flatten()
    expected = expected - expected.mean()
    actual = actual - actual.mean()
    return float((expected * actual).sum() / (expected.square().sum() * actual.square().sum()).sqrt())


def metrics(expected, actual):
    expected = expected.reshape(-1, expected.shape[-1])
    actual = actual.reshape_as(expected)
    row_pcc = [correlation(x, y) for x, y in zip(expected, actual)]
    agreement = expected.argmax(-1) == actual.argmax(-1)
    return {
        "whole_output_pcc": correlation(expected, actual),
        "minimum_row_pcc": min(row_pcc),
        "finite": bool(torch.isfinite(expected).all() and torch.isfinite(actual).all()),
        "row_count": len(row_pcc),
        "rows_below_0_995": [
            {"row": i, "pcc": pcc, "hidden_coordinate_argmax_equal": bool(agreement[i])}
            for i, pcc in enumerate(row_pcc)
            if pcc < 0.995
        ],
        "rows_below_0_99": [i for i, pcc in enumerate(row_pcc) if pcc < 0.99],
        "hidden_coordinate_argmax_agreement": float(agreement.double().mean()),
    }


def load_rows(path, kind):
    data = torch.load(path, map_location="cpu", weights_only=False)
    if kind == "QSA":
        return data["rows"]
    rows = [
        dict(
            length=34,
            metric_index=0,
            expected=data["expected_prefill"],
            actual=data["actual_prefill"],
            residual=data["prompt"],
        )
    ]
    for i, (expected, actual, residual) in enumerate(
        zip(data["expected_decode"], data["actual_decode"], data["decode_inputs"])
    ):
        rows.append(dict(length=34, metric_index=i + 1, expected=expected, actual=actual, residual=residual))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    report = {
        "policy": "Owner wrap-up: whole-output/per-layer-kind aggregate PCC>=0.995; row diagnostic floor>=0.99; preserve raw historical gates.",
        "scope": "Saved real-weight PLE34 and QSA34/66 outputs, prefill and three decode steps each; not all stage correctness gates.",
        "token_top1": {
            "applicable": False,
            "measured": False,
            "reason": "These isolated decoder layers return hidden residual streams, not vocabulary logits. No LM head or complete model is present in this stage. Hidden-coordinate argmax is recorded only as a diagnostic and is not token top-1 agreement.",
        },
        "raw_probe_status": "Original probes remain failed under their stricter every-row whole/contribution>=0.995 predicate; no raw artifact was edited.",
        "layers": [],
    }
    for kind, prefix in (("PLE", "ple34_control"), ("QSA", "qsa_tail_control")):
        cache = args.cache_root / f"{prefix}_best_writer_tail2_v1.pt"
        rows = load_rows(cache, kind)
        original = load_rows(args.cache_root / f"{prefix}_best_exact_qkv_cleanup_v1.pt", kind)
        baseline = load_rows(args.cache_root / f"{prefix}_baseline_v1.pt", kind)
        layer = {"kind": kind, "device_cache": str(cache), "rows": []}
        for row, old, base in zip(rows, original, baseline):
            assert row["length"] == old["length"] == base["length"]
            assert row["metric_index"] == old["metric_index"] == base["metric_index"]
            assert torch.equal(row["expected"], old["expected"])
            assert torch.equal(row["expected"], base["expected"])
            assert torch.equal(row["residual"], old["residual"])
            assert torch.equal(row["residual"], base["residual"])
            layer["rows"].append(
                {
                    "length": row["length"],
                    "phase": "prefill" if row["metric_index"] == 0 else f"decode{row['metric_index']-1}",
                    "whole": metrics(row["expected"], row["actual"]),
                    "contribution": metrics(row["expected"] - row["residual"], row["actual"] - row["residual"]),
                    "functional_baseline": metrics(base["expected"], base["actual"]),
                    "byte_equal_to_previous_fused": bool(
                        torch.equal(
                            row["actual"].contiguous().view(torch.uint8), old["actual"].contiguous().view(torch.uint8)
                        )
                    ),
                }
            )
        for phase, selected in (("all", rows), ("prefill", [row for row in rows if row["metric_index"] == 0])):
            layer[f"{phase}_aggregate_whole_output_pcc"] = correlation(
                torch.cat([row["expected"].flatten() for row in selected]),
                torch.cat([row["actual"].flatten() for row in selected]),
            )
        layer["whole_output_and_row_pcc_passed"] = all(
            row["whole"]["whole_output_pcc"] >= 0.995
            and row["whole"]["minimum_row_pcc"] >= 0.99
            and row["whole"]["finite"]
            for row in layer["rows"]
        )
        report["layers"].append(layer)
    report["whole_output_and_row_pcc_passed"] = all(row["whole_output_and_row_pcc_passed"] for row in report["layers"])
    assert "ttnn" not in sys.modules
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {"output": str(args.output), "whole_output_and_row_pcc_passed": report["whole_output_and_row_pcc_passed"]}
        )
    )


if __name__ == "__main__":
    main()
