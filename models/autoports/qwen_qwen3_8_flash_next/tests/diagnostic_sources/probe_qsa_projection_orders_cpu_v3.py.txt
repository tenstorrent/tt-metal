# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Fixed FP32 reduction-order controls, never an alternative acceptance oracle.

Every policy applies to every dot in full original 32x640 indexer projections.
Independent holdout chunks are selected by a recorded PRNG before evaluation.
Pair fused-emulation computes three-term sums in FP64 before FP32 rounding; it
is an arithmetic hypothesis, not a claim about any CPU or TT instruction.
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from .probe_qsa import load_reference
from .probe_qsa_fp64_indexer_control import digest, save_tensors, write_json

CRITICAL = [105568, 58528, 93024, 100032, 16736, 9184, 7840]
COORDINATES = [(105584, 328), (58531, 244), (9190, 595), (7842, 514)]


def tree(value, adjacent=False):
    tails = []
    while len(value) > 1:
        n = len(value)
        if n % 2:
            tails.append(value[-1])
        half = n // 2
        value = value[: 2 * half : 2] + value[1 : 2 * half : 2] if adjacent else value[:half] + value[half : 2 * half]
    result = value[0]
    for tail in tails:
        result = result + tail
    return result


def grouped(value, size):
    groups = value.reshape(-1, size, *value.shape[1:]).transpose(0, 1)
    reduced = tree(groups)
    result = torch.zeros_like(reduced[0])
    for item in reduced:
        result = result + item
    return result


def sequential_pairs(value, lanes, fused):
    accumulators = [torch.zeros_like(value[0]) for _ in range(lanes)]
    for index in range(0, len(value), 2):
        lane = (index // 2) % lanes
        if fused:
            accumulators[lane] = (
                accumulators[lane].double() + value[index].double() + value[index + 1].double()
            ).float()
        else:
            accumulators[lane] = accumulators[lane] + (value[index] + value[index + 1])
    return tree(torch.stack(accumulators))


def high_low_pairs(value, lanes):
    """Two separately rounded FP32 additions, high pair member then low."""
    accumulators = [torch.zeros_like(value[0]) for _ in range(lanes)]
    for index in range(0, len(value), 2):
        lane = (index // 2) % lanes
        accumulators[lane] = accumulators[lane] + value[index + 1]
        accumulators[lane] = accumulators[lane] + value[index]
    return tree(torch.stack(accumulators))


def blocked_high_low(value, block=1024):
    """Independent high/low pair partials, then ordered FP32 partial adds."""
    partials = [high_low_pairs(value[start : start + block], 1) for start in range(0, len(value), block)]
    result = partials[0]
    for partial in partials[1:]:
        result = result + partial
    return result


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--source-cache", type=Path, required=True)
    parser.add_argument("--tensor-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--include-high-low", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(8)
    assert "ttnn" not in sys.modules
    begin = time.monotonic()
    source = torch.load(args.source_cache, weights_only=True, mmap=True, map_location="cpu")
    _, model, _, _, provenance = load_reference(args.snapshot)
    assert source["key"]["reference_sha256"] == provenance["reference_sha256"]
    assert source["key"]["config_sha256"] == provenance["config_sha256"]
    assert source["key"]["snapshot"] == str(args.snapshot.resolve())
    assert source["key"]["length"] == 262144 and source["hidden"].shape == (1, 262144, 2560)
    seed = 120917
    rng = random.Random(seed)
    heldout = rng.sample([x for x in range(0, 262144, 32) if x not in CRITICAL], 8)
    policies = {"half_split_tree": lambda x: tree(x), "adjacent_tree": lambda x: tree(x, True)}
    for size in [32, 64, 128, 256]:
        policies[f"group{size}_tree_then_sequential"] = lambda x, size=size: grouped(x, size)
    for lanes in [1, 2, 4, 8, 16]:
        for fused in [False, True]:
            policies[
                f"pair_{lanes}lanes_" + ("one_round_emulation" if fused else "two_round")
            ] = lambda x, lanes=lanes, fused=fused: sequential_pairs(x, lanes, fused)
    if args.include_high_low:
        for lanes in [1, 2, 4, 8, 16]:
            policies[f"pair_{lanes}lanes_high_then_low"] = lambda x, lanes=lanes: high_low_pairs(x, lanes)
    if args.include_high_low:
        policies["block1024_high_low_then_partial_add"] = blocked_high_low
    report = {
        "scope": __doc__,
        "accepted": False,
        "device_executed": False,
        "status": "running",
        "identity": {
            "source_key": source["key"],
            "source_sha256": digest(args.source_cache),
            "harness_sha256": digest(Path(__file__)),
            "reference": provenance,
        },
        "critical_chunks": CRITICAL,
        "holdout_seed": seed,
        "holdout_chunks": heldout,
        "policies": list(policies) + ["fp32_linear", "fp64_rounded"],
        "chunks": [],
        "coordinates": [],
        "aggregate": {},
        "selection_rule": "Compare critical and independent holdout mismatch totals separately; no row-specific policy selection.",
    }
    saved = {"identity": report["identity"], "chunks": {}}
    weight = model.indexer.index_qk_proj.weight
    weight64 = weight.double()
    for start in CRITICAL + heldout:
        hidden = source["hidden"][:, start : start + 32]
        literal = F.linear(hidden, weight)[0]
        assert torch.equal(literal, F.linear(hidden, weight)[0])
        exact = F.linear(hidden.double(), weight64)[0]
        products = (hidden[0].T.float()[:, :, None] * weight.T.float()[:, None, :]).contiguous()
        outputs = {name: function(products) for name, function in policies.items()}
        outputs["fp32_linear"] = F.linear(hidden.float(), weight.float())[0]
        outputs["fp64_rounded"] = exact
        item = {
            "start": start,
            "partition": "critical" if start in CRITICAL else "holdout",
            "elements": literal.numel(),
            "metrics": {},
        }
        for name, output in outputs.items():
            different = output.bfloat16() != literal
            item["metrics"][name] = {
                "mismatches": int(different.sum()),
                "mismatch_fraction": float(different.float().mean()),
                "max_abs_fp64_dot_error": float((output.double() - exact).abs().max()),
            }
            totals = report["aggregate"].setdefault(
                name, {"critical": 0, "holdout": 0, "named_coordinate_mismatches": 0}
            )
            totals[item["partition"]] += int(different.sum())
        for position, column in COORDINATES:
            if start <= position < start + 32:
                row = position - start
                entry = {
                    "position": position,
                    "column": column,
                    "literal": float(literal[row, column]),
                    "fp64_dot": float(exact[row, column]),
                    "policies": {},
                }
                for name, output in outputs.items():
                    rounded = output[row, column].bfloat16()
                    matches = bool(rounded == literal[row, column])
                    entry["policies"][name] = {
                        "dot": float(output[row, column]),
                        "rounded": float(rounded),
                        "matches_literal": matches,
                    }
                    report["aggregate"][name]["named_coordinate_mismatches"] += int(not matches)
                report["coordinates"].append(entry)
        report["chunks"].append(item)
        saved["chunks"][start] = {"literal": literal, "fp64": exact, "outputs": outputs}
        write_json(args.output, report)
        print(
            json.dumps(
                {
                    "chunk": start,
                    "elapsed_seconds": time.monotonic() - begin,
                    "mismatches": {k: v["mismatches"] for k, v in item["metrics"].items()},
                }
            ),
            flush=True,
        )
        del products
    baseline = report["aggregate"]["half_split_tree"]
    for name, totals in report["aggregate"].items():
        totals["critical_fraction"] = totals["critical"] / (len(CRITICAL) * 32 * 640)
        totals["holdout_fraction"] = totals["holdout"] / (len(heldout) * 32 * 640)
        totals["candidate_for_followup"] = (
            totals["critical"] < baseline["critical"]
            and totals["holdout"] < baseline["holdout"]
            and totals["named_coordinate_mismatches"] == 0
        )
    report.update(
        status="completed_diagnostic",
        elapsed_seconds=time.monotonic() - begin,
        tensor_output=str(args.tensor_output),
        no_backend_instruction_claim=True,
    )
    saved["report"] = report
    save_tensors(args.tensor_output, saved)
    report["tensor_output_sha256"] = digest(args.tensor_output)
    assert "ttnn" not in sys.modules
    write_json(args.output, report)
    print(json.dumps({"aggregate": report["aggregate"], "elapsed_seconds": report["elapsed_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
