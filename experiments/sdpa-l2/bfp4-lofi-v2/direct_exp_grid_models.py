# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU native/direct exp-grid attribution with matched represented-P sums."""

import argparse
import hashlib
import importlib.util
import json
import math
import platform
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("direct_grid_exp", HERE / "exp_refiner_models.py")
EXP = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EXP)
MODEL, REPRO, BASE = EXP.MODEL, EXP.REPRO, EXP.BASE


def native_grid(delta):
    a = np.float32(np.float32(256) * np.float32(1.4426950408889634) * EXP.SCALE)
    b = np.float32(32500.818359375)
    transformed = (delta.float().double() * float(a) + float(b)).float()
    magnitude = torch.floor(transformed.double().abs() + 0.5).clamp_max(32767).int()
    linear = (magnitude << 15).contiguous().view(torch.float32)
    return torch.where(transformed >= 0, linear, 0).double()


def outputs(scores, v, coefficients):
    score = scores.reshape(128, -1, 512)
    maximum = score.amax(-1, keepdim=True).cummax(1).values
    delta = score - maximum
    correction = ((maximum - maximum[:, -1:]) / math.sqrt(128)).exp()
    linear, mantissa, _, _ = EXP.grid(delta)
    candidates = {
        "native_grid8": native_grid(delta),
        "direct_grid10": (linear.double() * 2.0**96).float().clamp_min(0).double(),
        "effective_p7_degree3": EXP.polynomial(linear, mantissa, coefficients["effective_p7_degree3"]),
    }
    result = {}
    for name, p in candidates.items():
        consumed = MODEL.round_significand(p, 7, "trunc").double()
        weighted = (consumed * correction).reshape(128, -1)
        result[name] = weighted @ v.double() / weighted.sum(-1, keepdim=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True)
    parser.add_argument("--lengths", nargs="+", type=int, default=[4096, 32768])
    parser.add_argument("--seed", type=int, default=1240)
    args = parser.parse_args()
    torch.set_num_threads(4)
    coefficients, _ = EXP.fits()
    previous = [json.loads(x) for x in (HERE / "exp-refiner-v1.jsonl").read_text().splitlines()]
    started, count = time.monotonic(), 0
    sources = [Path(__file__), Path(EXP.__file__), EXP.HEADER, HERE / "exp-refiner-v1.jsonl"]
    with (HERE / (args.label + ".jsonl")).open("x") as output:

        def emit(record):
            line = json.dumps(record, allow_nan=False)
            output.write(line + "\n")
            output.flush()
            print(line, flush=True)

        emit(
            dict(
                kind="provenance",
                args=vars(args),
                hostname=platform.node(),
                threads=4,
                contract="CPU FP32 grid emulation; FP64 QK/subtraction, online rescaling and PV/state; matched P trunc7; BF16 output; native8-bit no-refiner vs10-bit direct grid and compensated cubic",
                source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
            )
        )
        delta = torch.linspace(-20 * math.sqrt(128), 0, 65536, dtype=torch.float64)
        linear, _, _, _ = EXP.grid(delta)
        truth = (delta / math.sqrt(128)).exp()
        for name, p in (
            ("native_grid8", native_grid(delta)),
            ("direct_grid10", (linear.double() * 2.0**96).float().clamp_min(0).double()),
        ):
            ratio = p / truth
            gain = float(ratio.mean())
            relative = ratio / gain - 1
            emit(
                dict(
                    kind="scalar",
                    refiner=name,
                    common_relative_gain=gain,
                    normalized_relative_rms_pct=float(100 * relative.square().mean().sqrt()),
                    normalized_max_relative_pct=float(100 * relative.abs().max()),
                )
            )
        for length in args.lengths:
            for distribution in ("normal", "scaled_qk", "outliers", "common_k_centered"):
                q, k, v = [
                    x.squeeze().float()
                    for x in REPRO.make_inputs(
                        1,
                        128,
                        length,
                        128,
                        args.seed,
                        "common_k" if distribution == "common_k_centered" else distribution,
                    )
                ]
                reference = REPRO.reference(q, k, v)
                if distribution == "common_k_centered":
                    k = (k.double() - k.double().mean(0, keepdim=True)).float()
                for operands in ("exact_qk", "q7_kv5", "q7_kv4"):
                    qe = q if operands == "exact_qk" else MODEL.round_significand(q, 7)
                    ke, ve = k, v
                    if operands == "q7_kv5":
                        ke, ve = [MODEL.round_significand(x, 5) for x in (k, v)]
                    if operands == "q7_kv4":
                        ke, ve = [MODEL.quantize(x, 3, "host") for x in (k, v)]
                    results = outputs(qe.double() @ ke.double().T, ve, coefficients)
                    current = results["effective_p7_degree3"].bfloat16()
                    check = next(
                        r
                        for r in previous
                        if r.get("kind") == "attention"
                        and r["length"] == length
                        and r["distribution"] == distribution
                        and r["operands"] == operands
                        and r["refiner"] == "effective_p7_degree3"
                        and r["seed"] == args.seed
                    )
                    assert (
                        BASE.summary(current, reference)["l2_pct"] == check["l2_pct"]
                    ), "Previous cubic control must reproduce exactly"
                    for name, raw in results.items():
                        actual = raw.bfloat16()
                        emit(
                            dict(
                                kind="attention",
                                length=length,
                                seed=args.seed,
                                distribution=distribution,
                                operands=operands,
                                refiner=name,
                                **BASE.summary(actual, reference),
                                vs_current_cubic_l2_pct=BASE.summary(actual, current)["l2_pct"],
                                before_output_rounding_l2_pct=BASE.summary(raw, reference)["l2_pct"],
                            )
                        )
                        count += 1
            emit(dict(kind="length_completed", length=length, cases=count, seconds=time.monotonic() - started))
        emit(dict(kind="completed", cases=count, seconds=time.monotonic() - started))


if __name__ == "__main__":
    main()
