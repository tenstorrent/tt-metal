# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU-only HiFi2 operand/quantization attribution; no device operations.

Original BF16 Q/K/V are the FP64 reference. Q is RNE7; HiFi2 consumes
all BF16/BFP8 K/V bits and truncates logical-left P to seven significant
bits. Online softmax uses exact FP64 exponentials and matched P sums.
This deliberately excludes the device's cheap-exp/subtraction errors.
"""

import argparse
import hashlib
import importlib.util
import json
import math
import platform
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


MODEL = module("hi2_b8_model", HERE / "numerics.py")
REPRO = module("hi2_b8_repro", ROOT / "tests/ttnn/unit_tests/operations/sdpa/repro_sdpa_l2.py")


def summary(actual, expected):
    a, e = actual.double().flatten(), expected.double().flatten()
    gain = float(a.dot(e) / e.dot(e))
    return dict(
        l2_pct=float(100 * (a - e).norm() / e.norm()),
        gain=gain,
        gain_corrected_l2_pct=float(100 * (a / gain - e).norm() / e.norm()),
        pcc=float(torch.nn.functional.cosine_similarity(a - a.mean(), e - e.mean(), dim=0)),
    )


def attention(q, k, v, p_rule):
    q = q.double()
    maximum = torch.full((q.shape[0], 1), -torch.inf, dtype=torch.float64)
    denominator = torch.zeros_like(maximum)
    original_denominator = torch.zeros_like(maximum)
    numerator = torch.zeros_like(q)
    for start in range(0, k.shape[0], 512):
        scores = q @ k[start : start + 512].double().T / math.sqrt(q.shape[-1])
        updated = torch.maximum(maximum, scores.amax(-1, keepdim=True))
        correction = (maximum - updated).exp()
        p = (scores - updated).exp()
        pe = p if p_rule == "exact" else MODEL.round_significand(p, 7, "trunc").double()
        numerator = numerator * correction + pe @ v[start : start + 512].double()
        denominator = denominator * correction + pe.sum(-1, keepdim=True)
        original_denominator = original_denominator * correction + p.sum(-1, keepdim=True)
        maximum = updated
    return numerator / denominator, float((denominator / original_denominator).mean())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True)
    parser.add_argument("--lengths", type=int, nargs="+", default=[4096, 32768, 262144])
    parser.add_argument("--seed", type=int, default=1240)
    args = parser.parse_args()
    torch.set_num_threads(4)
    started = time.monotonic()
    sources = [Path(__file__), HERE / "numerics.py", MODEL.V1 / "probe.py", Path(REPRO.__file__)]
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
                contract="CPU-only FP64 matmul/exp/state; Q RNE7; P trunc7 and matched denominator; full BF16/BFP8 K/V; BF16 output; native RNA vs host RNE BFP8 groups16 along D",
                source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
            )
        )
        for length in args.lengths:
            q, k, v = [x.squeeze().float() for x in REPRO.make_inputs(1, 128, length, 128, args.seed, "normal")]
            reference = REPRO.reference(q, k, v)
            qe = MODEL.round_significand(q, 7)
            encoded = {"bf16": (k, v)}
            for name, route in (("rna", "device"), ("rne", "host")):
                encoded[name] = tuple(MODEL.quantize(x, 7, route) for x in (k, v))
                emit(
                    dict(
                        kind="representation",
                        length=length,
                        rounding=name,
                        k=summary(encoded[name][0], k),
                        v=summary(encoded[name][1], v),
                    )
                )
            cases = [("bf16_kv", "bf16", "bf16"), ("rna_kv", "rna", "rna"), ("rne_kv", "rne", "rne")]
            if length == 32768:
                cases += [
                    (f"{route}_{side}_only", route if side == "k" else "bf16", route if side == "v" else "bf16")
                    for route in ("rna", "rne")
                    for side in ("k", "v")
                ]
            for name, kroute, vroute in cases:
                for p_rule in (["trunc7", "exact"] if length == 32768 else ["trunc7"]):
                    raw, mass = attention(qe, encoded[kroute][0], encoded[vroute][1], p_rule)
                    emit(
                        dict(
                            kind="attention",
                            length=length,
                            seed=args.seed,
                            heads=1,
                            query_rows=128,
                            distribution="normal",
                            variant=name,
                            p_rule=p_rule,
                            denominator="matched",
                            **summary(raw.bfloat16(), reference),
                            before_output_rounding=summary(raw, reference),
                            p_mass_ratio=mass,
                            q=summary(qe, q),
                        )
                    )
            emit(dict(kind="length_completed", length=length, elapsed_seconds=time.monotonic() - started))
        emit(dict(kind="completed", seconds=time.monotonic() - started))


if __name__ == "__main__":
    main()
