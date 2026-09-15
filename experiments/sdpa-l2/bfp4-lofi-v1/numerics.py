# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Quantization-only attention ablations; FP64 arithmetic, NOT kernel timing.

Q128, D128, K512 online blocks. Quantizer must first be validated against the
device pack path that will produce each operand. BF16 inputs/output; no claim
that this models all finite-precision arithmetic in a future fused kernel.
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

from probe import metrics, quantize

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SPEC = importlib.util.spec_from_file_location("frontier", HERE.parent / "frontier-accuracy-v1/run.py")
FRONTIER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(FRONTIER)

# Magnitude-bit widths, not total storage widths. None means unquantized.
VARIANTS = {
    "all4": (3, 3, 3, 3),
    "q8k4p8v4": (7, 3, 7, 3),
    "qk4_only": (3, 3, None, None),
    "q8k4_only": (7, 3, None, None),
    "pv4_only": (None, None, 3, 3),
    "p8v4_only": (None, None, 7, 3),
    "unquantized": (None, None, None, None),
}


def hadamard(x, signs):
    y = x.double() * signs
    width = y.shape[-1]
    stride = 1
    while stride < width:
        z = y.reshape(*y.shape[:-1], -1, 2, stride)
        a, b = z[..., 0, :], z[..., 1, :]
        y = torch.stack((a + b, a - b), dim=-2).reshape(x.shape)
        stride *= 2
    return y / math.sqrt(width)


def preprocess(inputs, kind, seed):
    q, k, v = [x.squeeze().double() for x in inputs]
    q0 = torch.zeros((1, 128), dtype=torch.double)
    v0 = torch.zeros((1, 128), dtype=torch.double)
    if kind.startswith("center"):
        k = k - k.mean(0, keepdim=True)
        if kind in ("center_qk", "center_qkv", "center_qk_hadamard"):
            q0 = q.mean(0, keepdim=True)
            q = q - q0
        if kind == "center_qkv":
            v0 = v.mean(0, keepdim=True)
            v = v - v0
    # Global Q correction evaluated accurately using unquantized centered K.
    correction = q0 @ k.T / math.sqrt(128)
    if kind in ("hadamard", "center_qk_hadamard"):
        gen = torch.Generator().manual_seed(seed + 7000)
        signs = torch.randint(0, 2, (128,), generator=gen).double() * 2 - 1
        q, k = hadamard(q, signs), hadamard(k, signs)
    return q, k, v, correction, v0


def evaluate(prepared, widths, qkv_round, p_round, reference, block=512):
    q, k, v, correction, v0 = prepared
    qb, kb, pb, vb = widths
    q, k, v = [quantize(x, bits, qkv_round).double() if bits else x for x, bits in ((q, qb), (k, kb), (v, vb))]
    logits = q @ k.T / math.sqrt(128) + correction
    scores = logits.reshape(q.shape[0], -1, block)
    # Algebraically emulate sequential online maxima/rescaling in FP64.
    maxima = scores.amax(-1, keepdim=True).cummax(1).values
    p = (scores - maxima).exp()
    packed = quantize(p, pb, p_round).double() if pb else p
    alpha_to_final = (maxima - maxima[:, -1:]).exp()
    effective = (packed * alpha_to_final).reshape(q.shape[0], -1)
    numerator = effective @ v
    original_denominator = (p * alpha_to_final).sum((1, 2))[:, None]
    represented_denominator = effective.sum(-1, keepdim=True)
    records = []
    for denominator, value in (("original", original_denominator), ("matched", represented_denominator)):
        raw = numerator / value + v0
        actual = raw.bfloat16().double()
        row_l2 = 100 * (actual - reference).norm(dim=-1) / reference.norm(dim=-1).clamp_min(1e-300)
        records.append(
            dict(
                denominator=denominator,
                **metrics(actual, reference),
                row_p95_l2_pct=float(torch.quantile(row_l2, 0.95)),
                before_output_rounding_l2_pct=metrics(raw, reference)["l2_pct"],
                p_mass_ratio_mean=float((represented_denominator / original_denominator).mean()),
                p_zero_fraction=float((packed == 0).double().mean()),
            )
        )
    return records


def self_test():
    gen = torch.Generator().manual_seed(11)
    inputs = [torch.randn((1, 1, n, 128), generator=gen).bfloat16() for n in (128, 1024, 1024)]
    q, k, v = [x.squeeze().double() for x in inputs]
    ref = torch.softmax(q @ k.T / math.sqrt(128), dim=-1) @ v
    for kind in ("none", "center_k", "center_qk", "center_qkv", "hadamard", "center_qk_hadamard"):
        prepared = preprocess(inputs, kind, 11)
        records = evaluate(prepared, VARIANTS["unquantized"], "rne", "rne", ref)
        # Output BF16 rounding should be the only material error.
        for record in records:
            assert record["before_output_rounding_l2_pct"] < 1e-10, (kind, record)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    parser.add_argument("--qkv-round", required=True)
    parser.add_argument("--p-round", required=True)
    parser.add_argument("--lengths", nargs="+", type=int, default=[4096, 32768])
    parser.add_argument(
        "--distributions", nargs="+", default=["normal", "outliers", "common_q", "common_k", "common_v"]
    )
    parser.add_argument(
        "--preprocessing", nargs="+", default=["none", "center_k", "center_qk", "center_qkv", "hadamard"]
    )
    parser.add_argument("--seed", type=int, default=1240)
    args = parser.parse_args()
    torch.set_num_threads(8)
    self_test()
    path = HERE / (args.label + ".jsonl")
    assert not path.exists(), "Use a fresh label"
    started = time.monotonic()
    with path.open("x") as output:

        def emit(record):
            line = json.dumps(record, allow_nan=False)
            output.write(line + "\n")
            output.flush()
            print(line, flush=True)

        emit(
            dict(
                kind="provenance",
                hostname=platform.node(),
                arguments=vars(args),
                measurement="CPU quantization model with FP64 arithmetic; no device performance",
                source_sha256={
                    p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(__file__), HERE / "probe.py")
                },
            )
        )
        for length in args.lengths:
            assert length % 512 == 0
            for distribution in args.distributions:
                inputs = FRONTIER.inputs_for(length, args.seed, distribution)
                q, k, v = [x.squeeze().double() for x in inputs]
                ref = torch.softmax(q @ k.T / math.sqrt(128), dim=-1) @ v
                hashes = [hashlib.sha256(x.view(torch.uint16).numpy().tobytes()).hexdigest() for x in inputs]
                for preprocessing in args.preprocessing:
                    prepared = preprocess(inputs, preprocessing, args.seed)
                    for variant, widths in VARIANTS.items():
                        if preprocessing != "none" and variant not in ("all4", "q8k4p8v4"):
                            continue
                        for result in evaluate(prepared, widths, args.qkv_round, args.p_round, ref):
                            emit(
                                dict(
                                    kind="attention_model",
                                    length=length,
                                    seed=args.seed,
                                    distribution=distribution,
                                    preprocessing=preprocessing,
                                    variant=variant,
                                    input_sha256=hashes,
                                    **result,
                                )
                            )
        emit(dict(kind="completed", seconds=time.monotonic() - started))


if __name__ == "__main__":
    main()
