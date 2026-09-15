# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU-only Q prescale/RNE7 error-budget models; no device operations.

Q is multiplied in FP32 before RNE7, then interpreted with inverse-alpha
score scaling. P7, where enabled, is unchanged from the imported v2 model.
Remaining arithmetic is FP64; this does not model kernel accumulation/error.
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
SPEC = importlib.util.spec_from_file_location("prescale_v2_numerics", HERE / "numerics.py")
MODEL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODEL)
VARIANTS = ("q7_only", "qp7_exact_kv", "residual48_qp7")
ALPHAS = [1.0, 1.001, 1.002, 1.003, 1.0038, 1.00390625, 1.005, 1.01, 1.02, 1.05, 1.1, 1.2]


def effective_query(q, alpha):
    alpha32 = float(torch.tensor(alpha, dtype=torch.float32))
    stored = MODEL.round_significand(q.float() * alpha32, 7)
    assert torch.equal(stored, MODEL.round_significand(stored, 7, "trunc"))
    return stored.double() / alpha32, alpha32


def evaluate(q, k, v, variant, alpha, reference):
    qe, alpha32 = effective_query(q, alpha)
    scores = (qe @ k.T / math.sqrt(128)).reshape(128, -1, 512)
    maximum = scores.amax(-1, keepdim=True).cummax(1).values
    p = (scores - maximum).exp()
    pe = p if variant == "q7_only" else MODEL.effective(MODEL.encode(p, "e7", "device"), "e7", "left")
    to_final = (maximum - maximum[:, -1:]).exp()
    numerator = (pe * to_final).reshape(128, -1) @ v
    original_den = (p * to_final).sum((1, 2))[:, None]
    matched_den = (pe * to_final).sum((1, 2))[:, None]
    rows = []
    for name, denominator in (("original", original_den), ("matched", matched_den)):
        raw = numerator / denominator
        result = raw.bfloat16().double()
        row_l2 = 100 * (result - reference).norm(dim=-1) / reference.norm(dim=-1).clamp_min(1e-300)
        rows.append(
            dict(
                denominator=name,
                alpha=alpha,
                alpha_fp32=alpha32,
                **MODEL.metrics(result, reference),
                before_output_rounding_l2_pct=MODEL.metrics(raw, reference)["l2_pct"],
                row_p95_l2_pct=float(torch.quantile(row_l2, 0.95)),
                q_representation_l2_pct=MODEL.metrics(qe, q)["l2_pct"],
                q_gain_error_pct=float(100 * ((qe * q).sum() / q.square().sum() - 1)),
                p_mass_ratio=float((matched_den / original_den).mean()),
            )
        )
    return rows


def self_test():
    inputs = MODEL.V1_NUMERICS.FRONTIER.inputs_for(1024, 91, "normal")
    q, k, v = [x.squeeze().double() for x in inputs]
    reference = torch.softmax(q @ k.T / math.sqrt(128), -1) @ v
    for variant, existing in (("qp7_exact_kv", "pervalue_floor"), ("residual48_qp7", "residual48_pervalue")):
        formats = ("b4", "e5_b8") if variant == "residual48_qp7" else ("exact",)
        ke, ve = [sum(MODEL.components(x, formats, "host")) for x in (k, v)]
        actual = evaluate(q, ke, ve, variant, 1.0, reference)
        expected = MODEL.evaluate(inputs, existing, "none", "host", 91, reference)
        for a, e in zip(actual, expected):
            assert a["l2_pct"] == e["l2_pct"], (variant, a["l2_pct"], e["l2_pct"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True)
    parser.add_argument("--lengths", nargs="+", type=int, default=[4096, 32768])
    parser.add_argument("--seeds", nargs="+", type=int, default=[1244, 1245])
    parser.add_argument("--distributions", nargs="+", default=["normal"])
    parser.add_argument("--alphas", nargs="+", type=float, default=ALPHAS)
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=VARIANTS)
    args = parser.parse_args()
    assert all(math.isfinite(x) and x > 0 for x in args.alphas)
    torch.set_num_threads(4)
    self_test()
    started = time.monotonic()
    with (HERE / (args.label + ".jsonl")).open("x") as output:

        def emit(record):
            line = json.dumps(record, allow_nan=False)
            output.write(line + "\n")
            output.flush()
            print(line, flush=True)

        sources = [Path(__file__), HERE / "numerics.py", MODEL.V1 / "probe.py", MODEL.V1 / "numerics.py"]
        emit(
            dict(
                kind="provenance",
                args=vars(args),
                hostname=platform.node(),
                contract="CPU Q prescale FP32->RNE7, reciprocal-alpha score adjustment; FP64 subsequent arithmetic; NOT device accuracy/performance",
                source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
            )
        )
        for length in args.lengths:
            for seed in args.seeds:
                for distribution in args.distributions:
                    inputs = MODEL.V1_NUMERICS.FRONTIER.inputs_for(length, seed, distribution)
                    q, k, v = [x.squeeze().double() for x in inputs]
                    reference = torch.softmax(q @ k.T / math.sqrt(128), -1) @ v
                    residuals = [sum(MODEL.components(x, ("b4", "e5_b8"), "host")) for x in (k, v)]
                    for variant in args.variants:
                        ke, ve = residuals if variant == "residual48_qp7" else (k, v)
                        for alpha in args.alphas:
                            for row in evaluate(q, ke, ve, variant, alpha, reference):
                                emit(
                                    dict(
                                        kind="attention_model",
                                        length=length,
                                        seed=seed,
                                        distribution=distribution,
                                        variant=variant,
                                        **row,
                                    )
                                )
        emit(dict(kind="completed", seconds=time.monotonic() - started))


if __name__ == "__main__":
    main()
