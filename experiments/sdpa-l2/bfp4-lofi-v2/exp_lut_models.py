# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""CPU fit/model for a two-segment SFPLUTFP32 native-exp refiner."""

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch

import direct_exp_grid_models as D

HERE = Path(__file__).resolve().parent


def fit():
    coefficients, segments = [], []
    for lo, hi in ((1.0, 1.5), (1.5, 2.0)):
        m = np.linspace(lo, hi, 65536)
        target = np.exp2(m - 1) / m
        design = np.stack((m, np.ones_like(m)), axis=1) / target[:, None]
        exact = np.linalg.lstsq(design, np.ones_like(m), rcond=None)[0]
        realized = exact.astype(np.float16).astype(np.float32)
        relative = (realized[0] * m + realized[1]) / target - 1
        coefficients.append(realized.tolist())
        segments.append(
            dict(
                range=[lo, hi],
                fp16_coefficients=realized.tolist(),
                relative_rms_pct=float(100 * np.sqrt(np.mean(relative**2))),
                relative_max_pct=float(100 * np.max(np.abs(relative))),
            )
        )
    slopes = np.array([x[0] for x in coefficients], dtype=np.float16).view(np.uint16)
    intercepts = np.array([x[1] for x in coefficients], dtype=np.float16).view(np.uint16)
    packed = dict(
        slopes=f"0x{int(slopes[0]) | int(slopes[1]) << 16:08x}",
        intercepts=f"0x{int(intercepts[0]) | int(intercepts[1]) << 16:08x}",
    )
    return coefficients, segments, packed


def refine(linear, coefficients):
    x = linear.float().contiguous()
    mantissa = ((x.view(torch.int32) & 0x7FFFFF) | (127 << 23)).view(torch.float32)
    high = mantissa >= 1.5
    a = torch.where(high, coefficients[1][0], coefficients[0][0])
    b = torch.where(high, coefficients[1][1], coefficients[0][1])
    ratio = (mantissa.double() * a.double() + b.double()).float()
    return (x.double() * ratio.double()).float().clamp_min(0).double()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True)
    parser.add_argument("--fit-only", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    coefficients, segments, packed = fit()
    path = HERE / (args.label + ".jsonl")
    assert not path.exists()
    with path.open("x") as stream:

        def emit(value):
            stream.write(json.dumps(value, allow_nan=False) + "\n")
            stream.flush()
            print(json.dumps(value, allow_nan=False), flush=True)

        emit(
            dict(
                kind="fit",
                segments=segments,
                packed_fp16=packed,
                source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                contract="Native8-bit exp grid; FP16 LUT ratio exp2(m-1)/m on two segments; common exp scale cancels in softmax; no device claims",
            )
        )
        delta = torch.linspace(-80 * math.sqrt(128), 0, 262144, dtype=torch.float64)
        truth = (delta / math.sqrt(128)).exp()
        native = D.native_grid(delta)
        for name, value in (("native", native), ("lut2", refine(native, coefficients))):
            ratio = value / truth
            gain = float(ratio.mean())
            relative = ratio / gain - 1
            emit(
                dict(
                    kind="scalar",
                    exp=name,
                    common_gain=gain,
                    normalized_relative_rms_pct=float(100 * relative.square().mean().sqrt()),
                    normalized_relative_max_pct=float(100 * relative.abs().max()),
                )
            )
        if args.fit_only:
            return
        for seed in (1240, 1241):
            for distribution in ("normal", "outliers", "scaled_qk", "common_k_centered"):
                q, k, v = [
                    x.squeeze().float()
                    for x in D.REPRO.make_inputs(
                        1, 128, 32768, 128, seed, "common_k" if distribution == "common_k_centered" else distribution
                    )
                ]
                reference = D.REPRO.reference(q, k, v)
                if distribution == "common_k_centered":
                    k = (k.double() - k.double().mean(0, keepdim=True)).float()
                for precision in ("exact", "kv5", "kv4"):
                    qe = q if precision == "exact" else D.MODEL.round_significand(q, 7)
                    ke, ve = k, v
                    if precision == "kv5":
                        ke, ve = [D.MODEL.round_significand(x, 5) for x in (k, v)]
                    elif precision == "kv4":
                        ke, ve = [D.MODEL.quantize(x, 3, "host") for x in (k, v)]
                    score = (qe.double() @ ke.double().T).reshape(128, -1, 512)
                    maximum = score.amax(-1, keepdim=True).cummax(1).values
                    native = D.native_grid(score - maximum)
                    correction = ((maximum - maximum[:, -1:]) / math.sqrt(128)).exp()
                    for name, value in (("native", native), ("lut2", refine(native, coefficients))):
                        weights = (D.MODEL.round_significand(value, 7, "trunc").double() * correction).reshape(128, -1)
                        actual = (weights @ ve.double() / weights.sum(-1, keepdim=True)).bfloat16()
                        emit(
                            dict(
                                kind="attention",
                                seed=seed,
                                distribution=distribution,
                                operand_precision=precision,
                                exp=name,
                                metrics=D.REPRO.metrics(actual, reference),
                            )
                        )


if __name__ == "__main__":
    main()
