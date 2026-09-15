# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""CPU Q-centering/H16 interaction with original-K mean correction."""

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
SPEC = importlib.util.spec_from_file_location("center_diagonal", HERE / "diagonal_k_smoothing_models.py")
DIAG = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DIAG)
HAD, RISK, MODEL, REPRO = DIAG.HAD, DIAG.RISK, DIAG.MODEL, DIAG.REPRO
METHODS = ("none", "center", "h16", "center_bf16_h16", "center_rne7_h16", "h16_center_exact_bias")


def prepare(q, k, mean, method, signs):
    center = "center" in method
    rotate = "h16" in method
    if not rotate:
        return q - mean if center else q, k, 1
    kr = HAD.hadamard(k * signs, 16).bfloat16().float()
    if method in ("center_bf16_h16", "center_rne7_h16"):
        residual = q - mean
        # Both are BF16-storable inputs to the current BF16-input matmul.
        residual = residual.bfloat16().float() if method == "center_bf16_h16" else MODEL.round_significand(residual, 7)
        qr = HAD.hadamard(residual * signs, 16).bfloat16().float()
    else:
        qr = HAD.hadamard(q * signs, 16).bfloat16().float()
        if center:
            # Diagnostic: exact transformed original BF16 mean; do not silently
            # round to a different bias while retaining the original correction.
            transformed_mean = HAD.hadamard(mean.double() * signs.double(), 16)
            assert torch.equal(transformed_mean, transformed_mean.float().double())
            qr = qr - transformed_mean.float()
    return qr, kr, 16


def normalized_weights(score):
    score = score.reshape(128, -1, 512)
    maximum = score.amax(-1, keepdim=True).cummax(1).values
    p = MODEL.round_significand(RISK.GRID.native_grid(score - maximum), 7, "trunc").double()
    p *= ((maximum - maximum[:, -1:]) / math.sqrt(128)).exp()
    p = p.reshape(128, -1)
    return p / p.sum(-1, keepdim=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    signs = torch.randint(0, 2, (128,), generator=torch.Generator().manual_seed(HAD.SIGN_SEED)).float() * 2 - 1
    started, count = time.monotonic(), 0
    sources = [Path(__file__), Path(DIAG.__file__), Path(HAD.__file__), Path(RISK.__file__),
               Path(RISK.GRID.__file__), HERE / "numerics.py", MODEL.V1 / "probe.py", Path(REPRO.__file__)]
    previous = [json.loads(x) for x in (HERE / "qk-hadamard-native-models-v1.jsonl").read_text().splitlines()]
    with (HERE / (args.label + ".jsonl")).open("x") as output:
        def emit(record):
            line = json.dumps(record, allow_nan=False)
            output.write(line + "\n")
            output.flush()
            print(line, flush=True)

        emit(dict(kind="provenance", hostname=platform.node(), threads=4, length=32768, query_length=128,
                  heads=1, dim=128, seeds=[1240, 1241], methods=METHODS,
                  contract="Original BF16 QKV FP64 reference; original Q128 token mean rounded BF16; same bias subtraction and FP64 correction against ORIGINAL K; H16 signed unnormalized with BF16 outputs and additional explicit centered-input BF16 or RNE7 spill; final Q RNE7; K RNE BFP4; V RNE BFP4 or RNE5/nativeRNA BFP8/LoFi5; native exp grid; P trunc7 matched denominator; FP64 matmul/subtraction/correction/recurrence; BF16 output",
                  caveats="CPU only; no device FPU alignment, mean-generation error, correction matmul error or timing; after-rotation exact transformed-bias diagnostic requires FP32 bias unsupported by current BF16 bias helper",
                  source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}))
        for seed in (1240, 1241):
            for distribution in ("normal", "outliers", "common_q", "structured_balanced_qk"):
                if distribution == "structured_balanced_qk":
                    q, k, v = DIAG.inputs(seed, distribution)
                else:
                    q, k, v = [x.squeeze().float() for x in REPRO.make_inputs(1, 128, 32768, 128, seed, distribution)]
                reference = REPRO.reference(q, k, v)
                exact_score = q.double() @ k.double().T
                centered_exact = exact_score - exact_score.mean(-1, keepdim=True)
                mean = q.double().mean(0, keepdim=True).bfloat16().float()
                correction = mean.double() @ k.double().T
                values = {fmt: RISK.encode(v, fmt) for fmt in ("b8", "b4")}
                for method in METHODS:
                    qr, kr, width = prepare(q, k, mean, method, signs)
                    bias = correction if "center" in method else 0
                    preprocessing_score = qr.double() @ kr.double().T / width + bias
                    preprocessing_score -= preprocessing_score.mean(-1, keepdim=True)
                    qe, ke = MODEL.round_significand(qr, 7), RISK.encode(kr, "b4")
                    score = qe.double() @ ke.T / width + bias
                    p = normalized_weights(score)
                    score -= score.mean(-1, keepdim=True)
                    for vfmt, ve in values.items():
                        metrics = REPRO.metrics((p @ ve).bfloat16(), reference)
                        if distribution != "structured_balanced_qk" and method in ("none", "h16"):
                            old = next(r for r in previous if r.get("kind") == "attention" and r["distribution"] == distribution
                                       and r["seed"] == seed and r["v_format"] == vfmt and r["hadamard_width"] == width
                                       and not r["center_k"] and r["spill"] == ("none" if width == 1 else "bf16"))
                            assert metrics["l2_pct"] == old["l2_pct"], "Uncentered controls must reproduce prior study"
                        emit(dict(kind="attention", distribution=distribution, seed=seed, method=method, v_format=vfmt,
                                  **metrics,
                                  preprocessing_centered_score_l2_pct=float(100 * (preprocessing_score - centered_exact).norm() / centered_exact.norm()),
                                  quantized_centered_score_l2_pct=float(100 * (score - centered_exact).norm() / centered_exact.norm()),
                                  mean_bf16_rounding_l2_pct=float(100 * (mean - q.double().mean(0, keepdim=True)).norm() / q.double().mean(0, keepdim=True).norm()),
                                  q_residual_rms=float((q - mean).square().mean().sqrt()),
                                  p_entropy_mean=float(-(p * p.clamp_min(1e-300).log()).sum(-1).mean()),
                                  p_max_mean=float(p.max(-1).values.mean())))
                        count += 1
                emit(dict(kind="input_completed", seed=seed, distribution=distribution, cases=count,
                          seconds=time.monotonic() - started))
        emit(dict(kind="completed", cases=count, seconds=time.monotonic() - started))


if __name__ == "__main__":
    main()
