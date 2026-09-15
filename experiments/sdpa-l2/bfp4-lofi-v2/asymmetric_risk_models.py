# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Bounded CPU attribution of asymmetric LoFi K/V formats and native exp."""

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
SPEC = importlib.util.spec_from_file_location("risk_grid", HERE / "direct_exp_grid_models.py")
GRID = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GRID)
MODEL, REPRO = GRID.MODEL, GRID.REPRO


def encode(x, fmt):
    stored = MODEL.quantize(x, 3, "host") if fmt == "b4" else MODEL.encode(x, "e5_b8", "device")
    return MODEL.round_significand(stored, 5, "trunc").double()


def weights(q, k):
    score = (q.double() @ k.double().T).reshape(128, -1, 512)
    maximum = score.amax(-1, keepdim=True).cummax(1).values
    delta = score - maximum
    correction = ((maximum - maximum[:, -1:]) / math.sqrt(128)).exp()
    result = {}
    for name, p in (("exact", (delta / math.sqrt(128)).exp()), ("native", GRID.native_grid(delta))):
        pe = MODEL.round_significand(p, 7, "trunc").double()
        weighted = (pe * correction).reshape(128, -1)
        result[name] = weighted / weighted.sum(-1, keepdim=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    started, count = time.monotonic(), 0
    distributions = ("normal", "outliers", "scaled_qk", "common_q", "common_k",
                     "common_k_centered", "common_v", "channel_outlier_v", "constant_v")
    sources = [Path(__file__), Path(GRID.__file__), HERE / "numerics.py", MODEL.V1 / "probe.py", Path(REPRO.__file__)]
    with (HERE / (args.label + ".jsonl")).open("x") as output:
        def emit(record):
            line = json.dumps(record, allow_nan=False)
            output.write(line + "\n")
            output.flush()
            print(line, flush=True)

        emit(dict(kind="provenance", hostname=platform.node(), threads=4, seed=1240,
                  contract="CPU FP64 QK/subtraction, online correction and state; IEEE FP32 native exp grid or exact exp; Q RNE7; K/V RNE BFP4 or per-value RNE5 then native BFP8 RNA and LoFi trunc5; P trunc7 matched; BF16 final output; original BF16 reference",
                  inputs="N32K all nine distributions, normal4K control; H1/Q128/D128; channel_outlier_v multiplies every16th V channel by32; common offsets32",
                  caveats="No device cheap subtraction, BF16 recurrent state/compensation, P BF16 spill or SFPU-specific rounding; one seed, not qualification",
                  source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}))
        for length in (4096, 32768):
            for distribution in (("normal",) if length == 4096 else distributions):
                original_distribution = {"common_k_centered": "common_k", "channel_outlier_v": "normal"}.get(distribution, distribution)
                q, k, v = [x.squeeze().float() for x in REPRO.make_inputs(1, 128, length, 128, 1240, original_distribution)]
                if distribution == "channel_outlier_v":
                    v[:, ::16] *= 32
                reference = REPRO.reference(q, k, v)
                if distribution == "common_k_centered":
                    k = (k.double() - k.double().mean(0, keepdim=True)).float()
                qe = MODEL.round_significand(q, 7)
                keys, values = ({fmt: encode(x, fmt) for fmt in ("b8", "b4")} for x in (k, v))
                for kfmt, ke in keys.items():
                    normalized = weights(qe, ke)
                    for vfmt, ve in values.items():
                        mean_error = (ve - v.double()).mean(0)
                        for exp, p in normalized.items():
                            raw = p @ ve
                            actual = raw.bfloat16()
                            emit(dict(kind="attention", length=length, seed=1240, distribution=distribution,
                                      k_format=kfmt, v_format=vfmt, exp=exp, **REPRO.metrics(actual, reference),
                                      k_representation_l2_pct=REPRO.metrics(ke, k)["l2_pct"],
                                      v_representation_l2_pct=REPRO.metrics(ve, v)["l2_pct"],
                                      v_mean_error_rms=float(mean_error.square().mean().sqrt()),
                                      mean_bias_only_l2_pct=float(100 * mean_error.norm() * math.sqrt(128) / reference.norm()),
                                      diagnostic_v_mean_corrected_l2_pct=REPRO.metrics((raw - mean_error).bfloat16(), reference)["l2_pct"],
                                      p_entropy_mean=float(-(p * p.clamp_min(1e-300).log()).sum(-1).mean()),
                                      p_max_mean=float(p.max(-1).values.mean())))
                            count += 1
                emit(dict(kind="input_completed", length=length, distribution=distribution, count=count, seconds=time.monotonic() - started))
        emit(dict(kind="completed", cases=count, seconds=time.monotonic() - started))


if __name__ == "__main__":
    main()
