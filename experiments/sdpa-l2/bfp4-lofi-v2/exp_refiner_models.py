# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU polynomial/grid/attention models for lower-degree SDPA exp refiners.

Models the actual signed-INT16 -> shifted-float grid and folded coefficients,
not an ideal exponential in isolation. No device operations or speed claims.
"""

import argparse
import hashlib
import importlib.util
import json
import math
import platform
import re
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("exp_refiner_base", HERE / "hi2_b8_attribution_models.py")
BASE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BASE)
MODEL, REPRO = BASE.MODEL, BASE.REPRO
HEADER = HERE.parent / "hybrid-mixed-v1/candidate/tt_metal/hw/ckernels/blackhole/metal/llk_api/experimental/llk_sfpu/ckernel_sfpu_sdpa.h"
SCALE = np.float32(1 / math.sqrt(128))
GRID_A = np.float32(np.float32(1024) * np.float32(1.4426950408889634) * SCALE)
GRID_B = np.float32(31 * 1024 - 4 * (32512 - 32500.818359375))
COMMON = 2.0 ** (float(GRID_B) / 1024 - 31)


def fits():
    source = HEADER.read_text().split("inline void calculate_sdpa_exp_stream_effective()", 1)[1]
    source = source.split("TTI_SFPLOADI", 1)[0]
    current = [float(v) for v in re.findall(r"constexpr float [abcd] = ([^f;]+)f;", source)]
    assert len(current) == 8
    m = np.linspace(1, 2, 65536)
    result, records = {}, []
    for target_name, old in (("unbiased", current[:4]), ("effective_p7", current[4:])):
        target = np.exp2(m - 1)
        if target_name == "effective_p7":
            target = 0.995 * target + 1 / 128
        for degree in (1, 2, 3):
            design = np.vander(m, degree + 1) * (m / target)[:, None]
            coef = np.linalg.lstsq(design, np.ones_like(m), rcond=None)[0]
            if degree == 3:
                np.testing.assert_allclose(coef * 2.0**96, old, rtol=1e-9)
                coef = np.array(old) / 2.0**96
            folded = np.array(coef * 2.0**96, dtype=np.float32)
            realized = folded.astype(np.float64) / 2.0**96
            approximation = m * np.polyval(realized, m)
            relative = approximation / target - 1
            negative_m = np.linspace(-2, -1, 65536)
            assert (np.polyval(realized, negative_m) > 0).all(), "Underflow sign must survive refiner"
            name = f"{target_name}_degree{degree}"
            result[name] = folded.tolist()
            records.append(dict(kind="fit", name=name, target=target_name, degree=degree,
                                coefficients_descending=realized.tolist(), folded_fp32_coefficients=folded.tolist(),
                                relative_rms_pct=float(100 * np.sqrt(np.mean(relative**2))),
                                relative_max_abs_pct=float(100 * np.max(np.abs(relative))),
                                refined_mantissa_min=float(approximation.min()), refined_mantissa_max=float(approximation.max()),
                                negative_mantissa_sign_preserved=True))
    return result, records


def grid(delta):
    # IEEE FP32 FMA emulation: exact FP64 intermediate then one FP32 rounding.
    transformed = (delta.float().double() * float(GRID_A) + float(GRID_B)).float()
    magnitude = torch.floor(transformed.double().abs() + 0.5).clamp_max(32767).int()
    raw = (magnitude << 13).contiguous()
    linear = raw.view(torch.float32) * torch.where(transformed < 0, -1.0, 1.0)
    mantissa_bits = (linear.contiguous().view(torch.int32) & -2139095041) | (127 << 23)
    # -2139095041 == 0x807fffff interpreted as signed int32.
    mantissa = mantissa_bits.view(torch.float32)
    return linear, mantissa, magnitude, transformed


def polynomial(linear, mantissa, coefficients):
    value = torch.full_like(mantissa, coefficients[0])
    for coefficient in coefficients[1:]:
        value = (mantissa.double() * value.double() + coefficient).float()
    return (linear.double() * value.double()).float().clamp_min(0).double()


def outputs(scores, v, coefficients):
    score = scores.reshape(128, -1, 512)
    maximum = score.amax(-1, keepdim=True).cummax(1).values
    delta = score - maximum
    exact = (delta / math.sqrt(128)).exp()
    correction = ((maximum - maximum[:, -1:]) / math.sqrt(128)).exp()
    linear, mantissa, magnitude, transformed = grid(delta)
    grid_exact = torch.exp2(magnitude.double() / 1024 - 31)
    grid_exact = torch.where(transformed >= 0, grid_exact, 0)
    probabilities = {"exact_exp": exact, "grid_ideal_refiner": grid_exact}
    probabilities.update({name: polynomial(linear, mantissa, coef) for name, coef in coefficients.items()})
    result = {}
    for name, p in probabilities.items():
        consumed = MODEL.round_significand(p, 7, "trunc").double()
        weighted = (consumed * correction).reshape(128, -1)
        out = weighted @ v.double() / weighted.sum(-1, keepdim=True)
        result[name] = (out, dict(
            grid_negative_fraction=float((transformed < 0).double().mean()),
            grid_subnormal_linear_fraction=float(((magnitude < 1024) & (transformed >= 0)).double().mean()),
            true_mass_below_grid_range=float((exact * correction * (transformed < 0)).sum() / (exact * correction).sum()),
        ))
    return result


def scalar_grid_checks(coefficients):
    delta = torch.linspace(-20 * math.sqrt(128), 0, 65536, dtype=torch.float64)
    linear, mantissa, magnitude, transformed = grid(delta)
    assert bool((transformed > 1024).all())
    expected_linear = torch.ldexp(mantissa.double(), magnitude // 1024 - 127)
    torch.testing.assert_close(linear.double(), expected_linear, atol=0, rtol=0)
    truth = torch.exp(delta / math.sqrt(128)) * COMMON
    records = []
    variants = {"grid_ideal_refiner": torch.exp2(magnitude.double() / 1024 - 31)}
    variants.update({name: polynomial(linear, mantissa, coef) for name, coef in coefficients.items()})
    for name, value in variants.items():
        relative = value / truth - 1
        records.append(dict(kind="grid_check", name=name, logit_range=[-20, 0],
                            relative_rms_pct=float(100 * relative.square().mean().sqrt()),
                            relative_max_abs_pct=float(100 * relative.abs().max()),
                            mean_relative_pct=float(100 * relative.mean())))
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True)
    parser.add_argument("--lengths", nargs="+", type=int, default=[4096, 32768])
    parser.add_argument("--seed", type=int, default=1240)
    args = parser.parse_args()
    torch.set_num_threads(4)
    coefficients, fit_records = fits()
    grid_records = scalar_grid_checks(coefficients)
    started, count = time.monotonic(), 0
    sources = [Path(__file__), HEADER, Path(BASE.__file__), HERE / "numerics.py", MODEL.V1 / "probe.py", Path(REPRO.__file__)]
    with (HERE / (args.label + ".jsonl")).open("x") as output:
        def emit(record):
            line = json.dumps(record, allow_nan=False)
            output.write(line + "\n")
            output.flush()
            print(line, flush=True)

        emit(dict(kind="provenance", args=vars(args), hostname=platform.node(), threads=4,
                  grid_a=float(GRID_A), grid_b=float(GRID_B), common_exp_factor=COMMON,
                  contract="CPU exact QK/subtraction and online corrections; modeled FP32 exp-grid FMA, signed INT16 nearest-away, bitshift13, folded FP32 Horner FMA/MUL; P trunc7 matched; FP64 PV/state; BF16 output; no device",
                  caveats="IEEE FP32 rounding model, not bit-accurate SFPU exceptional/denormal arithmetic; no device cheap-subtraction or recurrent BF16 error",
                  source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}))
        for record in fit_records + grid_records:
            emit(record)
        for length in args.lengths:
            for distribution in ("normal", "scaled_qk", "outliers", "common_k_centered"):
                q, k, v = [x.squeeze().float() for x in REPRO.make_inputs(
                    1, 128, length, 128, args.seed, "common_k" if distribution == "common_k_centered" else distribution)]
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
                    result = outputs(qe.double() @ ke.double().T, ve, coefficients)
                    for name, (raw, details) in result.items():
                        actual = raw.bfloat16()
                        current = result["effective_p7_degree3"][0].bfloat16()
                        emit(dict(kind="attention", length=length, seed=args.seed, distribution=distribution,
                                  operands=operands, refiner=name, **BASE.summary(actual, reference), **details,
                                  before_output_rounding_l2_pct=BASE.summary(raw, reference)["l2_pct"],
                                  vs_current_compensated_cubic_l2_pct=BASE.summary(actual, current)["l2_pct"],
                                  vs_exact_exp_l2_pct=BASE.summary(actual, result["exact_exp"][0].bfloat16())["l2_pct"]))
                        count += 1
            emit(dict(kind="length_completed", length=length, cases=count, seconds=time.monotonic() - started))
        emit(dict(kind="completed", cases=count, seconds=time.monotonic() - started))


if __name__ == "__main__":
    main()
