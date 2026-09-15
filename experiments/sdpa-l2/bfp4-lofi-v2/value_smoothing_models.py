# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU-only reversible V smoothing and native-group RNE BFP4 attribution."""

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
SPEC = importlib.util.spec_from_file_location("value_smoothing_attribution", HERE / "hi2_b8_attribution_models.py")
BASE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BASE)
MODEL, REPRO = BASE.MODEL, BASE.REPRO
TRANSFORMS = ("none", "pow2_scale", "h16_fp32", "h16_bf16", "h128_fp32", "h128_bf16")
SIGN_SEED = 20260915


def hadamard(x, width):
    shape = x.shape
    y = x.reshape(-1, width)
    stride = 1
    while stride < width:
        z = y.reshape(y.shape[0], -1, 2, stride)
        a, b = z[..., 0, :], z[..., 1, :]
        y = torch.stack((a + b, a - b), dim=-2).reshape(-1, width)
        stride *= 2
    return (y / math.sqrt(width)).reshape(shape)


def signs():
    gen = torch.Generator().manual_seed(SIGN_SEED)
    return torch.randint(0, 2, (128,), generator=gen).float() * 2 - 1


def prepare(v, transform):
    if transform == "none":
        return v, lambda x: x, dict(scale_min=1.0, scale_max=1.0)
    if transform == "pow2_scale":
        # Max statistics and their reduction are deliberately not timed as a device kernel.
        scale = torch.exp2(torch.ceil(torch.log2(v.abs().amax(0).clamp_min(2.0**-126))))
        return v / scale, lambda x: x * scale.double(), dict(
            scale_min=float(scale.min()), scale_max=float(scale.max()),
            unique_scale_count=int(scale.unique().numel()))
    width = 16 if transform.startswith("h16_") else 128
    sign = signs()
    stored = hadamard(v * sign, width)  # FP32 butterflies and normalization.
    if transform.endswith("_bf16"):
        stored = stored.bfloat16().float()
    return stored, lambda x: hadamard(x.double(), width) * sign.double(), dict(hadamard_width=width)


def weights(q, k):
    score = (q.double() @ k.double().T / math.sqrt(128)).reshape(128, -1, 512)
    maximum = score.amax(-1, keepdim=True).cummax(1).values
    p = (score - maximum).exp()
    pe = MODEL.round_significand(p, 7).double()
    effective = (pe * (maximum - maximum[:, -1:]).exp()).reshape(128, -1)
    return effective / effective.sum(-1, keepdim=True)


def self_test():
    x = torch.randn((32, 128), generator=torch.Generator().manual_seed(91), dtype=torch.float64)
    sign = signs().double()
    for width in (16, 128):
        restored = hadamard(hadamard(x * sign, width), width) * sign
        torch.testing.assert_close(restored, x, atol=2e-14, rtol=2e-14)
    for transform in TRANSFORMS:
        prepared, inverse, _ = prepare(x.bfloat16().float(), transform)
        error = BASE.summary(inverse(prepared.double()), x.bfloat16())
        assert error["l2_pct"] < (0.3 if transform.endswith("bf16") else 0.0001), (transform, error)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True)
    parser.add_argument("--lengths", nargs="+", type=int, default=[4096, 32768])
    parser.add_argument("--seeds", nargs="+", type=int, default=[1240, 1241])
    parser.add_argument("--distributions", nargs="+", default=["normal", "outliers", "common_v", "channel_outlier_v"])
    args = parser.parse_args()
    torch.set_num_threads(4)
    self_test()
    started, cases = time.monotonic(), 0
    sources = [Path(__file__), Path(BASE.__file__), HERE / "numerics.py", MODEL.V1 / "probe.py", Path(REPRO.__file__)]
    with (HERE / (args.label + ".jsonl")).open("x") as output:
        def emit(record):
            line = json.dumps(record, allow_nan=False)
            output.write(line + "\n")
            output.flush()
            print(line, flush=True)

        emit(dict(kind="provenance", args=vars(args), hostname=platform.node(), threads=4,
                  contract="Original BF16 reference; Q RNE7; K host RNE BFP4 or exact; P RNE7 matched; FP32 V transform with optional BF16 spill; host RNE shared16 BFP4 V; FP64 attention/inverse; BF16 final output; CPU only",
                  sign_seed=SIGN_SEED, exact_k_subset="32K, first seed, all distributions",
                  channel_outlier_definition="Every16th V channel multiplied by32; original BF16 reference includes this exact power-of-two change",
                  source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}))
        for length in args.lengths:
            for seed in args.seeds:
                for distribution in args.distributions:
                    q, k, v = [x.squeeze().float() for x in REPRO.make_inputs(
                        1, 128, length, 128, seed, "normal" if distribution == "channel_outlier_v" else distribution)]
                    if distribution == "channel_outlier_v":
                        v[:, ::16] *= 32
                    reference = REPRO.reference(q, k, v)
                    qe = MODEL.round_significand(q, 7)
                    key_modes = ["b4"] + (["exact"] if length == 32768 and seed == args.seeds[0] else [])
                    for key_mode in key_modes:
                        ke = MODEL.quantize(k, 3, "host") if key_mode == "b4" else k
                        normalized = weights(qe, ke)
                        qp_baseline = normalized @ v.double()
                        for transform in TRANSFORMS:
                            prepared, inverse, details = prepare(v, transform)
                            encoded = MODEL.quantize(prepared, 3, "host")
                            assert torch.equal(encoded, MODEL.round_significand(encoded, 5, "trunc"))
                            represented = inverse(encoded.double())
                            error = represented - v.double()
                            mean_error = error.mean(0)
                            pre_inverse = normalized @ encoded.double()
                            raw = inverse(pre_inverse)
                            # A conventional BF16-output attention call incurs this additional spill.
                            bf16_intermediate = inverse(pre_inverse.bfloat16().double()).bfloat16()
                            actual = raw.bfloat16()
                            row_l2 = 100 * (actual.double() - reference).norm(dim=-1) / reference.norm(dim=-1)
                            emit(dict(kind="attention", length=length, seed=seed, distribution=distribution,
                                      key_mode=key_mode, transform=transform, **details,
                                      **BASE.summary(actual, reference),
                                      before_output_rounding_l2_pct=BASE.summary(raw, reference)["l2_pct"],
                                      bf16_before_inverse_l2_pct=BASE.summary(bf16_intermediate, reference)["l2_pct"],
                                      row_p95_l2_pct=float(torch.quantile(row_l2, 0.95)),
                                      exact_v_baseline_l2_pct=BASE.summary(qp_baseline.bfloat16(), reference)["l2_pct"],
                                      v_representation=BASE.summary(represented, v),
                                      prequant_roundtrip_l2_pct=BASE.summary(inverse(prepared.double()), v)["l2_pct"],
                                      v_mean_error_rms=float(mean_error.square().mean().sqrt()),
                                      v_mean_error_max_abs=float(mean_error.abs().max()),
                                      mean_bias_only_l2_pct=float(100 * mean_error.norm() * math.sqrt(128) / reference.norm()),
                                      mean_corrected_l2_pct=BASE.summary((raw - mean_error).bfloat16(), reference)["l2_pct"]))
                            cases += 1
                    emit(dict(kind="input_completed", length=length, seed=seed, distribution=distribution,
                              cases=cases, elapsed_seconds=time.monotonic() - started))
        emit(dict(kind="completed", cases=cases, seconds=time.monotonic() - started))


if __name__ == "__main__":
    main()
