# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""LoFi operand pre-rounding and residual-decomposition accuracy models.

This models representation and LoFi operand bit selection, not accumulation
rounding or device throughput. The v1 quantizers are immutable dependencies.
"""

import argparse
import hashlib
import importlib.util
import json
import math
import platform
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
V1 = HERE.parent / "bfp4-lofi-v1"
sys.path.insert(0, str(V1))
from probe import metrics, quantize

SPEC = importlib.util.spec_from_file_location("v1_numerics", V1 / "numerics.py")
V1_NUMERICS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(V1_NUMERICS)

# Q format, K components, P format, V components. Every matmul is LoFi.
# e7/e5 are per-value significant-bit rounding, stored without further loss.
VARIANTS = {
    "all4": ("b4", ("b4",), "b4", ("b4",)),
    "mixed84": ("b8", ("b4",), "b8", ("b4",)),
    "b8_raw": ("b8", ("b8",), "b8", ("b8",)),
    "b8_rne5": ("b8", ("e5_b8",), "b8", ("e5_b8",)),
    "pervalue_rne": ("e7", ("e5",), "e7", ("e5",)),
    "pervalue_rawp": ("e7", ("e5",), "fp32", ("e5",)),
    "kv4_rawp": ("e7", ("b4",), "fp32", ("b4",)),
    "b8_rne5_p7": ("b8", ("e5_b8",), "e7", ("e5_b8",)),
    "b8_rne5_rawp": ("b8", ("e5_b8",), "fp32", ("e5_b8",)),
    "q8_only_rawp": ("b8", ("e5",), "fp32", ("e5",)),
    "k8_only_rawp": ("e7", ("e5_b8",), "fp32", ("e5",)),
    "v8_only_rawp": ("e7", ("e5",), "fp32", ("e5_b8",)),
    "q7kv8_rawp": ("e7", ("e5_b8",), "fp32", ("e5_b8",)),
    "q7kv8_full": ("e7", ("b8",), "fp32", ("b8",)),
    "bf16kv8_full": ("bf16", ("b8",), "fp32", ("b8",)),
    "residual44": ("b8", ("b4", "b4"), "b8", ("b4", "b4")),
    "residual444": ("b8", ("b4", "b4", "b4"), "b8", ("b4", "b4", "b4")),
    "residual48": ("b8", ("b4", "e5_b8"), "b8", ("b4", "e5_b8")),
    "residual48_pervalue": ("e7", ("b4", "e5_b8"), "e7", ("b4", "e5_b8")),
    "residual44_pervalue": ("e7", ("b4", "b4"), "e7", ("b4", "b4")),
    "residual44_pervalue_konly": ("e7", ("b4", "b4"), "e7", ("e5",)),
    "residual44_pervalue_vonly": ("e7", ("e5",), "e7", ("b4", "b4")),
    "qk_floor": ("b8", ("exact",), "exact", ("exact",)),
    "p_floor": ("exact", ("exact",), "b8", ("exact",)),
    "qp_floor": ("b8", ("exact",), "b8", ("exact",)),
    "pervalue_floor": ("e7", ("exact",), "e7", ("exact",)),
    "exact": ("exact", ("exact",), "exact", ("exact",)),
}


def round_significand(x, bits, mode="rne"):
    """Finite normal-range per-value rounding with power-of-two steps."""
    x = x.float()
    # The FPU flushes subnormals; avoid underflow in the model's step itself.
    x = torch.where(x.abs() < 2.0**-126, 0.0, x)
    exponent = torch.frexp(x.abs())[1].float() - 1
    step = torch.exp2(exponent - (bits - 1))
    scaled = x.abs().double() / step.double()
    if mode == "rne":
        integer = scaled.round()
    elif mode == "rna":
        integer = (scaled + 0.5).floor()
    elif mode == "trunc":
        integer = scaled.floor()
    else:
        raise ValueError(mode)
    return (integer * step * x.sign()).float()


def encode(x, fmt, route):
    if fmt == "exact":
        return x.double()
    if fmt == "fp32":
        return x.float().double()
    if fmt == "bf16":
        return x.bfloat16().double()
    if fmt == "b4":
        return quantize(x, 3, route).double()
    if fmt == "b8":
        return quantize(x, 7, route).double()
    if fmt == "e5_b8":
        return quantize(round_significand(x, 5), 7, route).double()
    if fmt in ("e5", "e7"):
        return round_significand(x, int(fmt[1:])).double()
    raise ValueError(fmt)


def effective(x, fmt, side):
    # Exact is a deliberate higher-precision ablation, not a realizable LoFi
    # tensor. Other tensors are truncated to the phase-0 operand widths.
    return x if fmt == "exact" else round_significand(x, 7 if side == "left" else 5, "trunc").double()


def components(x, formats, route):
    result = []
    residual = x.double()
    for fmt in formats:
        stored = encode(residual, fmt, route)
        consumed = effective(stored, fmt, "right")
        result.append(consumed)
        # Decompose against what the FPU will consume, not the original CB.
        residual = residual - consumed
    return result


def evaluate(inputs, variant, preprocessing, route, seed, reference):
    match_v_mean = preprocessing.endswith("_vmatch")
    base_preprocessing = preprocessing.removesuffix("_vmatch")
    q, k, v, correction, v0 = V1_NUMERICS.preprocess(inputs, base_preprocessing, seed)
    qfmt, kformats, pfmt, vformats = VARIANTS[variant]
    qe = effective(encode(q, qfmt, route), qfmt, "left")
    ke = sum(components(k, kformats, route))
    ve = sum(components(v, vformats, route))
    if match_v_mean:
        # Preserve the original V column means after quantization, not merely
        # before it. Particularly important when BF16 ties acquire a shared
        # direction after subtracting a small column mean.
        v0 = v0 + (v - ve).mean(0, keepdim=True)
    scores = (qe @ ke.T / math.sqrt(128) + correction).reshape(128, -1, 512)
    maximum = scores.amax(-1, keepdim=True).cummax(1).values
    p = (scores - maximum).exp()
    pe = effective(encode(p, pfmt, "device"), pfmt, "left")
    alpha = (maximum - maximum[:, -1:]).exp()
    numerator = (pe * alpha).reshape(128, -1) @ ve
    exact_den = (p * alpha).sum((1, 2))[:, None]
    matched_den = (pe * alpha).sum((1, 2))[:, None]
    result = []
    for denom_name, denominator in (("original", exact_den), ("matched", matched_den)):
        output = (numerator / denominator + v0).bfloat16().double()
        row_error = (output - reference).norm(dim=-1) / reference.norm(dim=-1).clamp_min(1e-300) * 100
        result.append(dict(
            denominator=denom_name, **metrics(output, reference),
            row_p95_l2_pct=float(torch.quantile(row_error, 0.95)),
            k_representation_l2_pct=metrics(ke, k)["l2_pct"],
            v_representation_l2_pct=metrics(ve, v)["l2_pct"],
            p_mass_ratio=float((matched_den / exact_den).mean()),
            p_zero_fraction=float((pe == 0).double().mean()),
            qk_lofi_matmuls=len(kformats), pv_lofi_matmuls=len(vformats),
        ))
    return result


def self_test():
    gen = torch.Generator().manual_seed(77)
    x = torch.randn((128, 128), generator=gen)
    for bits in (5, 7):
        y = round_significand(x, bits)
        assert torch.equal(y, round_significand(y, bits, "trunc"))
    for route in ("host", "device"):
        y = encode(x, "e5_b8", route)
        assert torch.equal(y, effective(y, "e5_b8", "right"))
    inputs = V1_NUMERICS.FRONTIER.inputs_for(1024, 77, "normal")
    q, k, v = [t.squeeze().double() for t in inputs]
    ref = torch.softmax(q @ k.T / math.sqrt(128), -1) @ v
    for record in evaluate(inputs, "exact", "none", "host", 77, ref):
        assert abs(record["l2_pct"] - metrics(ref.bfloat16(), ref)["l2_pct"]) < 1e-10


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    parser.add_argument("--lengths", nargs="+", type=int, default=[4096, 32768])
    parser.add_argument("--seeds", nargs="+", type=int, default=[1240, 1241])
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument("--distributions", nargs="+", default=["normal", "outliers", "scaled_qk", "scaled_down"])
    parser.add_argument("--preprocessing", nargs="+", default=["none"])
    parser.add_argument("--qkv-route", choices=("host", "device"), default="host")
    args = parser.parse_args()
    torch.set_num_threads(8)
    self_test()
    path = HERE / (args.label + ".jsonl")
    assert not path.exists(), "Use a fresh label"
    start = time.monotonic()
    with path.open("x") as output:
        def emit(record):
            line = json.dumps(record, allow_nan=False)
            output.write(line + "\n")
            output.flush()
            print(line, flush=True)
        emit(dict(kind="provenance", args=vars(args), hostname=platform.node(),
                  contract="Quantization and LoFi operand-bit model; FP64 arithmetic; NOT performance",
                  source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in (Path(__file__), V1 / "probe.py", V1 / "numerics.py")}))
        for length in args.lengths:
            assert length % 512 == 0
            for seed in args.seeds:
                for distribution in args.distributions:
                    inputs = V1_NUMERICS.FRONTIER.inputs_for(length, seed, distribution)
                    q, k, v = [t.squeeze().double() for t in inputs]
                    reference = torch.softmax(q @ k.T / math.sqrt(128), -1) @ v
                    for preprocessing in args.preprocessing:
                        for variant in args.variants:
                            for record in evaluate(inputs, variant, preprocessing, args.qkv_route, seed, reference):
                                emit(dict(kind="attention_model", length=length, seed=seed, distribution=distribution,
                                          preprocessing=preprocessing, variant=variant, **record))
        emit(dict(kind="completed", seconds=time.monotonic() - start))


if __name__ == "__main__":
    main()
