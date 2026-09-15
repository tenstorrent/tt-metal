# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""CPU-only audit model of Blackhole's per-phase, eight-product alignment.

Internal research artifact: the detailed mechanism is based on local
ttsim-private/src/tensix.cpp (mvmul and fpu_accum_normalize_encode), not on
an IEEE FP32 assumption. No ttnn import, device access, or hardware execution.
Replay the exact random-input construction from matmul_fp32_floor.py and
compare model output hashes/metrics to its saved device records.
"""
import argparse
import hashlib
import json
from pathlib import Path

import torch


# Pinned files inspected when constructing this model. They are not loaded or
# imported on the CPU runner, where the private simulator checkout is absent.
MODEL_PROVENANCE = {
    "ttsim_git_commit": "8c47553e0c28e4b2239497ffa4f5a44f02e1683f",
    "ttmetal_git_commit": "637d956c8874d356c9080e467b9a1664133fa780",
    "isa_git_commit": "5287a62727350bcef35f7b411d1b8a706172ec4c",
    "source_sha256": {
        "ttsim-private/src/tensix.cpp": "73facffeb1e7d508a256dcc8cc1bba1120e870964283b1e5ef65c9ce71e7b9d5",
        "ttsim-private/tests/rtl/tt_fp_lane/sim_main.cpp": "7b29a2968c99e81cfb133163d1aea7f81218600c6397d06eecb8b4209811975a",
        "ttsim-private/tests/rtl/tt_fp_lane/fp_lane_top.sv": "c676cf5ab1364c5487749f08ab5a1763d1ea0fc043628cd8b9557d1aef80545e",
        "tt_llk_blackhole/llk_lib/llk_math_matmul.h": "41e1b2e27caa7177f4c8c8e8b7b8f0c3e1475a4faab7b8b9e34053f0b25afa3d",
        "tt_llk_blackhole/llk_lib/llk_math_common.h": "4acbd6def0385cbfb085c3c90afd3802586102672afc3dafac33f69f56409668",
        "tt_llk_blackhole/common/inc/cunpack_common.h": "ec0582f7e16f37c9d4d39706d882357e8920f44e3a9d90848a3120b94f1774f9",
    },
}


def raw32(x):
    return x.float().contiguous().view(torch.int32).long() & 0xFFFFFFFF


def align_magnitude(magnitude, distance, negative=None):
    """Align an integer magnitude; SOP negative ties differ from DST ties."""
    shift = distance.clamp(1, 30)
    correction = 0 if negative is None else negative.long()
    aligned = (magnitude + (1 << (shift - 1)) - correction) >> shift
    aligned = torch.where(distance >= 31, 0, aligned)
    return torch.where(distance > 0, aligned, magnitude)


def accumulate(groups, group_exponents, dst, group_shift=13):
    """Two signed SOPs plus FP32 DST, including intermediate integer rounding."""
    negative = groups < 0
    magnitude = groups.abs() << group_shift
    dst_sign = dst >> 31
    dst_exponent = (dst >> 23) & 255
    dst_magnitude = torch.where(dst_exponent == 0, 0, (dst & 0x7FFFFF) | 0x800000)
    shared = torch.maximum(group_exponents.amax(-1), dst_exponent)
    magnitude = align_magnitude(magnitude, shared[..., None] - group_exponents, negative)
    dst_magnitude = align_magnitude(dst_magnitude, shared - dst_exponent)
    signed_sum = (torch.where(negative, -magnitude, magnitude).sum(-1)
                  + torch.where(dst_sign.bool(), -dst_magnitude, dst_magnitude))
    sign, mantissa = (signed_sum < 0).long(), signed_sum.abs()
    # Integer values are <2^31, so float64 log2 safely computes this bit length.
    bit_length = torch.floor(torch.log2(mantissa.clamp_min(1).double())).long() + 1
    shift_left = 24 - bit_length
    exponent = shared - shift_left
    right = (-shift_left).clamp_min(1)
    rounded_right = (mantissa + (1 << (right - 1))) >> right
    normalized = torch.where(shift_left < 0, rounded_right, mantissa << shift_left.clamp_min(0))
    exponent += ((normalized & 0x1000000) != 0).long()
    result = (sign << 31) | (exponent << 23) | (normalized & 0x7FFFFF)
    result = torch.where(exponent >= 255, (sign << 31) | (255 << 23), result)
    return torch.where((shared <= 0) | (exponent <= 0) | (mantissa == 0), 0, result)


def dot16(srca, srcb, dst, phase, product_alignment=True):
    """srca=[N,16] is right operand; srcb=[M,16] is left operand."""
    a, b = raw32(srca)[None, :, :], raw32(srcb)[:, None, :]
    ea, eb = (a >> 23) & 255, (b >> 23) & 255
    # Source register significands have ten explicit bits; BF16 has low3 zero.
    ma, mb = ((a >> 13) & 1023) | 1024, ((b >> 13) & 1023) | 1024
    ma = ((ma >> 1) & 31) if phase & 1 else (ma >> 6)
    mb = ((mb & 15) << 3) if phase & 2 else (mb >> 4)
    zero = (ea == 0) | (eb == 0)
    exponents = torch.where(zero, 0, ea + eb - 127 - (5 if phase & 1 else 0) - (7 if phase & 2 else 0))
    products = torch.where(zero, 0, ma * mb)
    negative = ((a ^ b) >> 31).bool()
    shape = products.shape[:-1] + (2, 8)
    exponents, products, negative = [x.reshape(shape) for x in (exponents, products, negative)]
    group_exponents = exponents.amax(-1)
    distance = (group_exponents[..., None] - exponents).clamp_max(30)
    # Crucial loss: round each phase-product magnitude on its group's common
    # exponent grid before applying its sign or adding the eight products.
    # Ablation retains thirteen extra alignment bits, but leaves phase order
    # and the modeled final FP32 accumulation/normalization unchanged.
    extra_bits = 0 if product_alignment else 13
    aligned = ((products << (extra_bits + 1)) + (1 << distance)) >> (distance + 1)
    groups = torch.where(negative, -aligned, aligned).sum(-1)
    groups = torch.where(group_exponents <= 0, 0, groups)
    group_exponents = torch.where(group_exponents <= 0, 0, group_exponents)
    return accumulate(groups, group_exponents, dst, group_shift=13 - extra_bits)


def matmul(a, b, fidelity=4, order="tile_phase_half", product_alignment=True):
    a, b = a.reshape(a.shape[-2:]), b.reshape(b.shape[-2:])
    assert a.dtype == b.dtype == torch.bfloat16 and a.shape[1] == b.shape[0]
    assert a.shape[1] % 32 == 0
    dst = torch.zeros((a.shape[0], b.shape[1]), dtype=torch.int64)
    if order == "tile_phase_half":
        visits = [(tile + half, phase) for tile in range(0, a.shape[1], 32)
                  for phase in range(fidelity) for half in (0, 16)]
    elif order == "half_phase":
        visits = [(start, phase) for start in range(0, a.shape[1], 16) for phase in range(fidelity)]
    elif order == "phase_global":
        visits = [(start, phase) for phase in range(fidelity) for start in range(0, a.shape[1], 16)]
    else:
        raise ValueError(order)
    for start, phase in visits:
        dst = dot16(b[start:start + 16].T, a[:, start:start + 16], dst, phase, product_alignment)
    return dst.int().view(torch.float32).reshape(1, 1, a.shape[0], b.shape[1])


def cases():
    # Keep generator consumption exactly aligned with the device probe.
    generator = torch.Generator().manual_seed(1240)
    for case in ("identity", "one_term", "dense32", "dense128", "positive128"):
        reduction = 32 if case in ("identity", "one_term", "dense32") else 128
        a = torch.randn((1, 1, 32, reduction), generator=generator).bfloat16()
        b = torch.randn((1, 1, reduction, 64), generator=generator).bfloat16()
        if case == "identity":
            a = torch.eye(32).reshape(1, 1, 32, 32).bfloat16()
        if case == "one_term":
            a[..., 1:] = 0
            b[..., 1:, :] = 0
        if case == "positive128":
            a, b = a.abs(), b.abs()
        yield case, a, b


def metrics(actual, expected):
    a, b = actual.double(), expected.double()
    raw = raw32(actual)
    return dict(l2_pct=float(100 * (a - b).norm() / b.norm()),
                max_abs=float((a - b).abs().max()), exact=int((a == b).sum()), elements=a.numel(),
                low_bits_nonzero={str(n): int(((raw & ((1 << n) - 1)) != 0).sum()) for n in (8, 10, 12, 13, 16)},
                output_sha256=hashlib.sha256(actual.contiguous().numpy().tobytes()).hexdigest())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True)
    parser.add_argument("--device-records", type=Path, default=Path(__file__).with_name("matmul-fp32-floor-v1.jsonl"))
    parser.add_argument("--orders", nargs="+", choices=("tile_phase_half", "half_phase", "phase_global"),
                        default=("tile_phase_half", "half_phase", "phase_global"))
    args = parser.parse_args()
    assert Path(args.label).name == args.label
    path = Path(__file__).with_name(args.label + ".json")
    assert not path.exists(), "Use a fresh label"
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    device = [json.loads(line) for line in args.device_records.read_text().splitlines()]
    lookup = {(r["case"], r["fidelity"]): r for r in device if "case" in r}
    records = []
    for case, a, b in cases():
        expected = a.double() @ b.double()
        for fidelity, phases in (("LoFi", 1), ("HiFi2", 2), ("HiFi4", 4)):
            for order in args.orders:
                actual = matmul(a, b, phases, order)
                measured = lookup[(case, fidelity)]
                result = dict(case=case, fidelity=fidelity, order=order, **metrics(actual, expected))
                result.update(device_l2_pct=measured["l2_pct"], device_exact=measured["exact"],
                              device_hash_match=result["output_sha256"] == measured["output_sha256"],
                              device_low_bits_match=result["low_bits_nonzero"] == measured["low_bits_nonzero"])
                records.append(result)
                print("ALIGNMENT_MODEL", json.dumps(result), flush=True)
        ablated = matmul(a, b, 4, "tile_phase_half", product_alignment=False)
        records.append(dict(case=case, fidelity="HiFi4", ablation="extra13_product_alignment_bits", **metrics(ablated, expected)))
    discriminators = []
    for index in (1, 7, 8, 15, 16):
        a, b = torch.zeros((1, 1, 32, 32), dtype=torch.bfloat16), torch.zeros((1, 1, 32, 64), dtype=torch.bfloat16)
        a[..., 0], a[..., index] = 1, 2**-12
        b[..., 0, :], b[..., index, :] = 1, 1
        actual = matmul(a, b, 4)
        discriminators.append(dict(small_term_index=index, actual=float(actual[0, 0, 0, 0]), expected=1 + 2**-12))
    record = dict(cpu_only=True, threads=4, records=records, discriminators=discriminators,
                  model_provenance=MODEL_PROVENANCE,
                  device_provenance=[r for r in device if r.get("kind") == "provenance"],
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  device_records_sha256=hashlib.sha256(args.device_records.read_bytes()).hexdigest(),
                  model_source="Local ttsim-private/src/tensix.cpp; exact public API kernel replay order is inferred, tested by hash",
                  warning="Internal research model; not an IEEE guarantee or a public ISA precision specification")
    path.write_text(json.dumps(record, indent=2) + "\n")
    print("ALIGNMENT_DISCRIMINATORS", json.dumps(discriminators), flush=True)
    print("RESULT_PATH", path, flush=True)


if __name__ == "__main__":
    main()
