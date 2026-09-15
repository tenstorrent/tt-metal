# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Attribute native BFP8 rounding by route; independent, ordinary-BF16 models.

No frozen quantizer is changed. RNA means nearest, ties away from zero; RNE
means nearest, ties to even. Model names describe hypotheses, not route promises.
This is a conversion/numerics probe, not an attention or performance benchmark.
"""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def round_significand7(x, ties):
    """Seven TOTAL significant bits, on finite normal BF16 values or zero."""
    assert ties in ("even", "away")
    raw = x.float().contiguous().view(torch.int32).long() & 0xFFFFFFFF
    shift = 17  # FP32 has 24 total significant bits.
    bias = (1 << (shift - 1)) - 1
    increment = ((raw >> shift) & 1) if ties == "even" else torch.ones_like(raw)
    rounded = (raw + bias + increment) & 0xFFFE0000
    return rounded.int().view(torch.float32)


def shared_b8(x, ties):
    """Single shared-exponent BFP8 conversion of exactly BF16-representable input.

    Groups are 16 adjacent columns. Exponent is selected from the input to THIS
    stage, so preceding per-datum exponent carries are not silently ignored.
    Magnitude codes saturate at127. No exponent clamping or subnormal contract.
    """
    assert ties in ("even", "away")
    values = x.float()
    assert torch.equal(values.bfloat16().float(), values)
    assert torch.isfinite(values).all()
    assert ((values == 0) | (values.abs() >= 2.0**-126)).all()
    groups = values.reshape(-1, 16)
    raw = groups.abs().contiguous().view(torch.int32).long()
    exponent = (raw >> 23) & 255
    shared = exponent.amax(-1, keepdim=True)
    significand = torch.where(exponent == 0, 0, ((raw >> 16) & 127) | 128)
    shift = shared - exponent + 1
    safe_shift = shift.clamp_max(9)
    integer = significand >> safe_shift
    remainder = significand & ((1 << safe_shift) - 1)
    halfway = 1 << (safe_shift - 1)
    tie_up = ((integer & 1) != 0) if ties == "even" else torch.ones_like(integer, dtype=torch.bool)
    increment = (remainder > halfway) | ((remainder == halfway) & tie_up)
    integer = torch.where(shift > 8, 0, integer + increment.long()).clamp_max(127)
    return (torch.ldexp(integer.float(), (shared - 133).int()) * groups.sign()).reshape_as(values)


def make_input(count, distribution, seed):
    shape = (1, 1, count // 128, 128)
    if distribution == "normal":
        return torch.randn(shape, generator=torch.Generator().manual_seed(seed)).bfloat16()
    assert distribution == "dense_fixed_ties"
    group = torch.arange(count // 16).reshape(-1, 1)
    lane = torch.arange(16).reshape(1, -1)
    mantissa = (13 * group + 17 * lane) % 128
    delta = (lane + group // 128) % 8
    values = torch.ldexp(1 + mantissa.float() / 128, -delta.int()).clamp_max(1.75)
    # Exactly1.75 pins the group's exponent before AND after per-datum rounding.
    # All mantissas still appear at smaller relative exponents. Both signs, eight
    # relative exponents, zeros, and varied anchor columns exercise layout/ties.
    values[group.flatten(), group.flatten() % 16] = 1.75
    values *= torch.where((group + lane) % 2 == 0, 1.0, -1.0)
    values = torch.ldexp(values, (group % 17 - 8).int())
    values[::127] = 0
    return values.reshape(shape).bfloat16()


def metrics(actual, reference):
    a, r = actual.double().flatten(), reference.double().flatten()
    finite = bool(torch.isfinite(a).all())
    if not finite:
        return dict(finite=False, mismatch=int((actual != reference).sum()))
    power = r.square().sum()
    gain = (a * r).sum() / power
    error = a - r
    return dict(
        finite=True,
        mismatch=int((actual != reference).sum()),
        mismatch_pct=100 * int((actual != reference).sum()) / actual.numel(),
        gain=float(gain),
        gain_error_pct=float(100 * (gain - 1)),
        l2_pct=float(100 * error.norm() / power.sqrt()),
        gain_removed_l2_pct=float(100 * (a - gain * r).norm() / power.sqrt()),
        max_absolute_error=float(error.abs().max()),
        mean_error=float(error.mean()),
    )


def model_predictions(x):
    return {
        "single_shared_RNA": shared_b8(x, "away"),
        "perdatum_RNE7_then_shared_RNA": shared_b8(round_significand7(x, "even"), "away"),
        "perdatum_RNA7_then_shared_RNA": shared_b8(round_significand7(x, "away"), "away"),
        "single_shared_RNE": shared_b8(x, "even"),
    }


def device_outputs(device, x, cores):
    import ttnn

    spec = importlib.util.spec_from_file_location("b8_pipeline_preprocess", HERE / "preprocess.py")
    prep = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(prep)
    src = ttnn.from_torch(x, device=device, layout=ttnn.TILE_LAYOUT)
    outputs, contracts = {}, {}
    cast = ttnn.typecast(src, ttnn.bfloat8_b)
    outputs["typecast_bf16_to_b8"] = ttnn.to_torch(cast).float()
    contracts["typecast_bf16_to_b8"] = "High-level ttnn.typecast of BF16 device tensor; internal DST not forced"
    host = ttnn.from_torch(x.float(), dtype=ttnn.bfloat8_b, device=device, layout=ttnn.TILE_LAYOUT)
    outputs["host_pack_b8"] = ttnn.to_torch(host).float()
    contracts["host_pack_b8"] = "ttnn.from_torch FP32 host values equal to BF16 input; no device typecast"
    identities = {}
    for fp32_dst in (False, True):
        suffix = "fp32_dst" if fp32_dst else "bf16_dst"
        out, invoke, used_cores = prep.build(
            device, src, bits=8, output_format="b8", ncores=cores, batch=4, fp32_dst=fp32_dst
        )
        invoke()
        name = "preprocess_bits8_" + suffix
        outputs[name] = ttnn.to_torch(out).float()
        contracts[name] = (
            f"Existing preprocess.build(bits=8, output_format=b8, fp32_dst={fp32_dst}, batch=4); "
            f"{used_cores} cores. RNE8 SFPU step should be identity on BF16 input."
        )
        identity, identity_call, _ = prep.build(
            device, src, bits=8, output_format="bf16", ncores=cores, batch=4, fp32_dst=fp32_dst
        )
        identity_call()
        identities["bits8_to_bf16_" + suffix] = ttnn.to_torch(identity).float()
    return outputs, identities, contracts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True)
    parser.add_argument("--count", type=int, default=524288)
    parser.add_argument(
        "--distributions", nargs="+", choices=("normal", "dense_fixed_ties"), default=["normal", "dense_fixed_ties"]
    )
    parser.add_argument("--cores", type=int, default=110)
    parser.add_argument("--seed", type=int, default=1252)
    parser.add_argument("--host-only", action="store_true")
    args = parser.parse_args()
    assert Path(args.label).name == args.label, "Use a fresh plain label"
    assert args.count > 0 and args.count % 4096 == 0 and args.cores > 0
    assert len(set(args.distributions)) == len(args.distributions)
    path = HERE / (args.label + ".json")
    artifacts = [HERE / (args.label + "-" + distribution + ".pt") for distribution in args.distributions]
    assert not path.exists() and not any(p.exists() for p in artifacts), "Use a fresh label"
    torch.set_num_threads(8)
    device = None
    if not args.host_only:
        import ttnn

        device = ttnn.open_device(device_id=0)
    records, sanity_pass = [], True
    try:
        for distribution, artifact in zip(args.distributions, artifacts):
            x = make_input(args.count, distribution, args.seed)
            predictions = model_predictions(x)
            outputs, identities, contracts = (
                device_outputs(device, x, args.cores) if device is not None else ({}, {}, {})
            )
            identity_checks = {name: metrics(out, x) for name, out in identities.items()}
            sanity_pass &= all(check["finite"] and check["mismatch"] == 0 for check in identity_checks.values())
            routes, mismatch_examples = {}, {}
            for route, actual in outputs.items():
                sanity_pass &= bool(torch.isfinite(actual).all())
                comparisons = {name: metrics(actual, expected) for name, expected in predictions.items()}
                exact_models = [name for name, comparison in comparisons.items() if comparison["mismatch"] == 0]
                routes[route] = dict(
                    contract=contracts[route],
                    vs_input=metrics(actual, x),
                    vs_models=comparisons,
                    exact_matching_models=exact_models,
                    interpretation=(
                        "No listed model matches exactly" if not exact_models else "Exact on this dataset only"
                    ),
                )
                mismatch_examples[route] = {}
                for name, expected in predictions.items():
                    indices = (actual.flatten() != expected.flatten()).nonzero().flatten()[:64]
                    mismatch_examples[route][name] = dict(
                        flat_index=indices,
                        input=x.flatten()[indices],
                        actual=actual.flatten()[indices],
                        predicted=expected.flatten()[indices],
                        input_group=x.float().reshape(-1, 16)[indices // 16],
                    )
                print("B8_ROUTE", distribution, route, json.dumps(routes[route]), flush=True)
            pairs = {}
            names = list(outputs)
            for i, first in enumerate(names):
                for second in names[i + 1 :]:
                    pairs[first + "__vs__" + second] = metrics(outputs[first], outputs[second])
            torch.save(
                dict(
                    input_bf16=x,
                    downloaded_outputs=outputs,
                    bits8_identity_outputs=identities,
                    predicted_outputs=predictions,
                    mismatch_examples=mismatch_examples,
                ),
                artifact,
            )
            records.append(
                dict(
                    distribution=distribution,
                    input_shape=list(x.shape),
                    numel=x.numel(),
                    input_sha256=hashlib.sha256(x.contiguous().view(torch.uint16).numpy().tobytes()).hexdigest(),
                    model_vs_input={name: metrics(value, x) for name, value in predictions.items()},
                    routes=routes,
                    route_pairs=pairs,
                    identity_checks=identity_checks,
                    matrices_and_failure_examples=str(artifact.relative_to(ROOT)),
                )
            )
        sources = [Path(__file__).resolve(), HERE / "preprocess.py", *sorted((HERE / "preprocess").glob("*.cpp"))]
        result = dict(
            **vars(args),
            records=records,
            identity_and_finite_checks_pass=bool(sanity_pass),
            source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
            model_contract=(
                "Independent finite-normal-BF16/zero integer models; native groups16; exponent reselected at each stage; "
                "BFP8 magnitude127 saturation; no frozen-v1 exponent clamp. Gain=(actual·reference)/(reference·reference). "
                "Exact matches identify these datasets, not an architectural guarantee or all-range contract."
            ),
            note="Unknown rounding model is reported, not asserted away; complete outputs and mismatch groups are saved",
        )
        path.write_text(json.dumps(result, indent=2) + "\n")
        print("RESULT", json.dumps(result), flush=True)
        assert sanity_pass, "Nonfinite output or bits8 identity failure; downloaded matrices saved"
    finally:
        if device is not None:
            ttnn.close_device(device)


if __name__ == "__main__":
    main()
