# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Accuracy-only qualification of native-exp LoFi against an accurate control.

Private fullchip chain harness, fixed Q256/K512/D128 and existing input slots.
All input preprocessing executes on device. References use original BF16 inputs,
FP64 softmax and every KV row. No performance claim follows from this suite.
"""

import argparse
import gc
import hashlib
import json
from pathlib import Path

import torch
import ttnn

import fullchip as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
KNOWN = (
    "normal", "outliers", "scaled_qk", "scaled_down", "biased_v",
    "common_q", "common_k", "common_v", "constant_v", "uniform",
    "uniform_constant_v", "channel_k", "channel_v",
)


def make_inputs(length, heads, seed, distribution):
    assert distribution in KNOWN
    base = "normal" if distribution in ("scaled_down", "channel_k", "channel_v") else distribution
    values = F.REPRO.make_inputs(heads, length, length, 128, seed, base)
    if distribution == "scaled_down":
        values[:2] = [(x.float() * 0.25).bfloat16() for x in values[:2]]
    elif distribution in ("channel_k", "channel_v"):
        index = 1 if distribution == "channel_k" else 2
        values[index][..., ::16] *= 32
    return values


def source_hashes():
    files = [Path(__file__).resolve(), HERE / "fullchip.py", HERE / "preprocess.py",
             HERE / "numerics.py", HERE / "bfp4_round.py", HERE / "bfp8_round.py",
             HERE / "q_prescale.py", HERE / "center_mean.py", HERE / "center_preprocess.py",
             HERE / "safe_rescale.hpp", HERE / "fast_correction.hpp",
             HERE / "exp_native.hpp", HERE / "exp_refiner.hpp",
             HERE / "streaming/compute_streaming.hpp",
             HERE.parent / "bfp4-lofi-v1/probe.py", HERE.parent / "bfp4-lofi-v1/numerics.py",
             HERE.parent / "frontier-accuracy-v1/run.py",
             ROOT / "tests/ttnn/unit_tests/operations/sdpa/repro_sdpa_l2.py"]
    for directory in ("fullchip", "preprocess", "bfp4_round", "bfp8_round", "q_prescale", "center_preprocess"):
        files.extend(p for p in (HERE / directory).iterdir() if p.suffix in (".cpp", ".hpp", ".h"))
    frozen = ROOT / "experiments/sdpa-l2/hybrid-mixed-v1/candidate"
    files.extend(p for p in frozen.rglob("*") if p.suffix in (".hpp", ".h"))
    files.extend([
        ROOT / "ttnn/cpp/ttnn/kernel/dataflow/generate_bcast_scalar.hpp",
        ROOT / "tt_metal/tt-llk/tt_llk_blackhole/llk_lib/llk_math_matmul.h",
        ROOT / "tt_metal/tt-llk/tt_llk_blackhole/llk_lib/llk_unpack_AB_matmul.h",
        ROOT / "tt_metal/tt-llk/tt_llk_blackhole/llk_lib/llk_pack.h",
    ])
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(set(files))}


def metrics(actual, reference, v):
    mean_v = v.mean(dim=2, keepdim=True, dtype=torch.float64)
    ref_residual = reference - mean_v
    residual_norm = ref_residual.norm()
    error = actual.double() - reference
    constant = torch.equal(v, v[:, :, :1, :].expand_as(v))
    rounded = reference.bfloat16()
    return dict(
        original=F.REPRO.metrics(actual, reference),
        bf16_output_rounding_floor=F.REPRO.metrics(rounded, reference),
        residual_l2_pct=None if constant or residual_norm == 0 else float(100 * error.norm() / residual_norm),
        residual_bf16_rounding_floor_l2_pct=None if constant or residual_norm == 0 else
            float(100 * (rounded.double() - reference).norm() / residual_norm),
        residual_reference_rms=0.0 if constant else float(ref_residual.square().mean().sqrt()),
        absolute_error_rms=float(error.square().mean().sqrt()),
        residual_definition="Subtract same FP64 mean(original V) from both outputs; no gain fitting",
        residual_relative_undefined=constant or bool(residual_norm == 0),
    )


def run_variant(device, inputs, reference, rows, variant, center_k, args):
    config = argparse.Namespace(
        variant=variant, q_chunk=256, length=inputs[0].shape[2], heads=args.heads,
        cores=min(args.cores, args.heads * inputs[0].shape[2] // 256),
        q_prescale=1.0, center_k=center_k, mean_mode="bf16_fpu", b8_rne=False,
        bfp8_pack_precise=False, check_preprocess=args.check_preprocess,
        fix_correction=False, exp_degree=3, native_exp=variant.startswith("lofi_"),
        reader_chain=True, reader_split=False, reader_linear_k=False, read_barrier_tiles=2,
    )
    originals, prepared, output, attention, preprocess, combined, info = F.build(device, config, inputs)
    combined()
    actual = ttnn.to_torch(output).bfloat16()
    finite = bool(torch.isfinite(actual).all())
    record = dict(config=vars(config), kernel=info, finite=finite,
                  nonfinite_count=int((~torch.isfinite(actual)).sum()),
                  output_sha256=hashlib.sha256(actual.view(torch.uint16).numpy().tobytes()).hexdigest())
    if not finite:
        record["status"] = "FAIL_NONFINITE_NO_METRICS_OR_TIMING"
        return record
    record["metrics"] = metrics(actual[..., rows, :], reference, inputs[2])
    trace = ttnn.begin_trace_capture(device, cq_id=0)
    combined()
    ttnn.end_trace_capture(device, trace, cq_id=0)
    try:
        ttnn.execute_trace(device, trace, cq_id=0, blocking=True)
        record["trace_equal"] = torch.equal(actual, ttnn.to_torch(output))
        assert record["trace_equal"], "Device preprocessing + attention trace changed output"
    finally:
        ttnn.release_trace(device, trace)
    record["status"] = "FINITE_REPLAY_IDENTICAL"
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True)
    parser.add_argument("--lengths", type=int, nargs="+", default=[32768])
    parser.add_argument("--seeds", type=int, nargs="+", default=[1240, 1241])
    parser.add_argument("--distributions", nargs="+", choices=KNOWN, default=list(KNOWN))
    parser.add_argument("--variants", nargs="+", choices=("lofi_fp32_b8", "lofi_fp32_b4", "accurate"),
                        default=["lofi_fp32_b8", "lofi_fp32_b4", "accurate"])
    parser.add_argument("--heads", type=int, default=2)
    parser.add_argument("--cores", type=int, default=22)
    parser.add_argument("--sample-rows", type=int, default=128)
    parser.add_argument("--check-preprocess", action="store_true")
    args = parser.parse_args()
    assert Path(args.label).name == args.label
    assert args.heads > 0 and args.cores >= args.heads and args.cores % args.heads == 0
    assert all(n > 0 and n % 512 == 0 for n in args.lengths)
    assert args.sample_rows > 0
    path = HERE / (args.label + ".jsonl")
    assert not path.exists(), "Use a fresh label"
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    hashes = source_hashes()
    with path.open("x") as stream:
        def emit(value):
            stream.write(json.dumps(value, allow_nan=False) + "\n")
            stream.flush()
            print(json.dumps(value, allow_nan=False), flush=True)

        emit(dict(kind="provenance", args=vars(args), source_sha256=hashes,
                  scope="All heads/all KV, explicitly sampled Q rows; every device output finite/hash checked; no timing",
                  distributions="scaled_down multiplies BF16 Q and K by0.25; channel_k/v multiplies every16th channel by32; common offsets32"))
        for length in args.lengths:
            rows = torch.linspace(0, length - 1, min(args.sample_rows, length)).long().unique()
            for seed in args.seeds:
                for distribution in args.distributions:
                    inputs = make_inputs(length, args.heads, seed, distribution)
                    reference = F.REPRO.reference(inputs[0][..., rows, :], inputs[1], inputs[2])
                    input_hashes = [hashlib.sha256(x.view(torch.uint16).numpy().tobytes()).hexdigest() for x in inputs]
                    device = ttnn.open_device(device_id=0, trace_region_size=16777216)
                    try:
                        for variant in args.variants:
                            for centered in ([False, True] if distribution == "common_k" and variant.startswith("lofi_") else [False]):
                                record = run_variant(device, inputs, reference, rows, variant, centered, args)
                                assert source_hashes() == hashes, "Suite sources changed during qualification"
                                emit(dict(kind="qualification", length=length, seed=seed, distribution=distribution,
                                          variant=variant, center_k=centered, sampled_query_rows=rows.tolist(),
                                          original_input_sha256=input_hashes, **record))
                                gc.collect()
                    finally:
                        ttnn.close_device(device)
                    del inputs, reference
                    gc.collect()
        emit(dict(kind="complete", sources_unchanged=source_hashes() == hashes))


if __name__ == "__main__":
    main()
