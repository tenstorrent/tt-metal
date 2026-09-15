# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Isolated accurate-attention K-shift diagnostic; no kernel or SKU changes.

The unchanged fullchip accurate builder binds a private BF16 K buffer. This
experiment optionally overwrites ONLY that buffer with device-centered K,
always reading an independently uploaded immutable original-K device backup.
All mean/subtraction work is replayed and included in combined timing.
"""

import argparse
import gc
import hashlib
import json
from pathlib import Path

import torch
import ttnn

import center_mean as MEAN
import fullchip as F
import native_exp_qualification as QUAL

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def digest(tensor):
    return hashlib.sha256(tensor.contiguous().view(torch.uint16).numpy().tobytes()).hexdigest()


def source_hashes():
    hashes = QUAL.source_hashes()
    files = [Path(__file__).resolve(), HERE / "center_mean.py"]
    for relative in ("ttnn/cpp/ttnn/operations/eltwise/binary",
                     "ttnn/cpp/ttnn/operations/reduction/generic"):
        directory = ROOT / relative
        files.extend(directory.rglob("*.cpp"))
        files.extend(directory.rglob("*.hpp"))
    hashes.update({str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
    return hashes


def build(device, inputs, centered, args):
    config = argparse.Namespace(
        variant="accurate", q_chunk=256, length=inputs[0].shape[2], heads=args.heads,
        cores=min(args.cores, args.heads * inputs[0].shape[2] // 256),
        q_prescale=1.0, center_k=False, mean_mode="bf16_fpu", b8_rne=False,
        bfp8_pack_precise=False, check_preprocess=False, fix_correction=False,
        exp_degree=3, native_exp=False, reader_chain=True, reader_split=False,
        reader_linear_k=False, read_barrier_tiles=2,
    )
    private_inputs, prepared, output, attention, unused_preprocess, unused_combined, info = F.build(device, config, inputs)
    private_k = prepared[1]
    # For accurate mode the builder's "originals" and "prepared" lists alias.
    # They are PRIVATE WORKING COPIES here, not our immutable original input.
    assert private_inputs[1].buffer_address() == private_k.buffer_address()
    original_k = ttnn.from_torch(inputs[1], device=device, layout=ttnn.TILE_LAYOUT)
    assert original_k.buffer_address() != private_k.buffer_address()
    assert torch.equal(ttnn.to_torch(original_k).bfloat16(), inputs[1])
    assert torch.equal(ttnn.to_torch(private_k).bfloat16(), inputs[1])
    address = private_k.buffer_address()
    mean_invoke = None
    repeated_bias = None
    if centered:
        repeated_bias, mean_invoke = MEAN.build(device, original_k, "bf16_fpu")

    def subtract():
        bias = mean_invoke.last_mean
        assert bias is not None and bias.dtype == ttnn.bfloat16
        assert tuple(bias.shape) == (1, args.heads, 1, 128)
        assert private_k.buffer_address() == address
        # Current binary binding explicitly supports output_tensor and routes
        # BF16 fast_and_approximate_mode=False through SFPU with RNE output.
        result = ttnn.subtract(original_k, bias, fast_and_approximate_mode=False,
                               output_tensor=private_k)
        assert result.buffer_address() == address

    def preprocess():
        if centered:
            mean_invoke()
            subtract()

    def combined():
        preprocess()
        attention()

    # Retain all captured buffers/mean intermediates until trace release.
    combined.buffers = (private_inputs, prepared, original_k, repeated_bias, mean_invoke)
    return dict(config=config, kernel=info, original_k=original_k, private_k=private_k,
                output=output, attention=attention, preprocess=preprocess, combined=combined,
                mean=mean_invoke, subtract=subtract if centered else None,
                private_inputs=private_inputs, k_address=address)


def verify_preprocessing(state, inputs, centered):
    original = ttnn.to_torch(state["original_k"]).bfloat16()
    assert torch.equal(original, inputs[1]), "Immutable device original K changed"
    actual = ttnn.to_torch(state["private_k"]).bfloat16()
    assert bool(torch.isfinite(actual).all())
    assert state["private_k"].buffer_address() == state["k_address"]
    if centered:
        bias = ttnn.to_torch(state["mean"].last_mean).bfloat16()
        assert tuple(bias.shape) == (1, inputs[1].shape[1], 1, 128)
        assert bool(torch.isfinite(bias).all())
        # Device's actual rounded compact mean, never a host-generated bias.
        expected = (inputs[1].float() - bias.float()).bfloat16()
        ideal_shift = inputs[1].double() - bias.double()
        exact_mean = inputs[1].double().mean(dim=2, keepdim=True)
        mean_details = dict(
            accuracy_vs_fp64=F.REPRO.metrics(bias, exact_mean),
            mismatches_vs_fp64_mean_rounded_bf16=int((bias != exact_mean.bfloat16()).sum()),
            sha256=digest(bias),
        )
    else:
        bias, mean_details = None, None
        expected, ideal_shift = inputs[1], inputs[1].double()
    mismatches = int((actual != expected).sum())
    assert mismatches == 0, f"BF16 shifted-K oracle mismatch: {mismatches}"
    for index in (0, 2):
        assert torch.equal(ttnn.to_torch(state["private_inputs"][index]).bfloat16(), inputs[index])
    return actual, ideal_shift, dict(
        centered=centered, shifted_bf16_mismatches=mismatches, immutable_original_k_verified=True,
        original_k_sha256=digest(original), prepared_k_sha256=digest(actual), mean=mean_details,
        bf16_shift_rounding=F.REPRO.metrics(actual, ideal_shift),
        original_k_address=state["original_k"].buffer_address(), prepared_k_address=state["k_address"],
        address_contract="Only private builder K working copy changes; immutable device backup and host inputs unchanged",
    )


def run_case(device, inputs, reference, rows, centered, args):
    original_hashes = [digest(x) for x in inputs]
    state = build(device, inputs, centered, args)
    state["combined"]()
    actual_k, ideal_shift, preparation = verify_preprocessing(state, inputs, centered)
    actual = ttnn.to_torch(state["output"]).bfloat16()
    assert bool(torch.isfinite(actual).all()), "Nonfinite accurate output; no metrics/timing"
    ideal_shift_ref = F.REPRO.reference(inputs[0][..., rows, :], ideal_shift, inputs[2])
    shifted_bf16_ref = F.REPRO.reference(inputs[0][..., rows, :], actual_k, inputs[2])
    diagnostic = dict(
        original_reference=F.REPRO.metrics(actual[..., rows, :], reference),
        kernel_vs_actual_prepared_bf16_reference=F.REPRO.metrics(actual[..., rows, :], shifted_bf16_ref),
        bf16_shift_attention_drift=F.REPRO.metrics(shifted_bf16_ref, reference),
        exact_shift_attention_invariance=F.REPRO.metrics(ideal_shift_ref, reference),
        original_bf16_output_rounding_floor=F.REPRO.metrics(reference.bfloat16(), reference),
    )
    # A rounded constant mean is still an exact softmax invariance before the
    # centered-K BF16 spill. This explicit control catches wrong-axis shifts.
    assert diagnostic["exact_shift_attention_invariance"]["l2_pct"] < 1e-8
    trace = ttnn.begin_trace_capture(device, cq_id=0)
    state["combined"]()
    ttnn.end_trace_capture(device, trace, cq_id=0)
    try:
        # Two replays also detect accidental repeated subtraction in place.
        for _ in range(2):
            ttnn.execute_trace(device, trace, cq_id=0, blocking=True)
            assert torch.equal(actual, ttnn.to_torch(state["output"]).bfloat16())
            assert torch.equal(actual_k, ttnn.to_torch(state["private_k"]).bfloat16())
    finally:
        ttnn.release_trace(device, trace)
    timings = {}
    if args.iters:
        calls = {"attention": state["attention"]}
        if centered:
            calls.update(mean=state["mean"], subtract=state["subtract"], preprocessing=state["preprocess"])
        calls["combined"] = state["combined"]
        timings = {name: F.timed(device, call, args) for name, call in calls.items()}
    replay_k, _, replay_preparation = verify_preprocessing(state, inputs, centered)
    assert torch.equal(actual_k, replay_k)
    assert torch.equal(actual, ttnn.to_torch(state["output"]).bfloat16())
    assert preparation["mean"] == replay_preparation["mean"]
    assert [digest(x) for x in inputs] == original_hashes
    flops = 4 * args.heads * inputs[0].shape[2] ** 2 * 128
    return dict(
        center_k=centered, config=vars(state["config"]), kernel=state["kernel"],
        preparation=preparation, metrics=diagnostic, timings=timings, useful_attention_flops=flops,
        trace_equal=True, immutable_inputs_verified=True, output_sha256=digest(actual),
        mean_implementation="MEAN.build bf16_fpu; compact last_mean consumed; unused repeated-bias materialization is included in mean and combined cost" if centered else None,
        timing_scope="Device trace replay; immutable input uploads/allocation excluded; combined includes every device mean/subtraction and attention invocation. Mean+subtract are disjoint; preprocessing/combined are aggregate measurements.",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True)
    parser.add_argument("--lengths", type=int, nargs="+", default=[1024, 32768])
    parser.add_argument("--distributions", nargs="+", choices=("normal", "common_k"), default=["normal", "common_k"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[1240])
    parser.add_argument("--heads", type=int, default=2)
    parser.add_argument("--cores", type=int, default=22)
    parser.add_argument("--sample-rows", type=int, default=128, help="Lengths <=1024 always check all rows; larger lengths sample this many")
    parser.add_argument("--iters", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--trace-repeats", type=int, default=5)
    args = parser.parse_args()
    assert Path(args.label).name == args.label
    assert args.heads > 0 and args.cores >= args.heads and args.cores % args.heads == 0
    assert all(n > 0 and n % 512 == 0 for n in args.lengths)
    assert args.sample_rows > 0 and args.iters >= 0 and args.warmup >= 0 and args.trace_repeats > 0
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    hashes = source_hashes()
    with (HERE / (args.label + ".jsonl")).open("x") as stream:
        def emit(record):
            line = json.dumps(record, allow_nan=False)
            stream.write(line + "\n")
            stream.flush()
            print(line, flush=True)

        emit(dict(kind="provenance", args=vars(args), source_sha256=hashes,
                  purpose="Hardware-error attribution only; unchanged fullchip accurate control and locked kernels",
                  reference="Original BF16 Q/K/V, FP64 attention; actual BF16 K-shift and exact-shift references separate"))
        for length in args.lengths:
            count = length if length <= 1024 else min(args.sample_rows, length)
            rows = torch.linspace(0, length - 1, count).long().unique()
            for seed in args.seeds:
                for distribution in args.distributions:
                    inputs = F.REPRO.make_inputs(args.heads, length, length, 128, seed, distribution)
                    reference = F.REPRO.reference(inputs[0][..., rows, :], inputs[1], inputs[2])
                    device = ttnn.open_device(device_id=0, trace_region_size=16777216)
                    try:
                        for centered in (False, True):
                            record = run_case(device, inputs, reference, rows, centered, args)
                            assert source_hashes() == hashes, "Pinned sources changed"
                            emit(dict(kind="diagnostic", length=length, seed=seed, distribution=distribution,
                                      sampled_query_rows=rows.tolist(), all_output_finite=True,
                                      all_query_rows_referenced=len(rows) == length, **record))
                            gc.collect()
                    finally:
                        ttnn.close_device(device)
        emit(dict(kind="complete", sources_unchanged=source_hashes() == hashes))


if __name__ == "__main__":
    main()
