# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Original-V quantization plus device mean-error correction; no V centering.

The unchanged qualified BF16 fullchip builders quantize ORIGINAL V. This
wrapper computes delta=mean(original V)-mean(actual LoFi-consumed Vq) and
adds delta once after attention. Device means, BFP decode, optional SrcA5
truncation, subtraction/slice and output epilogue are explicitly timed.
This cancels only unweighted DC quantization error in ideal arithmetic.
"""
import argparse
import hashlib
import json
from pathlib import Path

import torch
import ttnn

import center_mean as MEAN
import effective_v_preprocess as EFFECTIVE
import value_centered_b8_fullchip as BASE8
import value_centered_fullchip as BASE4

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
REPRO = BASE8.REPRO


def tensor_sha256(tensor):
    """Bitwise BF16 hash, including signed zero; tensor must be on the CPU."""
    assert tensor.dtype == torch.bfloat16
    return hashlib.sha256(tensor.contiguous().view(torch.uint16).numpy().tobytes()).hexdigest()


def check_input_immutability(originals, inputs, expected_hashes):
    cpu_hashes = [tensor_sha256(x) for x in inputs]
    device_hashes = [tensor_sha256(ttnn.to_torch(x)) for x in originals]
    result = dict(
        cpu_input_sha256=cpu_hashes, device_input_sha256=device_hashes,
        cpu_inputs_unchanged=cpu_hashes == expected_hashes,
        device_inputs_unchanged=device_hashes == expected_hashes,
    )
    assert result["cpu_inputs_unchanged"], "Original CPU BF16 inputs changed"
    assert result["device_inputs_unchanged"], "Original device BF16 inputs differ from original CPU bits"
    return result


def qualify_trace_replays(device, combined, out, expected_hash, originals, inputs, input_hashes):
    """Always replay a one-invocation combined trace twice, including iters=0.

    Qualification is separate from timed() and its iteration count. Hashing
    outputs and downloading original tensors are correctness overhead only.
    """
    before = check_input_immutability(originals, inputs, input_hashes)
    trace = ttnn.begin_trace_capture(device, cq_id=0)
    combined()
    ttnn.end_trace_capture(device, trace, cq_id=0)
    replay_hashes = []
    try:
        for _ in range(2):
            ttnn.execute_trace(device, trace, cq_id=0, blocking=True)
            replay_hash = tensor_sha256(ttnn.to_torch(out))
            replay_hashes.append(replay_hash)
            assert replay_hash == expected_hash, "Combined correctness trace changed BF16 output bits"
    finally:
        ttnn.release_trace(device, trace)
    after = check_input_immutability(originals, inputs, input_hashes)
    return dict(
        explicit_combined_trace_replays=len(replay_hashes), invocations_per_replay=1,
        replay_output_sha256=replay_hashes, output_bitwise_equal=True,
        cpu_inputs_unchanged=True, device_inputs_unchanged=True,
        before=before, after=after,
        timing_scope="Mandatory correctness-only capture/replays, excluded from reported timings",
    )


def build_correction(device, original, packed, args):
    """Return compact BF16 delta, recompute callable, metadata.

    All tensors are stable device allocations except MEAN's internally
    retained reduction outputs. No host-computed bias is uploaded.
    """
    b8 = args.kv_formats == "b8_b8"
    original_bias, original_mean = MEAN.build(device, original, args.mean_mode)
    decoded = ttnn.allocate_tensor_on_device(
        original.shape, ttnn.bfloat16, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG
    )
    represented, truncate, truncate_cores = decoded, None, 0
    if b8:
        represented, truncate, truncate_cores = EFFECTIVE.build(device, decoded, args.cores)
    represented_bias, represented_mean = MEAN.build(device, represented, args.mean_mode)
    delta = ttnn.allocate_tensor_on_device(
        original_bias.shape, ttnn.bfloat16, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG
    )
    compact_shape = [1, original.shape[1], 1, 128]
    compact = ttnn.allocate_tensor_on_device(
        compact_shape, ttnn.bfloat16, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG
    )

    def invoke():
        original_mean()
        ttnn.typecast(packed, ttnn.bfloat16, output_tensor=decoded)
        if truncate is not None:
            truncate()
        represented_mean()
        ttnn.subtract(
            original_bias, represented_bias, output_tensor=delta, fast_and_approximate_mode=False
        )
        ttnn.slice(
            delta, [0, 0, 0, 0], compact_shape, [1, 1, 1, 1],
            output_tensor=compact, memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    # Retain every buffer and callable through all trace capture/replay.
    invoke.original_bias = original_bias
    invoke.represented_bias = represented_bias
    invoke.decoded = decoded
    invoke.represented = represented
    invoke.compact = compact
    invoke.keepalive = (original, packed, original_mean, represented_mean, truncate, delta)
    info = dict(
        mode="mean_error", precenter_v=False,
        formula="mean(original V) - mean(actual LoFi-consumed quantized ORIGINAL V)",
        original_mean={**original_mean.precision, "input": "Original device BF16 V"},
        represented_mean={**represented_mean.precision, "input": "Actual decoded V8 truncated to SrcA5" if b8 else "Actual decoded V4 (exact in SrcA5)"},
        effective_value_truncation="toward zero to five significant bits, NOT RNE" if b8 else None,
        effective_truncation_cores=truncate_cores,
        bias_precision="Subtract actual rounded BF16 means, then BF16 RNE delta",
        extra_rounding="BF16 attention output + BF16 delta -> BF16 RNE final output",
        limitation="Unweighted DC correction only; does not remove weighted quantization or recurrence error",
        common_mode_warning="Separate BF16 means can erase a small delta on a large common offset",
    )
    return compact, invoke, info


def check_correction(original, packed, invoke, b8):
    """Checks actual decode/consumed values and actual-mean subtraction exactly.

    FP64 means diagnose reduction/narrowing error; they are not fed to device.
    Signed zero is value-equivalent under the qualified effective-V contract.
    """
    actual_packed = ttnn.to_torch(packed).float()
    decoded = ttnn.to_torch(invoke.decoded).bfloat16()
    assert torch.equal(decoded.float(), actual_packed), "Packed V decode mismatch"
    represented = ttnn.to_torch(invoke.represented).bfloat16()
    effective_check = EFFECTIVE.check(represented, decoded) if b8 else None
    if not b8:
        assert torch.equal(EFFECTIVE.oracle(decoded).float(), decoded.float()), "V4 not exact in SrcA5"
    source_bias = ttnn.to_torch(invoke.original_bias).bfloat16()
    represented_bias = ttnn.to_torch(invoke.represented_bias).bfloat16()
    for bias in (source_bias, represented_bias):
        assert torch.equal(bias, bias[:, :, :1].expand_as(bias)), "Mean rows not repeated exactly"
    actual_delta = ttnn.to_torch(invoke.compact).bfloat16()
    expected_delta = (source_bias[:, :, :1].float() - represented_bias[:, :, :1].float()).bfloat16()
    mismatch = int((actual_delta != expected_delta).sum())
    assert mismatch == 0, "Bias subtraction/slice differs from BF16 RNE oracle"
    mu = original.double().mean(dim=2, keepdim=True)
    muq = represented.double().mean(dim=2, keepdim=True)
    ideal_delta = mu - muq
    error = actual_delta.double() - ideal_delta
    return dict(
        effective_value_check=effective_check, bias_mismatch=mismatch,
        quantizer_input="Original V, never precentered",
        ideal_delta_rms=float(ideal_delta.square().mean().sqrt()),
        ideal_delta_max_abs=float(ideal_delta.abs().max()),
        actual_delta_rms=float(actual_delta.double().square().mean().sqrt()),
        delta_error_rms_vs_fp64=float(error.square().mean().sqrt()),
        delta_error_max_abs_vs_fp64=float(error.abs().max()),
        original_mean_max_abs_vs_fp64=float((source_bias[:, :, :1].double() - mu).abs().max()),
        consumed_mean_max_abs_vs_fp64=float((represented_bias[:, :, :1].double() - muq).abs().max()),
        corrected_unweighted_mean_max_abs_error=float((muq + actual_delta.double() - mu).abs().max()),
    )


def build(device, args, inputs):
    """Same seven-item API as the qualified value-centered builders."""
    assert args.kv_formats in ("b8_b4", "b8_b8")
    assert args.destination in ("main_bf16", "fast_bf16"), "FP32 intentionally not enabled"
    assert args.correction_mode in ("none", "mean_error")
    base = BASE4 if args.kv_formats == "b8_b4" else BASE8
    base_args = argparse.Namespace(**vars(args))
    base_args.center_mode = "none"
    originals, tensors, core, attention, input_prep, _, info = base.build(device, base_args, inputs)
    bias = correction = None
    correction_info = dict(mode="none", precenter_v=False)
    final = core
    if args.correction_mode != "none":
        bias, correction, correction_info = build_correction(device, originals[2], tensors[2], args)
        correction()
        if args.check_preprocess:
            correction_info["check"] = check_correction(inputs[2], tensors[2], correction, args.kv_formats == "b8_b8")
            print("CORRECTION_CHECK", json.dumps(correction_info["check"]), flush=True)
        final = ttnn.allocate_tensor_on_device(
            core.shape, ttnn.bfloat16, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG
        )

    def epilogue():
        if bias is not None:
            ttnn.add(core, bias, output_tensor=final, fast_and_approximate_mode=False)

    def with_epilogue():
        attention()
        epilogue()

    def preprocess():
        input_prep()
        if correction is not None:
            correction()

    def combined():
        preprocess()
        with_epilogue()

    attention.core_output = core
    attention.epilogue = epilogue
    attention.with_epilogue = with_epilogue
    attention.epilogue_bias = bias
    preprocess.correction = correction
    combined.keepalive = (originals, tensors, final, attention, preprocess, correction, bias)
    info.update(value_mean_error_correction=correction_info, epilogue_materialized=bias is not None)
    return originals, tensors, final, attention, preprocess, combined, info


def source_files(destination, kv_formats):
    base = BASE4 if kv_formats == "b8_b4" else BASE8
    # Pin both imported wrappers and the complete BASE8 oracle/reference list,
    # also when BASE4 is selected (historical BASE4 reference pin was omitted).
    return sorted(set(base.source_files(destination) + BASE8.source_files(destination) + [
        Path(__file__).resolve(), Path(BASE4.__file__).resolve(), Path(BASE8.__file__).resolve(),
    ]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True)
    parser.add_argument("--destination", choices=("main_bf16", "fast_bf16"), default="fast_bf16")
    parser.add_argument("--denom-only", action="store_true")
    parser.add_argument("--kv-formats", choices=("b8_b4", "b8_b8"), default="b8_b4")
    parser.add_argument("--correction-mode", choices=("none", "mean_error"), default="mean_error")
    parser.add_argument("--mean-mode", choices=("bf16_fpu", "fp32_sfpu"), default="bf16_fpu")
    parser.add_argument("--length", type=int, default=1024)
    parser.add_argument("--heads", type=int, default=2)
    parser.add_argument("--cores", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1240)
    parser.add_argument("--distribution", default="normal")
    parser.add_argument("--common-mode", type=float, default=32)
    parser.add_argument("--sample-rows", type=int, default=128)
    parser.add_argument("--check-preprocess", action="store_true")
    parser.add_argument("--read-barrier-tiles", type=int, choices=(0, 1, 2, 4, 8, 16, 32, 64), default=2)
    parser.add_argument("--max-l2", type=float)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=0)
    parser.add_argument("--trace-repeats", type=int, default=1)
    args = parser.parse_args()
    assert args.length > 0 and args.length % 512 == 0
    assert args.heads > 0 and args.cores > 0 and args.sample_rows > 0
    assert args.iters >= 0 and args.warmup >= 0 and args.trace_repeats > 0
    assert not args.denom_only or args.destination == "fast_bf16"
    assert Path(args.label).name == args.label
    path = HERE / (args.label + ".json")
    assert not path.exists(), "Use a fresh label"
    torch.set_num_threads(8)
    inputs = REPRO.make_inputs(args.heads, args.length, args.length, 128, args.seed, args.distribution, args.common_mode)
    original_input_hashes = [tensor_sha256(x) for x in inputs]
    rows = torch.linspace(0, args.length - 1, min(args.sample_rows, args.length)).long().unique()
    reference = REPRO.reference(inputs[0][..., rows, :], inputs[1], inputs[2])
    sources = source_files(args.destination, args.kv_formats)
    pins = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    device = ttnn.open_device(device_id=0, trace_region_size=16777216)
    try:
        originals, tensors, out, attention, preprocess, combined, info = build(device, args, inputs)
        attention.with_epilogue()
        actual = ttnn.to_torch(out)
        assert torch.isfinite(actual).all()
        output_hash = tensor_sha256(actual)
        accuracy = REPRO.metrics(actual[..., rows, :], reference)
        centered_accuracy = BASE8.centered_output_metrics(actual[..., rows, :], reference, inputs[2])
        print("ACCURACY", json.dumps(accuracy), flush=True)
        print("CENTERED_OUTPUT_ACCURACY", json.dumps(centered_accuracy), flush=True)
        if args.max_l2 is not None:
            assert accuracy["l2_pct"] < args.max_l2, "Accuracy gate failed; do not time"
        replay_qualification = qualify_trace_replays(
            device, combined, out, output_hash, originals, inputs, original_input_hashes
        )
        timings = dict(
            attention=BASE8.timed(device, attention, args),
            preprocessing=BASE8.timed(device, preprocess, args),
            epilogue=BASE8.timed(device, attention.epilogue, args) if attention.epilogue_bias is not None else dict(median_ms=0.0, replay_ms=[]),
            attention_with_epilogue=BASE8.timed(device, attention.with_epilogue, args),
            combined=BASE8.timed(device, combined, args),
        )
        assert tensor_sha256(ttnn.to_torch(out)) == output_hash, "Timing changed output bits"
        final_immutability = check_input_immutability(originals, inputs, original_input_hashes)
        epilogue_check = None
        if attention.epilogue_bias is not None:
            core = ttnn.to_torch(attention.core_output).float()
            bias = ttnn.to_torch(attention.epilogue_bias).float()
            expected = (core + bias).bfloat16()
            epilogue_check = dict(mismatch=int((actual != expected).sum()), oracle="BF16 RNE of actual BF16 core + actual BF16 delta")
            assert epilogue_check["mismatch"] == 0, epilogue_check
        assert all(hashlib.sha256(p.read_bytes()).hexdigest() == pins[str(p.relative_to(ROOT))] for p in sources)
        flops = 4 * args.heads * args.length**2 * 128
        record = dict(
            **vars(args), **info, **timings, accuracy=accuracy,
            centered_output_accuracy=centered_accuracy, epilogue_check=epilogue_check,
            sampled_query_rows=rows.tolist(), useful_flops=flops,
            accuracy_scope="Original BF16 Q/K/V FP64 reference; sampled Q rows, all KV and heads",
            attention_tflops=flops / (timings["attention"]["median_ms"] * 1e9) if timings["attention"]["median_ms"] else None,
            combined_tflops=flops / (timings["combined"]["median_ms"] * 1e9) if timings["combined"]["median_ms"] else None,
            trace_equal=True, all_output_finite=True,
            correctness_trace_replays=replay_qualification["explicit_combined_trace_replays"],
            correctness_trace_qualification=replay_qualification,
            cpu_inputs_unchanged=True, device_inputs_unchanged=True,
            final_input_immutability=final_immutability,
            output_sha256=output_hash, original_input_sha256=original_input_hashes,
            source_sha256=pins, sources_unchanged=True,
            warning="Combined timing includes original Q/K/V quantization, both device means, decode, optional V8 trunc5, delta formation and BF16 output add; no V precentering and no benefit claim before measurement",
        )
        path.write_text(json.dumps(record, indent=2) + "\n")
        print("RESULT", json.dumps(record), flush=True)
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
