# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Device-generated K mean and repeated BF16 bias, optionally fused into a trace.

No host bias is uploaded. bf16_fpu uses BF16 inputs/scaler, HiFi4, FP32 DST;
fp32_sfpu includes device BF16->FP32 conversion and full-FP32 SFPU mean, then
explicit final BF16 conversion. Both materialize repeated rows with broadcast
add into a stable preallocated bias consumed by center_preprocess.
"""

import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path

import torch

import center_preprocess

HERE = Path(__file__).resolve().parent


def build(device, src, mode="bf16_fpu"):
    """Return (BF16 [1,H,32,128] bias, invoke); each invoke recomputes the mean.

    Source is BF16 TILE/DRAM [1,H,N,128]. No input values are downloaded.
    The returned invoke keeps small allocated mean intermediates alive so
    captured operations never reference freed/reused output buffers. Retain
    invoke until its traces are released; discard it when finished. Allocation
    occurs at dispatch/capture, not on trace replay. The full-size optional
    FP32 source and BF16 narrowing buffer are preallocated and reused.
    """
    import ttnn

    assert mode in ("bf16_fpu", "fp32_sfpu")
    assert src.dtype == ttnn.bfloat16 and len(src.shape) == 4
    assert src.shape[0] == 1 and src.shape[-1] == 128 and src.shape[2] % 32 == 0
    assert src.shape[2] > 0
    shape = [1, src.shape[1], 32, 128]
    compact_shape = [1, src.shape[1], 1, 128]
    zeros = ttnn.zeros(
        shape, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    bias = ttnn.allocate_tensor_on_device(shape, ttnn.bfloat16, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG)
    config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )
    wide_src, narrow_mean = None, None
    if mode == "fp32_sfpu":
        wide_src = ttnn.allocate_tensor_on_device(
            src.shape, ttnn.float32, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG
        )
        narrow_mean = ttnn.allocate_tensor_on_device(
            compact_shape, ttnn.bfloat16, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG
        )
    keepalive = []

    def invoke():
        source = src
        if wide_src is not None:
            # Cost belongs to this mean invocation, never a precomputed input.
            ttnn.typecast(src, ttnn.float32, output_tensor=wide_src)
            source = wide_src
        mean = ttnn.mean(
            source,
            dim=2,
            keepdim=True,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=config,
            fast_and_approximate_mode=False,
        )
        keepalive.append(mean)
        invoke.last_mean = mean
        if narrow_mean is not None:
            ttnn.typecast(mean, ttnn.bfloat16, output_tensor=narrow_mean)
            mean = narrow_mean
        # Logical H=1 is broadcast; padded rows of the reduction are NOT read
        # as valid means. Avoid repeat's TILE H=1 -> RM -> repeat -> TILE chain.
        ttnn.add(zeros, mean, output_tensor=bias)
        return bias

    invoke.last_mean = None
    invoke.intermediates = keepalive
    invoke.precision = dict(
        input="original BF16 K (device resident)",
        accumulation="FP32 DST",
        reduction="FPU/GMPOOL HiFi4" if mode == "bf16_fpu" else "full FP32 SFPU",
        reciprocal="1/N truncated to BF16 (not RNE)" if mode == "bf16_fpu" else "FP32 post-reduction 1/N",
        final_bias="BF16, explicit repeated column bias",
        full_input_fp32_cast_in_timing=mode == "fp32_sfpu",
        reduction_output_columns=src.shape[1] * 4,
    )
    return bias, invoke


def make_input(length, heads, distribution, seed):
    generator = torch.Generator().manual_seed(seed)
    values = torch.randn((1, heads, length, 128), generator=generator)
    head = torch.arange(heads).reshape(1, heads, 1, 1)
    column = torch.arange(128).reshape(1, 1, 1, 128)
    if distribution == "common_k":
        values += (head + 1) * 4.0 + (column % 13) * 0.125
    elif distribution == "wide":
        exponent = ((head * 17 + column * 7) % 161 - 80).int()
        values = torch.ldexp(1.5 + values * 0.25, exponent)
    elif distribution == "zeros":
        values.zero_()
    else:
        assert distribution == "normal"
    return values.bfloat16()


def measure(device, invoke, args):
    import ttnn

    if not args.iters:
        return []
    for _ in range(args.warmup):
        invoke()
    ttnn.synchronize_device(device)
    times = []
    if args.trace_repeats:
        trace = ttnn.begin_trace_capture(device, cq_id=0)
        for _ in range(args.trace_repeats):
            invoke()
        ttnn.end_trace_capture(device, trace, cq_id=0)
        try:
            # Captured mean outputs remain alive in the builder's closure.
            ttnn.execute_trace(device, trace, cq_id=0, blocking=True)
            for _ in range(args.iters):
                start = time.perf_counter()
                ttnn.execute_trace(device, trace, cq_id=0, blocking=True)
                times.append((time.perf_counter() - start) * 1000 / args.trace_repeats)
        finally:
            ttnn.release_trace(device, trace)
    else:
        for _ in range(args.iters):
            start = time.perf_counter()
            invoke()
            ttnn.synchronize_device(device)
            times.append((time.perf_counter() - start) * 1000)
    return times


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True)
    parser.add_argument("--mean-mode", choices=("bf16_fpu", "fp32_sfpu"), default="bf16_fpu")
    parser.add_argument("--quant-mode", choices=("b8_rne5", "b4_rne"), default="b8_rne5")
    parser.add_argument("--length", type=int, default=4096)
    parser.add_argument("--heads", type=int, default=3)
    parser.add_argument("--cores", type=int, default=4)
    parser.add_argument("--distribution", choices=("normal", "common_k", "wide", "zeros"), default="normal")
    parser.add_argument("--seed", type=int, default=1240)
    parser.add_argument("--iters", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument(
        "--trace-repeats", type=int, default=10, help="0 gives synchronized eager timing, including host overhead"
    )
    args = parser.parse_args()
    assert args.length > 0 and args.length % 32 == 0 and args.heads > 0
    assert args.iters >= 0 and args.warmup >= 0 and args.trace_repeats >= 0
    assert Path(args.label).name == args.label
    path = HERE / ("center-mean-" + args.label + ".json")
    assert not path.exists(), "Use a fresh label"
    torch.set_num_threads(8)
    x = make_input(args.length, args.heads, args.distribution, args.seed)
    import ttnn

    device = ttnn.open_device(device_id=0, trace_region_size=8388608 if args.iters and args.trace_repeats else 0)
    try:
        src = ttnn.from_torch(x, device=device, layout=ttnn.TILE_LAYOUT)
        bias, mean_invoke = build(device, src, args.mean_mode)
        mean_invoke()
        actual_bias = ttnn.to_torch(bias).bfloat16()
        assert torch.equal(actual_bias, actual_bias[:, :, :1, :].expand_as(actual_bias))
        assert bool(torch.isfinite(actual_bias.float()).all())
        # This host reduction is verification ONLY; never uploaded to device.
        reference64 = x.double().mean(dim=2, keepdim=True)
        golden_bf16 = reference64.bfloat16()
        actual_compact = actual_bias[:, :, :1, :].float()
        raw_mean = ttnn.to_torch(mean_invoke.last_mean).float()
        reciprocal32 = torch.tensor(1 / args.length, dtype=torch.float32)
        truncated_reciprocal = (reciprocal32.view(torch.int32) & -65536).view(torch.float32)
        mean_metrics = dict(
            bf16_bias_mismatches_vs_fp64=int((actual_compact != golden_bf16.float()).sum()),
            mean_l2_pct_vs_fp64=center_preprocess.l2_percent(actual_compact, reference64),
            mean_max_abs_vs_fp64=float((actual_compact.double() - reference64).abs().max()),
            ideal_bf16_rounding_l2_pct=center_preprocess.l2_percent(golden_bf16.float(), reference64),
            reduction_output_l2_pct_before_explicit_narrow=center_preprocess.l2_percent(raw_mean, reference64),
            bf16_truncated_reciprocal_relative_error=float(truncated_reciprocal) * args.length - 1,
        )
        output, center_invoke, cores = center_preprocess.build(device, src, bias, args.cores, args.quant_mode)
        center_invoke()
        expected = center_preprocess.quantize_oracle(center_preprocess.center_oracle(x, actual_bias), args.quant_mode)
        actual = ttnn.to_torch(output).float()
        mismatch = int((actual != expected).sum())
        print("DEVICE_MEAN_CENTER_CHECK", mismatch, mean_metrics, flush=True)
        assert mismatch == 0, "Centering must match the actual device-computed bias exactly"

        def chain():
            mean_invoke()
            center_invoke()

        mean_times = measure(device, mean_invoke, args)
        chain_times = measure(device, chain, args)
        if args.iters:
            assert torch.equal(actual_bias, ttnn.to_torch(bias).bfloat16())
            assert torch.equal(actual, ttnn.to_torch(output).float())
        files = [
            Path(__file__).resolve(),
            HERE / "center_preprocess.py",
            *sorted((HERE / "center_preprocess").glob("*.cpp")),
            *sorted((HERE / "center_preprocess").glob("*.hpp")),
        ]
        record = dict(
            **vars(args),
            actual_center_cores=cores,
            precision=mean_invoke.precision,
            mean_metrics=mean_metrics,
            centered_quantization_mismatches=mismatch,
            mean_median_ms=statistics.median(mean_times) if mean_times else None,
            mean_plus_center_median_ms=statistics.median(chain_times) if chain_times else None,
            mean_times_ms=mean_times,
            mean_plus_center_times_ms=chain_times,
            timing=(
                "trace replay wall-time per invocation"
                if args.trace_repeats
                else "synchronized eager, includes host overhead"
            ),
            host_precomputed_bias=False,
            source_sha256={str(p.relative_to(HERE)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        )
        path.write_text(json.dumps(record, indent=2) + "\n")
        print(json.dumps(record), flush=True)
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
