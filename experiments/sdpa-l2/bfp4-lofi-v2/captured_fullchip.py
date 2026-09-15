# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Private common-reader SDPA evaluation of a supplied captured BF16 Q/K/V artifact."""

import argparse
import gc
import hashlib
import json
from pathlib import Path

import torch

import captured_inputs as C

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
VARIANTS = ("main", "fast", "balanced", "accurate", "lofi_fp32_b8", "lofi_fp32_b4")


def configuration(variant, info, cores, check_preprocess):
    C.require(variant in VARIANTS, "Unsupported variant")
    _, heads, length, _ = info["shape"]
    C.require(cores >= heads and cores % heads == 0, "Requested cores must be a positive multiple of heads")
    return argparse.Namespace(
        variant=variant, q_chunk=256, length=length, heads=heads,
        cores=min(cores, heads * (length // 256)), q_prescale=1.0, center_k=False,
        mean_mode="bf16_fpu", b8_rne=False, bfp8_pack_precise=False,
        check_preprocess=check_preprocess, fix_correction=variant == "fast",
        exp_degree=3, native_exp=variant.startswith("lofi_"), reader_chain=True,
        reader_split=False, reader_linear_k=False, read_barrier_tiles=2)


def sources():
    paths = {Path(__file__).resolve(), HERE / "captured_inputs.py"}
    for name in ("fullchip", "preprocess", "numerics", "bfp4_round", "bfp8_round", "q_prescale",
                 "center_mean", "center_preprocess"):
        paths.add(HERE / (name + ".py"))
        directory = HERE / name
        if directory.is_dir():
            paths.update(p for p in directory.iterdir() if p.suffix in (".cpp", ".h", ".hpp"))
    for name in ("safe_rescale.hpp", "fast_correction.hpp", "exp_native.hpp", "exp_refiner.hpp",
                 "streaming/compute_streaming.hpp"):
        paths.add(HERE / name)
    paths.update((HERE.parent / "bfp4-lofi-v1" / name) for name in ("probe.py", "numerics.py"))
    paths.add(HERE.parent / "frontier-accuracy-v1/run.py")
    paths.add(ROOT / "tests/ttnn/unit_tests/operations/sdpa/repro_sdpa_l2.py")
    frozen = HERE.parent / "hybrid-mixed-v1/candidate"
    paths.update(p for p in frozen.rglob("*") if p.suffix in (".hpp", ".h"))
    # MAIN/FAST select different frozen headers in fullchip/compute.cpp, including
    # their own relative SFPU headers. The hybrid tree alone does not cover them.
    for directory in ("single-core-resident-v1/main", "bf16-denom-pair-v3/candidate"):
        paths.update(p for p in (HERE.parent / directory).rglob("*") if p.suffix in (".hpp", ".h"))
    for path in ("ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/dataflow/chain_link.hpp",
                 "ttnn/cpp/ttnn/kernel/dataflow/generate_bcast_scalar.hpp",
                 "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp",
                 "ttnn/cpp/ttnn/kernel_lib/dest_helpers.hpp",
                 "ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/sdpa_streaming_qktv.hpp",
                 "ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/q_chunk_remapping.hpp",
                 "ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/dataflow/chunked_prefill_utils.hpp",
                 "ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/sliding_window_geometry.hpp",
                 "ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/sliding_window_work_plan.hpp",
                 "tt_metal/hw/inc/api/compute/compute_kernel_hw_startup.h",
                 "tt_metal/hw/inc/api/compute/experimental/matmul_custom.h",
                 "tt_metal/hw/inc/api/compute/experimental/sdpa_sub_custom.h",
                 "tt_metal/tt-llk/tt_llk_blackhole/llk_lib/llk_math_matmul.h",
                 "tt_metal/tt-llk/tt_llk_blackhole/llk_lib/llk_unpack_AB_matmul.h",
                 "tt_metal/tt-llk/tt_llk_blackhole/llk_lib/llk_pack.h"):
        paths.add(ROOT / path)
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}


def verify_input_hashes(inputs, info):
    current = {k: C.tensor_sha256(v) for k, v in zip(("q", "k", "v"), inputs)}
    C.require(current == info["input_sha256"], "Original captured inputs changed")


def run_variant(F, ttnn, device, inputs, info, rows, reference, variant, args):
    config = configuration(variant, info, args.cores, args.check_preprocess)
    # Prevent F.build's hardware clamp from silently invalidating a head's chain.
    grid = device.compute_with_storage_grid_size()
    C.require(config.cores <= grid.x * grid.y, "Requested active cores exceed device grid; specify a smaller head multiple")
    originals, prepared, output, attention, preprocess, combined, kernel = F.build(device, config, inputs)
    combined()
    actual = ttnn.to_torch(output)
    C.require(actual.dtype is torch.bfloat16 and tuple(actual.shape) == tuple(info["shape"]),
              "Unexpected output shape/dtype")
    C.require(bool(torch.isfinite(actual).all()), "Nonfinite output; accuracy and timing rejected")
    output_hash = C.tensor_sha256(actual)
    metrics = C.metrics(actual[..., rows, :], reference, inputs[2])
    if args.max_l2 is not None:
        C.require(metrics["l2_pct"] is not None and metrics["l2_pct"] < args.max_l2,
                  "Explicit L2 gate failed; do not time this case")
    if args.min_pcc is not None:
        C.require(metrics["pcc"] is not None and metrics["pcc"] >= args.min_pcc,
                  "Explicit PCC gate failed; do not time this case")
    # Replay check always runs, including --iters0 smoke mode.
    trace = ttnn.begin_trace_capture(device, cq_id=0)
    combined()
    ttnn.end_trace_capture(device, trace, cq_id=0)
    try:
        ttnn.execute_trace(device, trace, cq_id=0, blocking=True)
        C.require(output_hash == C.tensor_sha256(ttnn.to_torch(output)), "Combined trace changed output bits")
    finally:
        ttnn.release_trace(device, trace)
    times = {}
    if args.iters:
        times["attention"] = F.timed(device, attention, args)
        times["preprocessing"] = F.timed(device, preprocess, args) if kernel["device_preprocessing"] else None
        times["combined"] = F.timed(device, combined, args) if kernel["device_preprocessing"] else times["attention"]
        C.require(output_hash == C.tensor_sha256(ttnn.to_torch(output)), "Timed replay changed output bits")
    verify_input_hashes(inputs, info)
    flops = info["useful_attention_flops"]
    result = dict(kind="captured_evaluation", variant=variant, config=vars(config), kernel=kernel,
                  original_input_sha256=info["input_sha256"], sampled_query_rows=rows,
                  metrics=metrics, all_output_finite=True, trace_equal=True, immutable_inputs_verified=True,
                  output_sha256=output_hash, timings=times, useful_attention_flops=flops,
                  preprocessing_exact_checked=bool(config.check_preprocess and kernel["device_preprocessing"]),
                  fast_private_correction_reset=config.fix_correction,
                  control_scope="Four established numerical choices rehosted in private common-reader harness; not frozen-driver replay",
                  reference_scope="Original captured BF16 Q/K/V; FP64; all heads and KV, explicit sampled Q rows")
    if times:
        result.update(attention_tflops=flops / (times["attention"]["median_ms"] * 1e9),
                      combined_tflops=flops / (times["combined"]["median_ms"] * 1e9))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--output", type=Path, required=True, help="Fresh JSONL path; exclusive create, never overwrite")
    parser.add_argument("--variants", choices=VARIANTS, nargs="+", default=list(VARIANTS))
    parser.add_argument("--cores", type=int, default=110)
    parser.add_argument("--sample-rows", type=int, default=128)
    parser.add_argument("--check-preprocess", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-file-bytes", type=int, default=4 * 1024**3)
    parser.add_argument("--max-l2", type=float)
    parser.add_argument("--min-pcc", type=float)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=5, help="Zero means correctness/replay smoke only")
    parser.add_argument("--trace-repeats", type=int, default=1)
    args = parser.parse_args()
    C.require(args.iters >= 0 and args.warmup >= 0 and args.trace_repeats > 0, "Invalid timing counts")
    C.require(len(args.variants) == len(set(args.variants)), "Duplicate variant")
    C.require(args.max_l2 is None or (0 < args.max_l2 < float("inf")), "Invalid L2 gate")
    C.require(args.min_pcc is None or -1 <= args.min_pcc <= 1, "Invalid PCC gate")
    C.require(not args.output.exists(), "Use a fresh output path")
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    inputs, info = C.load_capture(args.artifact, args.max_file_bytes)
    for variant in args.variants:
        configuration(variant, info, args.cores, args.check_preprocess)
    rows = C.select_rows(info["shape"][2], args.sample_rows)
    ref = C.reference(inputs, rows)
    # Host-only validation/reference above cannot open an accelerator.
    import ttnn
    import fullchip as F

    hashes = sources()
    with args.output.open("x") as stream:
        def emit(value):
            line = json.dumps(value, allow_nan=False)
            stream.write(line + "\n")
            stream.flush()
            print(line, flush=True)
        emit(dict(kind="provenance", capture=info,
                  args={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                  source_sha256=hashes, torch_version=str(torch.__version__),
                  scope="Private square noncausal Q256/K512/D128 chain harness; no model-level quality claim",
                  timing_scope="Trace replay wall time: attention or real device preprocessing+attention; excludes initial transfers/CPU reference",
                  flops_scope="Useful 4*H*N*N*128; not hardware instruction FLOPs, preprocessing FLOPs or measured FPU utilization"))
        for variant in args.variants:
            device = ttnn.open_device(device_id=0, trace_region_size=16777216)
            try:
                record = run_variant(F, ttnn, device, inputs, info, rows, ref, variant, args)
                C.require(sources() == hashes, "Sources changed during evaluation")
                emit(record)
            finally:
                ttnn.close_device(device)
            gc.collect()
        verify_input_hashes(inputs, info)
        C.require(sources() == hashes, "Sources changed during evaluation")
        emit(dict(kind="complete", sources_unchanged=True, original_inputs_unchanged=True,
                  evaluated_variants=args.variants))


if __name__ == "__main__":
    main()
