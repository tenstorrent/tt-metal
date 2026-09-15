# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Primitive-only HiFi4 correction probe: eight distinct BF16 means @ original K.

Host means and tiny-tile construction are permitted HERE and excluded from
timing. This is not device mean production or integrated attention timing.
"""

import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def metrics(actual, reference):
    a, r = actual.double().reshape(-1), reference.double().reshape(-1)
    assert bool(torch.isfinite(a).all()) and bool(torch.isfinite(r).all())
    d = a - r
    ac, rc = a - a.mean(), r - r.mean()
    denom = ac.norm() * rc.norm()
    return dict(
        l2_pct=float(100 * d.norm() / r.norm()) if r.norm() else None,
        max_abs=float(d.abs().max()),
        gain=float((a @ r) / (r @ r)) if r.norm() else None,
        pcc=float((ac @ rc) / denom) if denom else None,
    )


def host_inputs(length, seed, distribution, common_mode):
    generator = torch.Generator().manual_seed(seed)
    qblocks = torch.randn((8, 128, 128), generator=generator)
    if distribution == "common_q":
        qblocks += common_mode
    elif distribution == "structured":
        qblocks += torch.sin(torch.arange(128).float() / 9) * common_mode
    else:
        assert distribution == "normal"
    # Eight actual, distinct Q128 block means. Deliberately HOST generated;
    # source Q is first rounded BF16, mean computed FP64, final mean BF16.
    means = qblocks.bfloat16().double().mean(dim=1).bfloat16().reshape(1, 1, 8, 128)
    kgenerator = torch.Generator().manual_seed(seed + 1000)
    k = torch.randn((1, 1, length, 128), generator=kgenerator).bfloat16()
    return means, k


def case_inputs(means, case):
    if case == "split1":
        return [means[:, :, i : i + 1].contiguous() for i in range(8)], 1
    if case == "batched8":
        return [means], 8
    if case == "padded32":
        # One32-row call: eight distinct means, each repeated four times.
        # This is NOT the old eight32-row-call implementation.
        return [means.repeat_interleave(4, dim=2)], 32
    assert case == "legacy32"
    return [means[:, :, i : i + 1].expand(1, 1, 32, 128).contiguous() for i in range(8)], 32


def n_partition(length, cores, block_tiles):
    """Bound N ownership by the core budget without growing local CB blocks."""
    assert cores > 0 and block_tiles > 0 and block_tiles % 4 == 0
    assert length > 0 and length % (32 * block_tiles) == 0
    nt = length // 32
    # One1D-mcast worker owns one per_core_N partition; its internal output
    # blocks stay block_tiles wide. At256K/110cores/default16 this is80 owned
    # tiles and five16-tile blocks, not512 workers or a full-N circular buffer.
    owned_tiles = ((nt + cores * block_tiles - 1) // (cores * block_tiles)) * block_tiles
    active_cores = (nt + owned_tiles - 1) // owned_tiles
    return owned_tiles, active_cores


def build_case(device, means, k_device, case, cores=110, per_core_n=16):
    """Return outputs, invoke, input tensors, CPU operands, metadata.

    All allocations/upload/retiling are outside invoke. Each invocation
    writes all eight unique mean vectors' correction outputs.
    """
    import ttnn

    cpu_inputs, rows = case_inputs(means, case)
    length = k_device.shape[2]
    assert length % (32 * per_core_n) == 0 and per_core_n % 4 == 0
    hardware = device.compute_with_storage_grid_size()
    cores = min(cores, hardware.x * hardware.y)
    owned_n, active_cores = n_partition(length, cores, per_core_n)
    # All tiny-row cases have one logical M tile. The reuse-only factory
    # partitions M, requiring N==per_core_N; mcast-in0 instead partitions N.
    # Keep original K BF16/32-wide: mcast tiny-height compressed K and
    # transposed16-wide K are explicitly unsupported by current validation.
    assert k_device.dtype == ttnn.bfloat16
    coords = [ttnn.CoreCoord(i % hardware.x, i // hardware.x) for i in range(cores)]
    allowed = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in coords])
    config = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=(hardware.x, hardware.y),
        allowed_worker_cores=allowed,
        in0_block_w=4,
        out_subblock_h=1,
        out_subblock_w=4,
        out_block_h=1,
        out_block_w=per_core_n,
        per_core_M=1,
        per_core_N=owned_n,
        mcast_in0=True,
        fuse_batch=True,
        fused_activation=None,
    )
    compute = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )
    tile = ttnn.Tile([rows, 32])
    inputs = [
        ttnn.from_torch(
            x,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            tile=tile,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        for x in cpu_inputs
    ]
    # Host-uploaded zeros are only stable output allocation outside timing.
    # Explicit tile shape avoids silently allocating a standard32x32 output.
    outputs = [
        ttnn.from_torch(
            torch.zeros((1, 1, rows, length), dtype=torch.float32),
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            tile=tile,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        for _ in inputs
    ]

    def invoke():
        for src, out in zip(inputs, outputs):
            ttnn.matmul(
                src,
                k_device,
                transpose_b=True,
                dtype=ttnn.float32,
                program_config=config,
                compute_kernel_config=compute,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                optional_output_tensor=out,
            )

    per_phase = 16 if rows == 32 else 4
    info = dict(
        case=case,
        tile_shape=[rows, 32],
        logical_rows_per_call=rows,
        matmul_calls_per_invocation=len(inputs),
        distinct_mean_rows=8,
        program_config=str(config),
        active_cores_per_call=active_cores,
        allowed_cores=cores,
        partition="1D multicast means (in0), partition output N across workers",
        requested_per_core_n_tiles=per_core_n,
        effective_per_core_n_tiles=owned_n,
        out_block_n_tiles=per_core_n,
        n_blocks_per_worker=owned_n // per_core_n,
        buffering="Fixed1xout_block_n output blocks; per-core N ownership grows at long contexts, CB blocks do not",
        input_layout="BF16 original K32x32; BF16 mean tiny tiles",
        output_dtype="FP32",
        fidelity="HiFi4",
        fp32_dst=True,
        packer_l1_acc=False,
        correction_output_bytes=len(inputs) * rows * length * 4,
        useful_flops=2 * 8 * length * 128,
        physical_tile_row_equivalents_per_invocation=len(inputs) * (32 if rows == 32 else 8),
        estimated_mvmul_count=len(inputs) * (length // 32) * 4 * per_phase * 4,
        compulsory_k_read_bytes=len(inputs) * length * 128 * 2,
        k_read_warning="Algorithmic minimum per invocation, not measured traffic; no K read sharing across calls",
        producer_and_retiling_included=False,
        comparison_warning="padded32 is ONE32-row call for eight means; legacy32 is EIGHT32-row calls",
    )
    return outputs, invoke, inputs, cpu_inputs, info


def source_files():
    return [
        Path(__file__).resolve(),
        ROOT / "tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_math_matmul_api.h",
        ROOT / "tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_unpack_AB_matmul_api.h",
        ROOT / "tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_pack_tile_api.h",
        ROOT / "tt_metal/tt-llk/tt_llk_blackhole/llk_lib/llk_math_matmul.h",
        ROOT / "tt_metal/tt-llk/tt_llk_blackhole/llk_lib/llk_unpack_AB_matmul.h",
        ROOT / "tt_metal/tt-llk/tt_llk_blackhole/llk_lib/llk_pack.h",
        ROOT / "ttnn/cpp/ttnn/operations/matmul/device/factory/matmul_multicore_reuse_mcast_1d_program_factory.cpp",
        ROOT / "ttnn/cpp/ttnn/operations/matmul/device/factory/matmul_multicore_reuse_mcast_1d_program_factory.hpp",
        ROOT / "ttnn/cpp/ttnn/operations/matmul/device/kernels/compute/bmm_large_block_zm_fused_bias_activation.cpp",
        ROOT / "ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/reader_bmm_tile_layout_in0_sender_padding.cpp",
        ROOT / "ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/reader_bmm_tile_layout_in0_receiver.cpp",
        ROOT
        / "ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/reader_bmm_tile_layout_in1_sender_writer_padding.cpp",
        ROOT / "ttnn/cpp/ttnn/operations/matmul/device/matmul_device_operation.cpp",
        ROOT / "ttnn/cpp/ttnn/operations/matmul/matmul.cpp",
        ROOT / "ttnn/cpp/ttnn/operations/matmul/matmul_nanobind.cpp",
    ]


def measure(device, invoke, args):
    import ttnn

    if args.iters == 0:
        return []
    trace = ttnn.begin_trace_capture(device, cq_id=0)
    for _ in range(args.trace_repeats):
        invoke()
    ttnn.end_trace_capture(device, trace, cq_id=0)
    times = []
    try:
        for i in range(args.warmup + args.iters):
            start = time.perf_counter()
            ttnn.execute_trace(device, trace, cq_id=0, blocking=True)
            if i >= args.warmup:
                times.append(1000 * (time.perf_counter() - start) / args.trace_repeats)
    finally:
        ttnn.release_trace(device, trace)
    return times


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True)
    parser.add_argument("--length", type=int, default=1024)
    parser.add_argument(
        "--cases",
        nargs="+",
        choices=("split1", "batched8", "padded32", "legacy32"),
        default=["split1", "batched8", "padded32"],
    )
    parser.add_argument("--cores", type=int, default=110)
    parser.add_argument(
        "--per-core-n",
        type=int,
        default=16,
        help="N tiles per local output block/minimum worker partition; ownership grows in this quantum to fit --cores, without growing CB blocks",
    )
    parser.add_argument("--seed", type=int, default=1240)
    parser.add_argument("--distribution", choices=("normal", "common_q", "structured"), default="normal")
    parser.add_argument("--common-mode", type=float, default=32)
    parser.add_argument(
        "--max-l2",
        type=float,
        default=0.1,
        help="Primitive diagnostic gate; public HiFi4/FP32 baseline floor remains under investigation",
    )
    parser.add_argument("--iters", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--trace-repeats", type=int, default=10)
    args = parser.parse_args()
    assert args.cores > 0 and args.per_core_n > 0 and args.per_core_n % 4 == 0
    assert args.length > 0 and args.length % (32 * args.per_core_n) == 0
    assert args.iters >= 0 and args.warmup >= 0 and args.trace_repeats > 0
    assert len(set(args.cases)) == len(args.cases)
    assert Path(args.label).name == args.label
    path = HERE / (args.label + ".json")
    assert not path.exists(), "Use a fresh label"
    torch.set_num_threads(8)
    means, k = host_inputs(args.length, args.seed, args.distribution, args.common_mode)
    reference8 = means.double() @ k.double().transpose(-1, -2)
    files = source_files()
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    import ttnn

    device = ttnn.open_device(device_id=0, trace_region_size=8388608 if args.iters else 0)
    records = []
    canonical_first = None
    try:
        dk = ttnn.from_torch(
            k, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        assert torch.equal(ttnn.to_torch(dk), k), "Original BF16 K upload mismatch"
        for case in args.cases:
            outputs, invoke, tiny_inputs, cpu_inputs, info = build_case(
                device, means, dk, case, args.cores, args.per_core_n
            )
            for src, expected in zip(tiny_inputs, cpu_inputs):
                assert torch.equal(ttnn.to_torch(src), expected), "Partial-tile BF16 input conversion mismatch"
            for out in outputs:
                assert bool((ttnn.to_torch(out) == 0).all()), "Partial-tile FP32 zero output conversion mismatch"
            invoke()
            actuals, all_metrics = [], []
            for out, src in zip(outputs, cpu_inputs):
                actual = ttnn.to_torch(out).float()
                ref = src.double() @ k.double().transpose(-1, -2)
                result = metrics(actual, ref)
                assert result["l2_pct"] < args.max_l2, (case, result)
                actuals.append(actual)
                all_metrics.append(result)
            if case in ("split1", "legacy32"):
                canonical = torch.cat([a[:, :, :1] for a in actuals], dim=2)
            elif case == "padded32":
                canonical = actuals[0][:, :, ::4].contiguous()
            else:
                canonical = actuals[0]
            assert tuple(canonical.shape) == (1, 1, 8, args.length)
            comparison = metrics(canonical, canonical_first) if canonical_first is not None else None
            if canonical_first is None:
                canonical_first = canonical.clone()
            times = measure(device, invoke, args)
            for out, actual in zip(outputs, actuals):
                assert torch.equal(ttnn.to_torch(out).float(), actual), "Replay output changed"
            median = statistics.median(times) if times else None
            record = dict(
                **info,
                all_physical_rows_accuracy=all_metrics,
                unique_eight_rows_accuracy=metrics(canonical, reference8),
                versus_first_case=comparison,
                replay_ms=times,
                median_ms=median,
                unique_useful_tflops=info["useful_flops"] / (median * 1e9) if median else None,
                trace_equal=True,
                partial_tile_input_output_conversion_exact=True,
            )
            records.append(record)
            print("TINY_MEAN_CASE", json.dumps(record), flush=True)
        assert all(hashlib.sha256(p.read_bytes()).hexdigest() == hashes[str(p.relative_to(ROOT))] for p in files)
        result = dict(
            **vars(args),
            cases_results=records,
            source_sha256=hashes,
            host_means_sha256=hashlib.sha256(means.contiguous().view(torch.uint16).numpy().tobytes()).hexdigest(),
            original_k_sha256=hashlib.sha256(k.contiguous().view(torch.uint16).numpy().tobytes()).hexdigest(),
            timing_scope="Device matmul primitive only; host means/upload/retiling/producer/attention integration excluded",
            accuracy_scope="FP64 original BF16 mean @ original BF16 K; every physical output row and eight unique rows",
        )
        path.write_text(json.dumps(result, indent=2) + "\n")
        print("TINY_MEAN_RESULT", json.dumps(result), flush=True)
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
