# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Device ±1 Q/K transform; no centering and no input quantizer changes.

Common K must be centered separately before coarse quantization. Rotation alone
can severely worsen common K/Q, even though the unquantized identity is exact.
"""

import argparse
import hashlib
import json
import math
import statistics
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SIGN_SEED = 20260915
DIM = 128


def signs():
    return torch.randint(0, 2, (DIM,), generator=torch.Generator().manual_seed(SIGN_SEED)).double() * 2 - 1


def matrix(block_size):
    """FP64 matrix T=diag(signs) block_diag(H), with TT^T=block_size*I."""
    assert block_size in (16, 128)
    small = torch.ones((1, 1), dtype=torch.float64)
    while small.shape[0] < block_size:
        small = torch.cat((torch.cat((small, small), 1), torch.cat((small, -small), 1)), 0)
    return signs()[:, None] * torch.kron(torch.eye(DIM // block_size, dtype=torch.float64), small)


def oracle(src, block_size):
    """FP64, unnormalized transform; accepts arbitrary leading dimensions."""
    assert src.shape[-1] == DIM
    return src.double() @ matrix(block_size)


def source_files():
    """Primitive and principal stock TTNN/kernel/Blackhole LLK sources."""
    matmul = ROOT / "ttnn/cpp/ttnn/operations/matmul"
    api = ROOT / "tt_metal/hw/ckernels/blackhole/metal/llk_api"
    llk = ROOT / "tt_metal/tt-llk/tt_llk_blackhole/llk_lib"
    return [
        Path(__file__).resolve(),
        matmul / "matmul.cpp",
        matmul / "matmul.hpp",
        matmul / "device/matmul_device_operation.cpp",
        matmul / "device/matmul_device_operation_types.hpp",
        matmul / "device/factory/matmul_multicore_reuse_mcast_1d_program_factory.cpp",
        matmul / "device/factory/matmul_multicore_reuse_mcast_1d_program_factory.hpp",
        matmul / "device/kernels/compute/bmm_large_block_zm_fused_bias_activation.cpp",
        matmul / "device/kernels/dataflow/reader_bmm_tile_layout_in0_sender_padding.cpp",
        matmul / "device/kernels/dataflow/reader_bmm_tile_layout_in1_sender_writer_padding.cpp",
        matmul / "device/kernels/dataflow/reader_bmm_tile_layout_in1_receiver_writer_padding.cpp",
        ROOT / "tt_metal/hw/inc/api/compute/matmul.h",
        api / "llk_math_matmul_api.h",
        api / "llk_unpack_AB_matmul_api.h",
        api / "llk_pack_tile_api.h",
        api / "llk_pack_common_api.h",
        llk / "llk_math_matmul.h",
        llk / "llk_unpack_AB_matmul.h",
        llk / "llk_pack.h",
    ]


def build(device, src, block_size):
    """Return (rotated BF16 tensor, invoke, metadata), source [1,H,N,128].

    Source/output use tiled interleaved DRAM. H>=1 and N is a positive multiple
    of32. Only the fixed ±1 matrix is host-generated; every invoke reads the
    current source and performs a real device transform. No input is downloaded.
    All buffers are preallocated. Keep invoke alive until its traces are released.
    """
    import ttnn

    shape = list(src.shape)
    assert block_size in (16, 128)
    assert len(shape) == 4 and shape[0] == 1 and shape[1] > 0 and shape[-1] == DIM
    assert shape[2] > 0 and shape[2] % 32 == 0
    assert src.dtype == ttnn.bfloat16 and src.layout == ttnn.TILE_LAYOUT
    assert src.memory_config() == ttnn.DRAM_MEMORY_CONFIG

    # Flatten heads with rows: there is no cross-head mixing in a right matmul.
    # All dimensions are tile-aligned, so these reshapes are storage views.
    flat_shape = [1, 1, shape[1] * shape[2], DIM]
    flat_src = ttnn.reshape(src, flat_shape)
    out = ttnn.allocate_tensor_on_device(shape, ttnn.bfloat16, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG)
    flat_out = ttnn.reshape(out, flat_shape)
    host_matrix = matrix(block_size).bfloat16().reshape(1, 1, DIM, DIM)
    transform = ttnn.from_torch(
        host_matrix, device=device, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    grid = device.compute_with_storage_grid_size()
    grid_size = (grid.x, grid.y)
    cores = grid.x * grid.y
    mt = flat_shape[2] // 32
    # Bound CB storage independently of sequence length: each core iterates
    # fixed four-row blocks instead of buffering its whole M partition.
    out_block_h = 4
    per_core_m = math.ceil(mt / (cores * out_block_h)) * out_block_h
    program = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=grid_size,
        in0_block_w=4,
        out_subblock_h=1,
        out_subblock_w=4,
        out_block_h=out_block_h,
        out_block_w=4,
        per_core_M=per_core_m,
        per_core_N=4,
        mcast_in0=False,
        fuse_batch=True,
        fused_activation=None,
    )
    config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )

    def invoke():
        ttnn.matmul(
            flat_src,
            transform,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            program_config=program,
            compute_kernel_config=config,
            optional_output_tensor=flat_out,
        )
        return out

    # Explicit owners document trace lifetime, including source/output aliases.
    invoke.buffers = (src, flat_src, transform, out, flat_out)
    metadata = dict(
        block_size=block_size,
        sign_seed=SIGN_SEED,
        input_shape=shape,
        transform="T=diag(signs)*block_diag(Sylvester H); src@T; unnormalized",
        score_scale_divisor=block_size,
        centering=False,
        math_fidelity="HiFi4",
        fp32_dst=True,
        math_approx_mode=False,
        packer_l1_acc=False,
        input_dtype="BF16",
        output_dtype="BF16",
        matrix_dtype="BF16 exact +/-1 and0",
        host_constant_upload_in_invoke=False,
        device_input_transform=True,
        grid=list(grid_size),
        active_cores=math.ceil(mt / per_core_m),
        per_core_m_tiles=per_core_m,
        out_block_h_tiles=out_block_h,
        out_block_w_tiles=4,
        dense_matmul_flops=2 * shape[1] * shape[2] * DIM * DIM,
        implementation="Dense TTNN 128x128 HiFi4 matmul, including H16 block zeros; NOT a fast butterfly",
        matrix_sha256=hashlib.sha256(host_matrix.view(torch.uint16).numpy().tobytes()).hexdigest(),
        warning="No centering: quantized common K/Q can worsen severely. HiFi4 FP32 DST still has FPU alignment effects; ideal BF16 equality is diagnostic, not required.",
    )
    return out, invoke, metadata


def metrics(actual, expected):
    a, e = actual.double().reshape(-1), expected.double().reshape(-1)
    delta = a - e
    ac, ec = a - a.mean(), e - e.mean()
    denom = ac.norm() * ec.norm()
    return dict(
        finite=bool(torch.isfinite(a).all()),
        l2_pct=float(100 * delta.norm() / e.norm()) if e.norm() else None,
        max_abs=float(delta.abs().max()),
        reference_rms=float(e.square().mean().sqrt()),
        pcc=float(ac.dot(ec) / denom) if denom else None,
    )


def qk_diagnostic(original_q, original_k, rotated_q, rotated_k, block_size, rows=128, keys=512):
    """Bounded sampled unquantized QK error, including row-centered logits."""
    qi = torch.linspace(0, original_q.shape[-2] - 1, min(rows, original_q.shape[-2])).long().unique()
    ki = torch.linspace(0, original_k.shape[-2] - 1, min(keys, original_k.shape[-2])).long().unique()
    q, k = original_q[..., qi, :], original_k[..., ki, :]
    expected = q.double() @ k.double().transpose(-1, -2) / math.sqrt(DIM)
    ideal = oracle(q, block_size) @ oracle(k, block_size).transpose(-1, -2) / (block_size * math.sqrt(DIM))
    actual = (
        rotated_q[..., qi, :].double()
        @ rotated_k[..., ki, :].double().transpose(-1, -2)
        / (block_size * math.sqrt(DIM))
    )
    return dict(
        query_rows=qi.tolist(),
        key_rows=ki.tolist(),
        ideal_fp64_identity=metrics(ideal, expected),
        device_transformed_scores=metrics(actual, expected),
        device_row_centered_scores=metrics(
            actual - actual.mean(-1, keepdim=True), expected - expected.mean(-1, keepdim=True)
        ),
    )


def main():
    import ttnn

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True)
    parser.add_argument("--block-size", type=int, choices=(16, 128), default=16)
    parser.add_argument("--length", type=int, default=1024)
    parser.add_argument("--heads", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1240)
    parser.add_argument("--distribution", choices=("normal", "outliers", "common_q", "common_k"), default="normal")
    parser.add_argument("--iters", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--trace-repeats", type=int, default=10)
    args = parser.parse_args()
    assert args.length > 0 and args.length % 32 == 0 and args.heads > 0
    assert args.iters >= 0 and args.warmup >= 0 and args.trace_repeats > 0
    assert Path(args.label).name == args.label
    path = HERE / (args.label + ".json")
    assert not path.exists(), "Use a new result label"
    torch.set_num_threads(4)
    inputs = []
    for index in (0, 1):
        gen = torch.Generator().manual_seed(args.seed + index * 1000)
        shape = (1, args.heads, args.length, DIM)
        x = torch.randn(shape, generator=gen)
        if args.distribution == "outliers":
            x += 10 * torch.randn(shape, generator=gen) * (torch.rand(shape, generator=gen) < 0.001)
        if args.distribution == ("common_q", "common_k")[index]:
            x += 32
        inputs.append(x.bfloat16())
    device = ttnn.open_device(device_id=0, trace_region_size=16777216 if args.iters else 0)
    try:
        sources = [ttnn.from_torch(x, device=device, layout=ttnn.TILE_LAYOUT) for x in inputs]
        built = [build(device, x, args.block_size) for x in sources]

        def invoke_pair():
            for _, invoke, _ in built:
                invoke()

        invoke_pair()
        actual = [ttnn.to_torch(out) for out, _, _ in built]
        checks = []
        for src, transformed in zip(inputs, actual):
            expected = oracle(src, args.block_size)
            check = dict(
                vs_fp64=metrics(transformed, expected),
                bf16_rounding_only=metrics(expected.bfloat16(), expected),
                vs_ideal_bf16=metrics(transformed, expected.bfloat16()),
                ideal_bf16_mismatches=int((transformed != expected.bfloat16()).sum()),
            )
            assert check["vs_fp64"]["finite"], "Nonfinite transformed output"
            checks.append(check)
        diagnostics = qk_diagnostic(*inputs, *actual, args.block_size)
        times = []
        if args.iters:
            trace = ttnn.begin_trace_capture(device, cq_id=0)
            for _ in range(args.trace_repeats):
                invoke_pair()
            ttnn.end_trace_capture(device, trace, cq_id=0)
            try:
                for index in range(args.warmup + args.iters):
                    started = time.perf_counter()
                    ttnn.execute_trace(device, trace, cq_id=0, blocking=True)
                    if index >= args.warmup:
                        times.append((time.perf_counter() - started) * 1000 / args.trace_repeats)
            finally:
                ttnn.release_trace(device, trace)
            assert all(
                torch.equal(before, ttnn.to_torch(out)) for before, (out, _, _) in zip(actual, built)
            ), "Trace replay changed transform output"
        record = dict(
            **vars(args),
            transforms=[metadata for _, _, metadata in built],
            checks=checks,
            qk_invariance=diagnostics,
            median_pair_ms=statistics.median(times) if times else None,
            pair_replay_ms=times,
            trace_equal=True if times else None,
            timing="Two real device transforms per invocation; blocking trace replay wall-time; constant upload/allocation excluded; no input precomputation",
            source_sha256={
                str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files()
            },
        )
        with path.open("x") as output:
            output.write(json.dumps(record, indent=2) + "\n")
        print(json.dumps(record), flush=True)
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
