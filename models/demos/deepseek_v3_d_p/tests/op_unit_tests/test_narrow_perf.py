# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import csv
import os
import statistics
from pathlib import Path

import pytest
import torch

import ttnn
from models.common.utility_functions import profiler
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize(
    "M, K, N",
    [(2048, 7 * 1024, 512)],
)
@pytest.mark.parametrize(
    "dtype, layout",
    [(ttnn.bfloat16, ttnn.TILE_LAYOUT)],
)
def test_narrow_matmul_sequence(M, K, N, dtype, layout, device):
    """
    Sequence (run twice: warmup + measurement):
      1) 3 matmuls on the full input          : [M, K]     @ [K, N]
      2) ttnn.narrow on dim=0 to first half   : [M, K] -> [M // 2, K]
      3) 3 matmuls on the narrowed input      : [M//2, K]  @ [K, N]

    One input tensor is reused throughout; 6 distinct random weight tensors are used.
    First iteration is a warmup for both ttnn.matmul and ttnn.narrow; the second
    iteration is timed with the host-side profiler, and ttnn.narrow's host
    execution time is printed at the end of the test.
    """
    profiler.clear()
    profiler_key = "ttnn_narrow_host"

    assert M % 2 == 0
    M_half = M // 2

    torch.manual_seed(0)
    torch_dtype = torch.bfloat16

    torch_input = torch.randn((M, K), dtype=torch_dtype)
    torch_weights_pre = [torch.randn((K, N), dtype=torch_dtype) for _ in range(3)]
    torch_weights_post = [torch.randn((K, N), dtype=torch_dtype) for _ in range(3)]

    dram_mem_cfg = ttnn.MemoryConfig(buffer_type=ttnn.BufferType.DRAM)

    tt_input = ttnn.from_torch(torch_input, layout=layout, dtype=dtype, device=device, memory_config=dram_mem_cfg)
    tt_weights_pre = [
        ttnn.from_torch(w, layout=layout, dtype=dtype, device=device, memory_config=dram_mem_cfg)
        for w in torch_weights_pre
    ]
    tt_weights_post = [
        ttnn.from_torch(w, layout=layout, dtype=dtype, device=device, memory_config=dram_mem_cfg)
        for w in torch_weights_post
    ]

    # Warmup run: prime program caches for both ttnn.matmul and ttnn.narrow.
    _ = [ttnn.matmul(tt_input, w) for w in tt_weights_pre]
    _warmup_narrow = ttnn.narrow(tt_input, dim=0, start=0, length=M_half)
    _ = [ttnn.matmul(_warmup_narrow, w) for w in tt_weights_post]

    # Measurement run: profile ttnn.narrow on the second invocation.
    tt_outputs_pre = [ttnn.matmul(tt_input, w) for w in tt_weights_pre]

    profiler.start(profiler_key)
    tt_input_narrow = ttnn.narrow(tt_input, dim=0, start=0, length=M_half)
    profiler.end(profiler_key)

    tt_outputs_post = [ttnn.matmul(tt_input_narrow, w) for w in tt_weights_post]

    torch_input_narrow = torch.narrow(torch_input, 0, 0, M_half)
    torch_outputs_pre = [torch_input @ w for w in torch_weights_pre]
    torch_outputs_post = [torch_input_narrow @ w for w in torch_weights_post]

    for tt_out, torch_out in zip(tt_outputs_pre, torch_outputs_pre):
        assert_with_pcc(torch_out, ttnn.to_torch(tt_out), pcc=0.99)
    for tt_out, torch_out in zip(tt_outputs_post, torch_outputs_post):
        assert_with_pcc(torch_out, ttnn.to_torch(tt_out), pcc=0.99)

    narrow_host_us = profiler.get(profiler_key) * 1e6
    print(f"\nttnn.narrow host-side execution: {narrow_host_us:.3f} us")


@pytest.mark.parametrize(
    "M, K, N",
    [(2048, 7 * 1024, 512)],
)
@pytest.mark.parametrize(
    "dtype, layout",
    [(ttnn.bfloat16, ttnn.TILE_LAYOUT)],
)
def test_two_inputs_matmul_sequence(M, K, N, dtype, layout, device):
    """
    Baseline counterpart to test_narrow_matmul_sequence: instead of producing the
    half-size tensor with ttnn.narrow, both inputs are materialized on-device
    up front, then a sequence of 6 matmuls is dispatched.

    Layout (run twice: warmup + measurement):
      input_full = [M, K]      (3 matmuls against weights_pre)
      input_half = [M // 2, K] (3 matmuls against weights_post)

    The half input has the same numerical content as torch.narrow(input_full, 0, 0, M//2)
    so PCC checks line up with the narrow-based variant.
    """
    assert M % 2 == 0
    M_half = M // 2

    torch.manual_seed(0)
    torch_dtype = torch.bfloat16

    torch_input_full = torch.randn((M, K), dtype=torch_dtype)
    torch_input_half = torch.narrow(torch_input_full, 0, 0, M_half)
    torch_weights_pre = [torch.randn((K, N), dtype=torch_dtype) for _ in range(3)]
    torch_weights_post = [torch.randn((K, N), dtype=torch_dtype) for _ in range(3)]

    dram_mem_cfg = ttnn.MemoryConfig(buffer_type=ttnn.BufferType.DRAM)

    # Both inputs created before any matmul is dispatched.
    tt_input_full = ttnn.from_torch(
        torch_input_full, layout=layout, dtype=dtype, device=device, memory_config=dram_mem_cfg
    )
    tt_input_half = ttnn.from_torch(
        torch_input_half, layout=layout, dtype=dtype, device=device, memory_config=dram_mem_cfg
    )
    tt_weights_pre = [
        ttnn.from_torch(w, layout=layout, dtype=dtype, device=device, memory_config=dram_mem_cfg)
        for w in torch_weights_pre
    ]
    tt_weights_post = [
        ttnn.from_torch(w, layout=layout, dtype=dtype, device=device, memory_config=dram_mem_cfg)
        for w in torch_weights_post
    ]

    # Warmup run: prime program caches for ttnn.matmul on both input shapes.
    _ = [ttnn.matmul(tt_input_full, w) for w in tt_weights_pre]
    _ = [ttnn.matmul(tt_input_half, w) for w in tt_weights_post]

    # Measurement run: 6-matmul sequence.
    tt_outputs_pre = [ttnn.matmul(tt_input_full, w) for w in tt_weights_pre]
    tt_outputs_post = [ttnn.matmul(tt_input_half, w) for w in tt_weights_post]

    torch_outputs_pre = [torch_input_full @ w for w in torch_weights_pre]
    torch_outputs_post = [torch_input_half @ w for w in torch_weights_post]

    for tt_out, torch_out in zip(tt_outputs_pre, torch_outputs_pre):
        assert_with_pcc(torch_out, ttnn.to_torch(tt_out), pcc=0.99)
    for tt_out, torch_out in zip(tt_outputs_post, torch_outputs_post):
        assert_with_pcc(torch_out, ttnn.to_torch(tt_out), pcc=0.99)


def _summarize_us(samples_us):
    n = len(samples_us)
    return {
        "n": n,
        "mean_us": statistics.mean(samples_us),
        "median_us": statistics.median(samples_us),
        "min_us": min(samples_us),
        "max_us": max(samples_us),
        "stdev_us": statistics.stdev(samples_us) if n > 1 else 0.0,
    }


@pytest.mark.parametrize(
    "M, K, N",
    [(2048, 7 * 1024, 512)],
)
@pytest.mark.parametrize(
    "dtype, layout",
    [(ttnn.bfloat16, ttnn.TILE_LAYOUT)],
)
@pytest.mark.parametrize("iters", [100])
def test_narrow_perf_statistics(M, K, N, dtype, layout, iters, device):
    """
    Statistical comparison of the two pipelines, host-timed, after a warmup pass:
      - "with_narrow"   : 3 matmuls on full input, ttnn.narrow, 3 matmuls on narrowed input
      - "without_narrow": 3 matmuls on full input, 3 matmuls on pre-built half input

    Per-iteration host durations for the full 6-matmul sequence are recorded for
    `iters` iterations of each scenario, then written to
    `$TT_METAL_HOME/generated/profiler/narrow_perf_stats.csv` with one row per
    iteration. Summary stats (mean/median/min/max/stdev/diff) are printed.
    """
    assert M % 2 == 0
    M_half = M // 2

    torch.manual_seed(0)
    torch_dtype = torch.bfloat16

    torch_input_full = torch.randn((M, K), dtype=torch_dtype)
    torch_input_half = torch.narrow(torch_input_full, 0, 0, M_half)
    torch_weights_pre = [torch.randn((K, N), dtype=torch_dtype) for _ in range(3)]
    torch_weights_post = [torch.randn((K, N), dtype=torch_dtype) for _ in range(3)]

    dram_mem_cfg = ttnn.MemoryConfig(buffer_type=ttnn.BufferType.DRAM)

    tt_input_full = ttnn.from_torch(
        torch_input_full, layout=layout, dtype=dtype, device=device, memory_config=dram_mem_cfg
    )
    tt_input_half = ttnn.from_torch(
        torch_input_half, layout=layout, dtype=dtype, device=device, memory_config=dram_mem_cfg
    )
    tt_weights_pre = [
        ttnn.from_torch(w, layout=layout, dtype=dtype, device=device, memory_config=dram_mem_cfg)
        for w in torch_weights_pre
    ]
    tt_weights_post = [
        ttnn.from_torch(w, layout=layout, dtype=dtype, device=device, memory_config=dram_mem_cfg)
        for w in torch_weights_post
    ]

    # Warmup: prime program caches for matmul (both shapes) and narrow.
    _ = [ttnn.matmul(tt_input_full, w) for w in tt_weights_pre]
    _warmup_narrow = ttnn.narrow(tt_input_full, dim=0, start=0, length=M_half)
    _ = [ttnn.matmul(_warmup_narrow, w) for w in tt_weights_post]
    _ = [ttnn.matmul(tt_input_half, w) for w in tt_weights_post]
    ttnn.synchronize_device(device)

    profiler.clear()
    key_with = "pipeline_with_narrow"
    key_without = "pipeline_without_narrow"

    for _ in range(iters):
        ttnn.synchronize_device(device)
        profiler.start(key_with)
        _ = [ttnn.matmul(tt_input_full, w) for w in tt_weights_pre]
        tt_input_narrow = ttnn.narrow(tt_input_full, dim=0, start=0, length=M_half)
        _ = [ttnn.matmul(tt_input_narrow, w) for w in tt_weights_post]
        ttnn.synchronize_device(device)
        profiler.end(key_with)

        ttnn.synchronize_device(device)
        profiler.start(key_without)
        _ = [ttnn.matmul(tt_input_full, w) for w in tt_weights_pre]
        _ = [ttnn.matmul(tt_input_half, w) for w in tt_weights_post]
        ttnn.synchronize_device(device)
        profiler.end(key_without)

    samples_with_us = [t * 1e6 for t in profiler.times[key_with]]
    samples_without_us = [t * 1e6 for t in profiler.times[key_without]]
    diff_us = [a - b for a, b in zip(samples_with_us, samples_without_us)]

    tt_metal_home = os.environ.get("TT_METAL_HOME", str(Path.cwd()))
    csv_dir = Path(tt_metal_home) / "generated" / "profiler"
    csv_dir.mkdir(parents=True, exist_ok=True)
    csv_path = csv_dir / "narrow_perf_stats.csv"

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["iteration", "with_narrow_us", "without_narrow_us", "diff_us"])
        for i, (a, b, d) in enumerate(zip(samples_with_us, samples_without_us, diff_us)):
            writer.writerow([i, f"{a:.3f}", f"{b:.3f}", f"{d:.3f}"])

    stats_with = _summarize_us(samples_with_us)
    stats_without = _summarize_us(samples_without_us)
    stats_diff = _summarize_us(diff_us)

    print(f"\nCSV written to: {csv_path}")
    print(f"{'scenario':<22} {'n':>4} {'mean':>10} {'median':>10} {'min':>10} {'max':>10} {'stdev':>10}")
    for name, s in [
        ("with_narrow", stats_with),
        ("without_narrow", stats_without),
        ("diff (with-without)", stats_diff),
    ]:
        print(
            f"{name:<22} {s['n']:>4} "
            f"{s['mean_us']:>9.3f}u {s['median_us']:>9.3f}u "
            f"{s['min_us']:>9.3f}u {s['max_us']:>9.3f}u {s['stdev_us']:>9.3f}u"
        )


@pytest.mark.parametrize(
    "M_piece, K, N, num_pieces",
    # M_piece must be a multiple of lcm(TILE_HEIGHT=32, DRAM_BANKS=12) = 96 so every
    # successive narrow start lands on a bank-aligned page (narrow.cpp:120).
    [(1920, 7 * 1024, 512, 6)],
)
@pytest.mark.parametrize(
    "dtype, layout",
    [(ttnn.bfloat16, ttnn.TILE_LAYOUT)],
)
@pytest.mark.parametrize(
    "mesh_device",
    [(1, 1), (8, 1), (4, 2)],
    ids=["single", "8x1", "4x2"],
    indirect=True,
)
def test_narrow_chained_matmuls(M_piece, K, N, num_pieces, dtype, layout, mesh_device):
    """
    One big input tensor of shape [M_piece * num_pieces, K] is sliced into
    `num_pieces` successive [M_piece, K] chunks via ttnn.narrow on dim=0.
    Each chunk is fed into a matmul against a distinct weight tensor [K, N].

    Runs on three mesh shapes: 1x1 (single device), 8x1, and 4x2. Tensors are
    replicated across the mesh (ttnn.narrow only supports REPLICATED layouts;
    see narrow.cpp). Single warmup iteration followed by a single timed
    iteration; the host-side duration of the chained narrow+matmul sequence
    is printed.
    """
    M_total = M_piece * num_pieces

    torch.manual_seed(0)
    torch_dtype = torch.bfloat16

    torch_input = torch.randn((M_total, K), dtype=torch_dtype)
    torch_weights = [torch.randn((K, N), dtype=torch_dtype) for _ in range(num_pieces)]

    dram_mem_cfg = ttnn.MemoryConfig(buffer_type=ttnn.BufferType.DRAM)
    replicate = ttnn.ReplicateTensorToMesh(mesh_device)

    tt_input = ttnn.from_torch(
        torch_input,
        layout=layout,
        dtype=dtype,
        device=mesh_device,
        memory_config=dram_mem_cfg,
        mesh_mapper=replicate,
    )
    tt_weights = [
        ttnn.from_torch(
            w,
            layout=layout,
            dtype=dtype,
            device=mesh_device,
            memory_config=dram_mem_cfg,
            mesh_mapper=replicate,
        )
        for w in torch_weights
    ]

    # Warmup pass: prime program caches for ttnn.narrow and ttnn.matmul on this shape.
    for i in range(num_pieces):
        piece = ttnn.narrow(tt_input, dim=0, start=i * M_piece, length=M_piece)
        _ = ttnn.matmul(piece, tt_weights[i])
    ttnn.synchronize_device(mesh_device)

    profiler.clear()
    key = "chained_narrow_matmul"

    # Measurement pass: time the full narrow+matmul chain end-to-end.
    profiler.start(key)
    for i in range(num_pieces):
        piece = ttnn.narrow(tt_input, dim=0, start=i * M_piece, length=M_piece)
        _ = ttnn.matmul(piece, tt_weights[i])
    ttnn.synchronize_device(mesh_device)
    profiler.end(key)

    total_us = profiler.get(key) * 1e6
    print(
        f"\nchained narrow+matmul (x{num_pieces}) on mesh {tuple(mesh_device.shape)} "
        f"host execution: {total_us:.3f} us"
    )
