# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Device-perf sweep for tilize over REAL model shapes/configs.

Compares two implementations of the same op on the same cases:
  - "native"    -> ttnn.tilize (the in-tree C++ op)
  - "generated" -> ttnn.operations.tilize.tilize (the eval-generated op), if importable

Each case is one (shape, dtype, output_dtype, input mem-config, output mem-config,
use_multicore) drawn from actual model usage — see PROVENANCE on each CASE. Shapes
are grid-filling / DRAM-bound (unlike the tiny golden-suite correctness cells), so
this is where the perf numbers live.

HOW TO RUN (needs a healthy device + a rebuilt ttnn):
    scripts/run_safe_pytest.sh --profile --run-all \\
        tests/ttnn/unit_tests/operations/tilize/perf_sweep_tilize_model_shapes.py

`--profile` runs the Tracy device profiler and drops a per-op CSV (DEVICE KERNEL
DURATION [ns]); rows are separated by the parametrize id `impl-caseid`, so you get
native-vs-generated side by side per case. A wall-clock BenchmarkProfiler number is
also logged per case as a coarse cross-check.

The "generated" impl is skipped automatically when ttnn.operations.tilize is not
importable (i.e. before an eval run has generated it). Cases whose shard grid
exceeds the live device grid are skipped (device-portable).
"""

import time

import pytest
import torch
from loguru import logger

import ttnn

try:  # the eval-generated op; absent until a tilize eval run has produced it
    from ttnn.operations.tilize import tilize as _generated_tilize  # type: ignore

    _HAVE_GENERATED = True
except Exception:  # noqa: BLE001
    _generated_tilize = None
    _HAVE_GENERATED = False

WARMUP_ITERS = 5
TIMED_ITERS = 20


# --- mem-config builders (device-aware; skip when grid too big) -----------


def _skip_if_grid_too_big(device, grid):
    gs = device.compute_with_storage_grid_size()
    for cr in grid.ranges():
        e = cr.end_coord
        if e.x > gs.x - 1 or e.y > gs.y - 1:
            pytest.skip(f"grid end ({e.x},{e.y}) exceeds device compute grid ({gs.x},{gs.y})")


def interleaved(buffer):
    def build(device):
        return ttnn.DRAM_MEMORY_CONFIG if buffer == ttnn.BufferType.DRAM else ttnn.L1_MEMORY_CONFIG

    return build


def width_sharded(grid_ranges, shard_shape, buffer=ttnn.BufferType.L1, orientation=ttnn.ShardOrientation.ROW_MAJOR):
    def build(device):
        grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(*s), ttnn.CoreCoord(*e)) for (s, e) in grid_ranges})
        _skip_if_grid_too_big(device, grid)
        return ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED, buffer, ttnn.ShardSpec(grid, shard_shape, orientation)
        )

    return build


def height_sharded(grid_ranges, shard_shape, buffer=ttnn.BufferType.L1, orientation=ttnn.ShardOrientation.ROW_MAJOR):
    def build(device):
        grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(*s), ttnn.CoreCoord(*e)) for (s, e) in grid_ranges})
        _skip_if_grid_too_big(device, grid)
        return ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED, buffer, ttnn.ShardSpec(grid, shard_shape, orientation)
        )

    return build


def _torch_dtype(dt):
    return {
        ttnn.bfloat16: torch.bfloat16,
        ttnn.float32: torch.float32,
        ttnn.bfloat8_b: torch.bfloat16,
    }.get(dt, torch.bfloat16)


# --- model-derived cases --------------------------------------------------
# Each: id, shape, dtype (input, ROW_MAJOR), out_dtype (dtype= kwarg),
#       in_mem, out_mem, use_multicore, provenance.
CASES = [
    # CLASS A — DeepSeek V3 MLA (mla1d.py:2593; test_tilize.py:338)
    dict(
        id="dsv3_mla_wo_l1_interleaved",
        shape=[1, 1, 32, 16384],
        dtype=ttnn.bfloat16,
        out_dtype=ttnn.bfloat16,
        in_mem=interleaved(ttnn.BufferType.L1),
        out_mem=interleaved(ttnn.BufferType.L1),
        use_multicore=True,
        prov="deepseek_v3 mla1d.py:2593 (wo, L1 interleaved)",
    ),
    dict(
        id="dsv3_mla_wo_width_sharded",
        shape=[1, 1, 32, 16384],
        dtype=ttnn.bfloat16,
        out_dtype=ttnn.bfloat16,
        in_mem=width_sharded([((1, 0), (4, 1))], [32, 2048]),
        out_mem=width_sharded([((0, 0), (7, 1))], [32, 1024]),
        use_multicore=True,
        prov="deepseek_v3 test_tilize.py:379 (width-sharded 4x2->8x2)",
    ),
    dict(
        id="dsv3_unit_large_1x8x128x7168",
        shape=[1, 8, 128, 7168],
        dtype=ttnn.bfloat16,
        out_dtype=ttnn.bfloat16,
        in_mem=interleaved(ttnn.BufferType.L1),
        out_mem=interleaved(ttnn.BufferType.L1),
        use_multicore=True,
        prov="deepseek_v3 tests/unit/test_tilize.py:19-23",
    ),
    dict(
        id="dsv3_unit_large_8x1x32x7168",
        shape=[8, 1, 32, 7168],
        dtype=ttnn.bfloat16,
        out_dtype=ttnn.bfloat16,
        in_mem=interleaved(ttnn.BufferType.L1),
        out_mem=interleaved(ttnn.BufferType.L1),
        use_multicore=True,
        prov="deepseek_v3 tests/unit/test_tilize.py:19-23",
    ),
    dict(
        id="dsv3_small_dram_bf16",
        shape=[1, 1, 32, 256],
        dtype=ttnn.bfloat16,
        out_dtype=ttnn.bfloat16,
        in_mem=interleaved(ttnn.BufferType.DRAM),
        out_mem=interleaved(ttnn.BufferType.DRAM),
        use_multicore=True,
        prov="deepseek_v3 tests/unit/test_tilize.py:19",
    ),
    dict(
        id="dsv3_small_dram_fp32",
        shape=[1, 1, 32, 256],
        dtype=ttnn.float32,
        out_dtype=ttnn.float32,
        in_mem=interleaved(ttnn.BufferType.DRAM),
        out_mem=interleaved(ttnn.BufferType.DRAM),
        use_multicore=True,
        prov="deepseek_v3 tests/unit/test_tilize.py:20 (fp32)",
    ),
    dict(
        id="dsv3_mla_q_l1",
        shape=[1, 32, 16, 192],
        dtype=ttnn.bfloat16,
        out_dtype=ttnn.bfloat16,
        in_mem=interleaved(ttnn.BufferType.L1),
        out_mem=interleaved(ttnn.BufferType.L1),
        use_multicore=True,
        prov="deepseek_v3 mla1d.py:2475 (Q, L1)",
    ),
    # CLASS B — Falcon attention masks, large interleaved DRAM, cast to bf8b
    dict(
        id="falcon_mask_2048_bf16",
        shape=[1, 1, 2048, 2048],
        dtype=ttnn.bfloat16,
        out_dtype=ttnn.bfloat16,
        in_mem=interleaved(ttnn.BufferType.DRAM),
        out_mem=interleaved(ttnn.BufferType.DRAM),
        use_multicore=True,
        prov="falcon40b falcon_model.py:144 (causal mask, seq=2048)",
    ),
    dict(
        id="falcon_mask_2048_to_bf8b",
        shape=[1, 1, 2048, 2048],
        dtype=ttnn.bfloat16,
        out_dtype=ttnn.bfloat8_b,
        in_mem=interleaved(ttnn.BufferType.DRAM),
        out_mem=interleaved(ttnn.BufferType.DRAM),
        use_multicore=True,
        prov="falcon40b falcon_model.py:144 (mask cast -> bf8b)",
    ),
    # CLASS G — sharded unit/perf configs (test_tilize.py)
    dict(
        id="width_sharded_2d_32x16384",
        shape=[32, 16384],
        dtype=ttnn.bfloat16,
        out_dtype=ttnn.bfloat16,
        in_mem=width_sharded([((0, 0), (7, 7))], [32, 256]),
        out_mem=width_sharded([((0, 0), (7, 7))], [32, 256]),
        use_multicore=True,
        prov="test_tilize.py:155 (width-sharded 8x8, 64 cores)",
    ),
    dict(
        id="width_sharded_to_interleaved_32x256",
        shape=[32, 256],
        dtype=ttnn.bfloat16,
        out_dtype=ttnn.bfloat16,
        in_mem=width_sharded([((0, 0), (3, 0))], [32, 64]),
        out_mem=interleaved(ttnn.BufferType.L1),
        use_multicore=True,
        prov="test_tilize.py:198 (width-sharded -> interleaved)",
    ),
    # 2D interleaved (fp32 truncation path) + single-core comparison
    dict(
        id="fp32_512x512_multicore",
        shape=[512, 512],
        dtype=ttnn.float32,
        out_dtype=ttnn.float32,
        in_mem=interleaved(ttnn.BufferType.L1),
        out_mem=interleaved(ttnn.BufferType.L1),
        use_multicore=True,
        prov="test_tilize.py:107 (fp32 truncation)",
    ),
    dict(
        id="fp32_512x512_singlecore",
        shape=[512, 512],
        dtype=ttnn.float32,
        out_dtype=ttnn.float32,
        in_mem=interleaved(ttnn.BufferType.L1),
        out_mem=interleaved(ttnn.BufferType.L1),
        use_multicore=False,
        prov="test_tilize.py:107 (fp32, single-core)",
    ),
]


def _impls():
    impls = [("native", ttnn.tilize)]
    if _HAVE_GENERATED:
        impls.append(("generated", _generated_tilize))
    return impls


@pytest.mark.parametrize("impl_name,impl", _impls(), ids=[i[0] for i in _impls()])
@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_tilize_perf(impl_name, impl, case, device):
    """Warmup + timed device-perf loop for one (impl, case). Run under
    scripts/run_safe_pytest.sh --profile to capture DEVICE KERNEL DURATION."""
    from tracy import signpost

    shape = case["shape"]
    dtype = case["dtype"]
    out_dtype = case["out_dtype"]
    in_mem = case["in_mem"](device)
    out_mem = case["out_mem"](device)

    torch_input = torch.randn(shape, dtype=_torch_dtype(dtype))
    tt_input = ttnn.from_torch(
        torch_input, dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=in_mem
    )

    # Warmup (JIT + cache warm).
    for _ in range(WARMUP_ITERS):
        _ = impl(tt_input, memory_config=out_mem, dtype=out_dtype, use_multicore=case["use_multicore"])
    ttnn.synchronize_device(device)

    signpost(f"start-{impl_name}-{case['id']}")
    t0 = time.perf_counter()
    for _ in range(TIMED_ITERS):
        out = impl(tt_input, memory_config=out_mem, dtype=out_dtype, use_multicore=case["use_multicore"])
    ttnn.synchronize_device(device)
    t1 = time.perf_counter()
    signpost(f"stop-{impl_name}-{case['id']}")

    wall_us = (t1 - t0) / TIMED_ITERS * 1e6
    logger.info(
        f"[{impl_name}] {case['id']} shape={shape} mc={case['use_multicore']} "
        f"wall≈{wall_us:.1f} us/iter (device kernel ns in the --profile CSV) [{case['prov']}]"
    )
    assert out.layout == ttnn.TILE_LAYOUT
