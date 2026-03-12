# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Device test: Auto-fused gated_local_reduce vs hand-fused.

Builds the gated_local_reduce graph using auto-fusion infrastructure
(LOCAL_REDUCE + LOCAL_REDUCE + ELTWISE_MUL) and compares:
1. Correctness vs golden (PyTorch reference)
2. Correctness vs hand-fused kernel
3. Performance via timing (Tracy for detailed profiling)
"""

import time

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.deepseek_v3_b1.auto_fusion.graph import FusionGraph
from models.demos.deepseek_v3_b1.auto_fusion.specs.eltwise_mul import ELTWISE_MUL
from models.demos.deepseek_v3_b1.auto_fusion.specs.local_reduce import LOCAL_REDUCE
from models.demos.deepseek_v3_b1.auto_fusion.types import CBConfig
from models.demos.deepseek_v3_b1.fused_ops.gated_local_reduce.op import GatedLocalReduceOp


def create_tensors(device, tile_h, tile_w, group1_num_tiles, group2_num_tiles, seed=42):
    """Create input/output tensors on device for both hand-fused and auto-fused tests."""
    tile = ttnn.Tile([tile_h, tile_w])
    core = ttnn.CoreCoord(0, 0)
    core_grid = ttnn.CoreRangeSet({ttnn.CoreRange(core, core)})

    torch.manual_seed(seed)
    group1_inputs = [torch.randn((tile_h, tile_w), dtype=torch.bfloat16) for _ in range(group1_num_tiles)]
    group2_inputs = [torch.randn((tile_h, tile_w), dtype=torch.bfloat16) for _ in range(group2_num_tiles)]

    # Golden reference
    golden = GatedLocalReduceOp.golden(
        [t.float() for t in group1_inputs],
        [t.float() for t in group2_inputs],
    ).bfloat16()

    # Stack inputs into tensors
    torch_group1 = torch.cat(group1_inputs, dim=0)
    torch_group2 = torch.cat(group2_inputs, dim=0)

    # Create sharded tensors on device
    def make_sharded(torch_tensor, num_tiles):
        shard_shape = (num_tiles * tile_h, tile_w)
        shard_spec = ttnn.ShardSpec(core_grid, shard_shape, ttnn.ShardOrientation.ROW_MAJOR)
        mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, shard_spec)
        return ttnn.from_torch(
            torch_tensor,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=mem_config,
            tile=tile,
        )

    def make_output():
        shard_shape = (tile_h, tile_w)
        shard_spec = ttnn.ShardSpec(core_grid, shard_shape, ttnn.ShardOrientation.ROW_MAJOR)
        mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, shard_spec)
        return ttnn.from_torch(
            torch.zeros((tile_h, tile_w), dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=mem_config,
            tile=tile,
        )

    ttnn_group1 = make_sharded(torch_group1, group1_num_tiles)
    ttnn_group2 = make_sharded(torch_group2, group2_num_tiles)
    ttnn_output_hand = make_output()

    return golden, ttnn_group1, ttnn_group2, ttnn_output_hand, core_grid, tile


def run_hand_fused(ttnn_group1, ttnn_group2, ttnn_output, group1_num_tiles, group2_num_tiles):
    """Run the hand-fused gated_local_reduce kernel."""
    return GatedLocalReduceOp.op(
        ttnn_group1,
        ttnn_group2,
        ttnn_output,
        group1_num_tiles,
        group2_num_tiles,
    )


def run_auto_fused(
    device, ttnn_group1, ttnn_group2, core_grid, tile, tile_h, tile_w, group1_num_tiles, group2_num_tiles
):
    """Run the auto-fused gated_local_reduce via FusionGraph."""
    data_format = ttnn_group1.dtype
    tile_size = tile.get_tile_size(data_format)

    g = FusionGraph()

    # Phase 1: LocalReduce with SiLU (group1)
    g.add(
        "reduce1",
        LOCAL_REDUCE,
        cores=core_grid,
        ct_args={
            "num_tiles": group1_num_tiles,
            "apply_silu": True,
            "input_num_pages": group1_num_tiles,
            "math_fidelity": "HiFi4",
            "math_approx_mode": True,
        },
    )

    # Phase 2: LocalReduce without SiLU (group2)
    g.add(
        "reduce2",
        LOCAL_REDUCE,
        cores=core_grid,
        ct_args={
            "num_tiles": group2_num_tiles,
            "apply_silu": False,
            "input_num_pages": group2_num_tiles,
        },
    )

    # Phase 3: EltwiseMul (binary: reduce1_out * reduce2_out -> out)
    g.add(
        "mul",
        ELTWISE_MUL,
        cores=core_grid,
        ct_args={"num_tiles": 1},
        inputs={
            "in0": ("reduce1", "output"),
            "in1": ("reduce2", "output"),
        },
        cb_config={
            "out": CBConfig(
                page_size=tile_size,
                num_pages=1,
                data_format="bfloat16",
                tile_height=tile_h,
                tile_width=tile_w,
            ),
        },
    )

    # Create output tensor for auto-fused
    shard_shape = (tile_h, tile_w)
    shard_spec = ttnn.ShardSpec(core_grid, shard_shape, ttnn.ShardOrientation.ROW_MAJOR)
    output_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, shard_spec)
    ttnn_output = ttnn.from_torch(
        torch.zeros((tile_h, tile_w), dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=output_mem_config,
        tile=tile,
    )

    # Build IO tensor mapping
    io_tensors = {
        ("reduce1", "input"): ttnn_group1,
        ("reduce2", "input"): ttnn_group2,
        ("mul", "out"): ttnn_output,
    }

    # Build and run
    fused_op = g.build(device, io_tensors)
    result = fused_op.run()
    return result, fused_op


# ===========================================================================
# Correctness Tests
# ===========================================================================


@pytest.mark.parametrize(
    "tile_h, tile_w, group1_num_tiles, group2_num_tiles",
    [
        (32, 32, 2, 2),
        (32, 32, 4, 4),
        (32, 32, 8, 8),
        (32, 32, 4, 2),
    ],
)
def test_auto_fused_vs_golden(device, tile_h, tile_w, group1_num_tiles, group2_num_tiles):
    """Auto-fused gated_local_reduce matches golden reference."""
    golden, ttnn_group1, ttnn_group2, _, core_grid, tile = create_tensors(
        device, tile_h, tile_w, group1_num_tiles, group2_num_tiles
    )

    result, _ = run_auto_fused(
        device,
        ttnn_group1,
        ttnn_group2,
        core_grid,
        tile,
        tile_h,
        tile_w,
        group1_num_tiles,
        group2_num_tiles,
    )

    output_torch = ttnn.to_torch(result)
    pcc_threshold = 0.998
    passing, pcc_message = comp_pcc(golden, output_torch, pcc_threshold)
    logger.info(f"Auto-fused vs golden: {pcc_message}")
    assert passing, f"Auto-fused PCC too low: {pcc_message}"


@pytest.mark.parametrize(
    "tile_h, tile_w, group1_num_tiles, group2_num_tiles",
    [
        (32, 32, 4, 4),
    ],
)
def test_auto_fused_vs_hand_fused(device, tile_h, tile_w, group1_num_tiles, group2_num_tiles):
    """Auto-fused matches hand-fused kernel output."""
    golden, ttnn_group1, ttnn_group2, ttnn_output_hand, core_grid, tile = create_tensors(
        device, tile_h, tile_w, group1_num_tiles, group2_num_tiles
    )

    # Run hand-fused
    hand_result = run_hand_fused(
        ttnn_group1,
        ttnn_group2,
        ttnn_output_hand,
        group1_num_tiles,
        group2_num_tiles,
    )
    hand_torch = ttnn.to_torch(hand_result)

    # Need fresh tensors for auto-fused (hand-fused consumed the originals)
    _, ttnn_group1_2, ttnn_group2_2, _, _, _ = create_tensors(
        device, tile_h, tile_w, group1_num_tiles, group2_num_tiles
    )

    auto_result, _ = run_auto_fused(
        device,
        ttnn_group1_2,
        ttnn_group2_2,
        core_grid,
        tile,
        tile_h,
        tile_w,
        group1_num_tiles,
        group2_num_tiles,
    )
    auto_torch = ttnn.to_torch(auto_result)

    # Compare auto vs hand
    pcc_threshold = 0.999
    passing, pcc_message = comp_pcc(hand_torch, auto_torch, pcc_threshold)
    logger.info(f"Auto-fused vs hand-fused: {pcc_message}")
    assert passing, f"Auto vs hand PCC too low: {pcc_message}"


# ===========================================================================
# Performance Test
# ===========================================================================


@pytest.mark.parametrize(
    "tile_h, tile_w, group1_num_tiles, group2_num_tiles, single_run_only",
    [
        (32, 32, 4, 4, False),
        (32, 32, 8, 8, False),
    ],
)
def test_auto_fused_performance(device, tile_h, tile_w, group1_num_tiles, group2_num_tiles, single_run_only):
    """
    Performance comparison: auto-fused vs hand-fused.

    Measures host-side timing. For accurate device FW timing, use Tracy:
        export TT_METAL_DEVICE_PROFILER=1
        python -m tracy -r -m pytest tests/test_gated_local_reduce_device.py::test_auto_fused_performance -xvs
    """
    warmup = 3
    num_runs = 1 if single_run_only else 20

    # --- Hand-fused timing ---
    hand_times = []
    for i in range(warmup + num_runs):
        _, g1, g2, out, _, _ = create_tensors(device, tile_h, tile_w, group1_num_tiles, group2_num_tiles, seed=42 + i)
        ttnn.synchronize_device(device)
        t0 = time.perf_counter()
        run_hand_fused(g1, g2, out, group1_num_tiles, group2_num_tiles)
        ttnn.synchronize_device(device)
        t1 = time.perf_counter()
        if i >= warmup:
            hand_times.append(t1 - t0)

    # --- Auto-fused timing ---
    auto_times = []
    core = ttnn.CoreCoord(0, 0)
    core_grid = ttnn.CoreRangeSet({ttnn.CoreRange(core, core)})
    tile = ttnn.Tile([tile_h, tile_w])

    for i in range(warmup + num_runs):
        _, g1, g2, _, _, _ = create_tensors(device, tile_h, tile_w, group1_num_tiles, group2_num_tiles, seed=42 + i)
        ttnn.synchronize_device(device)
        t0 = time.perf_counter()
        run_auto_fused(
            device,
            g1,
            g2,
            core_grid,
            tile,
            tile_h,
            tile_w,
            group1_num_tiles,
            group2_num_tiles,
        )
        ttnn.synchronize_device(device)
        t1 = time.perf_counter()
        if i >= warmup:
            auto_times.append(t1 - t0)

    hand_avg = sum(hand_times) / len(hand_times) * 1000  # ms
    auto_avg = sum(auto_times) / len(auto_times) * 1000  # ms
    ratio = auto_avg / hand_avg if hand_avg > 0 else float("inf")

    logger.info(f"\n{'='*60}")
    logger.info(f"PERFORMANCE: group1={group1_num_tiles}, group2={group2_num_tiles}")
    logger.info(f"  Hand-fused:  {hand_avg:.3f} ms (avg of {num_runs} runs)")
    logger.info(f"  Auto-fused:  {auto_avg:.3f} ms (avg of {num_runs} runs)")
    logger.info(f"  Ratio:       {ratio:.2f}x (auto/hand, lower is better)")
    logger.info(f"{'='*60}")

    # We don't assert on performance — just report.
    # The goal is to be within 2x of hand-fused for host-side timing.
    # Device FW timing (Tracy) is the true comparison.
