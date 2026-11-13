# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""A DRAM-sharded matmul reads its weight at the same speed on every chip of a mesh.

The best reader core for a DRAM bank is the first unharvested worker column east of the bank, so it depends on each
chip's harvesting; chips of one mesh can differ (a QuietBox 2 does). The op used to place the readers once, for the
first chip, and run that program on all chips: a chip whose best reader is elsewhere then reads its banks from the
wrong side and runs the matmul tens of percent slower, and every collective after it waits for that chip.

The op now places each chip's readers for that chip (one program per chip where the placements differ, one program
for the whole mesh where they agree). This test checks that every chip computes the same result and that the slowest
chip is within _SPREAD of the fastest, with the real-time program profiler. On a mesh of identically harvested chips
it passes either way.
"""

import statistics

import pytest
import torch
from loguru import logger

import ttnn
from tt_py_test_utils_common.utility_functions import is_blackhole, skip_with_llk_assert, skip_with_watcher
from tests.ttnn.profiling.realtime_profiler_utils import profile_realtime_program, require_realtime_profiler

_ITERS = 5
_SPREAD = 0.10  # slowest chip / fastest chip - 1
# Weight-bound, like a decode projection: 71 MB of bf8 weights per chip and one 32-row activation tile.
_M, _K, _N = 32, 4096, 16384
_GRID = (8, 8)  # in0 width-shard and output cores


def _matmul(mesh_device, num_workers_per_dram_bank, use_multicore_pipeline):
    banks = mesh_device.dram_grid_size().x
    cores = _GRID[0] * _GRID[1]
    replicate = ttnn.ReplicateTensorToMesh(mesh_device)
    in0 = torch.randn(1, 1, _M, _K).bfloat16().float()
    in1 = torch.randn(1, 1, _K, _N).bfloat16().float()
    in1_grid = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(banks - 1, mesh_device.dram_grid_size().y - 1))}
    )
    in1_mem = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.DRAM,
        ttnn.ShardSpec(in1_grid, [_K, _N // banks], ttnn.ShardOrientation.ROW_MAJOR),
    )
    in1_t = ttnn.from_torch(
        in1,
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=in1_mem,
        mesh_mapper=replicate,
    )
    in0_t = ttnn.from_torch(
        in0, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device, mesh_mapper=replicate
    )
    in0_t = ttnn.interleaved_to_sharded(
        in0_t, _GRID, [_M, _K // cores], ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.ShardOrientation.ROW_MAJOR
    )
    program_config = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
        in0_block_w=_K // cores // 32,
        per_core_M=_M // 32,
        per_core_N=_N // cores // 32,
        fused_activation=None,
        num_workers_per_dram_bank=num_workers_per_dram_bank,
        use_multicore_pipeline=use_multicore_pipeline,
    )
    compute = ttnn.init_device_compute_kernel_config(
        mesh_device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=True,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )
    out_mem = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1)

    def run():
        return ttnn.matmul(
            in0_t,
            in1_t,
            program_config=program_config,
            memory_config=out_mem,
            dtype=ttnn.bfloat16,
            compute_kernel_config=compute,
        )

    return run, in0 @ in1


@pytest.mark.parametrize(
    "num_workers_per_dram_bank, use_multicore_pipeline",
    [(1, False), (2, False), (2, True)],
    ids=["one_reader", "two_readers", "two_readers_multicore"],
)
@pytest.mark.parametrize("mesh_device", [(1, 4)], indirect=True)
@pytest.mark.requires_host_iommu
@pytest.mark.skipif(not is_blackhole(), reason="DRAM reader placement is per chip on harvested Blackhole meshes")
@skip_with_llk_assert("No need to verify LLK asserts for performance tests.")
@skip_with_watcher("Watcher perturbs kernel timing; perf checks are not meaningful with it enabled.")
def test_dram_sharded_matmul_is_balanced_across_chips(mesh_device, num_workers_per_dram_bank, use_multicore_pipeline):
    require_realtime_profiler("DRAM-sharded matmul per-chip balance")
    torch.manual_seed(0)
    run, expected = _matmul(mesh_device, num_workers_per_dram_bank, use_multicore_pipeline)

    out = ttnn.sharded_to_interleaved(run(), ttnn.DRAM_MEMORY_CONFIG)
    results = [ttnn.to_torch(t).float() for t in ttnn.get_device_tensors(out)]
    for chip, actual in enumerate(results):
        pcc = torch.corrcoef(torch.stack([actual.flatten(), expected.flatten()]))[0, 1].item()
        assert pcc > 0.99, f"chip {chip}: PCC {pcc}"
        assert torch.equal(actual, results[0]), f"chip {chip} differs from chip 0 although the inputs are replicated"

    def run_all():
        for _ in range(_ITERS):
            run()

    _, records = profile_realtime_program(mesh_device, run_all, collect_all=True)
    per_chip = {}
    for record in records:
        if any("dram_sharded" in source for source in record["kernel_sources"]):
            per_chip.setdefault(record["chip_id"], []).append(record["duration_ns"])
    assert len(per_chip) == mesh_device.get_num_devices(), f"records for chips {sorted(per_chip)}"
    medians = {chip: statistics.median(ns) for chip, ns in sorted(per_chip.items())}
    logger.info(f"per-chip device time (ns, median of {_ITERS}): {medians}")
    spread = max(medians.values()) / min(medians.values()) - 1
    assert spread <= _SPREAD, f"slowest chip {spread:.1%} behind the fastest: {medians}"
