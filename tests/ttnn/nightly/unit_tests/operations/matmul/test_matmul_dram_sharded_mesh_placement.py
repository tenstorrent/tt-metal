# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""DRAM-sharded matmul on a 1x4 mesh: every chip gets the right result in about the same device time.

A chip's best reader core for a DRAM bank is the first unharvested worker column east of the bank, and logical
columns skip harvested ones, so on a mesh of differently harvested Blackhole chips (QuietBox 2) one placement for the
whole mesh -- the first chip's -- puts other chips' readers west of their banks, where NoC 0 delivers the bank's data
the long way round the row. Those chips then run the matmul 35-50% slower, and every collective after it waits for
them. The op now builds one program per chip with that chip's own placement when the chips' placements differ (and
one program when they agree). Cases: one reader per bank, two readers per bank (num_workers_per_dram_bank=2), and the
multi-core pipeline with two cores per bank (use_multicore_pipeline=True).

This test measures each chip's device time with the real-time profiler and requires the slowest chip to be within
_SPREAD of the fastest. On a mesh of identically harvested chips it passes either way. On a QuietBox 2 (2026-10-26)
with the first chip's placement, two readers per bank: chips 1 and 3 took 216 us against 149 us on chips 0 and 2
(45% spread, fails); with per-chip placement all four took 149 us. One reader per bank at this shape is not bound by
its DRAM reads (0.6% spread either way) and checks that the single-reader path stays balanced.
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
# Weight-bound, like a decode MoE projection: 71 MB of bf8 weights per chip, one 32-row activation tile.
_M, _K, _N = 32, 4096, 16384
_GRID = (8, 8)  # in0 width-shard and output cores


def _matmul_inputs(mesh_device, num_workers_per_dram_bank, use_multicore_pipeline):
    banks = mesh_device.dram_grid_size().x
    assert _N % (banks * num_workers_per_dram_bank) == 0
    cores = _GRID[0] * _GRID[1]
    in0 = torch.randn(1, 1, _M, _K).bfloat16().float()
    in1 = torch.randn(1, 1, _K, _N).bfloat16().float()
    replicate = ttnn.ReplicateTensorToMesh(mesh_device)

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
    ids=["one_worker", "two_workers", "two_workers_multicore"],
)
@pytest.mark.parametrize("mesh_device", [(1, 4)], indirect=True)
@pytest.mark.requires_host_iommu
@pytest.mark.skipif(not is_blackhole(), reason="per-chip DRAM reader placement matters on harvested Blackhole meshes")
@skip_with_llk_assert("No need to verify LLK asserts for performance tests.")
@skip_with_watcher("Watcher perturbs kernel timing; perf checks are not meaningful with it enabled.")
def test_dram_sharded_matmul_chips_balanced(mesh_device, num_workers_per_dram_bank, use_multicore_pipeline):
    require_realtime_profiler("DRAM-sharded matmul per-chip balance")
    torch.manual_seed(0)
    run, expected = _matmul_inputs(mesh_device, num_workers_per_dram_bank, use_multicore_pipeline)

    # every chip computes the same product
    out = ttnn.sharded_to_interleaved(run(), ttnn.DRAM_MEMORY_CONFIG)
    for chip, actual in enumerate(ttnn.get_device_tensors(out)):
        actual = ttnn.to_torch(actual).float()
        pcc = torch.corrcoef(torch.stack([actual.flatten(), expected.flatten()]))[0, 1].item()
        assert pcc > 0.99, f"chip {chip}: PCC {pcc}"

    def run_all():
        for _ in range(_ITERS):
            run()

    _, records = profile_realtime_program(mesh_device, run_all, collect_all=True)
    per_chip = {}
    for r in records:
        if any("dram_sharded" in source for source in r["kernel_sources"]):
            per_chip.setdefault(r["chip_id"], []).append(r["duration_ns"])
    assert len(per_chip) == mesh_device.get_num_devices(), f"records for chips {sorted(per_chip)}"
    medians = {chip: statistics.median(ns) for chip, ns in per_chip.items()}
    logger.info(f"per-chip device time (ns, median of {_ITERS}): {dict(sorted(medians.items()))}")
    spread = max(medians.values()) / min(medians.values()) - 1
    assert spread <= _SPREAD, f"slowest chip {spread:.1%} behind the fastest: {dict(sorted(medians.items()))}"
