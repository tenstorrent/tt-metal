# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Per-stage device time of the KDA grouped-scan schedule at K = V = 128.

Measures the stages whose cost depends on the number of time groups per head:
summary, group reduction (sequence-parallel only), group exclusive scan, and the
recurrent scan. One item is one stage at one folded geometry, so the critical
path of a schedule is the sum of its stage items. Uniform groups only: the
critical path of a ragged split equals that of its longest group.

Measurement harness, not a gate: it logs real-time profiler samples.
"""

from __future__ import annotations

import statistics
from collections.abc import Callable
from dataclasses import dataclass

import pytest
from loguru import logger

import ttnn
from models.common.utility_functions import run_for_blackhole, skip_with_llk_assert, skip_with_watcher
from tests.ttnn.nightly.unit_tests.operations.experimental.kda.recurrent_chunk_scan_test_utils import (
    CHUNK_SIZE,
    device_protocol,
    group_summary_height_sharded,
    host_protocol,
    initial_state,
    to_device,
)
from tests.ttnn.profiling.realtime_profiler_utils import profile_realtime_program
from tests.ttnn.unit_tests.operations.experimental.kda.kda_test_utils import make_actual_start

pytestmark = [
    run_for_blackhole(),
    pytest.mark.perf,
    pytest.mark.use_module_device({"l1_small_size": 24576}),
]

_DIM = 128
_SAMPLES = 5
# Preparation outputs stored as BF16 in the layer (KDA_PREP_OUTPUT_BF16_MASK).
_PREP_BF16 = frozenset({"kd", "q_decay", "final_decay"})


@dataclass(frozen=True)
class _Stage:
    stage: str
    heads: int
    groups: int
    group_chunks: int

    @property
    def folded_heads(self) -> int:
        return self.heads * self.groups

    @property
    def stage_id(self) -> str:
        return f"{self.stage}-h{self.heads}-g{self.groups}-n{self.group_chunks}"


_STAGES = (
    # Today at 32 heads: one group of 20 chunks; the recurrent scan splits V in two (64 cores).
    _Stage("scan", 32, 1, 20),
    _Stage("summary", 32, 1, 20),
    _Stage("affine_scan", 32, 1, 20),
    _Stage("reduce", 32, 1, 20),
    # Two groups of 10 (64 owners, full V per core).
    _Stage("scan", 32, 2, 10),
    _Stage("summary", 32, 2, 10),
    _Stage("affine_scan", 32, 2, 10),
    _Stage("reduce", 32, 2, 10),
    # Three groups of 7: the critical path of 7/7/6 (96 owners, full V per core).
    _Stage("scan", 32, 3, 7),
    _Stage("summary", 32, 3, 7),
    _Stage("affine_scan", 32, 3, 7),
    _Stage("reduce", 32, 3, 7),
    # Per-chunk slope points at fixed placement.
    _Stage("scan", 32, 1, 7),
    _Stage("scan", 32, 1, 10),
    _Stage("scan", 32, 3, 20),
    _Stage("scan", 32, 3, 10),
    _Stage("summary", 32, 1, 7),
    _Stage("summary", 32, 1, 10),
    _Stage("summary", 32, 3, 20),
    _Stage("summary", 32, 3, 10),
    # Kimi K3 (24 heads/chip) and GLM-5.3-Flash (16 heads/chip) at 640 rows: one group of 20
    # (the recurrent scan splits V in four) versus two groups of 10 (V split in two).
    *(
        _Stage(stage, heads, groups, 20 // groups)
        for heads in (24, 16)
        for groups in (1, 2)
        for stage in ("scan", "summary", "affine_scan", "reduce")
    ),
)


def _compute_config(device: ttnn.Device, fidelity: ttnn.MathFidelity) -> ttnn.DeviceComputeKernelConfig:
    # KDARecurrence: preparation/summary HiFi4, affine prefix and scan HiFi2 (config.py defaults).
    return ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=fidelity,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )


def _stage_runner(case: _Stage, device: ttnn.Device) -> Callable[[], list[ttnn.Tensor]]:
    actual_start = make_actual_start(device, 0)
    local_rows = case.groups * case.group_chunks * CHUNK_SIZE
    if case.stage in ("scan", "summary"):
        protocol = device_protocol(
            host_protocol(case.folded_heads, case.group_chunks, _DIM, _DIM, bf16_names=_PREP_BF16, seed=117),
            device,
        )
        if case.stage == "summary":
            memory = group_summary_height_sharded(device, case.folded_heads, _DIM)
            config = _compute_config(device, ttnn.MathFidelity.HiFi4)
            return lambda: list(
                ttnn.experimental.kda.summarize_chunk_recurrence(
                    *protocol,
                    groups_per_head=case.groups,
                    memory_config=memory,
                    compute_kernel_config=config,
                    actual_start=actual_start,
                )
            )
        entry = to_device(initial_state(case.folded_heads, _DIM, _DIM), device)
        tail = to_device(initial_state(case.heads, _DIM, _DIM), device)
        config = _compute_config(device, ttnn.MathFidelity.HiFi2)
        return lambda: list(
            ttnn.experimental.kda.recurrent_chunk_scan(
                *protocol,
                entry,
                groups_per_head=case.groups,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                compute_kernel_config=config,
                actual_start=actual_start,
                tail_entry_states=tail,
            )
        )

    memory = group_summary_height_sharded(device, case.folded_heads, _DIM)
    a = to_device(
        initial_state(case.folded_heads, _DIM, _DIM, seed=5), device, dtype=ttnn.bfloat16, memory_config=memory
    )
    b = to_device(
        initial_state(case.folded_heads, _DIM, _DIM, seed=6), device, dtype=ttnn.bfloat16, memory_config=memory
    )
    config = _compute_config(device, ttnn.MathFidelity.HiFi2)
    if case.stage == "affine_scan":
        state = to_device(initial_state(case.heads, _DIM, _DIM), device)
        return lambda: [
            ttnn.experimental.kda.affine_exclusive_scan(
                a,
                b,
                state,
                case.groups,
                local_rows=local_rows,
                tail_a=a,
                tail_b=b,
                tail_entry_states=state,
                actual_start=actual_start,
                memory_config=ttnn.L1_MEMORY_CONFIG,
                compute_kernel_config=config,
            )
        ]
    if case.stage == "reduce":
        return lambda: list(
            ttnn.experimental.kda.reduce_affine_transforms(
                a,
                b,
                case.groups,
                local_rows=local_rows,
                actual_start=actual_start,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                compute_kernel_config=config,
            )
        )
    raise ValueError(f"unknown stage {case.stage}")


@pytest.mark.requires_host_iommu
@skip_with_llk_assert("No need to verify LLK asserts for performance tests.")
@skip_with_watcher("Watcher perturbs kernel timing; perf checks are not meaningful with it enabled.")
@pytest.mark.parametrize("case", _STAGES, ids=lambda case: case.stage_id)
def test_grouped_scan_stage_device_time(case: _Stage, device: ttnn.Device) -> None:
    if not ttnn.device.IsProgramRealtimeProfilerActive():
        pytest.fail("Real-time profiler must be active for grouped-scan stage measurements")
    run = _stage_runner(case, device)
    for tensor in run():  # Compile and warm up outside the samples.
        ttnn.deallocate(tensor)
    samples = []
    for _ in range(_SAMPLES):
        outputs, record = profile_realtime_program(device, run)
        samples.append(record["duration_ns"])
        for tensor in outputs:
            ttnn.deallocate(tensor)
    logger.info(
        f"grouped scan stage {case.stage_id}: median_ns={statistics.median(samples):.0f}, "
        f"min_ns={min(samples):.0f}, max_ns={max(samples):.0f}, "
        f"samples_ns={[round(sample) for sample in samples]}"
    )
