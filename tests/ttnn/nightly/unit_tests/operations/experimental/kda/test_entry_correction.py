# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Single-chip accuracy, determinism and production performance of ttnn.experimental.kda.entry_correction.

On one chip the SP ring is a single rank, so a start inside the rank splits it: its chunks from the reset chunk on, and
its final state, take the tail's entry state.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import pytest
import torch
from loguru import logger

import ttnn
from tt_py_test_utils_common.utility_functions import run_for_blackhole, skip_with_llk_assert, skip_with_watcher
from models.demos.deepseek_v3_d_p.tt.kda.entry_correction import EntryCorrection
from tests.ttnn.nightly.unit_tests.operations.experimental.kda import kda_performance_model_test_utils as perf_model
from tests.ttnn.profiling.realtime_profiler_utils import profile_realtime_program
from tests.ttnn.unit_tests.operations.experimental.kda.kda_test_utils import (
    assert_accurate,
    assert_bit_identical,
    make_actual_start,
)

pytestmark = [
    run_for_blackhole(),
    pytest.mark.use_module_device({"l1_small_size": 24576}),
]

_CHUNK = 32
# Kimi-K3 on Galaxy SP8xTP4: one device's 640 tokens of 24 heads, over the model's 110 correction cores.
_ROWS, _HEADS, _KEY = 640, 24, 128
_CHUNKS = _ROWS // _CHUNK
_MAX_CORES = 110

_PERF_MARGIN = 0.05
_PRODUCTION_DURATION_NS: int | None = 19_685


@dataclass(frozen=True)
class _Interval:
    start: int
    end: int | None

    def valid_and_reset_chunks(self) -> tuple[int, int]:
        """The rank's valid chunks and the chunk its split tail starts at (0: no tail). On one rank the interval wraps
        within the rank: the head holds rows [offset, ROWS), the tail rows [0, offset)."""
        offset = self.start % _ROWS
        head_rows = _ROWS - offset
        valid_rows = _ROWS if self.end is None else min(self.end - self.start, _ROWS)
        reset = head_rows // _CHUNK if offset != 0 and valid_rows > head_rows else 0
        return -(-valid_rows // _CHUNK), reset


# From the sequence start every chunk enters at the entry state; "bounded" ends two chunks before the rank's end. A start
# 352 rows in splits the rank at chunk 9; "split-bounded" ends the interval three chunks into the tail. Over the 110
# cores a head's 20 chunks split five per core, so the reset falls inside the second core's chunks.
_INTERVALS = {
    "full": _Interval(0, None),
    "bounded": _Interval(0, _ROWS - 64),
    "split": _Interval(352, None),
    "split-bounded": _Interval(352, 352 + _ROWS - 256),
}


def _cores(device) -> list:
    grid = device.compute_with_storage_grid_size()
    return [ttnn.CoreCoord(x, y) for y in range(grid.y) for x in range(grid.x)][:_MAX_CORES]


def _correction(device, math_fidelity: ttnn.MathFidelity) -> EntryCorrection:
    return EntryCorrection(
        device,
        sequence_parallel_axis=0,
        batch_heads=_HEADS,
        num_chunks=_CHUNKS,
        key_dim=_KEY,
        local_rows=_ROWS,
        cores=_cores(device),
        math_fidelity=math_fidelity,
    )


def _host_inputs(seed: int) -> tuple[torch.Tensor, ...]:
    generator = torch.Generator().manual_seed(seed)
    zero = 0.1 * torch.randn(_HEADS, _CHUNKS, _CHUNK, _KEY, generator=generator)
    identity = 0.1 * torch.randn(_HEADS, _CHUNKS, _CHUNK, _KEY, generator=generator)
    final_a, final_b, entry, tail = (0.1 * torch.randn(_HEADS, _KEY, _KEY, generator=generator) for _ in range(4))
    return zero, identity, final_a, final_b, entry, tail


def _device_inputs(device, correction: EntryCorrection, host: tuple[torch.Tensor, ...]) -> tuple[ttnn.Tensor, ...]:
    def to_device(tensor, memory_config=ttnn.DRAM_MEMORY_CONFIG):
        return ttnn.from_torch(
            tensor, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device, memory_config=memory_config
        )

    zero, identity, *states = host
    return (
        to_device(zero, correction.terms_memory_config),
        to_device(identity, correction.terms_memory_config),
        *(to_device(state) for state in states),
    )


def _reference(host: tuple[torch.Tensor, ...], interval: _Interval) -> tuple[torch.Tensor, torch.Tensor, int]:
    """The valid chunk outputs and the final state, in FP64."""
    zero, identity, final_a, final_b, entry, tail = (tensor.double() for tensor in host)
    valid, reset = interval.valid_and_reset_chunks()
    output = zero + (identity - zero) @ entry.unsqueeze(1)
    if reset:
        output[:, reset:] = (zero + (identity - zero) @ tail.unsqueeze(1))[:, reset:]
    final = final_a @ (tail if reset else entry) + final_b
    return output[:, :valid].float(), final.float(), valid


def _entry_correction_ops() -> perf_model.FpuOps:
    # As the op's performance model counts it: per head, the chunk rows and the final state each multiply a [rows, K]
    # or [K, K] block by the K x K seed, after the chunk rows' differences FI - F0.
    return perf_model.FpuOps(
        matrix_flops=_HEADS * 2 * (_ROWS + _KEY) * _KEY * _KEY,
        add_ops=_HEADS * _ROWS * _KEY,
    )


# HiFi4 is exact enough to bound the op's own error tightly; HiFi2 is the model's scan fidelity.
_FIDELITIES = [
    pytest.param(ttnn.MathFidelity.HiFi4, 0.01, 1e-3, id="hifi4"),
    pytest.param(ttnn.MathFidelity.HiFi2, 0.01, 0.01, id="hifi2-production"),
]


@pytest.mark.parametrize("math_fidelity, output_nrmse, final_nrmse", _FIDELITIES)
@pytest.mark.parametrize("interval_id", list(_INTERVALS))
def test_entry_correction_accuracy_and_determinism(
    device: ttnn.Device,
    interval_id: str,
    math_fidelity: ttnn.MathFidelity,
    output_nrmse: float,
    final_nrmse: float,
) -> None:
    interval = _INTERVALS[interval_id]
    correction = _correction(device, math_fidelity)
    host = _host_inputs(seed=interval.start + (interval.end or 0))
    inputs = _device_inputs(device, correction, host)
    actual_start = make_actual_start(device, interval.start)
    actual_end = None if interval.end is None else make_actual_start(device, interval.end)
    expected_output, expected_final, valid = _reference(host, interval)

    results = []
    for _ in range(3):
        output, final = correction(*inputs, actual_start, actual_end)
        results.append((ttnn.to_torch(output).float()[:, :valid], ttnn.to_torch(final)))
        ttnn.deallocate(output)
        ttnn.deallocate(final)
    got_output, got_final = results[0]
    assert_accurate(
        expected_output,
        got_output,
        name=f"{interval_id} output",
        pcc_threshold=0.9999,
        nrmse_threshold=output_nrmse,
    )
    assert_accurate(
        expected_final,
        got_final,
        name=f"{interval_id} final",
        pcc_threshold=0.9999,
        nrmse_threshold=final_nrmse,
    )
    for repeat, (output, final) in enumerate(results[1:], start=1):
        assert_bit_identical(got_output, output, name=f"{interval_id} output repeat {repeat}")
        assert_bit_identical(got_final, final, name=f"{interval_id} final repeat {repeat}")


def test_entry_correction_cache_hit_reads_new_interval(device: ttnn.Device, isolated_program_cache: None) -> None:
    """actual_start and actual_end are read on device, not baked into the program: one cached program must serve an
    unsplit interval and a split, bounded one."""
    correction = _correction(device, ttnn.MathFidelity.HiFi4)
    host = _host_inputs(seed=7)
    inputs = _device_inputs(device, correction, host)
    entries = device.num_program_cache_entries()
    for interval_id in ("bounded", "split-bounded"):
        interval = _INTERVALS[interval_id]
        output, final = correction(
            *inputs, make_actual_start(device, interval.start), make_actual_start(device, interval.end)
        )
        expected_output, expected_final, valid = _reference(host, interval)
        assert_accurate(
            expected_output,
            ttnn.to_torch(output).float()[:, :valid],
            name=f"cached {interval_id} output",
            pcc_threshold=0.9999,
            nrmse_threshold=0.01,
        )
        assert_accurate(
            expected_final,
            ttnn.to_torch(final),
            name=f"cached {interval_id} final",
            pcc_threshold=0.9999,
            nrmse_threshold=1e-3,
        )
    assert device.num_program_cache_entries() == entries + 1


def test_entry_correction_rejects_aliased_entry_states(
    device: ttnn.Device, isolated_program_cache: None, expect_error: Callable
) -> None:
    """An unsplit interval never reads tail_entry, but the same tensor as entry would leave the cached program's
    bindings ambiguous: rejected, and a later call with distinct tensors is still correct."""
    correction = _correction(device, ttnn.MathFidelity.HiFi4)
    host = _host_inputs(seed=11)
    zero, identity, final_a, final_b, entry, tail = _device_inputs(device, correction, host)
    actual_start = make_actual_start(device, 0)
    with expect_error(RuntimeError, "entry and tail_entry must be different tensors"):
        correction(zero, identity, final_a, final_b, entry, entry, actual_start, None)
    output, final = correction(zero, identity, final_a, final_b, entry, tail, actual_start, None)
    expected_output, expected_final, valid = _reference(host, _INTERVALS["full"])
    assert_accurate(
        expected_output,
        ttnn.to_torch(output).float()[:, :valid],
        name="output after a rejected alias",
        pcc_threshold=0.9999,
        nrmse_threshold=0.01,
    )
    assert_accurate(
        expected_final,
        ttnn.to_torch(final),
        name="final after a rejected alias",
        pcc_threshold=0.9999,
        nrmse_threshold=1e-3,
    )


@pytest.mark.requires_host_iommu
@skip_with_llk_assert("No need to verify LLK asserts for performance tests.")
@skip_with_watcher("Watcher perturbs kernel timing; perf checks are not meaningful with it enabled.")
def test_entry_correction_production_performance(device: ttnn.Device) -> None:
    """The model's call: a split rank, so both seeds and the reset are exercised, at the scan's HiFi2."""
    if not ttnn.device.IsProgramRealtimeProfilerActive():
        pytest.fail("Real-time profiler must be active for entry-correction performance checks")
    math_fidelity = ttnn.MathFidelity.HiFi2
    correction = _correction(device, math_fidelity)
    inputs = _device_inputs(device, correction, _host_inputs(seed=117))
    interval = _INTERVALS["split"]
    actual_start = make_actual_start(device, interval.start)
    # Compile outside the measured window.
    for tensor in correction(*inputs, actual_start, None):
        ttnn.deallocate(tensor)

    (output, final), perf_record = profile_realtime_program(device, lambda: correction(*inputs, actual_start, None))
    duration_ns = perf_record["duration_ns"]
    assert tuple(output.shape) == (_HEADS, _CHUNKS, _CHUNK, _KEY)
    assert tuple(final.shape) == (_HEADS, _KEY, _KEY)
    performance = perf_model.performance(
        fpu=_entry_correction_ops(),
        sfpu=perf_model.SfpuOps(),
        inputs=inputs,
        outputs=(output, final),
        measured_ns=duration_ns,
        math_fidelity=math_fidelity,
    )
    logger.info(
        f"entry correction production h{_HEADS}-s{_ROWS}-k{_KEY}: measured_ns={duration_ns:.0f}, "
        f"reference={_PRODUCTION_DURATION_NS} ns, cores={len(correction.cores)}, "
        f"runtime_id={perf_record['runtime_id']}, work={performance.work}, "
        f"ideal_fpu_ns={performance.ideal_fpu_ns:.2f}, ideal_dram_ns={performance.ideal_dram_ns:.2f}, "
        f"ideal_ns={performance.ideal_ns:.2f}, "
        f"fpu_utilization_pct={performance.fpu_utilization_pct:.2f}, "
        f"dram_utilization_pct={performance.dram_utilization_pct:.2f}, "
        f"utilization_pct={performance.utilization_pct:.2f}"
    )
    if _PRODUCTION_DURATION_NS is None:
        pytest.fail(f"no reference duration: set _PRODUCTION_DURATION_NS from the measured {duration_ns:.0f} ns")
    lower = _PRODUCTION_DURATION_NS * (1 - _PERF_MARGIN)
    upper = _PRODUCTION_DURATION_NS * (1 + _PERF_MARGIN)
    assert lower <= duration_ns <= upper, (
        f"duration {duration_ns:.0f} ns outside [{lower:.0f}, {upper:.0f}] ns "
        f"(reference {_PRODUCTION_DURATION_NS} ns, margin +/- {_PERF_MARGIN * 100:.0f}%)"
    )
