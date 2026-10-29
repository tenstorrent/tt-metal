# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Single-chip accuracy, determinism and production performance of ttnn.experimental.kda.entry_correction.

On one chip the SP ring is a single rank, and the KDA chronology never splits a single rank (kda_chronology::derive):
every valid chunk, and the final state, take the entry state wherever the interval starts. The split tail and its reset
chunk need two ranks or more; the multi-chip relay and layer tests cover them.
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

    def valid_chunks(self) -> int:
        """The rank's valid chunks: an interval of end - start rows, or the whole rank without an end."""
        valid_rows = _ROWS if self.end is None else min(self.end - self.start, _ROWS)
        return -(-valid_rows // _CHUNK)


# "bounded" ends two chunks before the rank's end. "offset" starts 352 rows in, which on one rank changes nothing;
# "offset-bounded" ends 12 chunks after it, so a head's last cores (five chunks each over the 110 cores) have none.
_INTERVALS = {
    "full": _Interval(0, None),
    "bounded": _Interval(0, _ROWS - 64),
    "offset": _Interval(352, None),
    "offset-bounded": _Interval(352, 352 + _ROWS - 256),
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
    """The valid chunk outputs and the final state, in FP64. tail_entry is never read on one rank."""
    zero, identity, final_a, final_b, entry, _ = (tensor.double() for tensor in host)
    valid = interval.valid_chunks()
    output = zero + (identity - zero) @ entry.unsqueeze(1)
    final = final_a @ entry + final_b
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
    interval from the rank's start and a bounded one from inside it."""
    correction = _correction(device, ttnn.MathFidelity.HiFi4)
    host = _host_inputs(seed=7)
    inputs = _device_inputs(device, correction, host)
    entries = device.num_program_cache_entries()
    for interval_id in ("bounded", "offset-bounded"):
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
    """The model's call shape, starting inside the rank, at the scan's HiFi2."""
    if not ttnn.device.IsProgramRealtimeProfilerActive():
        pytest.fail("Real-time profiler must be active for entry-correction performance checks")
    math_fidelity = ttnn.MathFidelity.HiFi2
    correction = _correction(device, math_fidelity)
    inputs = _device_inputs(device, correction, _host_inputs(seed=117))
    interval = _INTERVALS["offset"]
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
