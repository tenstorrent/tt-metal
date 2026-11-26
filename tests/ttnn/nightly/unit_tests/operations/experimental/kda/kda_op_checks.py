# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Shared checks for KDA single-op tests (README.md).

An op test module holds its cases, tolerances and harness (``host_inputs``, ``reference``, ``device_inputs``,
``run``); each standard test is a few lines that call one check here. Op test modules import measurement,
determinism and profiler helpers only through this module, so moving a helper changes one import here.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import pytest
import torch

import ttnn
from models.demos.deepseek_v3_d_p.tests.kda.kda_test_utils import assert_accurate, assert_bit_identical
from models.demos.deepseek_v3_d_p.tests.measure.determinism import check_determinism, tensors
from models.demos.deepseek_v3_d_p.tests.measure.trace import TracedStep
from tests.ttnn.nightly.unit_tests.operations.experimental.kda import kda_performance_model_test_utils as perf_model
from tt_py_test_utils_common.realtime_profiler_utils import profile_realtime_program, require_realtime_profiler
from tt_py_test_utils_common.utility_functions import skip_with_llk_assert, skip_with_watcher

__all__ = [
    "Tolerance",
    "assert_bit_identical",
    "assert_cache_hit_rebinds",
    "assert_close",
    "assert_default_config_matches",
    "assert_deterministic",
    "assert_device_time",
    "assert_output_contract",
    "assert_programs_added",
    "case_params",
    "performance_test",
    "perf_model",
]


@dataclass(frozen=True)
class Tolerance:
    """Accuracy gate of one output: the clean value measured for it plus headroom (see assert_accurate)."""

    nrmse: float
    pcc: float = 0.999
    linf: float | None = None


def assert_close(expected: torch.Tensor, actual: torch.Tensor, tolerance: Tolerance, *, name: str) -> None:
    assert_accurate(
        expected.float(),
        actual.float(),
        name=name,
        pcc_threshold=tolerance.pcc,
        nrmse_threshold=tolerance.nrmse,
        linf_threshold=tolerance.linf,
    )


def case_params(*cases) -> list:
    """``pytest.param`` per case, with the case's id and the ``production`` mark on production cases.
    A case needs ``id`` and ``production`` attributes."""
    return [pytest.param(case, id=case.id, marks=[pytest.mark.production] if case.production else []) for case in cases]


def assert_output_contract(
    inputs: Sequence[ttnn.Tensor],
    call: Callable[[], ttnn.Tensor | Sequence[ttnn.Tensor]],
    *,
    dtypes: Sequence[ttnn.DataType],
    shapes: Sequence[tuple[int, ...]],
    memory_config: ttnn.MemoryConfig,
) -> list[ttnn.Tensor]:
    """Outputs have the promised dtype, TILE layout, memory configuration and shape, own distinct buffers that
    alias no input, and the call leaves every input unchanged. Returns the outputs."""
    snapshots = [ttnn.to_torch(tensor).clone() for tensor in inputs]
    outputs = tensors(call())
    assert len(outputs) == len(dtypes) == len(shapes), f"{len(outputs)} outputs, expected {len(dtypes)}"
    input_addresses = {tensor.buffer_address() for tensor in inputs}
    for index, (output, dtype, shape) in enumerate(zip(outputs, dtypes, shapes, strict=True)):
        assert output.dtype == dtype, f"output {index} dtype {output.dtype} != {dtype}"
        assert output.layout == ttnn.TILE_LAYOUT, f"output {index} layout {output.layout}"
        assert output.memory_config() == memory_config, f"output {index} memory {output.memory_config()}"
        assert tuple(output.shape) == shape, f"output {index} shape {tuple(output.shape)} != {shape}"
        assert output.buffer_address() not in input_addresses, f"output {index} aliases an input"
    assert len({output.buffer_address() for output in outputs}) == len(outputs), "outputs share a buffer"
    for index, (snapshot, tensor) in enumerate(zip(snapshots, inputs, strict=True)):
        assert_bit_identical(snapshot, ttnn.to_torch(tensor), name=f"input {index} unchanged")
    return outputs


def assert_deterministic(device: ttnn.MeshDevice, step: Callable[[], ttnn.Tensor | Sequence[ttnn.Tensor]]) -> None:
    """10 eager runs back to back, 10 replays of one captured trace, and the first replay against the first eager
    run agree on device (ttnn.ne: value equality, so -0 equals +0 and a NaN fails)."""
    failure = check_determinism(device, step, lambda: TracedStep(device, step))
    assert failure is None, failure


def assert_programs_added(
    device: ttnn.MeshDevice, calls: Sequence[Callable[[], object]], expected: Sequence[int | None]
) -> list:
    """Each call adds exactly its expected number of program-cache entries (None: not checked, e.g. a warm-up
    whose program an earlier test may have cached). Returns the calls' outputs."""
    outputs = []
    for index, (call, added) in enumerate(zip(calls, expected, strict=True)):
        before = device.num_program_cache_entries()
        outputs.append(call())
        ttnn.synchronize_device(device)
        actual = device.num_program_cache_entries() - before
        assert added is None or actual == added, f"call {index} added {actual} programs, expected {added}"
    return outputs


def assert_cache_hit_rebinds(
    device: ttnn.MeshDevice,
    run: Callable[[Sequence[ttnn.Tensor]], ttnn.Tensor | Sequence[ttnn.Tensor]],
    inputs: tuple[Sequence[ttnn.Tensor], Sequence[ttnn.Tensor]],
    expected: tuple[Sequence[torch.Tensor], Sequence[torch.Tensor]],
    tolerance: Tolerance,
) -> None:
    """A second call on new buffers reuses the first call's program and reads and writes the new buffers.
    The two input sets must hold different values, so stale reads would show."""
    first_inputs, second_inputs = inputs
    assert all(a.buffer_address() != b.buffer_address() for a, b in zip(first_inputs, second_inputs, strict=True))
    first, second = assert_programs_added(device, [lambda: run(first_inputs), lambda: run(second_inputs)], [None, 0])
    first, second = tensors(first), tensors(second)
    assert all(a.buffer_address() != b.buffer_address() for a, b in zip(first, second, strict=True))
    for label, outputs, goldens in (("cache miss", first, expected[0]), ("cache hit", second, expected[1])):
        for index, (output, golden) in enumerate(zip(outputs, goldens, strict=True)):
            assert_close(golden, ttnn.to_torch(output), tolerance, name=f"{label} output {index}")


def assert_default_config_matches(
    device: ttnn.MeshDevice,
    run_default: Callable[[], ttnn.Tensor | Sequence[ttnn.Tensor]],
    run_explicit: Callable[[], ttnn.Tensor | Sequence[ttnn.Tensor]],
) -> None:
    """Omitting the compute config is the documented default: the same program and bit-identical outputs."""
    default, explicit = assert_programs_added(device, [run_default, run_explicit], [None, 0])
    for index, (a, b) in enumerate(zip(tensors(default), tensors(explicit), strict=True)):
        assert_bit_identical(ttnn.to_torch(a), ttnn.to_torch(b), name=f"output {index} default vs explicit")


def performance_test(test):
    """Device-time gate: a host-IOMMU runner (the real-time profiler), no watcher or LLK asserts (they change
    kernel timing). The test name ends in ``_performance``, so the functional jobs' ``-k`` filter also skips it."""
    test = skip_with_watcher("Watcher perturbs kernel timing; perf checks are not meaningful with it enabled.")(test)
    test = skip_with_llk_assert("No need to verify LLK asserts for performance tests.")(test)
    return pytest.mark.requires_host_iommu(test)


def assert_device_time(
    device: ttnn.MeshDevice,
    call: Callable[[], ttnn.Tensor | Sequence[ttnn.Tensor]],
    *,
    reference_ns: float,
    margin: float,
    label: str,
    utilization: Callable[[ttnn.Tensor | Sequence[ttnn.Tensor], float], perf_model.KdaPerformance] | None = None,
):
    """The program's device time lies within ``reference_ns * (1 ± margin)``. Faster fails too: re-measure the
    reference, or the headroom hides the next regression. Prints the work-model utilization when given (perf
    jobs run at LOGURU_LEVEL=ERROR). Returns the call's output."""
    require_realtime_profiler(label)
    output, record = profile_realtime_program(device, call)
    duration_ns = record["duration_ns"]
    report = f"{label}: {duration_ns:.0f} ns (reference {reference_ns:.0f} ns, ±{margin:.0%})"
    if utilization is not None:
        performance = utilization(output, duration_ns)
        report += (
            f"; work={performance.work}, ideal_ns={performance.ideal_ns:.2f} "
            f"(fpu {performance.ideal_fpu_ns:.2f}, dram {performance.ideal_dram_ns:.2f}), "
            f"utilization={performance.utilization_pct:.2f}% "
            f"(fpu {performance.fpu_utilization_pct:.2f}%, dram {performance.dram_utilization_pct:.2f}%)"
        )
    print(report)
    lower, upper = reference_ns * (1 - margin), reference_ns * (1 + margin)
    assert lower <= duration_ns <= upper, f"{report} outside [{lower:.0f}, {upper:.0f}] ns"
    return output
