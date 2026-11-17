# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import ttnn


@pytest.mark.parametrize("legacy_args", [[], [("legacy.value", 3)]])
@pytest.mark.parametrize("blaze_args", [[], [("typed.value", 4)]])
def test_kernel_descriptor_named_compile_time_args(legacy_args, blaze_args):
    core = ttnn.CoreCoord(0, 0)
    kernel = ttnn.KernelDescriptor(
        kernel_source="kernel.cpp",
        core_ranges=ttnn.CoreRangeSet([ttnn.CoreRange(core, core)]),
        config=ttnn.DataMovementConfigDescriptor(),
        named_compile_time_args=legacy_args,
        blaze_named_compile_time_args=blaze_args,
    )

    assert kernel.named_compile_time_args == legacy_args
    assert kernel.blaze_named_compile_time_args == blaze_args


def test_kernel_descriptor_legacy_positional_constructor():
    core = ttnn.CoreCoord(0, 0)
    kernel = ttnn.KernelDescriptor(
        "kernel.cpp",
        ttnn.KernelDescriptor.SourceType.FILE_PATH,
        ttnn.CoreRangeSet([ttnn.CoreRange(core, core)]),
        [7],
        [("legacy.value", 3)],
        [],
        [],
        [9],
        None,
        ttnn.DataMovementConfigDescriptor(),
        [],
    )

    assert list(kernel.compile_time_args) == [7]
    assert list(kernel.common_runtime_args) == [9]
    assert kernel.named_compile_time_args == [("legacy.value", 3)]
    assert kernel.blaze_named_compile_time_args == []


def test_program_performance_descriptor_defaults():
    performance = ttnn.ProgramPerformanceDescriptor()

    assert performance.dram_bytes_read == 0
    assert performance.dram_bytes_written == 0
    assert performance.flops == 0
    assert performance.math_fidelity is None
    assert ttnn.ProgramDescriptor().performance is None


def test_program_descriptor_performance_round_trip():
    program = ttnn.ProgramDescriptor()
    program.performance = ttnn.ProgramPerformanceDescriptor(
        dram_bytes_read=7 << 30, dram_bytes_written=4096, flops=3 << 40, math_fidelity=ttnn.MathFidelity.LoFi
    )

    assert program.performance.dram_bytes_read == 7 << 30
    assert program.performance.dram_bytes_written == 4096
    assert program.performance.flops == 3 << 40
    assert program.performance.math_fidelity == ttnn.MathFidelity.LoFi

    program.performance = None
    assert program.performance is None


def _program_on_core(x, performance):
    core = ttnn.CoreCoord(x, 0)
    kernel = ttnn.KernelDescriptor(
        kernel_source="kernel.cpp",
        core_ranges=ttnn.CoreRangeSet([ttnn.CoreRange(core, core)]),
        config=ttnn.DataMovementConfigDescriptor(),
    )
    program = ttnn.ProgramDescriptor(kernels=[kernel])
    program.performance = performance
    return program


def test_merge_program_descriptors_sums_declared_performance():
    first = _program_on_core(
        0, ttnn.ProgramPerformanceDescriptor(dram_bytes_read=100, dram_bytes_written=10, flops=1000)
    )
    second = _program_on_core(1, ttnn.ProgramPerformanceDescriptor(dram_bytes_read=200, flops=500))
    undeclared = _program_on_core(2, None)

    merged = ttnn.merge_program_descriptors([first, undeclared, second])

    assert merged.performance.dram_bytes_read == 300
    assert merged.performance.dram_bytes_written == 10
    assert merged.performance.flops == 1500
    assert ttnn.merge_program_descriptors([undeclared, _program_on_core(3, None)]).performance is None
