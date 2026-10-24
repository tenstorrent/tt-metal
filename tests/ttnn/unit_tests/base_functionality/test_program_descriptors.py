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


@pytest.mark.parametrize(
    "opt_level",
    [
        ttnn.KernelDescriptor.BuildOptLevel.O0,
        ttnn.KernelDescriptor.BuildOptLevel.O1,
        ttnn.KernelDescriptor.BuildOptLevel.O2,
        ttnn.KernelDescriptor.BuildOptLevel.O3,
        ttnn.KernelDescriptor.BuildOptLevel.Os,
        ttnn.KernelDescriptor.BuildOptLevel.Ofast,
        ttnn.KernelDescriptor.BuildOptLevel.Oz,
    ],
)
def test_kernel_descriptor_build_opt_level_round_trip(opt_level):
    core = ttnn.CoreCoord(0, 0)
    kernel = ttnn.KernelDescriptor(
        kernel_source="kernel.cpp",
        core_ranges=ttnn.CoreRangeSet([ttnn.CoreRange(core, core)]),
        config=ttnn.DataMovementConfigDescriptor(),
        opt_level=opt_level,
    )

    assert kernel.opt_level == opt_level


def test_kernel_descriptor_build_opt_level_is_optional():
    kernel = ttnn.KernelDescriptor()
    assert kernel.opt_level is None

    kernel.opt_level = ttnn.KernelDescriptor.BuildOptLevel.Os
    assert kernel.opt_level == ttnn.KernelDescriptor.BuildOptLevel.Os
