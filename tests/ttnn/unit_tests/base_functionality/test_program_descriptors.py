# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import ttnn


@pytest.mark.parametrize("legacy_args", [[], [("shared.value", 3)]])
@pytest.mark.parametrize("blaze_args", [[], [("shared.value", 4)]])
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
