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


@pytest.mark.parametrize("processor", [None, 0, 1, 2])
def test_kernel_descriptor_copy_preserves_physical_risc_and_named_args(processor):
    core = ttnn.CoreCoord(0, 0)
    original = ttnn.KernelDescriptor(
        kernel_source="kernel.cpp",
        core_ranges=ttnn.CoreRangeSet([ttnn.CoreRange(core, core)]),
        config=ttnn.ComputeConfigDescriptor(),
        named_compile_time_args=[("legacy.value", 3)],
        blaze_named_compile_time_args=[("typed.value", 4)],
    )
    original.compute_processor = processor
    original.runtime_args_owner = 0 if processor in (1, 2) else None

    copied = ttnn.KernelDescriptor(original)
    assert copied.compute_processor == processor
    assert copied.runtime_args_owner == original.runtime_args_owner
    assert copied.named_compile_time_args == [("legacy.value", 3)]
    assert copied.blaze_named_compile_time_args == [("typed.value", 4)]

    copied.compute_processor = 1 if processor != 1 else 2
    copied.blaze_named_compile_time_args = [("typed.value", 5)]
    assert original.compute_processor == processor
    assert original.blaze_named_compile_time_args == [("typed.value", 4)]
