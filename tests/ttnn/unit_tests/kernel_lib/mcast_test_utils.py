# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Inspect serialized arguments through the descriptor attachment contract."""
import ttnn


def attach_for_inspection(family, cores, noc=ttnn.NOC.NOC_0, semaphores=()):
    kernel = ttnn.KernelDescriptor(
        kernel_source="inspection-only.cpp",
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=cores,
        config=ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_0, noc=noc),
    )
    descriptor = ttnn.ProgramDescriptor(semaphores=list(semaphores))
    family.attach(descriptor, "mcast", [kernel])
    return descriptor, kernel
