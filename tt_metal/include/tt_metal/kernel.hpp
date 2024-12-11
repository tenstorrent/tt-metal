// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "types.hpp"

//==================================================
//                 KERNEL EXECUTION
//==================================================

namespace tt::tt_metal {
namespace v1 {

/**
 * @brief Sets runtime arguments for a kernel.
 *
 * @param program The program containing the kernel.
 * @param kernel KernelHandle representing the kernel ID.
 * @param core_spec Specifies the cores where the runtime arguments will be set.
 * @param runtime_args The runtime arguments to be set.
 */
void SetRuntimeArgs(
    ProgramHandle& program, KernelHandle kernel, const CoreRangeSet& core_spec, RuntimeArgs runtime_args);

/**
 * @brief Sets common runtime arguments for a kernel, shared by all cores.
 *
 * @param program The program containing the kernel.
 * @param kernel KernelHandle representing the kernel ID.
 * @param runtime_args The runtime arguments to be set.
 */
void SetCommonRuntimeArgs(ProgramHandle& program, KernelHandle kernel, RuntimeArgs runtime_args);

/**
 * @brief Gets the runtime arguments for a kernel.
 *
 * @param program The program containing the kernel.
 * @param kernel KernelHandle representing the kernel ID.
 * @param logical_core The logical core coordinate.
 * @return Span of runtime arguments.
 */
RuntimeArgs GetRuntimeArgs(ProgramHandle& program, KernelHandle kernel, CoreCoord logical_core);

}  // namespace v1
}  // namespace tt::tt_metal
