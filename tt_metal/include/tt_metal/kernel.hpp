// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tt_metal/tt_stl/span.hpp"
#include "types.hpp"

//==================================================
//                 KERNEL EXECUTION
//==================================================

namespace tt::tt_metal {
namespace v1 {

/**
 * @brief Sets runtime arguments for a kernel.
 *
 * @param kernel KernelHandle representing the kernel ID.
 * @param core_spec Specifies the cores where the runtime arguments will be set.
 * @param runtime_args The runtime arguments to be set.
 */
void SetRuntimeArgs(KernelHandle kernel, const CoreRangeSet &core_spec, const RuntimeArgs &runtime_args);

/**
 * @brief Sets multiple runtime arguments of a kernel at once.
 *
 * @param kernel KernelHandle representing the kernel ID.
 * @param core_spec Vector of core coordinates where the runtime arguments will be set.
 * @param runtime_args The runtime arguments to be set.
 */
void SetRuntimeArgs(KernelHandle kernel, stl::Span<const CoreCoord> core_spec, const RuntimeArgs &runtime_args);

/**
 * @brief Sets common runtime arguments for a kernel, shared by all cores.
 *
 * @param kernel KernelHandle representing the kernel ID.
 * @param runtime_args The runtime arguments to be set.
 */
void SetCommonRuntimeArgs(KernelHandle kernel, const RuntimeArgs &runtime_args);

/**
 * @brief Gets the runtime arguments for a kernel.
 *
 * @param kernel KernelHandle representing the kernel ID.
 * @param logical_core The logical core coordinate.
 * @return Reference to RuntimeArgsData.
 */
RuntimeArgsData &GetRuntimeArgs(KernelHandle kernel, CoreCoord logical_core);

}  // namespace v1
}  // namespace tt::tt_metal
