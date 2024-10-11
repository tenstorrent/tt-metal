// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string_view>
#include "types.hpp"

#include "tt_metal/impl/kernels/kernel_types.hpp"
#include "tt_metal/impl/buffers/circular_buffer_types.hpp"

//==================================================
//                  INTERNAL PROGRAM MANAGEMENT
//==================================================

namespace tt::tt_metal{
namespace v1 {

//==================================================
//                 INTERNAL PROGRAM FUNCTIONS
//==================================================

/**
 * @brief Launches a program on the device.
 *
 * @param device The device on which to launch the program.
 * @param program The program to execute.
 * @param wait_until_cores_done If true, waits until cores have completed execution.
 * @param force_slow_dispatch If true, forces slow dispatch mode.
 */
void LaunchProgram(Device device, Program program, bool wait_until_cores_done = true, bool force_slow_dispatch = false);

} // namespace v1
} // namespace tt::tt_metal
