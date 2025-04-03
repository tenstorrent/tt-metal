// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <functional>
#include <memory>
#include <variant>

namespace tt {
namespace tt_metal {
class Buffer;
class CommandQueue;
}  // namespace tt_metal
}  // namespace tt

namespace tt::tt_metal {

// Note: LightMetalCompare functions could have been inside host_api.hpp / command_queue.cpp but seems better
// to not make as visible, since these are APIs used at light-metal capture time for verification purposes.

// clang-format off
/**
 * Reads a buffer from the device and captures return data as golden inside Light Metal Binary, and optionally returns to user.
 * When replaying Light Metal Binary, buffer is read and data is compared to the capture-time golden data.
 *
 * Return value: void
 *
 * | Argument       | Description                                                                       | Type                                | Valid Range                            | Required |
 * |----------------|-----------------------------------------------------------------------------------|-------------------------------------|----------------------------------------|----------|
 * | cq             | The command queue object which dispatches the command to the hardware             | CommandQueue &                      |                                        | Yes      |
 * | buffer         | The device buffer we are reading from                                             | Buffer & or std::shared_ptr<Buffer> |                                        | Yes      |
 * | dst            | The memory where the result will be stored, if provided                           | void*                               |                                        | No       |
 */
// clang-format on
void LightMetalCompareToCapture(
    CommandQueue& cq,
    const std::variant<std::reference_wrapper<Buffer>, std::shared_ptr<Buffer>>& buffer,
    void* dst = nullptr);

// clang-format off
/**
 * Accepts user-supplied golden data, stored inside Light Metal Binary.
 * When replaying Light Metal Binary, buffer is read and data is compared to the user-supplied golden data.
 *
 * Return value: void
 *
 * | Argument       | Description                                                                       | Type                                | Valid Range                            | Required |
 * |----------------|-----------------------------------------------------------------------------------|-------------------------------------|----------------------------------------|----------|
 * | cq             | The command queue object which dispatches the command to the hardware             | CommandQueue &                      |                                        | Yes      |
 * | buffer         | The device buffer we are reading from                                             | Buffer & or std::shared_ptr<Buffer> |                                        | Yes      |
 * | golden_data    | User supplied expected/golden data for buffer                                     | void*                               |                                        | Yes      |
 */
// clang-format on

void LightMetalCompareToGolden(
    CommandQueue& cq,
    const std::variant<std::reference_wrapper<Buffer>, std::shared_ptr<Buffer>>& buffer,
    void* golden_data);

}  // namespace tt::tt_metal
