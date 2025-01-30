// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "host_api_capture_helpers.hpp"
#include <tt-metalium/buffer.hpp>

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
inline void LightMetalCompareToCapture(
    CommandQueue& cq,
    const std::variant<std::reference_wrapper<Buffer>, std::shared_ptr<Buffer>>& buffer,
    void* dst = nullptr) {
    TRACE_FUNCTION_ENTRY();

    // If dst ptr is not provided, just allocate temp space for rd return capture/usage.
    std::vector<uint32_t> rd_data_tmp;
    if (!dst) {
        size_t buffer_size = std::holds_alternative<std::reference_wrapper<Buffer>>(buffer)
                                 ? std::get<std::reference_wrapper<Buffer>>(buffer).get().size()
                                 : std::get<std::shared_ptr<Buffer>>(buffer)->size();
        rd_data_tmp.resize(buffer_size / sizeof(uint32_t));
        dst = rd_data_tmp.data();
    }

    EnqueueReadBuffer(cq, buffer, dst, true);  // Blocking read to get golden value.
    TRACE_FUNCTION_CALL(CaptureLightMetalCompare, cq, buffer, dst, false);
}

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

inline void LightMetalCompareToGolden(
    CommandQueue& cq,
    const std::variant<std::reference_wrapper<Buffer>, std::shared_ptr<Buffer>>& buffer,
    void* golden_data) {
    TRACE_FUNCTION_ENTRY();
    TRACE_FUNCTION_CALL(CaptureLightMetalCompare, cq, buffer, golden_data, true);
}

}  // namespace tt::tt_metal
