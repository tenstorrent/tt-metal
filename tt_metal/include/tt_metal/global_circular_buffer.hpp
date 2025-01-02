// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tt_metal/impl/buffers/global_circular_buffer.hpp"
#include "tt_metal/types.hpp"
//==================================================
//        GLOBAL CIRCULAR BUFFER FUNCTIONS
//==================================================

namespace tt::tt_metal {
namespace v1 {

namespace experimental {

/**
 * @brief Allocates a global circular buffer in L1 on the device.
 *
 * @param device The device to create the global circular buffer on.
 * @param sender_receiver_core_mapping The mapping of remote sender to remote receiver cores for the circular buffer.
 * @param size Size of the global circular buffer per core in bytes.
 * @param buffer_type Buffer type to store the global circular buffer. Can only be an L1 buffer type.
 * @param sub_device_ids Sub-device IDs to wait on before writing the global circular buffer config to device. Defaults
 * to waiting on all sub-devices.
 * @return The allocated global circular buffer.
 */
GlobalCircularBuffer CreateGlobalCircularBuffer(
    Device* device,
    const std::unordered_map<CoreCoord, CoreRangeSet>& sender_receiver_core_mapping,
    uint32_t size,
    BufferType buffer_type = BufferType::L1,
    tt::stl::Span<const SubDeviceId> sub_device_ids = {});

}  // namespace experimental

}  // namespace v1
}  // namespace tt::tt_metal
