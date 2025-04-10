// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/global_circular_buffer_impl.hpp>
//==================================================
//        GLOBAL CIRCULAR BUFFER FUNCTIONS
//==================================================

namespace tt::tt_metal {
namespace experimental {

/**
 * @brief Allocates a global circular buffer in L1 on the device.
 *
 * @param device The device to create the global circular buffer on.
 * @param sender_receiver_core_mapping The mapping of remote sender to remote receiver cores for the circular buffer.
 * @param size Size of the global circular buffer per core in bytes.
 * @param buffer_type Buffer type to store the global circular buffer. Can only be an L1 buffer type.\
 * @return The allocated global circular buffer.
 */
GlobalCircularBuffer CreateGlobalCircularBuffer(
    IDevice* device,
    const std::vector<std::pair<CoreCoord, CoreRangeSet>>& sender_receiver_core_mapping,
    uint32_t size,
    BufferType buffer_type = BufferType::L1);

/**
 * @brief Creates a Circular Buffer in L1 memory of specified cores using the address space of the
 * global circular bufferand adds it to the program.
 *
 * @param program The program to which the buffer will be added.
 * @param core_spec Specifies the cores where the circular buffer will be configured.
 * @param config Configuration for the circular buffer.
 * @param global_circular_buffer Global circular buffer to use the address space and configuration of.
 * @return CBHandle representing the Circular Buffer ID.
 */
CBHandle CreateCircularBuffer(
    Program& program,
    const std::variant<CoreCoord, CoreRange, CoreRangeSet>& core_spec,
    const CircularBufferConfig& config,
    const GlobalCircularBuffer& global_circular_buffer);

/**
 * @brief Updates the address of a dynamic global circular buffer.
 *
 * @param program The program containing the circular buffer.
 * @param cb_handle Handle of the circular buffer.
 * @param buffer Dynamically allocated global L1 buffer that shares address space with the circular buffer.
 */
void UpdateDynamicCircularBufferAddress(
    Program& program, CBHandle cb_handle, const GlobalCircularBuffer& global_circular_buffer);

}  // namespace experimental

}  // namespace tt::tt_metal
