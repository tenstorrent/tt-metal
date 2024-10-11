// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "types.hpp"
#include "tt_metal/impl/buffers/buffer.hpp"
//==================================================
//                  BUFFER HANDLING
//==================================================

namespace tt::tt_metal{
namespace v1 {


/**
 * @brief Allocates an interleaved DRAM or L1 buffer on the device.
 *
 * @param config Configuration for the buffer.
 * @return Buffer handle to the allocated buffer.
 */
Buffer CreateBuffer(const InterleavedBufferConfig &config);

/**
 * @brief Allocates a buffer on the device.
 *
 * @param buffer The buffer to allocate.
 * @param bottom_up If true, allocates buffer from the bottom up.
 */
void AllocateBuffer(Buffer buffer, bool bottom_up);

/**
 * @brief Deallocates a buffer from the device.
 *
 * @param buffer The buffer to deallocate.
 */
void DeallocateBuffer(Buffer buffer);

/**
 * @brief Copies data from a host buffer into the specified device buffer.
 *
 * @param buffer Buffer to write data into.
 * @param host_buffer Host buffer containing data to copy.
 */
void WriteToBuffer(Buffer buffer, const std::vector<uint32_t> &host_buffer);

/**
 * @brief Copies data from a device buffer into a host buffer.
 *
 * @param buffer Buffer to read data from.
 * @param host_buffer Host buffer to copy data into.
 * @param shard_order If true, reads data in shard order.
 */
void ReadFromBuffer(Buffer buffer, std::vector<uint32_t> &host_buffer, bool shard_order = false);

/**
 * @brief Copies data from a specific shard of a device buffer into a host buffer.
 *
 * @param buffer Buffer to read data from.
 * @param host_buffer Host buffer to copy data into.
 * @param core_id ID of the core shard to read.
 */
void ReadFromShard(Buffer buffer, std::vector<uint32_t> &host_buffer, const uint32_t &core_id);


} // namespace v1
} // namespace tt::tt_metal
