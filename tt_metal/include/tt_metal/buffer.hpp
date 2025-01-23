// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <buffer.hpp>
#include <types.hpp>
//==================================================
//                  BUFFER HANDLING
//==================================================

namespace tt::tt_metal {
namespace v1 {

/**
 * @brief Allocates an interleaved DRAM or L1 buffer on the device.
 *
 * @param config Configuration for the buffer.
 * @return Buffer handle to the allocated buffer.
 */
BufferHandle CreateBuffer(InterleavedBufferConfig config);

/**
 * @brief Deallocates a buffer from the device.
 *
 * @param buffer The buffer to deallocate.
 */
void DeallocateBuffer(const BufferHandle& buffer);

/**
 * @brief Retrieves the ID of the specified buffer.
 *
 * @param buffer The buffer to get the ID from.
 * @return The unique ID of the buffer.
 */
std::size_t GetId(const BufferHandle& buffer);

/**
 * @brief Copies data from a host buffer into the specified device buffer.
 *
 * @param buffer Buffer to write data into.
 * @param host_buffer Host buffer containing data to copy.
 */
void WriteToBuffer(const BufferHandle& buffer, stl::Span<const std::byte> host_buffer);

/**
 * @brief Copies data from a device buffer into a host buffer.
 *
 * @param buffer Buffer to read data from.
 * @param host_buffer Host buffer to copy data into.
 * @param shard_order If true, reads data in shard order.
 */
void ReadFromBuffer(const BufferHandle& buffer, stl::Span<std::byte> host_buffer, bool shard_order = false);

/**
 * @brief Copies data from a specific shard of a device buffer into a host buffer.
 *
 * @param buffer Buffer to read data from.
 * @param host_buffer Host buffer to copy data into.
 * @param core_id ID of the core shard to read.
 */
void ReadFromShard(const BufferHandle& buffer, stl::Span<std::byte> host_buffer, std::uint32_t core_id);

}  // namespace v1
}  // namespace tt::tt_metal
