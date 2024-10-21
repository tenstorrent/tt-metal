// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tt_metal/impl/buffers/buffer.hpp"
#include "tt_metal/tt_stl/span.hpp"
#include "types.hpp"

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
Buffer CreateBuffer(InterleavedBufferConfig config);

/**
 * @brief Allocates a sharded DRAM or L1 buffer on the device.
 *
 * @param config Configuration for the buffer.
 * @return Buffer handle to the allocated buffer.
 */
Buffer CreateBuffer(ShardedBufferConfig config);

/**
 * @brief Allocates a buffer on the device.
 *
 * @param buffer The buffer to allocate.
 * @param config Whether to allocate buffer top down or bottom up.
 */
void AllocateBuffer(Buffer buffer, AllocationOrder config);

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
void WriteToBuffer(Buffer buffer, stl::Span<const std::byte> host_buffer);

/**
 * @brief Copies data from a device buffer into a host buffer.
 *
 * @param buffer Buffer to read data from.
 * @param host_buffer Host buffer to copy data into.
 * @param config Whether to read shards in mapped or unmapped order
 * @return Number of bytes written to host buffer
 */
std::size_t ReadFromBuffer(Buffer buffer, stl::Span<std::byte> host_buffer, ShardOrder config);

/**
 * @brief Copies data from a specific shard of a device buffer into a host buffer.
 *
 * @param buffer Buffer to read data from.
 * @param host_buffer Host buffer to copy data into.
 * @param core_id ID of the core shard to read.
 * @return Number of bytes written to host buffer
 */
std::size_t ReadFromShard(Buffer buffer, stl::Span<std::byte> host_buffer, std::uint32_t core_id);

}  // namespace v1
}  // namespace tt::tt_metal
