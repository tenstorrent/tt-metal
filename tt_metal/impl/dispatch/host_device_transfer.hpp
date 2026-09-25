// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <span>
#include <vector>

#include <tt-metalium/buffer.hpp>
#include <tt_stl/assert.hpp>
#include <tt_stl/span.hpp>

namespace tt::tt_metal {
class CoreRangeSet;
class IDevice;
}  // namespace tt::tt_metal

// Host <-> device transfers for a single device that bypass the command queue.
namespace tt::tt_metal::slow_dispatch {

// Copies `host_buffer` into `buffer`.
void WriteToBuffer(Buffer& buffer, ttsl::Span<const uint8_t> host_buffer);
template <typename DType>
void WriteToBuffer(Buffer& buffer, const std::vector<DType>& host_buffer) {
    WriteToBuffer(
        buffer,
        ttsl::Span<const uint8_t>(
            reinterpret_cast<const uint8_t*>(host_buffer.data()), host_buffer.size() * sizeof(DType)));
}
// Copies `host_buffer` into the shards of a sharded `buffer` that live on `logical_core_filter` cores only.
void WriteToBuffer(Buffer& buffer, ttsl::Span<const uint8_t> host_buffer, const CoreRangeSet& logical_core_filter);

// Copies the whole `buffer` into `host_buffer`, which must hold `buffer.size()` bytes.
void ReadFromBuffer(Buffer& buffer, uint8_t* host_buffer);
template <typename DType>
void ReadFromBuffer(Buffer& buffer, std::vector<DType>& host_buffer) {
    auto buffer_size = buffer.size();
    TT_FATAL(buffer_size % sizeof(DType) == 0, "Buffer size is not divisible by dtype size");
    host_buffer.resize(buffer_size / sizeof(DType));
    ReadFromBuffer(buffer, reinterpret_cast<uint8_t*>(host_buffer.data()));
}

// Copies the shard stored on the `core_id`-th core of a sharded `buffer` into `host_buffer`.
void ReadShard(Buffer& buffer, uint8_t* host_buffer, const uint32_t& core_id);

// Copies `host_buffer` to `address` within DRAM channel `dram_channel`. The address must be outside the reserved DRAM
// region.
bool WriteToDeviceDRAMChannel(
    IDevice& device, int dram_channel, uint32_t address, std::span<const uint8_t> host_buffer);
bool WriteToDeviceDRAMChannel(IDevice& device, int dram_channel, uint32_t address, std::vector<uint32_t>& host_buffer);

// Copies `host_buffer.size()` bytes from `address` within DRAM channel `dram_channel` into `host_buffer`.
bool ReadFromDeviceDRAMChannel(IDevice& device, int dram_channel, uint32_t address, std::span<uint8_t> host_buffer);
// Resizes `host_buffer` to hold `size` bytes and reads them.
bool ReadFromDeviceDRAMChannel(
    IDevice& device, int dram_channel, uint32_t address, uint32_t size, std::vector<uint32_t>& host_buffer);

}  // namespace tt::tt_metal::slow_dispatch
