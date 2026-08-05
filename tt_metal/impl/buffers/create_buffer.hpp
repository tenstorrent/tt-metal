// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>

#include <tt-metalium/buffer.hpp>
#include <tt-metalium/sub_device_types.hpp>

namespace tt::tt_metal {

struct BufferConfig {
    IDevice* device;
    DeviceAddr size;       // Size in bytes
    DeviceAddr page_size;  // Size of unit being interleaved. For non-interleaved buffers: size == page_size
    BufferType buffer_type;
};

using InterleavedBufferConfig = BufferConfig;

// clang-format off
/**
*  Creates a pre-allocated interleaved DRAM or L1 buffer with the global allocator on device
*
*  Return value: std::shared_ptr<Buffer>
*
*  | Argument        | Description                                                       | Type                      | Valid Range | Required |
*  |-----------------|------------------------------------------------------------------ |---------------------------|-------------|----------|
*  | config          | Config for the buffer                                             | InterleavedBufferConfig   |             | Yes      |
*/
// clang-format on
std::shared_ptr<Buffer> CreateBuffer(const InterleavedBufferConfig& config);

// clang-format off
/**
*  Creates a pre-allocated interleaved DRAM or L1 buffer with the global allocator on device
*
*  Return value: std::shared_ptr<Buffer>
*
*  | Argument        | Description                                                       | Type                      | Valid Range | Required |
*  |-----------------|------------------------------------------------------------------ |---------------------------|-------------|----------|
*  | config          | Config for the buffer                                             | InterleavedBufferConfig   |             | Yes      |
*  | address         | Device address of the buffer                                      | DeviceAddr                |             | No       |
*/
// clang-format on
std::shared_ptr<Buffer> CreateBuffer(const InterleavedBufferConfig& config, DeviceAddr address);

// clang-format off
/**
*  Creates a pre-allocated interleaved DRAM or L1 buffer on device
*
*  Return value: std::shared_ptr<Buffer>
*
*  | Argument        | Description                                                       | Type                      | Valid Range | Required |
*  |-----------------|------------------------------------------------------------------ |---------------------------|-------------|----------|
*  | config          | Config for the buffer                                             | InterleavedBufferConfig   |             | Yes      |
*  | sub_device_id   | The sub-device id to allocate on                                  | SubDeviceId               |             | No       |
*/
// clang-format on
std::shared_ptr<Buffer> CreateBuffer(const InterleavedBufferConfig& config, SubDeviceId sub_device_id);

// clang-format off
/**
*  Creates a pre-allocated sharded DRAM or L1 buffer with the global allocator on device
*
*  Return value: std::shared_ptr<Buffer>
*
*  | Argument        | Description                                                       | Type                      | Valid Range | Required |
*  |-----------------|------------------------------------------------------------------ |---------------------------|-------------|----------|
*  | config          | Config for the buffer                                             | ShardedBufferConfig       |             | Yes      |
*/
// clang-format on
std::shared_ptr<Buffer> CreateBuffer(const ShardedBufferConfig& config);

// clang-format off
/**
*  Creates a pre-allocated sharded DRAM or L1 buffer with the global allocator on device
*
*  Return value: std::shared_ptr<Buffer>
*
*  | Argument        | Description                                                       | Type                      | Valid Range | Required |
*  |-----------------|------------------------------------------------------------------ |---------------------------|-------------|----------|
*  | config          | Config for the buffer                                             | ShardedBufferConfig       |             | Yes      |
*  | address         | Device address of the buffer                                      | DeviceAddr                |             | No       |
*/
// clang-format on
std::shared_ptr<Buffer> CreateBuffer(const ShardedBufferConfig& config, DeviceAddr address);

// clang-format off
/**
*  Creates a pre-allocated sharded DRAM or L1 buffer on device
*
*  Return value: std::shared_ptr<Buffer>
*
*  | Argument        | Description                                                       | Type                      | Valid Range | Required |
*  |-----------------|------------------------------------------------------------------ |---------------------------|-------------|----------|
*  | config          | Config for the buffer                                             | ShardedBufferConfig       |             | Yes      |
*  | sub_device_id   | The sub-device id to allocate on                                  |                           |             | No       |
*/
// clang-format on
std::shared_ptr<Buffer> CreateBuffer(const ShardedBufferConfig& config, SubDeviceId sub_device_id);

}  // namespace tt::tt_metal
