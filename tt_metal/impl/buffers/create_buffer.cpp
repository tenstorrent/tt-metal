// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "create_buffer.hpp"

namespace tt::tt_metal {

std::shared_ptr<Buffer> CreateBuffer(const InterleavedBufferConfig& config) {
    return Buffer::create(config.device, config.size, config.page_size, config.buffer_type);
}

std::shared_ptr<Buffer> CreateBuffer(const InterleavedBufferConfig& config, DeviceAddr address) {
    return Buffer::create(config.device, address, config.size, config.page_size, config.buffer_type);
}

std::shared_ptr<Buffer> CreateBuffer(const InterleavedBufferConfig& config, SubDeviceId sub_device_id) {
    return Buffer::create(
        config.device, config.size, config.page_size, config.buffer_type, std::nullopt, std::nullopt, sub_device_id);
}

std::shared_ptr<Buffer> CreateBuffer(const ShardedBufferConfig& config) {
    return Buffer::create(
        config.device,
        config.size,
        config.page_size,
        config.buffer_type,
        BufferShardingArgs(config.shard_parameters, config.buffer_layout));
}

std::shared_ptr<Buffer> CreateBuffer(const ShardedBufferConfig& config, DeviceAddr address) {
    return Buffer::create(
        config.device,
        address,
        config.size,
        config.page_size,
        config.buffer_type,
        BufferShardingArgs(config.shard_parameters, config.buffer_layout));
}

std::shared_ptr<Buffer> CreateBuffer(const ShardedBufferConfig& config, SubDeviceId sub_device_id) {
    return Buffer::create(
        config.device,
        config.size,
        config.page_size,
        config.buffer_type,
        BufferShardingArgs(config.shard_parameters, config.buffer_layout),
        std::nullopt,
        sub_device_id);
}

}  // namespace tt::tt_metal
