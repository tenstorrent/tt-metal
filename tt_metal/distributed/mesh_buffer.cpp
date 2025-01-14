
// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/distributed/mesh_buffer.hpp"
#include "tt_metal/detail/tt_metal.hpp"
#include "tt_metal/tt_stl/overloaded.hpp"

namespace tt::tt_metal::distributed {
namespace {

void validate_mesh_buffer_config(const MeshBufferConfig& config, const MeshDevice& mesh_device) {
    if (std::holds_alternative<ReplicatedBufferConfig>(config)) {
        // Nothing to validate.
        return;
    }

    const auto& sharded_config = std::get<ShardedBufferConfig>(config);

    const auto [mesh_height, mesh_width] = mesh_device.shape();
    const auto [global_buffer_height, global_buffer_width] = sharded_config.global_buffer_shape;
    const auto [shard_height, shard_width] = sharded_config.shard_shape;
    TT_FATAL(
        (global_buffer_height % mesh_height == 0) and (global_buffer_width % mesh_width == 0),
        "Global buffer shape must be aligned with the mesh shape: requested buffer shape: ({}, {}), mesh "
        "shape: ({}, {})",
        global_buffer_height,
        global_buffer_width,
        mesh_height,
        mesh_width);

    TT_FATAL(
        (global_buffer_height % shard_height == 0) and (global_buffer_width % shard_width == 0),
        "Global buffer shape must be aligned with the shard shape: requested buffer shape: ({}, {}), shard "
        "shape: ({}, {})",
        global_buffer_height,
        global_buffer_width,
        shard_height,
        shard_width);

    const auto num_shard_rows = global_buffer_height / shard_height;
    const auto num_shard_cols = global_buffer_width / shard_width;
    TT_FATAL(
        num_shard_rows == mesh_height and num_shard_cols == mesh_width,
        "The number of shards must align with the mesh shape: number of shards: ({}, {}), mesh shape: ({}, {})",
        num_shard_rows,
        num_shard_cols,
        mesh_height,
        mesh_width);
}

}  // namespace

std::shared_ptr<MeshBuffer> MeshBuffer::create(
    const MeshBufferConfig& mesh_buffer_config,
    const DeviceLocalBufferConfig& device_local_layout,
    MeshDevice* mesh_device) {
    validate_mesh_buffer_config(mesh_buffer_config, *mesh_device);

    DeviceAddr device_local_size = std::visit(
        tt::stl::overloaded{
            [](const ReplicatedBufferConfig& c) { return c.buffer_size; },
            [mesh_device](const ShardedBufferConfig& config) {
                const auto [shard_height, shard_width] = config.shard_shape;
                return config.compute_datum_size_bytes() * shard_height * shard_width;
            }},
        mesh_buffer_config);

    auto mesh_buffer = std::shared_ptr<MeshBuffer>(
        new MeshBuffer(mesh_buffer_config, device_local_layout, device_local_size, mesh_device));
    mesh_buffer->allocate();

    return mesh_buffer;
}

void MeshBuffer::allocate() {
    // TODO: use mesh allocator, when available.
    buffers_.resize(mesh_device_->num_rows());
    for (int row = 0; row < mesh_device_->num_rows(); row++) {
        buffers_[row].reserve(mesh_device_->num_cols());
        for (int col = 0; col < mesh_device_->num_cols(); col++) {
            DeviceAddr page_size = device_local_config_.page_size;
            BufferType buffer_type = device_local_config_.buffer_type;
            TensorMemoryLayout buffer_layout = device_local_config_.buffer_layout;
            std::optional<ShardSpecBuffer> shard_parameters = device_local_config_.shard_parameters;
            bool bottom_up = device_local_config_.bottom_up;

            std::shared_ptr<Buffer> buffer = Buffer::create(
                mesh_device_->get_device(row, col),
                address_,
                device_local_size_,
                page_size,
                buffer_type,
                buffer_layout,
                shard_parameters,
                bottom_up);
            tt::tt_metal::detail::AllocateBuffer(buffer.get());
            buffers_[row].push_back(std::move(buffer));
        }
    }
}

void MeshBuffer::deallocate() {
    for (auto& row_buffers : buffers_) {
        for (auto& buffer : row_buffers) {
            tt::tt_metal::detail::DeallocateBuffer(buffer.get());
        }
    }
}

std::shared_ptr<Buffer> MeshBuffer::get_device_buffer(uint32_t logical_x, uint32_t logical_y) {
    TT_FATAL(
        logical_y < mesh_device_->num_rows() and logical_x < mesh_device_->num_cols(),
        "Logical coordinates must be within the bounds of the mesh: {}, {}, mesh shape: {}, {}",
        logical_y,
        logical_x,
        mesh_device_->num_rows(),
        mesh_device_->num_cols());
    return buffers_[logical_y][logical_x];
}

DeviceAddr MeshBuffer::global_size() const {
    return std::visit(
        tt::stl::overloaded{
            [&](const ReplicatedBufferConfig& config) { return config.buffer_size; },
            [&](const ShardedBufferConfig& config) { return config.global_buffer_size; }},
        config_);
}

MeshBufferLayout MeshBuffer::global_layout() const {
    return std::holds_alternative<ReplicatedBufferConfig>(config_) ? MeshBufferLayout::REPLICATED
                                                                   : MeshBufferLayout::SHARDED;
}

const ShardedBufferConfig& MeshBuffer::global_shard_spec() const {
    TT_FATAL(
        global_layout() == MeshBufferLayout::SHARDED, "Can only query the global shard spec for a sharded MeshBuffer");
    return std::get<ShardedBufferConfig>(config_);
}

}  // namespace tt::tt_metal::distributed
