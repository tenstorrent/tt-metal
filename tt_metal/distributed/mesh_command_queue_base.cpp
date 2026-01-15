// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "mesh_command_queue_base.hpp"

#include <mesh_device.hpp>
#include <mesh_event.hpp>
#include <api/tt-metalium/experimental/pinned_memory.hpp>
#include <optional>

#include "buffer.hpp"
#include "mesh_coord.hpp"
#include "tt_metal/distributed/mesh_workload_utils.hpp"
#include "tt_metal/impl/buffers/dispatch.hpp"
#include "tt_metal/impl/program/dispatch.hpp"
#include "tt_metal/impl/trace/dispatch.hpp"
#include "tt_metal/impl/dispatch/dispatch_query_manager.hpp"
#include "tt_metal/common/thread_pool.hpp"
#include "tt_cluster.hpp"
#include "dispatch/dispatch_settings.hpp"

namespace tt::tt_metal::distributed {

void MeshCommandQueueBase::write_sharded_buffer(const MeshBuffer& buffer, const void* src) {
    auto global_buffer_shape = buffer.global_shard_spec().global_buffer_shape;

    auto shard_shape = buffer.physical_shard_shape();
    auto datum_size_bytes = buffer.datum_size_bytes();

    auto stride_size_bytes = datum_size_bytes * global_buffer_shape.width();
    auto single_read_size = datum_size_bytes * shard_shape.width();
    auto total_read_size_per_shard = single_read_size * shard_shape.height();

    auto num_shards_x = global_buffer_shape.width() / shard_shape.width();
    auto num_shards_y = global_buffer_shape.height() / shard_shape.height();

    uint32_t num_devices_x = buffer.device()->num_cols();
    uint32_t num_devices_y = buffer.device()->num_rows();

    uint32_t device_x = 0;
    uint32_t device_y = 0;
    std::vector<uint32_t> shard_data = std::vector<uint32_t>(total_read_size_per_shard / sizeof(uint32_t), 0);
    const auto& [height_replicated, width_replicated] = buffer.replicated_dims();
    for (std::size_t shard_y = 0; shard_y < num_shards_y; shard_y++) {
        for (std::size_t shard_x = 0; shard_x < num_shards_x; shard_x++) {
            auto read_offset = (shard_x * single_read_size) + (shard_y * stride_size_bytes * shard_shape.height());
            uint32_t size_to_read = total_read_size_per_shard;
            uint32_t local_offset = 0;
            while (size_to_read) {
                std::memcpy(
                    shard_data.data() + (local_offset * (single_read_size / sizeof(uint32_t))),
                    (uint8_t*)(src) + read_offset + (local_offset * stride_size_bytes),
                    single_read_size);
                size_to_read -= single_read_size;
                local_offset++;
            }

            if (height_replicated and width_replicated) {
                for (std::size_t replicated_device_x = 0; replicated_device_x < num_devices_x; replicated_device_x++) {
                    for (std::size_t replicated_device_y = 0; replicated_device_y < num_devices_y;
                         replicated_device_y++) {
                        this->write_shard_to_device(
                            buffer,
                            MeshCoordinate(replicated_device_y, replicated_device_x),
                            shard_data.data(),
                            /*region=*/std::nullopt);
                    }
                }
            } else if (height_replicated or width_replicated) {
                if (buffer.global_shard_spec().shard_orientation == ShardOrientation::ROW_MAJOR) {
                    for (auto replicated_device_y = 0; replicated_device_y < num_devices_y; replicated_device_y++) {
                        this->write_shard_to_device(
                            buffer,
                            MeshCoordinate(replicated_device_y, device_x),
                            shard_data.data(),
                            /*region=*/std::nullopt);
                    }
                    device_x++;
                } else {
                    for (auto replicated_device_x = 0; replicated_device_x < num_devices_x; replicated_device_x++) {
                        this->write_shard_to_device(
                            buffer,
                            MeshCoordinate(device_y, replicated_device_x),
                            shard_data.data(),
                            /*region=*/std::nullopt);
                    }
                    device_y++;
                }
            } else {
                this->write_shard_to_device(
                    buffer, MeshCoordinate(device_y, device_x), shard_data.data(), /*region=*/std::nullopt);
                if (buffer.global_shard_spec().shard_orientation == ShardOrientation::ROW_MAJOR) {
                    if (++device_x == num_devices_x) {
                        device_x = 0;
                        ++device_y;
                    }
                } else {
                    if (++device_y == num_devices_y) {
                        device_y = 0;
                        ++device_x;
                    }
                }
            }
        }
    }
}

void MeshCommandQueueBase::read_sharded_buffer(MeshBuffer& buffer, void* dst) {
    const auto& [height_replicated, width_replicated] = buffer.replicated_dims();
    TT_FATAL(
        not(height_replicated or width_replicated), "Cannot read a MeshBuffer that is replicated along any dimension.");
    auto global_buffer_shape = buffer.global_shard_spec().global_buffer_shape;
    auto shard_shape = buffer.physical_shard_shape();
    auto datum_size_bytes = buffer.datum_size_bytes();

    const auto stride_size_bytes = datum_size_bytes * global_buffer_shape.width();
    const auto single_write_size = datum_size_bytes * shard_shape.width();
    const uint64_t total_write_size_per_shard = single_write_size * shard_shape.height();
    auto num_shards_x = global_buffer_shape.width() / shard_shape.width();
    auto num_shards_y = global_buffer_shape.height() / shard_shape.height();
    uint32_t num_devices_x = buffer.device()->num_cols();
    uint32_t num_devices_y = buffer.device()->num_rows();

    uint32_t device_x = 0;
    uint32_t device_y = 0;

    std::vector<uint32_t> shard_data = std::vector<uint32_t>(total_write_size_per_shard / sizeof(uint32_t), 0);
    for (std::size_t shard_y = 0; shard_y < num_shards_y; shard_y++) {
        for (std::size_t shard_x = 0; shard_x < num_shards_x; shard_x++) {
            std::unordered_map<IDevice*, uint32_t> num_txns_per_device = {};
            this->read_shard_from_device(
                buffer,
                MeshCoordinate(device_y, device_x),
                shard_data.data(),
                /*pinned_memory=*/nullptr,
                /*region=*/std::nullopt,
                num_txns_per_device);
            this->submit_memcpy_request(num_txns_per_device, true);
            uint64_t write_offset =
                (shard_x * single_write_size) + (shard_y * stride_size_bytes * shard_shape.height());
            uint64_t size_to_write = total_write_size_per_shard;
            uint32_t local_offset = 0;
            while (size_to_write) {
                std::memcpy(
                    (uint8_t*)(dst) + write_offset + (local_offset * stride_size_bytes),
                    shard_data.data() + (local_offset * (single_write_size / sizeof(uint32_t))),
                    single_write_size);
                local_offset++;
                size_to_write -= single_write_size;
            }
            if (buffer.global_shard_spec().shard_orientation == ShardOrientation::ROW_MAJOR) {
                if (++device_x == num_devices_x) {
                    device_x = 0;
                    ++device_y;
                }
            } else {
                if (++device_y == num_devices_y) {
                    device_y = 0;
                    ++device_x;
                }
            }
        }
    }
}

void MeshCommandQueueBase::enqueue_write_shard_to_sub_grid(
    const MeshBuffer& buffer,
    const void* host_data,
    const MeshCoordinateRange& device_range,
    bool blocking,
    std::optional<BufferRegion> region) {
    auto lock = lock_api_function_();
    if (buffer.global_layout() == MeshBufferLayout::REPLICATED) {
        // Multi-Threaded writes supported for Replicated buffers.
        // Currently not supported when doing TT-Mesh Native sharding, since we
        // rely on TTNN to perform sharding and call enqueue_write_shards
        auto dispatch_lambda = [this, &buffer, host_data, &region](const MeshCoordinate& coord) {
            this->write_shard_to_device(buffer, coord, host_data, region);
        };
        for (const auto& coord : device_range) {
            if (mesh_device_->is_local(coord)) {
                dispatch_thread_pool_->enqueue(
                    [&dispatch_lambda, coord]() { dispatch_lambda(coord); }, mesh_device_->get_device(coord)->id());
            }
        }
        dispatch_thread_pool_->wait();
    } else {
        this->write_sharded_buffer(buffer, host_data);
    }

    if (blocking) {
        this->finish_nolock();
    }
}

void MeshCommandQueueBase::enqueue_write_mesh_buffer(
    const std::shared_ptr<MeshBuffer>& buffer, const void* host_data, bool blocking) {
    MeshCoordinateRange mesh_device_extent(buffer->device()->shape());
    this->enqueue_write_shard_to_sub_grid(*buffer, host_data, mesh_device_extent, blocking);
}

void MeshCommandQueueBase::enqueue_read_mesh_buffer(
    void* host_data, const std::shared_ptr<MeshBuffer>& buffer, bool blocking) {
    TT_FATAL(
        (buffer->global_layout() == MeshBufferLayout::SHARDED) || (buffer->device()->num_devices() == 1),
        "Can only read a Sharded MeshBuffer from a MeshDevice or a Replicated MeshBuffer from a Unit-Mesh.");
    TT_FATAL(
        blocking, "Non-Blocking reads are not supported through {}. Use enqueue_read_shards_instead.", __FUNCTION__);
    if (buffer->global_layout() == MeshBufferLayout::SHARDED) {
        auto lock = lock_api_function_();
        this->read_sharded_buffer(*buffer, host_data);
    } else {
        std::vector<distributed::ShardDataTransfer> shard_data_transfers = {
            distributed::ShardDataTransfer{MeshCoordinate(0, 0)}.host_data(host_data)};
        // enqueue_read_shards will call lock_api_function_(), no need to call it here
        this->enqueue_read_shards(shard_data_transfers, buffer, blocking);
    }
}

void MeshCommandQueueBase::enqueue_write_shards_nolock(
    const std::shared_ptr<MeshBuffer>& buffer,
    const std::vector<distributed::ShardDataTransfer>& shard_data_transfers,
    bool blocking) {
    // TODO: #17215 - this API is used by TTNN, as it currently implements rich ND sharding API for multi-devices.
    // In the long run, the multi-device sharding API in Metal will change, and this will most likely be replaced.
    auto dispatch_lambda = [&shard_data_transfers, &buffer, this](uint32_t shard_idx) {
        const auto& shard_data_transfer = shard_data_transfers[shard_idx];
        this->write_shard_to_device(
            *buffer, shard_data_transfer.shard_coord(), shard_data_transfer.host_data(), shard_data_transfer.region());
    };

    for (std::size_t shard_idx = 0; shard_idx < shard_data_transfers.size(); shard_idx++) {
        auto shard_coord = shard_data_transfers[shard_idx].shard_coord();
        if (mesh_device_->is_local(shard_coord)) {
            dispatch_thread_pool_->enqueue(
                [&dispatch_lambda, shard_idx]() { dispatch_lambda(shard_idx); },
                mesh_device_->get_device(shard_coord)->id());
        }
    }
    dispatch_thread_pool_->wait();

    if (blocking) {
        this->finish_nolock();
    }
}

void MeshCommandQueueBase::enqueue_write_shards(
    const std::shared_ptr<MeshBuffer>& mesh_buffer,
    const std::vector<distributed::ShardDataTransfer>& shard_data_transfers,
    bool blocking) {
    auto lock = lock_api_function_();
    this->enqueue_write_shards_nolock(mesh_buffer, shard_data_transfers, blocking);
}

void MeshCommandQueueBase::enqueue_write(
    const std::shared_ptr<MeshBuffer>& mesh_buffer, const DistributedHostBuffer& host_buffer, bool blocking) {
    auto lock = lock_api_function_();
    // Iterate over global coordinates; skip host-remote coordinates, as per `host_buffer` configuration.
    std::vector<distributed::ShardDataTransfer> shard_data_transfers;
    for (const auto& host_buffer_coord : host_buffer.shard_coords()) {
        auto buf = host_buffer.get_shard(host_buffer_coord);
        if (buf.has_value()) {
            shard_data_transfers.push_back(distributed::ShardDataTransfer{MeshCoordinate(host_buffer_coord)}
                                               .host_data(buf->view_bytes().data())
                                               .region(BufferRegion(0, buf->view_bytes().size())));
        }
    }

    this->enqueue_write_shards_nolock(mesh_buffer, shard_data_transfers, blocking);
}

void MeshCommandQueueBase::enqueue_read_shards_nolock(
    const std::vector<distributed::ShardDataTransfer>& shard_data_transfers,
    const std::shared_ptr<MeshBuffer>& buffer,
    bool blocking) {
    // TODO: #17215 - this API is used by TTNN, as it currently implements rich ND sharding API for multi-devices.
    // In the long run, the multi-device sharding API in Metal will change, and this will most likely be replaced.
    std::unordered_map<IDevice*, uint32_t> num_txns_per_device = {};
    bool has_pinned_memory = false;
    for (const auto& shard_data_transfer : shard_data_transfers) {
        if (mesh_device_->is_local(shard_data_transfer.shard_coord())) {
            auto pinned_memory = experimental::ShardDataTransferGetPinnedMemory(shard_data_transfer);
            has_pinned_memory = has_pinned_memory || pinned_memory != nullptr;
            this->read_shard_from_device(
                *buffer,
                shard_data_transfer.shard_coord(),
                shard_data_transfer.host_data(),
                pinned_memory,
                shard_data_transfer.region(),
                num_txns_per_device);
        }
    }
    this->submit_memcpy_request(num_txns_per_device, blocking);

    if (!blocking && has_pinned_memory) {
        auto event = this->enqueue_record_event_to_host_nolock();
        for (const auto& shard_data_transfer : shard_data_transfers) {
            if (mesh_device_->is_local(shard_data_transfer.shard_coord())) {
                auto pinned_memory = experimental::ShardDataTransferGetPinnedMemory(shard_data_transfer);
                if (pinned_memory != nullptr) {
                    pinned_memory->add_barrier_event(event);
                }
            }
        }
    }
}

void MeshCommandQueueBase::enqueue_read_shards(
    const std::vector<distributed::ShardDataTransfer>& shard_data_transfers,
    const std::shared_ptr<MeshBuffer>& mesh_buffer,
    bool blocking) {
    auto lock = lock_api_function_();
    this->enqueue_read_shards_nolock(shard_data_transfers, mesh_buffer, blocking);
}

void MeshCommandQueueBase::enqueue_read(
    const std::shared_ptr<MeshBuffer>& buffer,
    DistributedHostBuffer& host_buffer,
    const std::optional<std::unordered_set<MeshCoordinate>>& shards,
    bool blocking) {
    auto lock = lock_api_function_();
    std::vector<distributed::ShardDataTransfer> shard_data_transfers;
    for (const auto& coord : MeshCoordinateRange(buffer->device()->shape())) {
        if (shards.has_value() && !shards->contains(coord)) {
            continue;
        }

        auto buf = host_buffer.get_shard(coord);
        if (buf.has_value()) {
            shard_data_transfers.push_back(distributed::ShardDataTransfer{coord}
                                               .host_data(buf->view_bytes().data())
                                               .region(BufferRegion(0, buf->view_bytes().size())));
        }
    }

    this->enqueue_read_shards_nolock(shard_data_transfers, buffer, blocking);
}

void MeshCommandQueue::enqueue_write_shards(
    const std::shared_ptr<MeshBuffer>& mesh_buffer,
    const std::vector<ShardDataTransfer>& shard_data_transfers,
    bool blocking) {
    std::vector<distributed::ShardDataTransfer> distributed_shard_data_transfers;
    distributed_shard_data_transfers.reserve(shard_data_transfers.size());
    for (const auto& shard_data_transfer : shard_data_transfers) {
        distributed_shard_data_transfers.push_back(distributed::ShardDataTransfer{shard_data_transfer.shard_coord}
                                                       .host_data(shard_data_transfer.host_data)
                                                       .region(shard_data_transfer.region));
    }
    this->enqueue_write_shards(mesh_buffer, distributed_shard_data_transfers, blocking);
}

void MeshCommandQueue::enqueue_read_shards(
    const std::vector<ShardDataTransfer>& shard_data_transfers,
    const std::shared_ptr<MeshBuffer>& mesh_buffer,
    bool blocking) {
    std::vector<distributed::ShardDataTransfer> distributed_shard_data_transfers;
    distributed_shard_data_transfers.reserve(shard_data_transfers.size());
    for (const auto& shard_data_transfer : shard_data_transfers) {
        distributed_shard_data_transfers.push_back(distributed::ShardDataTransfer{shard_data_transfer.shard_coord}
                                                       .host_data(shard_data_transfer.host_data)
                                                       .region(shard_data_transfer.region));
    }
    this->enqueue_read_shards(distributed_shard_data_transfers, mesh_buffer, blocking);
}

}  // namespace tt::tt_metal::distributed
