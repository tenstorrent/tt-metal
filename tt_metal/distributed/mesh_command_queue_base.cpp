// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt_stl/fmt.hpp>
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
#include "tt_metal/impl/threading/thread_pool.hpp"
#include "tt_cluster.hpp"
#include "dispatch/dispatch_settings.hpp"
#include "tt_metal/distributed/mesh_device_impl.hpp"

namespace tt::tt_metal::distributed {

tt::TargetDevice MeshCommandQueueBase::get_target_device_type() const {
    return tt::tt_metal::MetalContext::instance(this->device()->impl().get_context_id())
        .get_cluster()
        .get_target_device_type();
}

void MeshCommandQueueBase::enqueue_write_shard_to_sub_grid(
    const MeshBuffer& buffer,
    const void* host_data,
    const MeshCoordinateRange& device_range,
    bool blocking,
    std::optional<BufferRegion> region) {
    auto lock = lock_api_function_();
    // The same data is written to every device in the range. Any cross-device sharding
    // is performed by TTNN, which calls enqueue_write_shards directly.
    auto dispatch_lambda = [this, &buffer, host_data, &region](const MeshCoordinate& coord) {
        this->write_shard_to_device(buffer, coord, host_data, region, {}, nullptr);
    };
    for (const auto& coord : device_range) {
        if (mesh_device_->impl().is_local(coord)) {
            dispatch_thread_pool_->enqueue(
                [&dispatch_lambda, coord]() { dispatch_lambda(coord); }, mesh_device_->impl().get_device(coord)->id());
        }
    }
    dispatch_thread_pool_->wait();

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
    TT_FATAL(buffer->device()->num_devices() == 1, "Can only read a replicated MeshBuffer from a Unit-Mesh.");
    TT_FATAL(
        blocking, "Non-Blocking reads are not supported through {}. Use enqueue_read_shards_instead.", __FUNCTION__);
    std::vector<distributed::ShardDataTransfer> shard_data_transfers = {
        distributed::ShardDataTransfer{MeshCoordinate::zero_coordinate(buffer->device()->shape().dims())}.host_data(
            host_data)};
    // enqueue_read_shards will call lock_api_function_(), no need to call it here
    this->enqueue_read_shards(shard_data_transfers, buffer, blocking);
}

void MeshCommandQueueBase::enqueue_write_shards_nolock(
    MeshBuffer& buffer,
    const std::vector<distributed::ShardDataTransfer>& shard_data_transfers,
    bool blocking,
    const tt::tt_metal::CoreRangeSet* logical_core_filter) {
    // TODO: #17215 - this API is used by TTNN, as it currently implements rich ND sharding API for multi-devices.
    // In the long run, the multi-device sharding API in Metal will change, and this will most likely be replaced.

    // Track if any transfer actually used pinned memory
    std::atomic<bool> any_pinned_used = false;

    auto dispatch_lambda =
        [&shard_data_transfers, &buffer, &any_pinned_used, logical_core_filter, this](uint32_t shard_idx) {
            const auto& shard_data_transfer = shard_data_transfers[shard_idx];
            bool pinned_used = this->write_shard_to_device(
                buffer,
                shard_data_transfer.shard_coord(),
                shard_data_transfer.host_data(),
                shard_data_transfer.region(),
                {},
                experimental::ShardDataTransferGetPinnedMemory(shard_data_transfer),
                logical_core_filter);
            if (pinned_used) {
                any_pinned_used.store(true, std::memory_order_relaxed);
            }
        };

    for (std::size_t shard_idx = 0; shard_idx < shard_data_transfers.size(); shard_idx++) {
        auto shard_coord = shard_data_transfers[shard_idx].shard_coord();
        if (mesh_device_->impl().is_local(shard_coord)) {
            dispatch_thread_pool_->enqueue(
                [&dispatch_lambda, shard_idx]() { dispatch_lambda(shard_idx); },
                mesh_device_->impl().get_device(shard_coord)->id());
        }
    }
    dispatch_thread_pool_->wait();

    if (any_pinned_used.load(std::memory_order_relaxed)) {
        this->invalidate_prefetcher_cache_after_pinned_write();
    }

    if (blocking) {
        this->finish_nolock();
    } else if (any_pinned_used.load(std::memory_order_relaxed)) {
        // If any transfer used pinned memory, add barrier event to all pinned memory objects
        auto event = this->enqueue_record_event_to_host_nolock();
        for (const auto& shard_data_transfer : shard_data_transfers) {
            if (mesh_device_->is_local(shard_data_transfer.shard_coord())) {
                auto pinned_memory = experimental::ShardDataTransferGetPinnedMemory(shard_data_transfer);
                if (pinned_memory) {
                    pinned_memory->add_barrier_event(event);
                }
            }
        }
    }
}

void MeshCommandQueueBase::enqueue_write_shards(
    const std::shared_ptr<MeshBuffer>& mesh_buffer,
    const std::vector<distributed::ShardDataTransfer>& shard_data_transfers,
    bool blocking) {
    auto lock = lock_api_function_();
    this->enqueue_write_shards_nolock(*mesh_buffer, shard_data_transfers, blocking, nullptr);
}

void MeshCommandQueueBase::enqueue_write(
    const std::shared_ptr<MeshBuffer>& mesh_buffer, const DistributedHostBuffer& host_buffer, bool blocking) {
    this->enqueue_write_with_core_filter(*mesh_buffer, host_buffer, blocking, nullptr);
}

void MeshCommandQueueBase::enqueue_write_with_core_filter(
    MeshBuffer& mesh_buffer,
    const DistributedHostBuffer& host_buffer,
    bool blocking,
    const tt::tt_metal::CoreRangeSet* logical_core_filter) {
    auto lock = lock_api_function_();
    // Iterate over global coordinates; skip host-remote coordinates, as per `host_buffer` configuration.
    std::vector<distributed::ShardDataTransfer> shard_data_transfers;
    for (const auto& host_buffer_coord : host_buffer.shard_coords()) {
        auto buf = host_buffer.get_shard(host_buffer_coord);
        if (buf.has_value()) {
            auto shard_data_transfer = distributed::ShardDataTransfer{MeshCoordinate(host_buffer_coord)}
                                           .host_data(buf->view_bytes().data())
                                           .region(BufferRegion(0, buf->view_bytes().size()));
            experimental::ShardDataTransferSetPinnedMemory(
                shard_data_transfer, experimental::HostBufferGetPinnedMemory(*buf));
            shard_data_transfers.push_back(std::move(shard_data_transfer));
        }
    }

    this->enqueue_write_shards_nolock(mesh_buffer, shard_data_transfers, blocking, logical_core_filter);
}

void MeshCommandQueueBase::enqueue_read_shards_nolock(
    const std::vector<distributed::ShardDataTransfer>& shard_data_transfers,
    const std::shared_ptr<MeshBuffer>& buffer,
    bool blocking,
    std::vector<MemoryPin> memory_pins) {
    // TODO: #17215 - this API is used by TTNN, as it currently implements rich ND sharding API for multi-devices.
    // In the long run, the multi-device sharding API in Metal will change, and this will most likely be replaced.
    std::unordered_map<IDevice*, uint32_t> num_txns_per_device = {};
    bool has_pinned_memory = false;
    for (const auto& shard_data_transfer : shard_data_transfers) {
        if (mesh_device_->impl().is_local(shard_data_transfer.shard_coord())) {
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
    this->submit_memcpy_request(num_txns_per_device, blocking, std::move(memory_pins));

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
    // For non-blocking reads, capture a MemoryPin for each shard so the host
    // buffer stays alive until the async reader thread finishes the memcpy
    // (fixes use-after-free, issue #43638). For blocking reads finish_nolock()
    // ensures the copy is complete before we return, so no pin is needed.
    std::vector<MemoryPin> memory_pins;
    for (const auto& coord : MeshCoordinateRange(buffer->device()->shape())) {
        if (shards.has_value() && !shards->contains(coord)) {
            continue;
        }

        auto buf = host_buffer.get_shard(coord);
        if (buf.has_value()) {
            if (!blocking) {
                memory_pins.push_back(buf->pin());
            }
            auto xfer = distributed::ShardDataTransfer{coord}
                            .host_data(buf->view_bytes().data())
                            .region(BufferRegion(0, buf->view_bytes().size()));
            experimental::ShardDataTransferSetPinnedMemory(xfer, experimental::HostBufferGetPinnedMemory(*buf));
            shard_data_transfers.push_back(std::move(xfer));
        }
    }

    this->enqueue_read_shards_nolock(shard_data_transfers, buffer, blocking, std::move(memory_pins));
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
