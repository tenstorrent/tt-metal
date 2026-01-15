// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "mesh_command_queue.hpp"

#include "tt_metal/common/thread_pool.hpp"

#include <mutex>
#include <functional>

namespace tt::tt_metal::distributed {

class MeshCommandQueueBase : public MeshCommandQueue {
protected:
    std::shared_ptr<ThreadPool>
        dispatch_thread_pool_;  // Thread pool used to dispatch to the Mesh (used by main thread)
    std::function<std::lock_guard<std::mutex>()> lock_api_function_;

    // Helper functions for reading and writing individual shards
    virtual void write_shard_to_device(
        const MeshBuffer& buffer,
        const MeshCoordinate& device_coord,
        const void* src,
        const std::optional<BufferRegion>& region,
        tt::stl::Span<const SubDeviceId> sub_device_ids = {}) = 0;
    virtual void read_shard_from_device(
        const MeshBuffer& buffer,
        const MeshCoordinate& device_coord,
        void* dst,
        std::shared_ptr<experimental::PinnedMemory> pinned_memory,
        const std::optional<BufferRegion>& region,
        std::unordered_map<IDevice*, uint32_t>& num_txns_per_device,
        tt::stl::Span<const SubDeviceId> sub_device_ids = {}) = 0;
    virtual void submit_memcpy_request(std::unordered_map<IDevice*, uint32_t>& num_txns_per_device, bool blocking) = 0;
    // Must be called with lock_api_function_() held.
    virtual void finish_nolock(tt::stl::Span<const SubDeviceId> sub_device_ids = {}) = 0;
    virtual MeshEvent enqueue_record_event_to_host_nolock(
        tt::stl::Span<const SubDeviceId> sub_device_ids = {},
        const std::optional<MeshCoordinateRange>& device_range = std::nullopt) = 0;

private:
    // Helper functions for read and write entire Sharded-MeshBuffers
    void write_sharded_buffer(const MeshBuffer& buffer, const void* src);
    void read_sharded_buffer(MeshBuffer& buffer, void* dst);

    // Must be called with lock_api_function_() held.
    void enqueue_read_shards_nolock(
        const std::vector<distributed::ShardDataTransfer>& shard_data_transfers,
        const std::shared_ptr<MeshBuffer>& mesh_buffer,
        bool blocking);
    // Must be called with lock_api_function_() held.
    void enqueue_write_shards_nolock(
        const std::shared_ptr<MeshBuffer>& mesh_buffer,
        const std::vector<distributed::ShardDataTransfer>& shard_data_transfers,
        bool blocking);

public:
    MeshCommandQueueBase(
        MeshDevice* mesh_device,
        uint32_t id,
        std::shared_ptr<ThreadPool> dispatch_thread_pool,
        std::function<std::lock_guard<std::mutex>()> lock_api_function) :
        MeshCommandQueue(mesh_device, id),
        dispatch_thread_pool_(std::move(dispatch_thread_pool)),
        lock_api_function_(std::move(lock_api_function)) {}

    void enqueue_write_shard_to_sub_grid(
        const MeshBuffer& buffer,
        const void* host_data,
        const MeshCoordinateRange& device_range,
        bool blocking,
        std::optional<BufferRegion> region = std::nullopt) override;
    void enqueue_write_mesh_buffer(
        const std::shared_ptr<MeshBuffer>& buffer, const void* host_data, bool blocking) override;
    void enqueue_write_shards(
        const std::shared_ptr<MeshBuffer>& mesh_buffer,
        const std::vector<distributed::ShardDataTransfer>& shard_data_transfers,
        bool blocking) override;
    void enqueue_write(
        const std::shared_ptr<MeshBuffer>& mesh_buffer,
        const DistributedHostBuffer& host_buffer,
        bool blocking) override;

    // MeshBuffer Read APIs
    void enqueue_read_mesh_buffer(void* host_data, const std::shared_ptr<MeshBuffer>& buffer, bool blocking) override;
    void enqueue_read_shards(
        const std::vector<distributed::ShardDataTransfer>& shard_data_transfers,
        const std::shared_ptr<MeshBuffer>& mesh_buffer,
        bool blocking) override;
    void enqueue_read(
        const std::shared_ptr<MeshBuffer>& mesh_buffer,
        DistributedHostBuffer& host_buffer,
        const std::optional<std::unordered_set<MeshCoordinate>>& shards,
        bool blocking) override;

    // Returns true if the CQ is in use (has had commands enqueued).
    virtual bool in_use() { return false; }
};

}  // namespace tt::tt_metal::distributed
