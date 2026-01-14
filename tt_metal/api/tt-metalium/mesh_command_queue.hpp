// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdint.h>
#include <atomic>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <optional>
#include <queue>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <variant>
#include <vector>

#include <tt_stl/span.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/mesh_buffer.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/mesh_trace_id.hpp>
#include <tt-metalium/distributed_host_buffer.hpp>
#include <tt-metalium/mesh_workload.hpp>
#include <tt-metalium/sub_device_types.hpp>
#include <tt-metalium/vector_aligned.hpp>
#include <umd/device/types/core_coordinates.hpp>

namespace tt::tt_metal {
class IDevice;
class SystemMemoryManager;
class WorkerConfigBufferMgr;
namespace distributed {
class MeshDevice;
class MeshWorkload;
}  // namespace distributed
struct ProgramCommandSequence;
namespace experimental {

class ShardDataTransferHelper;
}  // namespace experimental
}  // namespace tt::tt_metal

namespace tt::tt_metal::distributed {

class MeshEvent;
class MeshTraceDescriptor;
struct MeshBufferReadDescriptor;
struct MeshReadEventDescriptor;
struct MeshCoreDataReadDescriptor;

using MeshCompletionReaderVariant =
    std::variant<MeshBufferReadDescriptor, MeshReadEventDescriptor, MeshCoreDataReadDescriptor>;

class ShardDataTransfer;

// THREAD SAFETY: All methods are thread safe.
class MeshCommandQueue {
    // Main interface to dispatch data and workloads to a MeshDevice
    // Currently only supports dispatching workloads and relies on the
    // tt::tt_metal::CommandQueue.
    // Additional support for Reads and Writes to be added
protected:
    MeshDevice* mesh_device_ = nullptr;
    uint32_t id_ = 0;

    MeshCommandQueue(MeshDevice* mesh_device, uint32_t id) : mesh_device_(mesh_device), id_(id) {}

public:
    MeshCommandQueue(const MeshCommandQueue& other) = delete;
    MeshCommandQueue& operator=(const MeshCommandQueue& other) = delete;

    virtual ~MeshCommandQueue() = default;

    MeshDevice* device() const { return mesh_device_; }
    uint32_t id() const { return id_; }
    virtual std::optional<MeshTraceId> trace_id() const = 0;
    virtual WorkerConfigBufferMgr& get_config_buffer_mgr(uint32_t index) = 0;
    virtual void enqueue_mesh_workload(MeshWorkload& mesh_workload, bool blocking) = 0;

    // Specifies host data to be written to or read from a MeshBuffer shard.
    struct [[deprecated("Use distributed::ShardDataTransfer instead.")]] ShardDataTransfer {
        MeshCoordinate shard_coord;
        void* host_data = nullptr;
        std::optional<BufferRegion> region;
    };

    // MeshBuffer Write APIs
    virtual void enqueue_write_shard_to_sub_grid(
        const MeshBuffer& buffer,
        const void* host_data,
        const MeshCoordinateRange& device_range,
        bool blocking,
        std::optional<BufferRegion> region = std::nullopt) = 0;
    virtual void enqueue_write_mesh_buffer(
        const std::shared_ptr<MeshBuffer>& buffer, const void* host_data, bool blocking) = 0;
    virtual void enqueue_write(
        const std::shared_ptr<MeshBuffer>& mesh_buffer, const DistributedHostBuffer& host_buffer, bool blocking) = 0;
    virtual void enqueue_write_shards(
        const std::shared_ptr<MeshBuffer>& mesh_buffer,
        const std::vector<distributed::ShardDataTransfer>& shard_data_transfers,
        bool blocking) = 0;
    [[deprecated("Use enqueue_write_shards with distributed::ShardDataTransfer instead.")]]
    void enqueue_write_shards(
        const std::shared_ptr<MeshBuffer>& mesh_buffer,
        const std::vector<ShardDataTransfer>& shard_data_transfers,
        bool blocking);

    // MeshBuffer Read APIs
    virtual void enqueue_read_mesh_buffer(
        void* host_data, const std::shared_ptr<MeshBuffer>& buffer, bool blocking) = 0;
    virtual void enqueue_read_shards(
        const std::vector<distributed::ShardDataTransfer>& shard_data_transfers,
        const std::shared_ptr<MeshBuffer>& mesh_buffer,
        bool blocking) = 0;
    // TODO: does "enqueue" make sense anymore? Return the object by value instead.
    virtual void enqueue_read(
        const std::shared_ptr<MeshBuffer>& mesh_buffer,
        DistributedHostBuffer& host_buffer,
        const std::optional<std::unordered_set<MeshCoordinate>>& shards,
        bool blocking) = 0;
    [[deprecated("Use enqueue_read_shards with distributed::ShardDataTransfer instead.")]]
    void enqueue_read_shards(
        const std::vector<ShardDataTransfer>& shard_data_transfers,
        const std::shared_ptr<MeshBuffer>& mesh_buffer,
        bool blocking);

    virtual MeshEvent enqueue_record_event(
        tt::stl::Span<const SubDeviceId> sub_device_ids = {},
        const std::optional<MeshCoordinateRange>& device_range = std::nullopt) = 0;
    virtual MeshEvent enqueue_record_event_to_host(
        tt::stl::Span<const SubDeviceId> sub_device_ids = {},
        const std::optional<MeshCoordinateRange>& device_range = std::nullopt) = 0;
    virtual void enqueue_wait_for_event(const MeshEvent& sync_event) = 0;
    virtual void finish(tt::stl::Span<const SubDeviceId> sub_device_ids = {}) = 0;
    virtual void reset_worker_state(
        bool reset_launch_msg_state,
        uint32_t num_sub_devices,
        const vector_aligned<uint32_t>& go_signal_noc_data,
        const std::vector<std::pair<CoreRangeSet, uint32_t>>& core_go_message_mapping) = 0;
    virtual void record_begin(const MeshTraceId& trace_id, const std::shared_ptr<MeshTraceDescriptor>& ctx) = 0;
    virtual void record_end() = 0;
    virtual void enqueue_trace(const MeshTraceId& trace_id, bool blocking) = 0;

    // Internal function.
    virtual void wait_for_completion(bool) {}
    // May only be called after wait_for_completion has been called on both command queues on the device.
    virtual void finish_and_reset_in_use() {}
};

// Specifies host data to be written to or read from a MeshBuffer shard.
class ShardDataTransfer {
private:
    MeshCoordinate shard_coord_;
    void* host_data_ = nullptr;
    std::optional<BufferRegion> region_;
    std::shared_ptr<experimental::PinnedMemory> pinned_memory_ = nullptr;
    friend class experimental::ShardDataTransferHelper;

public:
    explicit ShardDataTransfer(const MeshCoordinate& shard_coord) : shard_coord_(shard_coord) {}
    explicit ShardDataTransfer(const MeshCommandQueue::ShardDataTransfer& shard_data_transfer) :
        shard_coord_(shard_data_transfer.shard_coord),
        host_data_(shard_data_transfer.host_data),
        region_(shard_data_transfer.region) {}

    MeshCoordinate shard_coord() const { return shard_coord_; }
    void* host_data() const { return host_data_; }
    std::optional<BufferRegion> region() const { return region_; }

    ShardDataTransfer& shard_coord(const MeshCoordinate& shard_coord) {
        shard_coord_ = shard_coord;
        return *this;
    }
    ShardDataTransfer& host_data(void* host_data) {
        host_data_ = host_data;
        return *this;
    }
    ShardDataTransfer& region(std::optional<BufferRegion> region) {
        region_ = region;
        return *this;
    }
};

}  // namespace tt::tt_metal::distributed
