// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "mesh_command_queue.hpp"

#include <tt-metalium/experimental/core_subset_write/mesh_command_queue.hpp>

#include "tt_metal/impl/dispatch/host_transfer_stats.hpp"
#include "tt_metal/impl/dispatch/vector_aligned.hpp"
#include "tt_metal/impl/threading/thread_pool.hpp"
#include "tt_target_device.hpp"

#include <tt_stl/assert.hpp>

#include <atomic>
#include <chrono>
#include <mutex>
#include <functional>
#include <utility>

namespace tt::tt_metal::distributed {

// Identifies one L1 location on one device of the mesh: (mesh coord, virtual
// core, device address). Used by per-core enqueue APIs.
struct DeviceMemoryAddress {
    MeshCoordinate device_coord;
    CoreCoord virtual_core_coord;
    DeviceAddr address{};
};

class MeshCommandQueueBase : public MeshCommandQueue {
protected:
    std::shared_ptr<ThreadPool>
        dispatch_thread_pool_;  // Thread pool used to dispatch to the Mesh (used by main thread)
    std::function<std::lock_guard<std::mutex>()> lock_api_function_;

    // Helper functions for reading and writing individual shards
    // Returns true if pinned memory was used for the transfer
    virtual bool write_shard_to_device(
        const MeshBuffer& buffer,
        const MeshCoordinate& device_coord,
        const void* src,
        const std::optional<BufferRegion>& region,
        ttsl::Span<const SubDeviceId> sub_device_ids = {},
        std::shared_ptr<experimental::PinnedMemory> pinned_memory = nullptr,
        const tt::tt_metal::CoreRangeSet* logical_core_filter = nullptr) = 0;
    virtual void read_shard_from_device(
        const MeshBuffer& buffer,
        const MeshCoordinate& device_coord,
        void* dst,
        std::shared_ptr<experimental::PinnedMemory> pinned_memory,
        const std::optional<BufferRegion>& region,
        std::unordered_map<IDevice*, uint32_t>& num_txns_per_device,
        ttsl::Span<const SubDeviceId> sub_device_ids = {}) = 0;
    virtual void submit_memcpy_request(
        std::unordered_map<IDevice*, uint32_t>& num_txns_per_device,
        bool blocking,
        std::vector<MemoryPin> memory_pins = {}) = 0;
    // Must be called with lock_api_function_() held.
    virtual void finish_nolock(ttsl::Span<const SubDeviceId> sub_device_ids = {}) = 0;
    virtual MeshEvent enqueue_record_event_to_host_nolock(
        ttsl::Span<const SubDeviceId> sub_device_ids = {},
        const std::optional<MeshCoordinateRange>& device_range = std::nullopt) = 0;
    virtual void invalidate_prefetcher_cache_after_pinned_write() {}

    tt::TargetDevice get_target_device_type() const;

    // Host transfer stats (TT_METAL_HOST_TRANSFER_STATS). Each transfer API call is one transfer.
    enum class TransferOp : uint8_t { BufferWrite, BufferRead, CoreWrite, CoreRead, ProgramBinaries };
    // Open and close the transfer of one API call while stats are on, for queues whose transfers finish after the
    // call returns. Called with lock_api_function_() held, so the transfers of one queue never overlap.
    // end_transfer_stats is called after the transfer's commands are issued and before a blocking call waits for
    // them.
    virtual void begin_transfer_stats(TransferOp /*op*/) {}
    virtual void end_transfer_stats(TransferOp /*op*/, uint64_t /*bytes*/) {}
    // Adds one shard's bytes to the transfer open on this queue. Shard writes and reads call it once they are past
    // the checks that skip the transfer.
    void count_transfer_bytes(uint64_t bytes) {
        if (host_transfer_stats_->enabled()) {
            transfer_bytes_.fetch_add(bytes, std::memory_order_relaxed);
        }
    }
    // Bytes a shard write copies from host: with a core filter, only the pages bound for cores in it.
    static uint64_t host_bytes_written(
        Buffer& shard_view, uint64_t region_size, const tt::tt_metal::CoreRangeSet* logical_core_filter);

    // Covers the transfer of one API call while stats are on: begin_transfer_stats on construction and
    // end_transfer_stats with the counted bytes, plus `bytes`, on close() or destruction.
    class ScopedTransferStats {
    public:
        ScopedTransferStats(MeshCommandQueueBase& cq, TransferOp op, uint64_t bytes = 0);
        ~ScopedTransferStats();
        void close();

        ScopedTransferStats(const ScopedTransferStats&) = delete;
        ScopedTransferStats& operator=(const ScopedTransferStats&) = delete;
        ScopedTransferStats(ScopedTransferStats&&) = delete;
        ScopedTransferStats& operator=(ScopedTransferStats&&) = delete;

    private:
        MeshCommandQueueBase* cq_ = nullptr;
        TransferOp op_;
        uint64_t bytes_;
    };

    HostTransferStats* host_transfer_stats_;

private:
    std::atomic<uint64_t> transfer_bytes_{0};

    // Helper functions for read and write entire Sharded-MeshBuffers
    void write_sharded_buffer(const MeshBuffer& buffer, const void* src);
    void read_sharded_buffer(MeshBuffer& buffer, void* dst);

    // Must be called with lock_api_function_() held.
    void enqueue_read_shards_nolock(
        const std::vector<ShardDataTransfer>& shard_data_transfers,
        const std::shared_ptr<MeshBuffer>& mesh_buffer,
        bool blocking,
        std::vector<MemoryPin> memory_pins = {});
    // Must be called with lock_api_function_() held.
    void enqueue_write_shards_nolock(
        MeshBuffer& mesh_buffer,
        const std::vector<ShardDataTransfer>& shard_data_transfers,
        bool blocking,
        const tt::tt_metal::CoreRangeSet* logical_core_filter = nullptr);

    void enqueue_write_with_core_filter(
        MeshBuffer& mesh_buffer,
        const DistributedHostBuffer& host_buffer,
        bool blocking,
        const tt::tt_metal::CoreRangeSet* logical_core_filter);

    friend void tt::tt_metal::experimental::core_subset_write::enqueue_write(
        tt::tt_metal::distributed::MeshCommandQueue& cq,
        tt::tt_metal::distributed::MeshBuffer& mesh_buffer,
        const tt::tt_metal::DistributedHostBuffer& host_buffer,
        bool blocking,
        const tt::tt_metal::CoreRangeSet& logical_core_filter);

public:
    MeshCommandQueueBase(
        MeshDevice* mesh_device,
        uint32_t id,
        std::shared_ptr<ThreadPool> dispatch_thread_pool,
        std::function<std::lock_guard<std::mutex>()> lock_api_function);

    // While one is alive, buffer writes made on this thread count as kernel binary uploads in host transfer stats.
    class ScopedProgramBinaryUpload {
    public:
        ScopedProgramBinaryUpload() : previous_(std::exchange(active_, true)) {}
        ~ScopedProgramBinaryUpload() { active_ = previous_; }
        static bool active() { return active_; }

        ScopedProgramBinaryUpload(const ScopedProgramBinaryUpload&) = delete;
        ScopedProgramBinaryUpload& operator=(const ScopedProgramBinaryUpload&) = delete;
        ScopedProgramBinaryUpload(ScopedProgramBinaryUpload&&) = delete;
        ScopedProgramBinaryUpload& operator=(ScopedProgramBinaryUpload&&) = delete;

    private:
        inline static thread_local bool active_ = false;
        bool previous_;
    };

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
        const std::vector<ShardDataTransfer>& shard_data_transfers,
        bool blocking) override;
    void enqueue_write(
        const std::shared_ptr<MeshBuffer>& mesh_buffer,
        const DistributedHostBuffer& host_buffer,
        bool blocking) override;

    // MeshBuffer Read APIs
    void enqueue_read_mesh_buffer(void* host_data, const std::shared_ptr<MeshBuffer>& buffer, bool blocking) override;
    void enqueue_read_shards(
        const std::vector<ShardDataTransfer>& shard_data_transfers,
        const std::shared_ptr<MeshBuffer>& mesh_buffer,
        bool blocking) override;
    void enqueue_read(
        const std::shared_ptr<MeshBuffer>& mesh_buffer,
        DistributedHostBuffer& host_buffer,
        const std::optional<std::unordered_set<MeshCoordinate>>& shards,
        bool blocking) override;

    // Returns true if the CQ is in use (has had commands enqueued).
    virtual bool in_use() { return false; }

    // Resets this queue's dispatch state. `reset_launch_msg_state` additionally resets state that is shared
    // by all hardware CQs (the worker launch message ring buffer and the GO mailboxes), so exactly one CQ may
    // be given it, and only once every other CQ has been drained. The implementation drains the CQs that are
    // not given it, so callers must reset the CQs in order and pass it to the last one. Must be called with
    // the MeshDevice api lock held.
    virtual void reset_worker_state(
        bool reset_launch_msg_state,
        uint32_t num_sub_devices,
        const std::vector<std::pair<CoreRangeSet, uint32_t>>& core_go_message_mapping,
        ttsl::Span<const uint32_t> workers_per_sub_device) = 0;

    // Write `value` (a uint32 counter) to one L1 address on each of `targets`,
    // ordered after prior worker programs / buffer writes on this queue. Each
    // target's `address` is the full device destination. Must be called with the
    // MeshDevice api lock already held — unlike the other queue APIs this does NOT
    // re-lock, so the caller can keep the lock across the surrounding sequence (the
    // Tensor prefetcher's WaitForCq needs the counter bump and the WAIT_CQ enqueue
    // to be atomic). Fast/slow dispatch perform the write; the dummy (inactive-rank)
    // queue is a no-op.
    virtual void enqueue_write_dram_core_counter(
        ttsl::Span<const DeviceMemoryAddress> targets,
        uint32_t value,
        bool blocking,
        ttsl::Span<const SubDeviceId> sub_device_ids = {}) = 0;

    virtual void wait_for_completion(bool) {}
    // May only be called after wait_for_completion has been called on both command queues on the device.
    virtual void finish_and_reset_in_use() {}
};

}  // namespace tt::tt_metal::distributed
