// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "mesh_command_queue_base.hpp"

namespace tt::tt_metal::distributed {

class SDMeshCommandQueue final : public MeshCommandQueueBase {
private:
    // Distributed context used to synchronize operations done by all active ranks on the given mesh device.
    std::shared_ptr<distributed::multihost::DistributedContext> active_distributed_context_;

protected:
    bool write_shard_to_device(
        const MeshBuffer& buffer,
        const MeshCoordinate& device_coord,
        const void* src,
        const std::optional<BufferRegion>& region,
        ttsl::Span<const SubDeviceId> sub_device_ids = {},
        std::shared_ptr<experimental::PinnedMemory> pinned_memory = nullptr) override;
    void read_shard_from_device(
        const MeshBuffer& buffer,
        const MeshCoordinate& device_coord,
        void* dst,
        std::shared_ptr<experimental::PinnedMemory> pinned_memory,
        const std::optional<BufferRegion>& region,
        std::unordered_map<IDevice*, uint32_t>& num_txns_per_device,
        ttsl::Span<const SubDeviceId> sub_device_ids = {}) override;
    void submit_memcpy_request(std::unordered_map<IDevice*, uint32_t>& num_txns_per_device, bool blocking) override;
    void finish_nolock(ttsl::Span<const SubDeviceId> sub_device_ids = {}) override;
    MeshEvent enqueue_record_event_to_host_nolock(
        ttsl::Span<const SubDeviceId> sub_device_ids = {},
        const std::optional<MeshCoordinateRange>& device_range = std::nullopt) override;

public:
    SDMeshCommandQueue(
        MeshDevice* mesh_device,
        uint32_t id,
        std::function<std::lock_guard<std::mutex>()> lock_api_function,
        std::shared_ptr<distributed::multihost::DistributedContext> distributed_context);
    ~SDMeshCommandQueue() override = default;

    std::optional<MeshTraceId> trace_id() const override;

    WorkerConfigBufferMgr& get_config_buffer_mgr(uint32_t index) override;
    void enqueue_mesh_workload(MeshWorkload& mesh_workload, bool blocking) override;

    MeshEvent enqueue_record_event(
        ttsl::Span<const SubDeviceId> sub_device_ids = {},
        const std::optional<MeshCoordinateRange>& device_range = std::nullopt) override;
    MeshEvent enqueue_record_event_to_host(
        ttsl::Span<const SubDeviceId> sub_device_ids = {},
        const std::optional<MeshCoordinateRange>& device_range = std::nullopt) override;
    void enqueue_wait_for_event(const MeshEvent& sync_event) override;
    void finish(ttsl::Span<const SubDeviceId> sub_device_ids = {}) override;
    void reset_worker_state(
        bool reset_launch_msg_state,
        uint32_t num_sub_devices,
        const vector_aligned<uint32_t>& go_signal_noc_data,
        const std::vector<std::pair<CoreRangeSet, uint32_t>>& core_go_message_mapping) override;
    void record_begin(const MeshTraceId& trace_id, const std::shared_ptr<MeshTraceDescriptor>& ctx) override;
    void record_end() override;
    void enqueue_trace(const MeshTraceId& trace_id, bool blocking) override;

    void enable_asynchronous_slow_dispatch();
    void disable_asynchronous_slow_dispatch();

private:
    void wait_for_cores_idle();

    std::unordered_map<ChipId, std::vector<std::vector<CoreCoord>>> logical_cores_for_previous_workload_;

    bool asynchronous_slow_dispatch_enabled_ = false;
};

}  // namespace tt::tt_metal::distributed
