// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "mesh_command_queue_base.hpp"

namespace tt::tt_metal::distributed {

class SDMeshCommandQueue final : public MeshCommandQueueBase {
protected:
    void write_shard_to_device(
        const MeshBuffer& buffer,
        const MeshCoordinate& device_coord,
        const void* src,
        const std::optional<BufferRegion>& region,
        tt::stl::Span<const SubDeviceId> sub_device_ids = {}) override;
    void read_shard_from_device(
        const MeshBuffer& buffer,
        const MeshCoordinate& device_coord,
        void* dst,
        std::shared_ptr<experimental::PinnedMemory> pinned_memory,
        const std::optional<BufferRegion>& region,
        std::unordered_map<IDevice*, uint32_t>& num_txns_per_device,
        tt::stl::Span<const SubDeviceId> sub_device_ids = {}) override;
    void submit_memcpy_request(std::unordered_map<IDevice*, uint32_t>& num_txns_per_device, bool blocking) override;
    void finish_nolock(tt::stl::Span<const SubDeviceId> sub_device_ids = {}) override;
    MeshEvent enqueue_record_event_to_host_nolock(
        tt::stl::Span<const SubDeviceId> sub_device_ids = {},
        const std::optional<MeshCoordinateRange>& device_range = std::nullopt) override;

public:
    SDMeshCommandQueue(
        MeshDevice* mesh_device, uint32_t id, std::function<std::lock_guard<std::mutex>()> lock_api_function);
    ~SDMeshCommandQueue() override = default;

    std::optional<MeshTraceId> trace_id() const override;

    WorkerConfigBufferMgr& get_config_buffer_mgr(uint32_t index) override;
    void enqueue_mesh_workload(MeshWorkload& mesh_workload, bool blocking) override;

    MeshEvent enqueue_record_event(
        tt::stl::Span<const SubDeviceId> sub_device_ids = {},
        const std::optional<MeshCoordinateRange>& device_range = std::nullopt) override;
    MeshEvent enqueue_record_event_to_host(
        tt::stl::Span<const SubDeviceId> sub_device_ids = {},
        const std::optional<MeshCoordinateRange>& device_range = std::nullopt) override;
    void enqueue_wait_for_event(const MeshEvent& sync_event) override;
    void finish(tt::stl::Span<const SubDeviceId> sub_device_ids = {}) override;
    void reset_worker_state(
        bool reset_launch_msg_state,
        uint32_t num_sub_devices,
        const vector_aligned<uint32_t>& go_signal_noc_data,
        const std::vector<std::pair<CoreRangeSet, uint32_t>>& core_go_message_mapping) override;
    void record_begin(const MeshTraceId& trace_id, const std::shared_ptr<MeshTraceDescriptor>& ctx) override;
    void record_end() override;
    void enqueue_trace(const MeshTraceId& trace_id, bool blocking) override;
};

}  // namespace tt::tt_metal::distributed
