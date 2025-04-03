// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "mesh_command_queue_base.hpp"

#include <tt-metalium/command_queue.hpp>
#include <tt-metalium/multi_producer_single_consumer_queue.hpp>

namespace tt::tt_metal::distributed {

struct MeshReadEventDescriptor;
struct MeshBufferReadDescriptor;

using MeshCompletionReaderVariant = std::variant<MeshBufferReadDescriptor, MeshReadEventDescriptor>;

class FDMeshCommandQueue final : public MeshCommandQueueBase {
private:
    void populate_read_descriptor_queue();
    void populate_virtual_program_dispatch_core();
    void populate_dispatch_core_type();
    CoreCoord virtual_program_dispatch_core() const;
    CoreType dispatch_core_type() const;

    void increment_num_entries_in_completion_queue();
    MeshEvent enqueue_record_event_helper(
        tt::stl::Span<const SubDeviceId> sub_device_ids,
        bool notify_host,
        const std::optional<MeshCoordinateRange>& device_range = std::nullopt);
    // Trace capture utility functions
    // Captures dispatch commands associated with running a program on a Virtual Mesh subgrid
    // inside the appropriate trace staging vector (corresponding to the specified subgrid)
    void capture_program_trace_on_subgrid(
        const MeshCoordinateRange& sub_grid,
        ProgramCommandSequence& program_cmd_seq,
        bool stall_first,
        bool stall_before_program,
        uint32_t workload_runtime_id);
    // For a given MeshWorkload, a subgrid is unused if no programs are run on it. Go signals
    // must be sent to this subgrid, to ensure consistent global state across the Virtual Mesh.
    // When running trace, the dispatch commands responsible for forwarding go signals must be
    // captured on these subgrids.
    void capture_go_signal_trace_on_unused_subgrids(
        const MeshCoordinateRange& active_sub_grids,
        const SubDeviceId& sub_device_id,
        uint32_t expected_num_workers_completed,
        bool mcast_go_signals,
        bool unicast_go_signals);
    // Workload dispatch utility functions
    // Write dispatch commands associated with running a program on a Virtual Mesh subgrid
    void write_program_cmds_to_subgrid(
        const MeshCoordinateRange& sub_grid,
        ProgramCommandSequence& program_cmd_seq,
        bool stall_first,
        bool stall_before_program,
        std::unordered_set<uint32_t>& chip_ids_in_workload,
        uint32_t workload_runtime_id);
    // For a given MeshWorkload, a subgrid is unused if no programs are run on it.  Go signals
    // must be sent to this subgrid, to ensure consistent global state across the Virtual Mesh.
    // This function generates and writes dispatch commands forwarding go signals to these subgrids.
    void write_go_signal_to_unused_sub_grids(
        std::unordered_set<uint32_t>& chip_ids_in_workload,
        const SubDeviceId& sub_device_id,
        uint32_t expected_num_workers_completed,
        bool mcast_go_signals,
        bool unicast_go_signals);
    // Clear the num_workers_completed counter on the dispatcher cores corresponding to this CQ.
    void clear_expected_num_workers_completed();
    // Access a reference system memory manager, which acts as a global host side state manager for
    // specific MeshCommandQueue attributes.
    // TODO: All Mesh level host state managed by this class should be moved out, since its not
    // tied to system memory anyway. Move out:
    // 1. Event ID managment.
    // 2. Bypass mode tracker.
    SystemMemoryManager& reference_sysmem_manager();
    MultiProducerSingleConsumerQueue<CompletionReaderVariant>& get_read_descriptor_queue(IDevice* device);

    // Shared across all MeshCommandQueue instances for a MeshDevice.
    std::shared_ptr<DispatchArray<LaunchMessageRingBufferState>> worker_launch_message_buffer_state_;

    DispatchArray<uint32_t> expected_num_workers_completed_;
    DispatchArray<tt::tt_metal::WorkerConfigBufferMgr> config_buffer_mgr_;

    DispatchArray<LaunchMessageRingBufferState> worker_launch_message_buffer_state_reset_;
    DispatchArray<uint32_t> expected_num_workers_completed_reset_;
    DispatchArray<tt::tt_metal::WorkerConfigBufferMgr> config_buffer_mgr_reset_;

    // The following data structures are only popiulated when the MeshCQ is being used to trace workloads
    // i.e. between record_begin() and record_end() being called
    std::optional<MeshTraceId> trace_id_;
    std::shared_ptr<MeshTraceDescriptor> trace_ctx_;
    std::vector<MeshTraceStagingMetadata> ordered_mesh_trace_md_;

    CoreCoord dispatch_core_;
    CoreType dispatch_core_type_ = CoreType::WORKER;
    // MeshCommandQueues and the MeshDevice share thread-pools for dispatching to and reading from the Mesh
    std::shared_ptr<ThreadPool>
        reader_thread_pool_;  // Thread pool used to read from the Mesh (used by the Completion Queue Reader thread)

    // Member Vars used to control the execution of the Completion Queue Reader thread

    // TODO: Explore other thread-safe data-structures for these queues.
    // Main thread submits request to the completion queue reader through this queue
    MultiProducerSingleConsumerQueue<MeshCompletionReaderVariant> completion_queue_reads_;
    // Main thread pushes to a queue per physical device, specifying the buffer read configuration that
    // must be used by the completion queue reader. Only used for reading buffer data from the Mesh.
    std::unordered_map<uint32_t, std::unique_ptr<MultiProducerSingleConsumerQueue<CompletionReaderVariant>>>
        read_descriptors_;
    // CV used by main thread to notify completion queue reader of work
    std::condition_variable reader_thread_cv_;
    std::mutex reader_thread_cv_mutex_;
    // CV used by the completion queue reader to notify the main thread that all work is completed
    std::condition_variable reads_processed_cv_;
    std::mutex reads_processed_cv_mutex_;
    // Number of outstanding reads to be completed by the completion queue reader
    std::atomic<uint32_t> num_outstanding_reads_ = 0;
    // Exit signal for the completion queue reader
    std::atomic<bool> exit_condition_ = false;
    // Completion Queue Reader thread
    std::thread completion_queue_reader_thread_;
    // Global Mutex (used by both CQs) to safely use the reader_thread_pool_
    inline static std::mutex reader_thread_pool_mutex_;
    // Used to Maintain state: Mark/Check if this data structure is being used for dispatch.
    // This is temporary - will not be needed when we MeshCommandQueue is the only dispatch interface.
    bool in_use_ = false;

protected:
    void write_shard_to_device(
        Buffer* shard_view,
        const void* src,
        const BufferRegion& region,
        tt::stl::Span<const SubDeviceId> sub_device_ids = {}) override;
    void read_shard_from_device(
        Buffer* shard_view,
        void* dst,
        const BufferRegion& region,
        std::unordered_map<IDevice*, uint32_t>& num_txns_per_device,
        tt::stl::Span<const SubDeviceId> sub_device_ids = {}) override;
    void submit_memcpy_request(std::unordered_map<IDevice*, uint32_t>& num_txns_per_device, bool blocking) override;

public:
    FDMeshCommandQueue(
        MeshDevice* mesh_device,
        uint32_t id,
        std::shared_ptr<ThreadPool>& dispatch_thread_pool,
        std::shared_ptr<ThreadPool>& reader_thread_pool,
        std::shared_ptr<DispatchArray<LaunchMessageRingBufferState>>& worker_launch_message_buffer_state);

    ~FDMeshCommandQueue() override;

    WorkerConfigBufferMgr& get_config_buffer_mgr(uint32_t index) override { return config_buffer_mgr_[index]; };
    void enqueue_mesh_workload(MeshWorkload& mesh_workload, bool blocking) override;

    MeshEvent enqueue_record_event(
        tt::stl::Span<const SubDeviceId> sub_device_ids = {},
        const std::optional<MeshCoordinateRange>& device_range = std::nullopt) override;
    MeshEvent enqueue_record_event_to_host(
        tt::stl::Span<const SubDeviceId> sub_device_ids = {},
        const std::optional<MeshCoordinateRange>& device_range = std::nullopt) override;
    void enqueue_wait_for_event(const MeshEvent& sync_event) override;
    void drain_events_from_completion_queue();
    void verify_reported_events_after_draining(const MeshEvent& event);
    void finish(tt::stl::Span<const SubDeviceId> sub_device_ids = {}) override;
    void reset_worker_state(
        bool reset_launch_msg_state,
        uint32_t num_sub_devices,
        const vector_aligned<uint32_t>& go_signal_noc_data) override;
    void record_begin(const MeshTraceId& trace_id, const std::shared_ptr<MeshTraceDescriptor>& ctx) override;
    void record_end() override;
    void enqueue_trace(const MeshTraceId& trace_id, bool blocking) override;
    // Main function (event loop) for the Completion Queue Reader
    void read_completion_queue();
    // Helper function - read events from Completion Queue
    void read_completion_queue_event(MeshReadEventDescriptor& read_event_descriptor);
    // Helper function - read buffer data from Completion Queue
    void copy_buffer_data_to_user_space(MeshBufferReadDescriptor& read_buffer_descriptor);
};

}  // namespace tt::tt_metal::distributed
