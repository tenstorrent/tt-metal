// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "mesh_command_queue_base.hpp"

#include "impl/dispatch/command_queue.hpp"

#include "tt_metal/common/multi_producer_single_consumer_queue.hpp"
#include "dispatch/cq_shared_state.hpp"
#include "dispatch/dispatch_settings.hpp"
#include "dispatch/launch_message_ring_buffer_state.hpp"
#include "dispatch/worker_config_buffer.hpp"
#include "mesh_trace.hpp"
#include "tt_metal/impl/dispatch/ringbuffer_cache.hpp"
#include "tt_metal/impl/program/dispatch.hpp"

// Forward declaration of the FDMeshCQTestAccessor class
// This is used to access the system memory manager from cq test fixtures
namespace tt::tt_metal::tt_dispatch_tests::Common {
class FDMeshCQTestAccessor;
}  // namespace tt::tt_metal::tt_dispatch_tests::Common

namespace tt::tt_metal::distributed {

struct MeshReadEventDescriptor;
struct MeshBufferReadDescriptor;
struct MeshCoreDataReadDescriptor;

using MeshCompletionReaderVariant =
    std::variant<MeshBufferReadDescriptor, MeshReadEventDescriptor, MeshCoreDataReadDescriptor>;

struct DeviceMemoryAddress {
    MeshCoordinate device_coord;
    CoreCoord virtual_core_coord;
    DeviceAddr address{};
};

class FDMeshCommandQueue final : public MeshCommandQueueBase {
private:
    // This class can now access private members of FDMeshCommandQueue
    // This is used to access the system memory manager from cq test fixtures
    friend class tt_dispatch_tests::Common::FDMeshCQTestAccessor;

    void populate_read_descriptor_queue();
    void populate_virtual_program_dispatch_core();
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
        uint32_t program_runtime_id);
    // Captures a dispatch command to reset the expected number of workers. Used when the worker
    // counter on the host overflows.
    void capture_expected_worker_count_reset_cmd(uint32_t previous_expected_workers, SubDeviceId sub_device);
    // For a given MeshWorkload, a subgrid is unused if no programs are run on it. Go signals
    // must be sent to this subgrid, to ensure consistent global state across the Virtual Mesh.
    // When running trace, the dispatch commands responsible for forwarding go signals must be
    // captured on these subgrids.
    void capture_go_signal_trace_on_unused_subgrids(
        const MeshCoordinateRangeSet& active_grids_set,
        const SubDeviceId& sub_device_id,
        uint32_t expected_num_workers_completed,
        bool mcast_go_signals,
        bool unicast_go_signals,
        const program_dispatch::ProgramDispatchMetadata& dispatch_md);
    // Workload dispatch utility functions
    // Write dispatch commands associated with running a program on a Virtual Mesh subgrid
    void write_program_cmds_to_subgrid(
        const MeshCoordinateRange& sub_grid,
        ProgramCommandSequence& program_cmd_seq,
        bool stall_first,
        bool stall_before_program,
        std::unordered_set<uint32_t>& chip_ids_in_workload,
        uint32_t program_runtime_id);
    // For a given MeshWorkload, a subgrid is unused if no programs are run on it.  Go signals
    // must be sent to this subgrid, to ensure consistent global state across the Virtual Mesh.
    // This function generates and writes dispatch commands forwarding go signals to these subgrids.
    void write_go_signal_to_unused_sub_grids(
        std::unordered_set<uint32_t>& chip_ids_in_workload,
        const SubDeviceId& sub_device_id,
        uint32_t expected_num_workers_completed,
        bool mcast_go_signals,
        bool unicast_go_signals,
        const program_dispatch::ProgramDispatchMetadata& dispatch_md);
    // When the device profiler is not enabled, launch messages are identical across all physical devices running the
    // same program, to reduce state managed on host. When the profiler is enabled, the host_assigned_id field in the
    // launch message must be unique across physical devices to accurately capture program execution time on host and
    // device. This API is responsible for updating the launch message before writing it to each device (see
    // tt_metal/api/tt-metalium/dev_msgs.h for a description of how the host_assigned_id field is generated).
    void update_launch_messages_for_device_profiler(
        ProgramCommandSequence& program_cmd_seq, uint32_t program_runtime_id, IDevice* device);
    // Clear the num_workers_completed counter on the dispatcher cores corresponding to this CQ.
    void clear_expected_num_workers_completed();
    // Access a reference system memory manager, which acts as a global host side state manager for
    // specific MeshCommandQueue attributes.
    // TODO: All Mesh level host state managed by this class should be moved out, since its not
    // tied to system memory anyway. Move out:
    // 1. Event ID management.
    // 2. Bypass mode tracker.
    SystemMemoryManager& reference_sysmem_manager();
    MultiProducerSingleConsumerQueue<CompletionReaderVariant>& get_read_descriptor_queue(IDevice* device);

    void submit_core_data_memcpy_request(
        const ReadCoreDataDescriptor& read_descriptor,
        const MeshCoordinate& device_coord,
        bool blocking,
        tt::stl::Span<const SubDeviceId> sub_device_ids = {});

    // Shared across all MeshCommandQueue instances for a MeshDevice.
    std::shared_ptr<CQSharedState> cq_shared_state_;

    DispatchArray<uint32_t> expected_num_workers_completed_{};
    DispatchArray<tt::tt_metal::WorkerConfigBufferMgr> config_buffer_mgr_;

    DispatchArray<LaunchMessageRingBufferState> worker_launch_message_buffer_state_reset_;
    DispatchArray<uint32_t> expected_num_workers_completed_reset_{};
    DispatchArray<tt::tt_metal::WorkerConfigBufferMgr> config_buffer_mgr_reset_;

    // The following data structures are only popiulated when the MeshCQ is being used to trace workloads
    // i.e. between record_begin() and record_end() being called
    std::optional<MeshTraceId> trace_id_;
    std::shared_ptr<MeshTraceDescriptor> trace_ctx_;
    std::vector<MeshTraceStagingMetadata> ordered_mesh_trace_md_;

    CoreCoord dispatch_core_;
    const CoreType dispatch_core_type_;
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
    std::atomic<bool> in_use_ = false;

    const uint32_t prefetcher_dram_aligned_block_size_;
    const uint64_t prefetcher_cache_sizeB_;
    const uint32_t prefetcher_dram_aligned_num_blocks_;
    const uint32_t prefetcher_cache_manager_size_;
    // The prefetcher cache manager is used to track the state of the prefetcher cache.
    std::unique_ptr<RingbufferCacheManager> prefetcher_cache_manager_;
    // The backup prefetcher cache manager is used to stash away the prefetcher cache state during trace recording.
    std::unique_ptr<RingbufferCacheManager> dummy_prefetcher_cache_manager_;

    // Used to define when the exception should be handled.
    // The goal is to not throw exceptions in loop and do it just once
    // Once this is set to true, whoever gets to the point to handle the exception,
    // it will and set this back to false, so no other thread tries to handle it themselves
    std::atomic<bool> should_handle_exception_{false};

    // Used to store the exception pointer.
    // Since the exception is captured inside a different thread that the main one,
    // we need to store it and let the main thread handle it
    // So python can catch it
    std::exception_ptr thread_exception_ptr_;
    // Exceptions are not compatible with std::atomic, so we need a muted to store it.
    // Since a reader thread will be setting this while the main thread will be handling it,
    // it must be thread safe
    std::mutex exception_mutex_;

    // We are in an unrecoverable state, we need to close as many processes as possible.
    // When this is true, we are generally breaking locks and doing a bit of cleaning
    // so the main thread can handle the exception
    std::atomic<bool> thread_exception_state_ = false;

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
    FDMeshCommandQueue(
        MeshDevice* mesh_device,
        uint32_t id,
        std::shared_ptr<ThreadPool>& dispatch_thread_pool,
        std::shared_ptr<ThreadPool>& reader_thread_pool,
        std::shared_ptr<CQSharedState>& cq_shared_state,
        std::function<std::lock_guard<std::mutex>()> lock_api_function);

    ~FDMeshCommandQueue() override;

    std::optional<MeshTraceId> trace_id() const override { return this->trace_id_; }

    WorkerConfigBufferMgr& get_config_buffer_mgr(uint32_t index) override { return config_buffer_mgr_[index]; };
    void enqueue_mesh_workload(MeshWorkload& mesh_workload, bool blocking) override;

    // TODO: This will error out for SD mesh command queues
    // - Need to add equivalent APIs for SD and expose via mesh command queue base or mesh command queue
    void enqueue_write_shard_to_core(
        DeviceMemoryAddress address,
        const void* src,
        uint32_t size_bytes,
        bool blocking,
        tt::stl::Span<const SubDeviceId> sub_device_ids = {});
    void enqueue_read_shard_from_core(
        DeviceMemoryAddress address,
        void* dst,
        uint32_t size_bytes,
        bool blocking,
        tt::stl::Span<const SubDeviceId> sub_device_ids = {});

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
        const vector_aligned<uint32_t>& go_signal_noc_data,
        const std::vector<std::pair<CoreRangeSet, uint32_t>>& core_go_message_mapping) override;
    void record_begin(const MeshTraceId& trace_id, const std::shared_ptr<MeshTraceDescriptor>& ctx) override;
    void record_end() override;
    void enqueue_trace(const MeshTraceId& trace_id, bool blocking) override;
    // Main function (event loop) for the Completion Queue Reader
    void read_completion_queue();
    // Helper function - read events from Completion Queue
    void read_completion_queue_event(MeshReadEventDescriptor& read_event_descriptor);
    // Helper function - read buffer data from Completion Queue
    void copy_buffer_data_to_user_space(MeshBufferReadDescriptor& read_buffer_descriptor);
    // Helper function - read L1 data from Completion Queue
    void read_l1_data_from_completion_queue(MeshCoreDataReadDescriptor& read_l1_data_descriptor);

    // Prefetcher Cache Manager APIs
    std::pair<bool, size_t> query_prefetcher_cache(uint64_t workload_id, uint32_t lengthB);
    void reset_prefetcher_cache_manager();
    int get_prefetcher_cache_sizeB() const;

    void wait_for_completion(bool reset_launch_msg_state) override;
    void finish_and_reset_in_use() override;
    bool in_use() override { return in_use_.load(); }
};

}  // namespace tt::tt_metal::distributed
