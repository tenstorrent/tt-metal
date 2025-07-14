// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <thread>
#include <variant>

#include "buffer.hpp"
#include "cq_shared_state.hpp"
#include "command_queue.hpp"
#include "command_queue_interface.hpp"
#include "core_coord.hpp"
#include "dispatch_settings.hpp"
#include "event.hpp"
#include "host_runtime_commands.hpp"
#include "launch_message_ring_buffer_state.hpp"
#include "tt-metalium/program.hpp"
#include <tt_stl/span.hpp>
#include "sub_device_types.hpp"
#include "trace/trace_buffer.hpp"
#include <umd/device/tt_core_coordinates.h>
#include "vector_aligned.hpp"
#include "worker_config_buffer.hpp"
#include "trace/trace_node.hpp"
#include "tt_metal/impl/buffers/dispatch.hpp"
#include "tt_metal/common/multi_producer_single_consumer_queue.hpp"
#include "ringbuffer_cache.hpp"

namespace tt {
namespace tt_metal {
class IDevice;
class Program;
class SystemMemoryManager;
enum NOC : uint8_t;
}  // namespace tt_metal
}  // namespace tt

namespace tt::tt_metal {

class HWCommandQueue : public CommandQueue {
public:
    HWCommandQueue(
        IDevice* device,
        std::shared_ptr<CQSharedState> cq_shared_state,
        uint32_t id,
        NOC noc_index,
        uint32_t completion_queue_reader_core = 0);

    ~HWCommandQueue() override;

    const CoreCoord& virtual_enqueue_program_dispatch_core() const override;

    void record_begin(uint32_t tid, const std::shared_ptr<TraceDescriptor>& ctx) override;
    void record_end() override;

    void reset_worker_state(
        bool reset_launch_msg_state,
        uint32_t num_sub_devices,
        const vector_aligned<uint32_t>& go_signal_noc_data,
        const std::vector<std::pair<CoreRangeSet, uint32_t>>& core_go_message_mapping) override;

    void set_go_signal_noc_data_and_dispatch_sems(
        uint32_t num_dispatch_sems, const vector_aligned<uint32_t>& noc_mcast_unicast_data) override;

    uint32_t id() const override;
    std::optional<uint32_t> tid() const override;

    SystemMemoryManager& sysmem_manager() override;

    void terminate() override;

    // This function is temporarily needed since MeshCommandQueue relies on the CommandQueue object
    WorkerConfigBufferMgr& get_config_buffer_mgr(uint32_t index) override;

    void enqueue_trace(uint32_t trace_id, bool blocking) override;
    void enqueue_program(Program& program, bool blocking) override;
    void enqueue_read_buffer(
        const std::variant<std::reference_wrapper<Buffer>, std::shared_ptr<Buffer>>& buffer,
        void* dst,
        const BufferRegion& region,
        bool blocking,
        tt::stl::Span<const SubDeviceId> sub_device_ids = {}) override;

    void enqueue_record_event(
        const std::shared_ptr<Event>& event, tt::stl::Span<const SubDeviceId> sub_device_ids = {}) override;
    void enqueue_wait_for_event(const std::shared_ptr<Event>& sync_event) override;

    void enqueue_write_buffer(
        const std::variant<std::reference_wrapper<Buffer>, std::shared_ptr<Buffer>>& buffer,
        HostDataType src,
        const BufferRegion& region,
        bool blocking,
        tt::stl::Span<const SubDeviceId> sub_device_ids = {}) override;

    void enqueue_read_from_core(
        const CoreCoord& virtual_core,
        void* dst,
        DeviceAddr address,
        uint32_t size_bytes,
        bool blocking,
        tt::stl::Span<const SubDeviceId> sub_device_ids = {});

    void enqueue_write_to_core(
        const CoreCoord& virtual_core,
        const void* src,
        DeviceAddr address,
        uint32_t size_bytes,
        bool blocking,
        tt::stl::Span<const SubDeviceId> sub_device_ids = {});

    void finish(tt::stl::Span<const SubDeviceId> sub_device_ids) override;

    IDevice* device() override;

private:
    uint32_t id_;
    uint32_t size_B_;
    uint32_t completion_queue_reader_core_ = 0;
    std::optional<uint32_t> tid_;
    std::shared_ptr<TraceDescriptor> trace_ctx_;
    std::thread completion_queue_thread_;
    SystemMemoryManager& manager_;

    std::vector<TraceNode> trace_nodes_;

    // Shared across all CommandQueue instances for a Device.
    std::shared_ptr<CQSharedState> cq_shared_state_;

    DispatchArray<tt::tt_metal::WorkerConfigBufferMgr> config_buffer_mgr_;
    // Expected value of DISPATCH_MESSAGE_ADDR in dispatch core L1
    //  Value in L1 incremented by worker to signal completion to dispatch. Value on host is set on each enqueue program
    //  call
    DispatchArray<uint32_t> expected_num_workers_completed_;

    std::atomic<bool> exit_condition_;
    std::atomic<uint32_t> num_entries_in_completion_q_;  // issue queue writer thread increments this when an issued
                                                         // command is expected back in the completion queue
    std::atomic<uint32_t> num_completed_completion_q_reads_;  // completion queue reader thread increments this after
                                                              // reading an entry out of the completion queue

    MultiProducerSingleConsumerQueue<CompletionReaderVariant> issued_completion_q_reads_;
    // These values are used to reset the host side launch message wptr after a trace is captured
    // Trace capture is a fully host side operation, but it modifies the state of the wptrs above
    // To ensure that host and device are not out of sync, we reset the wptrs to their original values
    // post trace capture.
    DispatchArray<LaunchMessageRingBufferState> worker_launch_message_buffer_state_reset_;
    DispatchArray<uint32_t> expected_num_workers_completed_reset_;
    DispatchArray<tt::tt_metal::WorkerConfigBufferMgr> config_buffer_mgr_reset_;
    IDevice* device_;

    std::condition_variable reader_thread_cv_;
    std::mutex reader_thread_cv_mutex_;

    std::condition_variable reads_processed_cv_;
    std::mutex reads_processed_cv_mutex_;
    CoreType get_dispatch_core_type();

    CoreCoord virtual_enqueue_program_dispatch_core_;
    CoreCoord completion_queue_writer_core_;
    NOC noc_index_;

    const uint32_t prefetcher_dram_aligned_block_size_;
    const uint64_t prefetcher_cache_sizeB_;
    const uint32_t prefetcher_dram_aligned_num_blocks_;
    const uint32_t prefetcher_cache_manager_size_;
    // The prefetcher cache manager is used to track the state of the prefetcher cache.
    std::unique_ptr<RingbufferCacheManager> prefetcher_cache_manager_;

    // The backup prefetcher cache manager is used to stash away the prefetcher cache state during trace recording.
    // Trace recording will change the state of the host side cache manager, without actually enqueueing the
    // corresponding commands, which would cause the bookkeeping to go out of sync from the prefetcher cache. Hence we
    // will use the following variable to swap out the cache manager into a backup variable before starting trace
    // recording. At the end of the recording, we will reset the cache manager, and swap it with the backup.
    std::unique_ptr<RingbufferCacheManager> dummy_prefetcher_cache_manager_;

    void allocate_trace_programs();
    void read_completion_queue();

    // sub_device_ids only needs to be passed when blocking and there are specific sub_devices to wait on
    template <typename T>
    void enqueue_command(T& command, bool blocking, tt::stl::Span<const SubDeviceId> sub_device_ids);

    void increment_num_entries_in_completion_q();
    void set_exit_condition();

    std::pair<bool, size_t> query_prefetcher_cache(uint64_t pgm_id, uint32_t lengthB);

    void reset_prefetcher_cache_manager();

    int get_prefetcher_cache_sizeB() const;
};

}  // namespace tt::tt_metal
