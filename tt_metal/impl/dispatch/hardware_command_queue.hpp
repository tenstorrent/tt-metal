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
#include "core_coord.hpp"
#include "dispatch_settings.hpp"
#include "event.hpp"
#include "host_runtime_commands.hpp"
#include "launch_message_ring_buffer_state.hpp"
#include "tt-metalium/program.hpp"
#include <tt_stl/span.hpp>
#include "sub_device_types.hpp"
#include "trace/trace_buffer.hpp"
#include <umd/device/types/core_coordinates.hpp>
#include "vector_aligned.hpp"
#include "worker_config_buffer.hpp"
#include "tt_metal/impl/buffers/dispatch.hpp"
#include "tt_metal/common/multi_producer_single_consumer_queue.hpp"
#include "ringbuffer_cache.hpp"

namespace tt::tt_metal {
class IDevice;
class Program;
class SystemMemoryManager;
enum NOC : uint8_t;
struct Event;
struct TraceDescriptor;
}  // namespace tt::tt_metal

namespace tt::tt_metal {


class HWCommandQueue {
public:
    HWCommandQueue(
        IDevice* device,
        std::shared_ptr<CQSharedState> cq_shared_state,
        uint32_t id,
        NOC noc_index,
        uint32_t completion_queue_reader_core = 0);

    ~HWCommandQueue();

    const CoreCoord& virtual_enqueue_program_dispatch_core() const;

    void set_go_signal_noc_data_and_dispatch_sems(
        uint32_t num_dispatch_sems, const vector_aligned<uint32_t>& noc_mcast_unicast_data);

    uint32_t id() const;

    SystemMemoryManager& sysmem_manager();

    // This function is temporarily needed since MeshCommandQueue relies on the HWCommandQueue object
    WorkerConfigBufferMgr& get_config_buffer_mgr(uint32_t index);

    IDevice* device();

    // needed interface items
    void terminate();
    void finish(tt::stl::Span<const SubDeviceId> sub_device_ids);

private:
    uint32_t id_;
    uint32_t completion_queue_reader_core_ = 0;
    std::thread completion_queue_thread_;
    SystemMemoryManager& manager_;

    // Shared across all CommandQueue instances for a Device.
    std::shared_ptr<CQSharedState> cq_shared_state_;

    DispatchArray<tt::tt_metal::WorkerConfigBufferMgr> config_buffer_mgr_;
    // Expected value of DISPATCH_MESSAGE_ADDR in dispatch core L1
    //  Value in L1 incremented by worker to signal completion to dispatch. Value on host is set on each enqueue program
    //  call
    DispatchArray<uint32_t> expected_num_workers_completed_{};

    std::atomic<bool> exit_condition_;
    std::atomic<uint32_t> num_entries_in_completion_q_;  // issue queue writer thread increments this when an issued
                                                         // command is expected back in the completion queue
    std::atomic<uint32_t> num_completed_completion_q_reads_;  // completion queue reader thread increments this after
                                                              // reading an entry out of the completion queue

    MultiProducerSingleConsumerQueue<CompletionReaderVariant> issued_completion_q_reads_;
    IDevice* device_;

    std::condition_variable reader_thread_cv_;
    std::mutex reader_thread_cv_mutex_;

    std::condition_variable reads_processed_cv_;
    std::mutex reads_processed_cv_mutex_;
    CoreType get_dispatch_core_type();

    CoreCoord virtual_enqueue_program_dispatch_core_;

    const uint32_t prefetcher_dram_aligned_block_size_;
    const uint64_t prefetcher_cache_sizeB_;
    const uint32_t prefetcher_dram_aligned_num_blocks_;
    const uint32_t prefetcher_cache_manager_size_;
    // The prefetcher cache manager is used to track the state of the prefetcher cache.
    std::unique_ptr<RingbufferCacheManager> prefetcher_cache_manager_;

    void read_completion_queue();

    void increment_num_entries_in_completion_q();
    void set_exit_condition();

    std::pair<bool, size_t> query_prefetcher_cache(uint64_t pgm_id, uint32_t lengthB);

    void reset_prefetcher_cache_manager();

    void enqueue_record_event(const std::shared_ptr<Event>& event, tt::stl::Span<const SubDeviceId> sub_device_ids);
};

}  // namespace tt::tt_metal
