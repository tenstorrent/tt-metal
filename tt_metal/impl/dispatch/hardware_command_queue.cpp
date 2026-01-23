// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "hardware_command_queue.hpp"

#include <device.hpp>
#include <event.hpp>
// Because we are a Friend of Program, accessing Program::get_program_transfer_info() and Program::get_kernels_buffer()
// MUST REMOVE
#include <tt-metalium/program.hpp>
#include "sub_device_types.hpp"
#include "trace/trace_buffer.hpp"
#include <tracy/Tracy.hpp>
#include <tt-metalium/allocator.hpp>
#include <tt_stl/overloaded.hpp>
#include <algorithm>
#include <unordered_map>
#include <utility>
#include <vector>

#include <tt_stl/assert.hpp>
#include "buffers/dispatch.hpp"
#include "cq_shared_state.hpp"
#include "device/dispatch.hpp"
#include "dispatch/device_command.hpp"
#include "dispatch_settings.hpp"
#include "impl/context/metal_context.hpp"
#include "dispatch/host_runtime_commands.hpp"
#include "event/dispatch.hpp"
#include "hal_types.hpp"
#include <tt-logger/tt-logger.hpp>
#include <tt_stl/strong_type.hpp>
#include "system_memory_manager.hpp"
#include "tt_metal/impl/program/dispatch.hpp"
#include "tt_metal/impl/trace/dispatch.hpp"
#include "tt_metal/impl/allocator/allocator.hpp"
#include <umd/device/types/xy_pair.hpp>
#include "data_collection.hpp"
#include "ringbuffer_cache.hpp"
#include "program/dispatch.hpp"
#include <tt-metalium/graph_tracking.hpp>
#include <impl/debug/dprint_server.hpp>
#include <impl/debug/watcher_server.hpp>
#include <impl/dispatch/dispatch_mem_map.hpp>

namespace tt::tt_metal {
enum NOC : uint8_t;
}  // namespace tt::tt_metal

namespace tt::tt_metal {
namespace {

// Binds a device worker/reader thread to a CPU core, determined using round-robin.
void set_device_thread_affinity(std::thread& thread_, int cpu_core_for_worker) {
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(cpu_core_for_worker, &cpuset);
    int rc = pthread_setaffinity_np(thread_.native_handle(), sizeof(cpu_set_t), &cpuset);
    if (rc) {
        log_warning(
            tt::LogMetal,
            "Unable to bind worker thread to CPU Core. May see performance degradation. Error Code: {}",
            rc);
    }
}

}  // namespace

HWCommandQueue::HWCommandQueue(
    IDevice* device,
    std::shared_ptr<CQSharedState> cq_shared_state,
    uint32_t id,
    NOC /*noc_index*/,
    uint32_t completion_queue_reader_core) :
    id_(id),

    completion_queue_reader_core_(completion_queue_reader_core),
    manager_(device->sysmem_manager()),
    cq_shared_state_(std::move(cq_shared_state)),
    num_entries_in_completion_q_(0),
    num_completed_completion_q_reads_(0),
    device_(device),
    prefetcher_dram_aligned_block_size_(MetalContext::instance().hal().get_alignment(HalMemType::DRAM)),
    prefetcher_cache_sizeB_(
        MetalContext::instance().dispatch_mem_map(this->get_dispatch_core_type()).ringbuffer_size()),
    prefetcher_dram_aligned_num_blocks_(prefetcher_cache_sizeB_ / prefetcher_dram_aligned_block_size_),
    prefetcher_cache_manager_size_(
        1 << (std::bit_width(std::min(1024u, std::max(2u, prefetcher_dram_aligned_num_blocks_ >> 4))) - 1)),
    prefetcher_cache_manager_(std::make_unique<RingbufferCacheManager>(
        prefetcher_dram_aligned_block_size_, prefetcher_dram_aligned_num_blocks_, prefetcher_cache_manager_size_)),
    dummy_prefetcher_cache_manager_(std::make_unique<RingbufferCacheManager>(
        prefetcher_dram_aligned_block_size_, prefetcher_dram_aligned_num_blocks_, prefetcher_cache_manager_size_)) {
    ZoneScopedN("CommandQueue_constructor");

    ChipId mmio_device_id =
        tt::tt_metal::MetalContext::instance().get_cluster().get_associated_mmio_device(device_->id());
    uint16_t channel =
        tt::tt_metal::MetalContext::instance().get_cluster().get_assigned_channel_for_device(device_->id());
    this->size_B_ =
        tt::tt_metal::MetalContext::instance().get_cluster().get_host_channel_size(mmio_device_id, channel) /
        device_->num_hw_cqs();
    if (tt::tt_metal::MetalContext::instance().get_cluster().is_galaxy_cluster()) {
        // Galaxy puts 4 devices per host channel until umd can provide one channel per device.
        this->size_B_ = this->size_B_ / 4;
    }

    CoreCoord enqueue_program_dispatch_core;
    CoreType core_type = MetalContext::instance().get_dispatch_core_manager().get_dispatch_core_type();
    if (this->device_->num_hw_cqs() == 1 or core_type == CoreType::WORKER) {
        // dispatch_s exists with this configuration. Workers write to dispatch_s
        enqueue_program_dispatch_core =
            MetalContext::instance().get_dispatch_core_manager().dispatcher_s_core(device_->id(), channel, id);
    } else {
        if (device_->is_mmio_capable()) {
            enqueue_program_dispatch_core =
                MetalContext::instance().get_dispatch_core_manager().dispatcher_core(device_->id(), channel, id);
        } else {
            enqueue_program_dispatch_core =
                MetalContext::instance().get_dispatch_core_manager().dispatcher_d_core(device_->id(), channel, id);
        }
    }
    this->virtual_enqueue_program_dispatch_core_ =
        device_->virtual_core_from_logical_core(enqueue_program_dispatch_core, core_type);

    tt_cxy_pair completion_q_writer_location =
        MetalContext::instance().get_dispatch_core_manager().completion_queue_writer_core(
            device_->id(), channel, this->id_);

    this->completion_queue_writer_core_ = CoreCoord(completion_q_writer_location.x, completion_q_writer_location.y);

    this->exit_condition_ = false;
    std::thread completion_queue_thread = std::thread(&HWCommandQueue::read_completion_queue, this);
    this->completion_queue_thread_ = std::move(completion_queue_thread);
    // Set the affinity of the completion queue reader.
    set_device_thread_affinity(this->completion_queue_thread_, this->completion_queue_reader_core_);
    program_dispatch::reset_config_buf_mgrs_and_expected_workers(
        this->config_buffer_mgr_,
        this->expected_num_workers_completed_,
        DispatchSettings::DISPATCH_MESSAGE_ENTRIES,
        device_->allocator_impl()->get_config().l1_unreserved_base);
}

uint32_t HWCommandQueue::id() const { return this->id_; }

std::optional<uint32_t> HWCommandQueue::tid() const { return this->tid_; }

SystemMemoryManager& HWCommandQueue::sysmem_manager() { return this->manager_; }

void HWCommandQueue::set_go_signal_noc_data_and_dispatch_sems(
    uint32_t num_dispatch_sems, const vector_aligned<uint32_t>& noc_mcast_unicast_data) {
    program_dispatch::set_num_worker_sems_on_dispatch(device_, this->manager_, id_, num_dispatch_sems);
    program_dispatch::set_go_signal_noc_data_on_dispatch(device_, noc_mcast_unicast_data, this->manager_, id_);
}

HWCommandQueue::~HWCommandQueue() {
    ZoneScopedN("HWCommandQueue_destructor");
    if (this->exit_condition_) {
        this->completion_queue_thread_.join();  // We errored out already prior
    } else {
        TT_ASSERT(
            this->issued_completion_q_reads_.empty(),
            "There should be no reads in flight after closing our completion queue thread");
        TT_ASSERT(
            this->num_entries_in_completion_q_ == this->num_completed_completion_q_reads_,
            "There shouldn't be any commands in flight after closing our completion queue thread. Num uncompleted "
            "commands: {}",
            this->num_entries_in_completion_q_ - this->num_completed_completion_q_reads_);
        this->set_exit_condition();
        this->completion_queue_thread_.join();
    }
}

void HWCommandQueue::increment_num_entries_in_completion_q() {
    // Increment num_entries_in_completion_q and inform reader thread
    // that there is work in the completion queue to process
    std::lock_guard lock(this->reader_thread_cv_mutex_);
    this->num_entries_in_completion_q_++;
    this->reader_thread_cv_.notify_one();
}

void HWCommandQueue::set_exit_condition() {
    std::lock_guard lock(this->reader_thread_cv_mutex_);
    this->exit_condition_ = true;
    this->reader_thread_cv_.notify_one();
}

IDevice* HWCommandQueue::device() { return this->device_; }

CoreType HWCommandQueue::get_dispatch_core_type() {
    return MetalContext::instance().get_dispatch_core_manager().get_dispatch_core_type();
}

void HWCommandQueue::enqueue_record_event(
    const std::shared_ptr<Event>& event, tt::stl::Span<const SubDeviceId> sub_device_ids) {
    ZoneScopedN("HWCommandQueue_enqueue_record_event");

    TT_FATAL(!this->manager_.get_bypass_mode(), "Enqueue Record Event cannot be used with tracing");

    // Populate event struct for caller. When async queues are enabled, this is in child thread, so consumers
    // of the event must wait for it to be ready (ie. populated) here. Set ready flag last. This couldn't be
    // in main thread otherwise event_id selection would get out of order due to main/worker thread timing.
    event->cq_id = this->id_;
    event->event_id = this->manager_.get_next_event(this->id_);
    event->device = this->device_;
    event->ready = true;

    sub_device_ids = buffer_dispatch::select_sub_device_ids(this->device_, sub_device_ids);
    event_dispatch::issue_record_event_commands(
        device_,
        device_->id(),
        event->event_id,
        id_,
        device_->num_hw_cqs(),
        this->manager_,
        sub_device_ids,
        this->expected_num_workers_completed_);
    this->issued_completion_q_reads_.push(
        std::make_shared<CompletionReaderVariant>(std::in_place_type<ReadEventDescriptor>, event->event_id));
    this->increment_num_entries_in_completion_q();

    auto& sub_device_cq_owner = cq_shared_state_->sub_device_cq_owner;
    for (const auto& sub_device_id : sub_device_ids) {
        auto& sub_device_entry = sub_device_cq_owner[*sub_device_id];
        sub_device_entry.recorded_event(event->event_id, event->cq_id);
    }
}

void HWCommandQueue::read_completion_queue() {
    ChipId mmio_device_id =
        tt::tt_metal::MetalContext::instance().get_cluster().get_associated_mmio_device(this->device_->id());
    uint16_t channel =
        tt::tt_metal::MetalContext::instance().get_cluster().get_assigned_channel_for_device(this->device_->id());
    while (true) {
        uint32_t num_events_to_read = 0;
        {
            std::unique_lock<std::mutex> lock(this->reader_thread_cv_mutex_);
            this->reader_thread_cv_.wait(lock, [this] {
                return this->num_entries_in_completion_q_ > this->num_completed_completion_q_reads_ or
                       this->exit_condition_;
            });
            if (this->num_entries_in_completion_q_ > this->num_completed_completion_q_reads_) {
                num_events_to_read = this->num_entries_in_completion_q_ - this->num_completed_completion_q_reads_;
            }
        }
        if (num_events_to_read > 0) {
            ZoneScopedN("CompletionQueueReader");
            for (uint32_t i = 0; i < num_events_to_read; i++) {
                ZoneScopedN("CompletionQueuePopulated");
                auto read_descriptor = *(this->issued_completion_q_reads_.pop());
                {
                    ZoneScopedN("CompletionQueueWait");
                    this->manager_.completion_queue_wait_front(
                        this->id_, this->exit_condition_);  // CQ DISPATCHER IS NOT HANDSHAKING WITH HOST RN
                }
                if (this->exit_condition_) {  // Early exit
                    return;
                }

                std::visit(
                    ttsl::overloaded{
                        [&, this](const ReadBufferDescriptor& read_descriptor) {
                            ZoneScopedN("CompletionQueueReadData");
                            buffer_dispatch::copy_completion_queue_data_into_user_space(
                                read_descriptor, mmio_device_id, channel, id_, manager_, exit_condition_);
                        },
                        [&, this](ReadEventDescriptor& read_descriptor) {
                            ZoneScopedN("CompletionQueueReadEvent");
                            event_dispatch::read_events_from_completion_queue(
                                read_descriptor, mmio_device_id, this->device_->id(), channel, id_, manager_);
                        },
                        [&, this](const ReadCoreDataDescriptor& read_descriptor) {
                            ZoneScopedN("CompletionQueueReadCoreData");
                            device_dispatch::read_core_data_from_completion_queue(
                                read_descriptor, mmio_device_id, channel, id_, manager_, exit_condition_);
                        },
                        [](std::monostate) {},
                    },
                    read_descriptor);
            }
            {
                std::unique_lock<std::mutex> lock(this->reads_processed_cv_mutex_);
                this->num_completed_completion_q_reads_ += num_events_to_read;
                this->reads_processed_cv_.notify_one();
            }
        } else if (this->exit_condition_) {
            return;
        }
    }
}

void HWCommandQueue::finish(tt::stl::Span<const SubDeviceId> /*sub_device_ids*/) {
    TT_FATAL(false, "HWCommandQueue::finish is disabled and should not be used.");
}

const CoreCoord& HWCommandQueue::virtual_enqueue_program_dispatch_core() const {
    return this->virtual_enqueue_program_dispatch_core_;
}

void HWCommandQueue::terminate() {
    ZoneScopedN("HWCommandQueue_terminate");
    TT_FATAL(!this->manager_.get_bypass_mode(), "Terminate cannot be used with tracing");
    log_debug(tt::LogDispatch, "Terminating dispatch kernels for command queue {}", this->id_);
    auto command = EnqueueTerminateCommand(this->id_, this->device_, this->manager_);
    command.process();
}

WorkerConfigBufferMgr& HWCommandQueue::get_config_buffer_mgr(uint32_t index) { return config_buffer_mgr_[index]; }

std::pair<bool, size_t> HWCommandQueue::query_prefetcher_cache(uint64_t pgm_id, uint32_t lengthB) {
    auto result = prefetcher_cache_manager_->get_cache_offset(pgm_id, lengthB);
    TT_FATAL(
        result.has_value(),
        "Prefetcher cache query failed. Cache size: {}, requested: {}",
        this->prefetcher_cache_manager_->get_cache_sizeB(),
        lengthB);
    return std::make_pair(result.value().is_cached, result.value().offset * this->prefetcher_dram_aligned_block_size_);
}

void HWCommandQueue::reset_prefetcher_cache_manager() { prefetcher_cache_manager_->reset(); }

int HWCommandQueue::get_prefetcher_cache_sizeB() const { return this->prefetcher_cache_manager_->get_cache_sizeB(); }

}  // namespace tt::tt_metal
