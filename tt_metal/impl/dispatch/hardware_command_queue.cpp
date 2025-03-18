// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "hardware_command_queue.hpp"

#include <device.hpp>
#include "dprint_server.hpp"
#include <event.hpp>
#include <tt_stl/overloaded.hpp>
#include <trace_buffer.hpp>
#include <tt-metalium/command_queue_interface.hpp>
#include <tt-metalium/dispatch_settings.hpp>

#include "tt_cluster.hpp"

#include "work_executor.hpp"

// Because we are a Friend of Program, accessing Program::get_program_transfer_info() and Program::get_kernels_buffer()
// MUST REMOVE
#include <program_impl.hpp>

#include "tt_metal/impl/debug/watcher_server.hpp"
#include "tt_metal/impl/program/dispatch.hpp"
#include "tt_metal/impl/trace/dispatch.hpp"
#include "tt_metal/impl/dispatch/dispatch_query_manager.hpp"

#include "rtoptions.hpp"

namespace tt::tt_metal {
namespace {

Buffer& get_buffer_object(const std::variant<std::reference_wrapper<Buffer>, std::shared_ptr<Buffer>>& buffer) {
    return std::visit(
        tt::stl::overloaded{
            [](const std::shared_ptr<Buffer>& b) -> Buffer& { return *b; },
            [](const std::reference_wrapper<Buffer>& b) -> Buffer& { return b.get(); }},
        buffer);
}

}  // namespace

HWCommandQueue::HWCommandQueue(
    IDevice* device,
    std::shared_ptr<DispatchArray<LaunchMessageRingBufferState>>& worker_launch_message_buffer_state,
    uint32_t id,
    NOC noc_index,
    uint32_t completion_queue_reader_core) :
    manager_(device->sysmem_manager()),
    completion_queue_thread_{},
    completion_queue_reader_core_(completion_queue_reader_core) {
    ZoneScopedN("CommandQueue_constructor");
    this->device_ = device;
    this->worker_launch_message_buffer_state_ = worker_launch_message_buffer_state;
    this->id_ = id;
    this->noc_index_ = noc_index;
    this->num_entries_in_completion_q_ = 0;
    this->num_completed_completion_q_reads_ = 0;

    chip_id_t mmio_device_id = tt::Cluster::instance().get_associated_mmio_device(device_->id());
    uint16_t channel = tt::Cluster::instance().get_assigned_channel_for_device(device_->id());
    this->size_B_ = tt::Cluster::instance().get_host_channel_size(mmio_device_id, channel) / device_->num_hw_cqs();
    if (tt::Cluster::instance().is_galaxy_cluster()) {
        // Galaxy puts 4 devices per host channel until umd can provide one channel per device.
        this->size_B_ = this->size_B_ / 4;
    }

    CoreCoord enqueue_program_dispatch_core;
    CoreType core_type = dispatch_core_manager::instance().get_dispatch_core_type();
    if (this->device_->num_hw_cqs() == 1 or core_type == CoreType::WORKER) {
        // dispatch_s exists with this configuration. Workers write to dispatch_s
        enqueue_program_dispatch_core = dispatch_core_manager::instance().dispatcher_s_core(device_->id(), channel, id);
    } else {
        if (device_->is_mmio_capable()) {
            enqueue_program_dispatch_core =
                dispatch_core_manager::instance().dispatcher_core(device_->id(), channel, id);
        } else {
            enqueue_program_dispatch_core =
                dispatch_core_manager::instance().dispatcher_d_core(device_->id(), channel, id);
        }
    }
    this->virtual_enqueue_program_dispatch_core_ =
        device_->virtual_core_from_logical_core(enqueue_program_dispatch_core, core_type);

    tt_cxy_pair completion_q_writer_location =
        dispatch_core_manager::instance().completion_queue_writer_core(device_->id(), channel, this->id_);

    this->completion_queue_writer_core_ = CoreCoord(completion_q_writer_location.x, completion_q_writer_location.y);

    this->exit_condition_ = false;
    std::thread completion_queue_thread = std::thread(&HWCommandQueue::read_completion_queue, this);
    this->completion_queue_thread_ = std::move(completion_queue_thread);
    // Set the affinity of the completion queue reader.
    set_device_thread_affinity(this->completion_queue_thread_, this->completion_queue_reader_core_);
    program_dispatch::reset_config_buf_mgrs_and_expected_workers(
        this->config_buffer_mgr_, this->expected_num_workers_completed_, DispatchSettings::DISPATCH_MESSAGE_ENTRIES);
}

uint32_t HWCommandQueue::id() const { return this->id_; }

std::optional<uint32_t> HWCommandQueue::tid() const { return this->tid_; }

SystemMemoryManager& HWCommandQueue::sysmem_manager() { return this->manager_; }

void HWCommandQueue::reset_worker_state(
    bool reset_launch_msg_state, uint32_t num_sub_devices, const vector_memcpy_aligned<uint32_t>& go_signal_noc_data) {
    TT_FATAL(!this->manager_.get_bypass_mode(), "Cannot reset worker state during trace capture");
    // TODO: This could be further optimized by combining all of these into a single prefetch entry
    // Currently each one will be pushed into its own prefetch entry
    program_dispatch::reset_worker_dispatch_state_on_device(
        device_,
        this->manager_,
        id_,
        this->virtual_enqueue_program_dispatch_core_,
        this->expected_num_workers_completed_,
        reset_launch_msg_state);
    program_dispatch::set_num_worker_sems_on_dispatch(device_, this->manager_, id_, num_sub_devices);
    program_dispatch::set_go_signal_noc_data_on_dispatch(device_, go_signal_noc_data, this->manager_, id_);
    // expected_num_workers_completed is reset on the dispatcher, as part of this step - this must be reflected
    // on host, along with the config_buf_manager being reset, since we wait for all programs across SubDevices
    // to complete as part of resetting the worker state
    program_dispatch::reset_config_buf_mgrs_and_expected_workers(
        this->config_buffer_mgr_, this->expected_num_workers_completed_, device_->num_sub_devices());
    if (reset_launch_msg_state) {
        std::for_each(
            this->worker_launch_message_buffer_state_->begin(),
            this->worker_launch_message_buffer_state_->begin() + num_sub_devices,
            std::mem_fn(&LaunchMessageRingBufferState::reset));
    }
}

void HWCommandQueue::set_go_signal_noc_data_and_dispatch_sems(
    uint32_t num_dispatch_sems, const vector_memcpy_aligned<uint32_t>& noc_mcast_unicast_data) {
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

template <typename T>
void HWCommandQueue::enqueue_command(T& command, bool blocking, tt::stl::Span<const SubDeviceId> sub_device_ids) {
    command.process();
    if (blocking) {
        this->finish(sub_device_ids);
    }
}

// Read buffer command is enqueued in the issue region and device writes requested buffer data into the completion
// region
void HWCommandQueue::enqueue_read_buffer(
    const std::variant<std::reference_wrapper<Buffer>, std::shared_ptr<Buffer>>& buffer,
    void* dst,
    const BufferRegion& region,
    bool blocking,
    tt::stl::Span<const SubDeviceId> sub_device_ids) {
    ZoneScopedN("HWCommandQueue_read_buffer");
    TT_FATAL(!this->manager_.get_bypass_mode(), "Enqueue Read Buffer cannot be used with tracing");
    Buffer& buffer_obj = get_buffer_object(buffer);
    sub_device_ids = buffer_dispatch::select_sub_device_ids(this->device_, sub_device_ids);

    if (is_sharded(buffer_obj.buffer_layout())) {
        // Forward data from each core to the completion queue.
        // Then have the completion queue reader thread copy this data to user space.
        auto dispatch_params = buffer_dispatch::initialize_sharded_buf_read_dispatch_params(
            buffer_obj, this->id_, this->expected_num_workers_completed_, region);
        auto cores = buffer_dispatch::get_cores_for_sharded_buffer(
            dispatch_params.width_split, dispatch_params.buffer_page_mapping, buffer_obj);
        for (uint32_t core_id = 0; core_id < buffer_obj.num_cores(); ++core_id) {
            buffer_dispatch::copy_sharded_buffer_from_core_to_completion_queue(
                core_id,
                buffer_obj,
                dispatch_params,
                sub_device_ids,
                cores[core_id],
                dispatch_core_manager::instance().get_dispatch_core_type());
            if (dispatch_params.pages_per_txn > 0) {
                this->issued_completion_q_reads_.push(
                    buffer_dispatch::generate_sharded_buffer_read_descriptor(dst, dispatch_params, buffer_obj));
                this->increment_num_entries_in_completion_q();
            }
        }
    } else {
        // Forward data from device to the completion queue.
        // Then have the completion queue reader thread copy this data to user space.
        buffer_dispatch::BufferReadDispatchParamsVariant dispatch_params_variant =
            buffer_dispatch::initialize_interleaved_buf_read_dispatch_params(
                buffer_obj, this->id_, this->expected_num_workers_completed_, region);

        buffer_dispatch::BufferReadDispatchParams* dispatch_params = std::visit(
            [](auto& val) { return static_cast<buffer_dispatch::BufferReadDispatchParams*>(&val); },
            dispatch_params_variant);

        buffer_dispatch::copy_interleaved_buffer_to_completion_queue(
            *dispatch_params, buffer_obj, sub_device_ids, dispatch_core_manager::instance().get_dispatch_core_type());
        if (dispatch_params->pages_per_txn > 0) {
            this->issued_completion_q_reads_.push(
                buffer_dispatch::generate_interleaved_buffer_read_descriptor(dst, dispatch_params, buffer_obj));
            this->increment_num_entries_in_completion_q();
        }
    }
    if (blocking) {
        this->finish(sub_device_ids);
    }
}

void HWCommandQueue::enqueue_write_buffer(
    const std::variant<std::reference_wrapper<Buffer>, std::shared_ptr<Buffer>>& buffer,
    HostDataType src,
    const BufferRegion& region,
    bool blocking,
    tt::stl::Span<const SubDeviceId> sub_device_ids) {
    ZoneScopedN("HWCommandQueue_write_buffer");
    TT_FATAL(!this->manager_.get_bypass_mode(), "Enqueue Write Buffer cannot be used with tracing");
    // Top level API to accept different variants for buffer and src
    // For shared pointer variants, object lifetime is guaranteed at least till the end of this function
    auto* data = std::visit(
        tt::stl::overloaded{
            [](const void* raw_data) -> const void* { return raw_data; },
            [](const auto& data) -> const void* { return data->data(); }},
        src);
    Buffer& buffer_obj = get_buffer_object(buffer);

    sub_device_ids = buffer_dispatch::select_sub_device_ids(this->device_, sub_device_ids);
    auto dispatch_core_type = dispatch_core_manager::instance().get_dispatch_core_type();

    buffer_dispatch::write_to_device_buffer(
        data, buffer_obj, region, this->id_, this->expected_num_workers_completed_, dispatch_core_type, sub_device_ids);

    if (blocking) {
        this->finish(sub_device_ids);
    }
}

CoreType HWCommandQueue::get_dispatch_core_type() { return dispatch_core_manager::instance().get_dispatch_core_type(); }

void HWCommandQueue::enqueue_program(Program& program, bool blocking) {
    ZoneScopedN("HWCommandQueue_enqueue_program");
    std::vector<SubDeviceId> sub_device_ids = {program.determine_sub_device_ids(device_)};
    TT_FATAL(sub_device_ids.size() == 1, "Programs must be executed on a single sub-device");
    // Finalize Program: Compute relative offsets for data structures (semaphores, kernel binaries, etc) in L1
    program_dispatch::finalize_program_offsets(program, device_);

    if (program.get_program_binary_status(device_->id()) == ProgramBinaryStatus::NotSent) {
        // Write program binaries to device if it hasn't previously been cached
        program.allocate_kernel_bin_buf_on_device(device_);
        if (program.get_program_transfer_info().binary_data.size()) {
            const BufferRegion buffer_region(0, program.get_kernels_buffer(device_)->size());
            this->enqueue_write_buffer(
                *program.get_kernels_buffer(device_),
                program.get_program_transfer_info().binary_data.data(),
                buffer_region,
                false);
        }
        program.set_program_binary_status(device_->id(), ProgramBinaryStatus::InFlight);
    }
    // Lower the program to device: Generate dispatch commands.
    // Values in these commands will get updated based on kernel config ring
    // buffer state at runtime.
    program.generate_dispatch_commands(device_);
    program.set_last_used_command_queue_for_testing(this);

#ifdef DEBUG
    if (tt::llrt::RunTimeOptions::get_instance().get_validate_kernel_binaries()) {
        TT_FATAL(!this->manager_.get_bypass_mode(), "Tracing cannot be used while validating program binaries");
        if (const auto buffer = program.get_kernels_buffer(device_)) {
            std::vector<uint32_t> read_data(buffer->page_size() * buffer->num_pages() / sizeof(uint32_t));
            const BufferRegion region(0, buffer->size());
            this->enqueue_read_buffer(*buffer, read_data.data(), region, true);
            TT_FATAL(
                program.get_program_transfer_info().binary_data == read_data,
                "Binary for program to be executed is corrupted. Another program likely corrupted this binary");
        }
    }
#endif
    auto sub_device_id = sub_device_ids[0];
    auto sub_device_index = *sub_device_id;

    // Snapshot of expected workers from previous programs, used for dispatch_wait cmd generation.
    uint32_t expected_workers_completed = this->manager_.get_bypass_mode()
                                              ? this->trace_ctx_->descriptors[sub_device_id].num_completion_worker_cores
                                              : this->expected_num_workers_completed_[sub_device_index];
    if (this->manager_.get_bypass_mode()) {
        if (program.runs_on_noc_multicast_only_cores()) {
            this->trace_ctx_->descriptors[sub_device_id].num_traced_programs_needing_go_signal_multicast++;
            this->trace_ctx_->descriptors[sub_device_id].num_completion_worker_cores +=
                device_->num_worker_cores(HalProgrammableCoreType::TENSIX, sub_device_id);
        }
        if (program.runs_on_noc_unicast_only_cores()) {
            this->trace_ctx_->descriptors[sub_device_id].num_traced_programs_needing_go_signal_unicast++;
            this->trace_ctx_->descriptors[sub_device_id].num_completion_worker_cores +=
                device_->num_worker_cores(HalProgrammableCoreType::ACTIVE_ETH, sub_device_id);
        }
    } else {
        if (program.runs_on_noc_multicast_only_cores()) {
            this->expected_num_workers_completed_[sub_device_index] +=
                device_->num_worker_cores(HalProgrammableCoreType::TENSIX, sub_device_id);
        }
        if (program.runs_on_noc_unicast_only_cores()) {
            this->expected_num_workers_completed_[sub_device_index] +=
                device_->num_worker_cores(HalProgrammableCoreType::ACTIVE_ETH, sub_device_id);
        }
    }

    auto& worker_launch_message_buffer_state = (*this->worker_launch_message_buffer_state_)[*sub_device_id];
    auto command = EnqueueProgramCommand(
        this->id_,
        this->device_,
        this->noc_index_,
        program,
        this->virtual_enqueue_program_dispatch_core_,
        this->manager_,
        this->get_config_buffer_mgr(sub_device_index),
        expected_workers_completed,
        // The assembled program command will encode the location of the launch messages in the ring buffer
        worker_launch_message_buffer_state.get_mcast_wptr(),
        worker_launch_message_buffer_state.get_unicast_wptr(),
        sub_device_id);
    // Update wptrs for tensix and eth launch message in the device class
    if (program.runs_on_noc_multicast_only_cores()) {
        worker_launch_message_buffer_state.inc_mcast_wptr(1);
    }
    if (program.runs_on_noc_unicast_only_cores()) {
        worker_launch_message_buffer_state.inc_unicast_wptr(1);
    }
    this->enqueue_command(command, blocking, sub_device_ids);

#ifdef DEBUG
    if (tt::llrt::RunTimeOptions::get_instance().get_validate_kernel_binaries()) {
        TT_FATAL(!this->manager_.get_bypass_mode(), "Tracing cannot be used while validating program binaries");
        if (const auto buffer = program.get_kernels_buffer(device_)) {
            std::vector<uint32_t> read_data(buffer->page_size() * buffer->num_pages() / sizeof(uint32_t));
            const BufferRegion region(0, buffer->size());
            this->enqueue_read_buffer(*buffer, read_data.data(), region, true);
            TT_FATAL(
                program.get_program_transfer_info().binary_data == read_data,
                "Binary for program that executed is corrupted. This program likely corrupted its own binary.");
        }
    }
#endif

    log_trace(
        tt::LogMetal,
        "Created EnqueueProgramCommand (active_cores: {} bypass_mode: {} expected_workers_completed: {})",
        program.get_program_transfer_info().num_active_cores,
        this->manager_.get_bypass_mode(),
        expected_workers_completed);
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
        event->event_id,
        id_,
        device_->num_hw_cqs(),
        this->manager_,
        sub_device_ids,
        this->expected_num_workers_completed_);
    this->issued_completion_q_reads_.push(
        std::make_shared<CompletionReaderVariant>(std::in_place_type<ReadEventDescriptor>, event->event_id));
    this->increment_num_entries_in_completion_q();
}

void HWCommandQueue::enqueue_wait_for_event(const std::shared_ptr<Event>& sync_event) {
    ZoneScopedN("HWCommandQueue_enqueue_wait_for_event");
    event_dispatch::issue_wait_for_event_commands(id_, sync_event->cq_id, this->manager_, sync_event->event_id);
}

void HWCommandQueue::enqueue_trace(const uint32_t trace_id, bool blocking) {
    ZoneScopedN("HWCommandQueue_enqueue_trace");

    auto trace_inst = this->device_->get_trace(trace_id);
    auto descriptor = trace_inst->desc;
    auto buffer = trace_inst->buffer;
    uint32_t num_sub_devices = descriptor->sub_device_ids.size();

    auto cmd_sequence_sizeB = trace_dispatch::compute_trace_cmd_size(num_sub_devices);

    trace_dispatch::TraceDispatchMetadata dispatch_md(
        cmd_sequence_sizeB,
        descriptor->descriptors,
        descriptor->sub_device_ids,
        buffer->page_size(),
        buffer->num_pages(),
        buffer->address());

    trace_dispatch::issue_trace_commands(
        device_,
        device_->sysmem_manager(),
        dispatch_md,
        id_,
        this->expected_num_workers_completed_,
        virtual_enqueue_program_dispatch_core_);

    trace_dispatch::update_worker_state_post_trace_execution(
        trace_inst->desc->descriptors,
        *this->worker_launch_message_buffer_state_,
        this->config_buffer_mgr_,
        this->expected_num_workers_completed_);

    if (blocking) {
        this->finish(trace_inst->desc->sub_device_ids);
    }
}

void HWCommandQueue::read_completion_queue() {
    chip_id_t mmio_device_id = tt::Cluster::instance().get_associated_mmio_device(this->device_->id());
    uint16_t channel = tt::Cluster::instance().get_assigned_channel_for_device(this->device_->id());
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
                    [&](auto&& read_descriptor) {
                        using T = std::decay_t<decltype(read_descriptor)>;
                        if constexpr (std::is_same_v<T, ReadBufferDescriptor>) {
                            ZoneScopedN("CompletionQueueReadData");
                            buffer_dispatch::copy_completion_queue_data_into_user_space(
                                read_descriptor,
                                mmio_device_id,
                                channel,
                                this->id_,
                                this->manager_,
                                this->exit_condition_);
                        } else if constexpr (std::is_same_v<T, ReadEventDescriptor>) {
                            ZoneScopedN("CompletionQueueReadEvent");
                            event_dispatch::read_events_from_completion_queue(
                                read_descriptor, mmio_device_id, channel, this->id_, this->manager_);
                        }
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

void HWCommandQueue::finish(tt::stl::Span<const SubDeviceId> sub_device_ids) {
    ZoneScopedN("HWCommandQueue_finish");
    tt::log_debug(tt::LogDispatch, "Finish for command queue {}", this->id_);
    std::shared_ptr<Event> event = std::make_shared<Event>();
    this->enqueue_record_event(event, sub_device_ids);
    if (tt::llrt::RunTimeOptions::get_instance().get_test_mode_enabled()) {
        while (this->num_entries_in_completion_q_ > this->num_completed_completion_q_reads_) {
            if (DPrintServerHangDetected()) {
                // DPrint Server hang, early exit. We're in test mode, so main thread will assert.
                this->set_exit_condition();
                return;
            } else if (tt::watcher_server_killed_due_to_error()) {
                // Illegal NOC txn killed watcher, early exit. We're in test mode, so main thread will assert.
                this->set_exit_condition();
                return;
            }
        }
    } else {
        std::unique_lock<std::mutex> lock(this->reads_processed_cv_mutex_);
        this->reads_processed_cv_.wait(
            lock, [this] { return this->num_entries_in_completion_q_ == this->num_completed_completion_q_reads_; });
    }
}

const CoreCoord& HWCommandQueue::virtual_enqueue_program_dispatch_core() const {
    return this->virtual_enqueue_program_dispatch_core_;
}

void HWCommandQueue::record_begin(const uint32_t tid, const std::shared_ptr<TraceDescriptor>& ctx) {
    // Clear host dispatch state, since when trace runs we will reset the launch_msg_ring_buffer,
    // worker_config_buffer, etc.
    trace_dispatch::reset_host_dispatch_state_for_trace(
        device_->num_sub_devices(),
        *this->worker_launch_message_buffer_state_,
        this->expected_num_workers_completed_,
        this->config_buffer_mgr_,
        this->worker_launch_message_buffer_state_reset_,
        this->expected_num_workers_completed_reset_,
        this->config_buffer_mgr_reset_);

    // Record commands using bypass mode
    this->tid_ = tid;
    this->trace_ctx_ = std::move(ctx);
    this->manager_.set_bypass_mode(true, true);  // start trace capture
}

void HWCommandQueue::record_end() {
    auto& trace_data = this->trace_ctx_->data;
    trace_data = std::move(this->manager_.get_bypass_data());
    // Add trace end command to terminate the trace buffer
    DeviceCommand command_sequence(hal.get_alignment(HalMemType::HOST));
    command_sequence.add_prefetch_exec_buf_end();
    for (int i = 0; i < command_sequence.size_bytes() / sizeof(uint32_t); i++) {
        trace_data.push_back(((uint32_t*)command_sequence.data())[i]);
    }
    // Copy the desc keys into a separate vector. When enqueuing traces, we sometimes need to pass sub-device ids
    // separately
    this->trace_ctx_->sub_device_ids.reserve(this->trace_ctx_->descriptors.size());
    for (const auto& [id, _] : this->trace_ctx_->descriptors) {
        auto index = *id;
        this->trace_ctx_->sub_device_ids.push_back(id);
    }
    this->tid_ = std::nullopt;
    this->trace_ctx_ = nullptr;

    // Reset the expected workers, launch msg buffer state, and config buffer mgr to their original value,
    // so device can run programs after a trace was captured. This is needed since trace capture modifies the state on
    // host, even though device doesn't run any programs.
    trace_dispatch::load_host_dispatch_state(
        device_->num_sub_devices(),
        *this->worker_launch_message_buffer_state_,
        this->expected_num_workers_completed_,
        this->config_buffer_mgr_,
        this->worker_launch_message_buffer_state_reset_,
        this->expected_num_workers_completed_reset_,
        this->config_buffer_mgr_reset_);
    this->manager_.set_bypass_mode(false, true);  // stop trace capture
}

void HWCommandQueue::terminate() {
    ZoneScopedN("HWCommandQueue_terminate");
    TT_FATAL(!this->manager_.get_bypass_mode(), "Terminate cannot be used with tracing");
    tt::log_debug(tt::LogDispatch, "Terminating dispatch kernels for command queue {}", this->id_);
    auto command = EnqueueTerminateCommand(this->id_, this->device_, this->manager_);
    this->enqueue_command(command, false, {});
}

WorkerConfigBufferMgr& HWCommandQueue::get_config_buffer_mgr(uint32_t index) { return config_buffer_mgr_[index]; }

}  // namespace tt::tt_metal
