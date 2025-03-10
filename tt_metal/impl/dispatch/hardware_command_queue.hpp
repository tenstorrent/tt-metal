// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <condition_variable>
#include <cstdint>
#include <memory>
#include <thread>

#include "command_queue.hpp"
#include "host_runtime_commands.hpp"
#include "command_queue_interface.hpp"
#include "multi_producer_single_consumer_queue.hpp"
#include "worker_config_buffer.hpp"
#include "program_impl.hpp"
#include "trace_buffer.hpp"

#include "tt_metal/impl/buffers/dispatch.hpp"

namespace tt::tt_metal {

class HWCommandQueue : public CommandQueue {
public:
    HWCommandQueue(IDevice* device, uint32_t id, NOC noc_index, uint32_t completion_queue_reader_core = 0);

    ~HWCommandQueue() override;

    const CoreCoord& virtual_enqueue_program_dispatch_core() const override;

    void record_begin(const uint32_t tid, const std::shared_ptr<TraceDescriptor>& ctx) override;
    void record_end() override;

    void reset_worker_state(
        bool reset_launch_msg_state,
        uint32_t num_sub_devices,
        const vector_memcpy_aligned<uint32_t>& go_signal_noc_data) override;

    void set_go_signal_noc_data_and_dispatch_sems(
        uint32_t num_dispatch_sems, const vector_memcpy_aligned<uint32_t>& noc_mcast_unicast_data) override;

    uint32_t id() const override;
    std::optional<uint32_t> tid() const override;

    SystemMemoryManager& sysmem_manager() override;

    void terminate() override;

    // This function is temporarily needed since MeshCommandQueue relies on the CommandQueue object
    WorkerConfigBufferMgr& get_config_buffer_mgr(uint32_t index) override;

    void enqueue_trace(const uint32_t trace_id, bool blocking) override;
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

    void finish(tt::stl::Span<const SubDeviceId> sub_device_ids) override;

    IDevice* device() override;

private:
    uint32_t id_;
    uint32_t size_B;
    uint32_t completion_queue_reader_core = 0;
    std::optional<uint32_t> tid_;
    std::shared_ptr<TraceDescriptor> trace_ctx;
    std::thread completion_queue_thread;
    SystemMemoryManager& manager;
    std::array<tt::tt_metal::WorkerConfigBufferMgr, DispatchSettings::DISPATCH_MESSAGE_ENTRIES> config_buffer_mgr;
    // Expected value of DISPATCH_MESSAGE_ADDR in dispatch core L1
    //  Value in L1 incremented by worker to signal completion to dispatch. Value on host is set on each enqueue program
    //  call
    std::array<uint32_t, DispatchSettings::DISPATCH_MESSAGE_ENTRIES> expected_num_workers_completed;

    volatile bool exit_condition;
    volatile uint32_t num_entries_in_completion_q;  // issue queue writer thread increments this when an issued command
                                                    // is expected back in the completion queue
    volatile uint32_t num_completed_completion_q_reads;  // completion queue reader thread increments this after reading
                                                         // an entry out of the completion queue

    MultiProducerSingleConsumerQueue<CompletionReaderVariant> issued_completion_q_reads;
    // These values are used to reset the host side launch message wptr after a trace is captured
    // Trace capture is a fully host side operation, but it modifies the state of the wptrs above
    // To ensure that host and device are not out of sync, we reset the wptrs to their original values
    // post trace capture.
    std::array<LaunchMessageRingBufferState, DispatchSettings::DISPATCH_MESSAGE_ENTRIES>
        worker_launch_message_buffer_state_reset;
    std::array<uint32_t, DispatchSettings::DISPATCH_MESSAGE_ENTRIES> expected_num_workers_completed_reset;
    std::array<tt::tt_metal::WorkerConfigBufferMgr, DispatchSettings::DISPATCH_MESSAGE_ENTRIES>
        config_buffer_mgr_reset;
    IDevice* device_;

    std::condition_variable reader_thread_cv;
    std::mutex reader_thread_cv_mutex;

    std::condition_variable reads_processed_cv;
    std::mutex reads_processed_cv_mutex;
    CoreType get_dispatch_core_type();

    CoreCoord virtual_enqueue_program_dispatch_core_;
    CoreCoord completion_queue_writer_core_;
    NOC noc_index_;

    void read_completion_queue();

    // sub_device_ids only needs to be passed when blocking and there are specific sub_devices to wait on
    template <typename T>
    void enqueue_command(T& command, bool blocking, tt::stl::Span<const SubDeviceId> sub_device_ids);

    void increment_num_entries_in_completion_q();
    void set_exit_condition();
};

}  // namespace tt::tt_metal
