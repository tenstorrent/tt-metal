// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <condition_variable>
#include <memory>
#include <thread>

#include "command_queue_interface.hpp"
#include "lock_free_queue.hpp"
#include "worker_config_buffer.hpp"
#include "program_impl.hpp"
#include "trace_buffer.hpp"

namespace tt::tt_metal {
inline namespace v0 {

// Forward declarations for defining friend relations.
class CommandQueue;
class Event;

}  // namespace v0

namespace detail {

// Used so the host knows how to properly copy data into user space from the completion queue (in hugepages)
struct ReadBufferDescriptor {
    TensorMemoryLayout buffer_layout;
    uint32_t page_size;
    uint32_t padded_page_size;
    std::shared_ptr<const BufferPageMapping> buffer_page_mapping;
    void* dst;
    uint32_t dst_offset;
    uint32_t num_pages_read;
    uint32_t cur_dev_page_id;

    ReadBufferDescriptor(
        TensorMemoryLayout buffer_layout,
        uint32_t page_size,
        uint32_t padded_page_size,
        void* dst,
        uint32_t dst_offset,
        uint32_t num_pages_read,
        uint32_t cur_dev_page_id,
        const std::shared_ptr<const BufferPageMapping>& buffer_page_mapping = nullptr) :
        buffer_layout(buffer_layout),
        page_size(page_size),
        padded_page_size(padded_page_size),
        buffer_page_mapping(buffer_page_mapping),
        dst(dst),
        dst_offset(dst_offset),
        num_pages_read(num_pages_read),
        cur_dev_page_id(cur_dev_page_id) {}
};

// Used so host knows data in completion queue is just an event ID
struct ReadEventDescriptor {
    uint32_t event_id;
    uint32_t global_offset;

    explicit ReadEventDescriptor(uint32_t event) : event_id(event), global_offset(0) {}

    void set_global_offset(uint32_t offset) { global_offset = offset; }
    uint32_t get_global_event_id() { return global_offset + event_id; }
};

using CompletionReaderVariant = std::variant<std::monostate, ReadBufferDescriptor, ReadEventDescriptor>;

}  // namespace detail

class HWCommandQueue {
public:
    HWCommandQueue(IDevice* device, uint32_t id, NOC noc_index);

    ~HWCommandQueue();

    CoreCoord virtual_enqueue_program_dispatch_core;
    CoreCoord completion_queue_writer_core;
    NOC noc_index;
    volatile bool is_dprint_server_hung();
    volatile bool is_noc_hung();

    void record_begin(const uint32_t tid, std::shared_ptr<detail::TraceDescriptor> ctx);
    void record_end();
    void set_num_worker_sems_on_dispatch(uint32_t num_worker_sems);
    void set_go_signal_noc_data_on_dispatch(const vector_memcpy_aligned<uint32_t>& go_signal_noc_data);

    void reset_worker_state(
        bool reset_launch_msg_state,
        uint32_t num_sub_devices,
        const vector_memcpy_aligned<uint32_t>& go_signal_noc_data);

    uint32_t get_id() const;
    std::optional<uint32_t> get_tid() const;

    SystemMemoryManager& sysmem_manager();

    void terminate();

    // These functions are temporarily needed since MeshCommandQueue relies on the CommandQueue object
    uint32_t get_expected_num_workers_completed_for_sub_device(uint32_t sub_device_index) const;
    void set_expected_num_workers_completed_for_sub_device(uint32_t sub_device_index, uint32_t num_workers);
    WorkerConfigBufferMgr& get_config_buffer_mgr(uint32_t index);

private:
    uint32_t id;
    uint32_t size_B;
    std::optional<uint32_t> tid;
    std::shared_ptr<detail::TraceDescriptor> trace_ctx;
    std::thread completion_queue_thread;
    SystemMemoryManager& manager;
    std::array<tt::tt_metal::WorkerConfigBufferMgr, dispatch_constants::DISPATCH_MESSAGE_ENTRIES> config_buffer_mgr;
    // Expected value of DISPATCH_MESSAGE_ADDR in dispatch core L1
    //  Value in L1 incremented by worker to signal completion to dispatch. Value on host is set on each enqueue program
    //  call
    std::array<uint32_t, dispatch_constants::DISPATCH_MESSAGE_ENTRIES> expected_num_workers_completed;

    volatile bool exit_condition;
    volatile bool dprint_server_hang = false;
    volatile bool illegal_noc_txn_hang = false;
    volatile uint32_t num_entries_in_completion_q;  // issue queue writer thread increments this when an issued command
                                                    // is expected back in the completion queue
    volatile uint32_t num_completed_completion_q_reads;  // completion queue reader thread increments this after reading
                                                         // an entry out of the completion queue

    LockFreeQueue<detail::CompletionReaderVariant> issued_completion_q_reads;
    // These values are used to reset the host side launch message wptr after a trace is captured
    // Trace capture is a fully host side operation, but it modifies the state of the wptrs above
    // To ensure that host and device are not out of sync, we reset the wptrs to their original values
    // post trace capture.
    std::array<LaunchMessageRingBufferState, dispatch_constants::DISPATCH_MESSAGE_ENTRIES>
        worker_launch_message_buffer_state_reset;
    std::array<uint32_t, dispatch_constants::DISPATCH_MESSAGE_ENTRIES> expected_num_workers_completed_reset;
    std::array<tt::tt_metal::WorkerConfigBufferMgr, dispatch_constants::DISPATCH_MESSAGE_ENTRIES>
        config_buffer_mgr_reset;
    IDevice* device;

    std::condition_variable reader_thread_cv;
    std::mutex reader_thread_cv_mutex;

    std::condition_variable reads_processed_cv;
    std::mutex reads_processed_cv_mutex;
    CoreType get_dispatch_core_type();

    void reset_worker_dispatch_state_on_device(bool reset_launch_msg_state);
    void reset_config_buffer_mgr(const uint32_t num_entries);

    void read_completion_queue();

    // sub_device_ids only needs to be passed when blocking and there are specific sub_devices to wait on
    template <typename T>
    void enqueue_command(T& command, bool blocking, tt::stl::Span<const SubDeviceId> sub_device_ids);

    void enqueue_read_buffer(
        std::shared_ptr<Buffer>& buffer,
        void* dst,
        const BufferRegion& region,
        bool blocking,
        tt::stl::Span<const SubDeviceId> sub_device_ids = {});
    void enqueue_read_buffer(
        Buffer& buffer,
        void* dst,
        const BufferRegion& region,
        bool blocking,
        tt::stl::Span<const SubDeviceId> sub_device_ids = {});
    void enqueue_write_buffer(
        const std::variant<std::reference_wrapper<Buffer>, std::shared_ptr<Buffer>>& buffer,
        HostDataType src,
        const BufferRegion& region,
        bool blocking,
        tt::stl::Span<const SubDeviceId> sub_device_ids = {});
    void enqueue_write_buffer(
        Buffer& buffer,
        const void* src,
        const BufferRegion& region,
        bool blocking,
        tt::stl::Span<const SubDeviceId> sub_device_ids = {});
    void enqueue_program(Program& program, bool blocking);
    void enqueue_record_event(
        const std::shared_ptr<Event>& event,
        bool clear_count = false,
        tt::stl::Span<const SubDeviceId> sub_device_ids = {});
    void enqueue_wait_for_event(const std::shared_ptr<Event>& sync_event, bool clear_count = false);
    void enqueue_trace(const uint32_t trace_id, bool blocking);
    void finish(tt::stl::Span<const SubDeviceId> sub_device_ids);
    void increment_num_entries_in_completion_q();
    void set_exit_condition();

    friend void EnqueueTraceImpl(CommandQueue& cq, uint32_t trace_id, bool blocking);
    friend void EnqueueProgramImpl(CommandQueue& cq, Program& program, bool blocking);
    friend void EnqueueReadBufferImpl(
        CommandQueue& cq,
        std::variant<std::reference_wrapper<Buffer>, std::shared_ptr<Buffer>> buffer,
        void* dst,
        const BufferRegion& region,
        bool blocking);
    friend void EnqueueWriteBufferImpl(
        CommandQueue& cq,
        const std::variant<std::reference_wrapper<Buffer>, std::shared_ptr<Buffer>>& buffer,
        HostDataType src,
        const BufferRegion& region,
        bool blocking);
    friend void EnqueueGetBufferAddrImpl(void* dst_buf_addr, const Buffer* buffer);
    friend void EnqueueRecordEventImpl(
        CommandQueue& cq, const std::shared_ptr<Event>& event, tt::stl::Span<const SubDeviceId> sub_device_ids);
    friend void EnqueueWaitForEventImpl(CommandQueue& cq, const std::shared_ptr<Event>& event);
    friend void FinishImpl(CommandQueue& cq, tt::stl::Span<const SubDeviceId> sub_device_ids);
    friend CommandQueue;
    friend detail::Program_;
    friend void CaptureEnqueueProgram(CommandQueue& cq, Program& program, bool blocking);
};

}  // namespace tt::tt_metal
