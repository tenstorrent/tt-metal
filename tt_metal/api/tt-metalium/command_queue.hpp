// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <condition_variable>
#include <cstdint>
#include <memory>
#include <thread>

#include "worker_config_buffer.hpp"
#include "trace_buffer.hpp"
#include "memcpy.hpp"
#include "command_queue_interface.hpp"

namespace tt::tt_metal {

inline namespace v0 {
class Event;
class Program;
class Kernel;
}  // namespace v0

class CommandQueue {
public:
    virtual ~CommandQueue() = default;

    virtual const CoreCoord& virtual_enqueue_program_dispatch_core() const = 0;

    virtual void record_begin(const uint32_t tid, const std::shared_ptr<TraceDescriptor>& ctx) = 0;
    virtual void record_end() = 0;

    virtual void reset_worker_state(
        bool reset_launch_msg_state,
        uint32_t num_sub_devices,
        const vector_memcpy_aligned<uint32_t>& go_signal_noc_data) = 0;

    virtual void set_go_signal_noc_data_and_dispatch_sems(
        uint32_t num_dispatch_sems, const vector_memcpy_aligned<uint32_t>& noc_mcast_unicast_data) = 0;

    virtual uint32_t id() const = 0;
    virtual std::optional<uint32_t> tid() const = 0;

    virtual SystemMemoryManager& sysmem_manager() = 0;

    virtual void terminate() = 0;

    virtual IDevice* device() = 0;

    // This function is temporarily needed since MeshCommandQueue relies on the CommandQueue object
    virtual WorkerConfigBufferMgr& get_config_buffer_mgr(uint32_t index) = 0;

    virtual void enqueue_trace(const uint32_t trace_id, bool blocking) = 0;

    virtual void enqueue_program(Program& program, bool blocking) = 0;

    virtual void enqueue_read_buffer(
        const std::variant<std::reference_wrapper<Buffer>, std::shared_ptr<Buffer>>& buffer,
        void* dst,
        const BufferRegion& region,
        bool blocking,
        tt::stl::Span<const SubDeviceId> sub_device_ids = {}) = 0;

    virtual void enqueue_record_event(
        const std::shared_ptr<Event>& event, tt::stl::Span<const SubDeviceId> sub_device_ids = {}) = 0;

    virtual void enqueue_wait_for_event(const std::shared_ptr<Event>& sync_event) = 0;

    virtual void enqueue_write_buffer(
        const std::variant<std::reference_wrapper<Buffer>, std::shared_ptr<Buffer>>& buffer,
        HostDataType src,
        const BufferRegion& region,
        bool blocking,
        tt::stl::Span<const SubDeviceId> sub_device_ids = {}) = 0;

    virtual void finish(tt::stl::Span<const SubDeviceId> sub_device_ids) = 0;
};

struct ReadBufferDescriptor;
struct ReadEventDescriptor;

using CompletionReaderVariant = std::variant<std::monostate, ReadBufferDescriptor, ReadEventDescriptor>;

}  // namespace tt::tt_metal
