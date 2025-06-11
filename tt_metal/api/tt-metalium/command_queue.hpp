// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <condition_variable>
#include <cstdint>
#include <memory>
#include <thread>

#include <tt-metalium/command_queue_interface.hpp>

#include <tt-metalium/vector_aligned.hpp>

namespace tt::tt_metal {

class Event;
class Program;
class Kernel;
class TraceDescriptor;
class HWCommandQueue;

class CommandQueue {
public:
    virtual ~CommandQueue() = default;

    virtual void record_begin(const uint32_t tid, const std::shared_ptr<TraceDescriptor>& ctx) = 0;
    virtual void record_end() = 0;

    virtual uint32_t id() const = 0;

    virtual IDevice* device() = 0;

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

private:
    // Ensure only HWCommandQueue can inherit from CommandQueue.
    CommandQueue() = default;
    friend class HWCommandQueue;
};

struct ReadBufferDescriptor;
struct ReadEventDescriptor;
struct ReadCoreDataDescriptor;
using CompletionReaderVariant =
    std::variant<std::monostate, ReadBufferDescriptor, ReadEventDescriptor, ReadCoreDataDescriptor>;

}  // namespace tt::tt_metal
