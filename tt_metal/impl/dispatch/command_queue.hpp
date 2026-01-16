// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <cstdint>
#include <memory>
#include <variant>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/vector_aligned.hpp>
#include <tt-metalium/sub_device_types.hpp>
#include <tt-metalium/buffer.hpp>

#include <tt_stl/span.hpp>

namespace tt::tt_metal {

struct Event;
class Program;
class IDevice;
class SystemMemoryManager;
class WorkerConfigBufferMgr;
struct TraceDescriptor;

class CommandQueue {
public:
    virtual ~CommandQueue() = default;

    virtual const CoreCoord& virtual_enqueue_program_dispatch_core() const = 0;

    virtual void record_begin(uint32_t tid, const std::shared_ptr<TraceDescriptor>& ctx) = 0;
    virtual void record_end() = 0;

    virtual void set_go_signal_noc_data_and_dispatch_sems(
        uint32_t num_dispatch_sems, const vector_aligned<uint32_t>& noc_mcast_unicast_data) = 0;

    virtual uint32_t id() const = 0;
    virtual std::optional<uint32_t> tid() const = 0;

    virtual SystemMemoryManager& sysmem_manager() = 0;

    virtual IDevice* device() = 0;

    // This function is temporarily needed since MeshCommandQueue relies on the CommandQueue object
    virtual WorkerConfigBufferMgr& get_config_buffer_mgr(uint32_t index) = 0;

    // needed interface items
    virtual void terminate() = 0;
    virtual void finish(tt::stl::Span<const SubDeviceId> sub_device_ids) = 0;
};

struct ReadBufferDescriptor;
struct ReadEventDescriptor;
struct ReadCoreDataDescriptor;
using CompletionReaderVariant =
    std::variant<std::monostate, ReadBufferDescriptor, ReadEventDescriptor, ReadCoreDataDescriptor>;

}  // namespace tt::tt_metal
