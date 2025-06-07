// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "host_runtime_commands.hpp"

#include <assert.hpp>
#include <buffer.hpp>
#include <event.hpp>
#include <host_api.hpp>
#include <tt-logger/tt-logger.hpp>
#include <tt_metal.hpp>
#include <chrono>
#include <fstream>
#include <functional>
#include <memory>
#include <string>
#include <thread>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

#include "command_queue.hpp"
#include "device.hpp"
#include "dispatch/device_command.hpp"
#include "impl/context/metal_context.hpp"
#include "dprint_server.hpp"
#include "hal_types.hpp"
#include "lightmetal/host_api_capture_helpers.hpp"
#include "tt-metalium/program.hpp"
#include <tt_stl/span.hpp>
#include "system_memory_manager.hpp"
#include "tracy/Tracy.hpp"
#include "tt_metal/impl/debug/watcher_server.hpp"
#include "tt_metal/impl/dispatch/data_collection.hpp"
#include "tt_metal/impl/dispatch/kernels/cq_commands.hpp"
#include "tt_metal/impl/program/dispatch.hpp"
#include "tt_metal/impl/program/program_command_sequence.hpp"

namespace tt {
namespace tt_metal {
class WorkerConfigBufferMgr;
enum NOC : uint8_t;
}  // namespace tt_metal
}  // namespace tt

using namespace tt::tt_metal;

namespace tt::tt_metal {

namespace detail {

bool DispatchStateCheck(bool isFastDispatch) {
    static bool fd = isFastDispatch;
    TT_FATAL(fd == isFastDispatch, "Mixing fast and slow dispatch is prohibited!");
    return fd;
}

Buffer& GetBufferObject(const std::variant<std::reference_wrapper<Buffer>, std::shared_ptr<Buffer>>& buffer) {
    return std::visit(
        [&](auto&& b) -> Buffer& {
            using type_buf = std::decay_t<decltype(b)>;
            if constexpr (std::is_same_v<type_buf, std::shared_ptr<Buffer>>) {
                return *b;
            } else {
                return b.get();
            }
        },
        buffer);
}

void ValidateBufferRegion(
    const std::variant<std::reference_wrapper<Buffer>, std::shared_ptr<Buffer>>& buffer, const BufferRegion& region) {
    Buffer& buffer_obj = GetBufferObject(buffer);

    TT_FATAL(
        buffer_obj.is_valid_region(region),
        "Buffer region with offset {} and size {} is invalid.",
        region.offset,
        region.size);
}
}  // namespace detail

inline uint32_t get_packed_write_max_unicast_sub_cmds(IDevice* device) {
    return device->compute_with_storage_grid_size().x * device->compute_with_storage_grid_size().y;
}

EnqueueProgramCommand::EnqueueProgramCommand(
    uint32_t command_queue_id,
    IDevice* device,
    NOC noc_index,
    Program& program,
    CoreCoord& dispatch_core,
    SystemMemoryManager& manager,
    WorkerConfigBufferMgr& config_buffer_mgr,
    uint32_t expected_num_workers_completed,
    uint32_t multicast_cores_launch_message_wptr,
    uint32_t unicast_cores_launch_message_wptr,
    SubDeviceId sub_device_id) :
    command_queue_id(command_queue_id),
    noc_index(noc_index),
    manager(manager),
    config_buffer_mgr(config_buffer_mgr),
    expected_num_workers_completed(expected_num_workers_completed),
    program(program),
    dispatch_core(dispatch_core),
    multicast_cores_launch_message_wptr(multicast_cores_launch_message_wptr),
    unicast_cores_launch_message_wptr(unicast_cores_launch_message_wptr),
    sub_device_id(sub_device_id) {
    this->device = device;
    this->dispatch_core_type = MetalContext::instance().get_dispatch_core_manager().get_dispatch_core_type();
    this->packed_write_max_unicast_sub_cmds = get_packed_write_max_unicast_sub_cmds(this->device);
}

void EnqueueProgramCommand::process() {
    // Dispatch metadata contains runtime information based on
    // the kernel config ring buffer state
    program_dispatch::ProgramDispatchMetadata dispatch_metadata;

    // Compute the total number of workers this program uses
    uint32_t num_workers = 0;
    if (program.runs_on_noc_multicast_only_cores()) {
        num_workers += device->num_worker_cores(HalProgrammableCoreType::TENSIX, this->sub_device_id);
    }
    if (program.runs_on_noc_unicast_only_cores()) {
        num_workers += device->num_worker_cores(HalProgrammableCoreType::ACTIVE_ETH, this->sub_device_id);
    }
    // Reserve space for this program in the kernel config ring buffer
    program_dispatch::reserve_space_in_kernel_config_buffer(
        this->config_buffer_mgr,
        program.impl().get_program_config_sizes(),
        program.get_program_binary_status(device->id()),
        num_workers,
        this->expected_num_workers_completed,
        dispatch_metadata);

    RecordProgramRun(program.get_id());

    // Access the program dispatch-command cache
    uint64_t command_hash = *device->get_active_sub_device_manager_id();
    auto& cached_program_command_sequence = program.get_cached_program_command_sequences().at(command_hash);
    // Update the generated dispatch commands based on the state of the CQ and the ring buffer
    program_dispatch::update_program_dispatch_commands(
        program.impl(),
        cached_program_command_sequence,
        this->multicast_cores_launch_message_wptr,
        this->unicast_cores_launch_message_wptr,
        this->expected_num_workers_completed,
        this->dispatch_core,
        this->dispatch_core_type,
        this->sub_device_id,
        dispatch_metadata,
        program.get_program_binary_status(device->id()));
    // Issue dispatch commands for this program
    program_dispatch::write_program_command_sequence(
        cached_program_command_sequence,
        this->manager,
        this->command_queue_id,
        this->dispatch_core_type,
        dispatch_metadata.stall_first,
        dispatch_metadata.stall_before_program);
    // Kernel Binaries are committed to DRAM, the first time the program runs on device. Reflect this on host.
    program.set_program_binary_status(device->id(), ProgramBinaryStatus::Committed);
}

EnqueueTerminateCommand::EnqueueTerminateCommand(
    uint32_t command_queue_id, IDevice* device, SystemMemoryManager& manager) :
    command_queue_id(command_queue_id), device(device), manager(manager) {}

void EnqueueTerminateCommand::process() {
    // CQ_PREFETCH_CMD_RELAY_INLINE + CQ_DISPATCH_CMD_TERMINATE
    // CQ_PREFETCH_CMD_TERMINATE
    uint32_t cmd_sequence_sizeB = MetalContext::instance().hal().get_alignment(HalMemType::HOST);

    // dispatch and prefetch terminate commands each needs to be a separate fetch queue entry
    void* cmd_region = this->manager.issue_queue_reserve(cmd_sequence_sizeB, this->command_queue_id);
    HugepageDeviceCommand dispatch_d_command_sequence(cmd_region, cmd_sequence_sizeB);
    dispatch_d_command_sequence.add_dispatch_terminate(DispatcherSelect::DISPATCH_MASTER);
    this->manager.issue_queue_push_back(cmd_sequence_sizeB, this->command_queue_id);
    this->manager.fetch_queue_reserve_back(this->command_queue_id);
    this->manager.fetch_queue_write(cmd_sequence_sizeB, this->command_queue_id);
    if (MetalContext::instance().get_dispatch_query_manager().dispatch_s_enabled()) {
        // Terminate dispatch_s if enabled
        cmd_region = this->manager.issue_queue_reserve(cmd_sequence_sizeB, this->command_queue_id);
        HugepageDeviceCommand dispatch_s_command_sequence(cmd_region, cmd_sequence_sizeB);
        dispatch_s_command_sequence.add_dispatch_terminate(DispatcherSelect::DISPATCH_SUBORDINATE);
        this->manager.issue_queue_push_back(cmd_sequence_sizeB, this->command_queue_id);
        this->manager.fetch_queue_reserve_back(this->command_queue_id);
        this->manager.fetch_queue_write(cmd_sequence_sizeB, this->command_queue_id);
    }
    cmd_region = this->manager.issue_queue_reserve(cmd_sequence_sizeB, this->command_queue_id);
    HugepageDeviceCommand prefetch_command_sequence(cmd_region, cmd_sequence_sizeB);
    prefetch_command_sequence.add_prefetch_terminate();
    this->manager.issue_queue_push_back(cmd_sequence_sizeB, this->command_queue_id);
    this->manager.fetch_queue_reserve_back(this->command_queue_id);
    this->manager.fetch_queue_write(cmd_sequence_sizeB, this->command_queue_id);
}

void EnqueueWriteBuffer(
    CommandQueue& cq,
    const std::variant<std::reference_wrapper<Buffer>, std::shared_ptr<Buffer>>& buffer,
    std::vector<uint32_t>& src,
    bool blocking) {
    // TODO(agrebenisan): Move to deprecated
    EnqueueWriteBuffer(cq, std::move(buffer), src.data(), blocking);
}

void EnqueueReadBuffer(
    CommandQueue& cq,
    const std::variant<std::reference_wrapper<Buffer>, std::shared_ptr<Buffer>>& buffer,
    void* dst,
    bool blocking) {
    LIGHT_METAL_TRACE_FUNCTION_ENTRY();
    LIGHT_METAL_TRACE_FUNCTION_CALL(CaptureEnqueueReadBuffer, cq, buffer, dst, blocking);
    Buffer& buffer_obj = detail::GetBufferObject(buffer);
    BufferRegion region(0, buffer_obj.size());
    EnqueueReadSubBuffer(cq, buffer, dst, region, blocking);
}

void EnqueueReadSubBuffer(
    CommandQueue& cq,
    std::variant<std::reference_wrapper<Buffer>, std::shared_ptr<Buffer>> buffer,
    void* dst,
    const BufferRegion& region,
    bool blocking) {
    detail::DispatchStateCheck(true);
    detail::ValidateBufferRegion(buffer, region);

    std::visit(
        [&](auto&& b) {
            using T = std::decay_t<decltype(b)>;
            if constexpr (
                std::is_same_v<T, std::reference_wrapper<Buffer>> || std::is_same_v<T, std::shared_ptr<Buffer>>) {
                cq.enqueue_read_buffer(b, dst, region, blocking);
            }
        },
        buffer);
}

void EnqueueWriteBuffer(
    CommandQueue& cq,
    const std::variant<std::reference_wrapper<Buffer>, std::shared_ptr<Buffer>>& buffer,
    HostDataType src,
    bool blocking) {
    LIGHT_METAL_TRACE_FUNCTION_ENTRY();
    LIGHT_METAL_TRACE_FUNCTION_CALL(CaptureEnqueueWriteBuffer, cq, buffer, src, blocking);
    Buffer& buffer_obj = detail::GetBufferObject(buffer);
    BufferRegion region(0, buffer_obj.size());
    EnqueueWriteSubBuffer(cq, buffer, std::move(src), region, blocking);
}

void EnqueueWriteSubBuffer(
    CommandQueue& cq,
    const std::variant<std::reference_wrapper<Buffer>, std::shared_ptr<Buffer>>& buffer,
    HostDataType src,
    const BufferRegion& region,
    bool blocking) {
    detail::DispatchStateCheck(true);
    detail::ValidateBufferRegion(buffer, region);

    cq.enqueue_write_buffer(std::move(buffer), std::move(src), region, blocking);
}

void EnqueueProgram(CommandQueue& cq, Program& program, bool blocking) {
    ZoneScoped;
    LIGHT_METAL_TRACE_FUNCTION_ENTRY();
    LIGHT_METAL_TRACE_FUNCTION_CALL(CaptureEnqueueProgram, cq, program, blocking);
    detail::DispatchStateCheck(true);

    IDevice* device = cq.device();
    detail::CompileProgram(device, program);
    program.allocate_circular_buffers(device);
    detail::ValidateCircularBufferRegion(program, device);
    cq.enqueue_program(program, blocking);
    // Program relinquishes ownership of all global buffers its using, once its been enqueued. Avoid mem
    // leaks on device.
    program.release_buffers();
}

void EnqueueRecordEvent(
    CommandQueue& cq, const std::shared_ptr<Event>& event, tt::stl::Span<const SubDeviceId> sub_device_ids) {
    detail::DispatchStateCheck(true);
    cq.enqueue_record_event(event, sub_device_ids);
}

void EnqueueWaitForEvent(CommandQueue& cq, const std::shared_ptr<Event>& event) {
    detail::DispatchStateCheck(true);
    event->wait_until_ready();  // Block until event populated. Worker thread.
    log_trace(
        tt::LogMetal,
        "EnqueueWaitForEvent() issued on Event(device_id: {} cq_id: {} event_id: {}) from device_id: {} cq_id: {}",
        event->device->id(),
        event->cq_id,
        event->event_id,
        cq.device()->id(),
        cq.id());
    cq.enqueue_wait_for_event(event);
}

void EventSynchronize(const std::shared_ptr<Event>& event) {
    detail::DispatchStateCheck(true);
    event->wait_until_ready();  // Block until event populated. Parent thread.
    log_trace(
        tt::LogMetal,
        "Issuing host sync on Event(device_id: {} cq_id: {} event_id: {})",
        event->device->id(),
        event->cq_id,
        event->event_id);

    while (event->device->sysmem_manager().get_last_completed_event(event->cq_id) < event->event_id) {
        if (tt::tt_metal::MetalContext::instance().rtoptions().get_test_mode_enabled() &&
            tt::watcher_server_killed_due_to_error()) {
            TT_FATAL(
                false,
                "Command Queue could not complete EventSynchronize. See {} for details.",
                tt::watcher_get_log_file_name());
            return;
        }
        std::this_thread::sleep_for(std::chrono::microseconds(5));
    }
}

bool EventQuery(const std::shared_ptr<Event>& event) {
    detail::DispatchStateCheck(true);
    event->wait_until_ready();  // Block until event populated. Parent thread.
    bool event_completed = event->device->sysmem_manager().get_last_completed_event(event->cq_id) >= event->event_id;
    log_trace(
        tt::LogMetal,
        "Returning event_completed: {} for host query on Event(device_id: {} cq_id: {} event_id: {})",
        event_completed,
        event->device->id(),
        event->cq_id,
        event->event_id);
    return event_completed;
}

void Finish(CommandQueue& cq, tt::stl::Span<const SubDeviceId> sub_device_ids) {
    LIGHT_METAL_TRACE_FUNCTION_ENTRY();
    LIGHT_METAL_TRACE_FUNCTION_CALL(CaptureFinish, cq, sub_device_ids);
    detail::DispatchStateCheck(true);
    cq.finish(sub_device_ids);
    // If in testing mode, don't need to check dprint/watcher errors, since the tests will induce/handle them.
    if (!MetalContext::instance().rtoptions().get_test_mode_enabled()) {
        TT_FATAL(
            !(DPrintServerHangDetected()),
            "Command Queue could not finish: device hang due to unanswered DPRINT WAIT.");
        TT_FATAL(
            !(tt::watcher_server_killed_due_to_error()),
            "Command Queue could not finish: device hang due to illegal NoC transaction. See {} for details.",
            tt::watcher_get_log_file_name());
    }
}

void EnqueueTrace(CommandQueue& cq, uint32_t trace_id, bool blocking) {
    LIGHT_METAL_TRACE_FUNCTION_ENTRY();
    LIGHT_METAL_TRACE_FUNCTION_CALL(CaptureEnqueueTrace, cq, trace_id, blocking);
    detail::DispatchStateCheck(true);
    TT_FATAL(cq.device()->get_trace(trace_id) != nullptr, "Trace instance {} must exist on device", trace_id);
    cq.enqueue_trace(trace_id, blocking);
}

}  // namespace tt::tt_metal

std::ostream& operator<<(std::ostream& os, const EnqueueCommandType& type) {
    switch (type) {
        case EnqueueCommandType::ENQUEUE_READ_BUFFER: os << "ENQUEUE_READ_BUFFER"; break;
        case EnqueueCommandType::ENQUEUE_WRITE_BUFFER: os << "ENQUEUE_WRITE_BUFFER"; break;
        case EnqueueCommandType::ENQUEUE_PROGRAM: os << "ENQUEUE_PROGRAM"; break;
        case EnqueueCommandType::ENQUEUE_TRACE: os << "ENQUEUE_TRACE"; break;
        case EnqueueCommandType::ENQUEUE_RECORD_EVENT: os << "ENQUEUE_RECORD_EVENT"; break;
        case EnqueueCommandType::ENQUEUE_WAIT_FOR_EVENT: os << "ENQUEUE_WAIT_FOR_EVENT"; break;
        case EnqueueCommandType::FINISH: os << "FINISH"; break;
        case EnqueueCommandType::FLUSH: os << "FLUSH"; break;
        default: TT_THROW("Invalid command type!");
    }
    return os;
}
