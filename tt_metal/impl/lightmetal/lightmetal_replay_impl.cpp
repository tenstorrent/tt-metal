// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "lightmetal_replay_impl.hpp"

#include <iostream>
#include "light_metal_binary_generated.h"
#include "command_generated.h"
#include <tt-logger/tt-logger.hpp>

#include <host_api.hpp>
#include "env_lib.hpp"
#include <tt-metalium/tt_metal.hpp>
#include "trace/trace_buffer.hpp"
#include <tt-metalium/command_queue.hpp>
#include <tt-metalium/device.hpp>
#include "flatbuffer/base_types_from_flatbuffer.hpp"
#include "flatbuffer/program_types_from_flatbuffer.hpp"
#include "flatbuffer/buffer_types_from_flatbuffer.hpp"

namespace tt::tt_metal {

//////////////////////////////////////
// Helper Functions                 //
//////////////////////////////////////

TraceDescriptor from_flatbuffer(const flatbuffer::TraceDescriptor* fb_desc) {
    if (!fb_desc) {
        std::cerr << "TraceDescriptor is null." << std::endl;
        return {};
    }

    TraceDescriptor trace_desc;

    // Deserialize trace_data
    if (auto trace_data_fb = fb_desc->trace_data()) {
        trace_desc.data.assign(trace_data_fb->begin(), trace_data_fb->end());
    }

    // Deserialize sub_device_descriptors
    if (auto sub_device_descriptors_fb = fb_desc->sub_device_descriptors()) {
        for (const auto* mapping : *sub_device_descriptors_fb) {
            if (mapping) {
                TraceWorkerDescriptor descriptor;
                descriptor.num_completion_worker_cores = mapping->descriptor()->num_completion_worker_cores();
                descriptor.num_traced_programs_needing_go_signal_multicast =
                    mapping->descriptor()->num_traced_programs_needing_go_signal_multicast();
                descriptor.num_traced_programs_needing_go_signal_unicast =
                    mapping->descriptor()->num_traced_programs_needing_go_signal_unicast();

                // Add the descriptor to the map
                trace_desc.descriptors[SubDeviceId{mapping->sub_device_id()}] = descriptor;
            }
        }
    }

    // Deserialize sub_device_ids
    if (auto sub_device_ids_fb = fb_desc->sub_device_ids()) {
        for (const auto id : *sub_device_ids_fb) {
            trace_desc.sub_device_ids.emplace_back(SubDeviceId{id});
        }
    }

    return trace_desc;
}

namespace detail {

//////////////////////////////////////
// LightMetalReplay Class           //
//////////////////////////////////////

LightMetalReplayImpl::LightMetalReplayImpl(LightMetalBinary&& binary, IDevice* device) :
    binary_(std::move(binary)), fb_binary_(nullptr), device_(device) {
    if (binary_.is_empty()) {
        log_warning(tt::LogMetalTrace, "Empty LightMetalBinary provided to LightMetalReplay.");
    }

    show_reads_ = parse_env("TT_LIGHT_METAL_SHOW_READS", false);
    disable_checking_ = parse_env("TT_LIGHT_METAL_DISABLE_CHECKING", false);
    fb_binary_ = parse_flatbuffer_binary();  // Parse and store the FlatBuffer binary
}

// Needs access to BufferMap, so part of LightMetalReplay class
std::shared_ptr<RuntimeArgs> LightMetalReplayImpl::rt_args_from_flatbuffer(
    const FlatbufferRuntimeArgVector flatbuffer_args) {
    auto runtime_args = std::make_shared<RuntimeArgs>();

    for (const auto& flatbuffer_arg : *flatbuffer_args) {
        const auto* runtime_arg = flatbuffer_arg;
        TT_FATAL(runtime_arg, "Null RuntimeArg in FlatBuffer vector");

        // Determine the type of the RuntimeArg
        switch (runtime_arg->value_type()) {
            case tt::tt_metal::flatbuffer::RuntimeArgValue::UInt32Value: {
                // Extract UInt32Value
                const auto* uint32_value = runtime_arg->value_as_UInt32Value();
                TT_FATAL(uint32_value, "Failed to read UInt32Value");
                runtime_args->emplace_back(uint32_value->value());
                break;
            }
            case tt::tt_metal::flatbuffer::RuntimeArgValue::BufferGlobalId: {
                // Extract BufferGlobalId
                const auto* buffer_global_id = runtime_arg->value_as_BufferGlobalId();
                TT_FATAL(buffer_global_id, "Failed to read BufferGlobalId");
                uint32_t global_id = buffer_global_id->id();
                auto buffer = get_buffer_from_map(global_id);
                TT_FATAL(buffer, "Buffer w/ global_id: {} not previously created", global_id);
                runtime_args->emplace_back(buffer.get());
                break;
            }
            case tt::tt_metal::flatbuffer::RuntimeArgValue::NONE: {
                TT_THROW("Unknown RuntimeArgValue type NONE in FlatBuffer");
            }
        }
    }

    return runtime_args;
}

const tt::tt_metal::flatbuffer::LightMetalBinary* LightMetalReplayImpl::parse_flatbuffer_binary() {
    try {
        const uint8_t* data_ptr = binary_.get_data().data();
        size_t size = binary_.get_data().size();

        // Verify the FlatBuffer data.
        flatbuffers::Verifier verifier(data_ptr, size);
        if (!tt::tt_metal::flatbuffer::VerifyLightMetalBinaryBuffer(verifier)) {
            std::cerr << "Failed to verify FlatBuffer data." << std::endl;
            return nullptr;
        }

        // Parse and return the FlatBuffer object.
        return tt::tt_metal::flatbuffer::GetLightMetalBinary(data_ptr);
    } catch (const std::exception& e) {
        std::cerr << "Exception while parsing FlatBuffer binary: " << e.what() << std::endl;
        return nullptr;
    }
}

// Return a TraceDescriptor for a given trace_id from the FlatBuffer binary.
std::optional<TraceDescriptor> LightMetalReplayImpl::get_trace_by_id(uint32_t target_trace_id) {
    if (const auto* trace_descriptors = fb_binary_ ? fb_binary_->trace_descriptors() : nullptr) {
        if (const auto* fb_trace_desc_by_id = trace_descriptors->LookupByKey(target_trace_id)) {
            if (const auto* fb_desc = fb_trace_desc_by_id->desc()) {
                return from_flatbuffer(fb_desc);
            }
        }
    }

    std::cerr << "Failed to find trace_id: " << target_trace_id << " in binary." << std::endl;
    return std::nullopt;
}

//////////////////////////////////////
// Object Map Public Accessors      //
//////////////////////////////////////

void LightMetalReplayImpl::add_buffer_to_map(
    uint32_t global_id, const std::shared_ptr<::tt::tt_metal::Buffer>& buffer) {
    if (buffer_map_.find(global_id) != buffer_map_.end()) {
        log_warning(tt::LogMetalTrace, "Buffer with global_id: {} already exists in map.", global_id);
    }
    buffer_map_[global_id] = buffer;  // Shared ownership
}

std::shared_ptr<::tt::tt_metal::Buffer> LightMetalReplayImpl::get_buffer_from_map(uint32_t global_id) const {
    auto it = buffer_map_.find(global_id);
    return it != buffer_map_.end() ? it->second : nullptr;
}

void LightMetalReplayImpl::remove_bufer_from_map(uint32_t global_id) { buffer_map_.erase(global_id); }

void LightMetalReplayImpl::add_program_to_map(
    uint32_t global_id, const std::shared_ptr<::tt::tt_metal::Program>& program) {
    if (program_map_.find(global_id) != program_map_.end()) {
        log_warning(tt::LogMetalTrace, "Program with global_id: {} already exists in map.", global_id);
    }
    program_map_[global_id] = program;  // Shared ownership
}

std::shared_ptr<::tt::tt_metal::Program> LightMetalReplayImpl::get_program_from_map(uint32_t global_id) const {
    auto it = program_map_.find(global_id);
    return it != program_map_.end() ? it->second : nullptr;
}

void LightMetalReplayImpl::remove_program_from_map(uint32_t global_id) { program_map_.erase(global_id); }

void LightMetalReplayImpl::add_kernel_handle_to_map(uint32_t global_id, ::tt::tt_metal::KernelHandle kernel_id) {
    if (kernel_handle_map_.find(global_id) != kernel_handle_map_.end()) {
        log_warning(tt::LogMetalTrace, "KernelHandle with global_id: {} already exists in map.", global_id);
    }
    kernel_handle_map_[global_id] = kernel_id;  // Shared ownership
}

::tt::tt_metal::KernelHandle LightMetalReplayImpl::get_kernel_handle_from_map(uint32_t global_id) const {
    auto it = kernel_handle_map_.find(global_id);
    return it != kernel_handle_map_.end() ? it->second : UINT32_MAX;
}

void LightMetalReplayImpl::remove_kernel_handle_from_map(uint32_t global_id) { kernel_handle_map_.erase(global_id); }

void LightMetalReplayImpl::add_kernel_to_map(
    uint32_t global_id, const std::shared_ptr<::tt::tt_metal::Kernel>& kernel) {
    if (kernel_map_.find(global_id) != kernel_map_.end()) {
        log_warning(tt::LogMetalTrace, "Kernel with global_id: {} already exists in map.", global_id);
    }
    kernel_map_[global_id] = kernel;  // Shared ownership
}

std::shared_ptr<::tt::tt_metal::Kernel> LightMetalReplayImpl::get_kernel_from_map(uint32_t global_id) const {
    auto it = kernel_map_.find(global_id);
    return it != kernel_map_.end() ? it->second : nullptr;
}

void LightMetalReplayImpl::remove_kernel_from_map(uint32_t global_id) { kernel_map_.erase(global_id); }

void LightMetalReplayImpl::add_cb_handle_to_map(uint32_t global_id, ::tt::tt_metal::CBHandle cb_handle) {
    if (cb_handle_map_.find(global_id) != cb_handle_map_.end()) {
        log_warning(tt::LogMetalTrace, "CBHandle with global_id: {} already exists in map.", global_id);
    }
    cb_handle_map_[global_id] = cb_handle;  // Shared ownership
}

::tt::tt_metal::CBHandle LightMetalReplayImpl::get_cb_handle_from_map(uint32_t global_id) const {
    auto it = cb_handle_map_.find(global_id);
    return it != cb_handle_map_.end() ? it->second : UINT32_MAX;
}

void LightMetalReplayImpl::remove_cb_handle_from_map(uint32_t global_id) { cb_handle_map_.erase(global_id); }

//////////////////////////////////////
// Device Setup/Teardown            //
//////////////////////////////////////

// TODO (kmabee) - Hardcode for now, eventually capture/replay "systemdesc" from binary.
// Alternatively, user can manage device open/close and pass to replay library.
void LightMetalReplayImpl::setup_devices() {
    log_debug(tt::LogMetalTrace, "LightMetalReplay(setup_devices) - Using hardcoded CreateDevices() as temp hack.");
    TT_FATAL(!device_, "Device already setup in LightMetalReplay, no need to call setup_devices()");
    const size_t trace_region_size = 4096;  // Default is 0
    const auto dispatch_core_type = tt_metal::DispatchCoreType::WORKER;
    const chip_id_t mmio_device_id = 0;
    auto devices_map = tt::tt_metal::detail::CreateDevices(
        {mmio_device_id}, 1, DEFAULT_L1_SMALL_SIZE, trace_region_size, dispatch_core_type);
    this->device_ = devices_map.at(mmio_device_id);
}

// TODO (kmabee) - Hardcode for now, eventually capture/replay "systemdesc" from binary or let user call.
void LightMetalReplayImpl::close_devices() { CloseDevice(this->device_); }

// Clear object maps for items not deallocated/destroyed naturally during replay.
// Later can update these to be asserts once all paths covered properly.
void LightMetalReplayImpl::clear_object_maps() {
    // Later can update these to be asserts.
    if (buffer_map_.size()) {
        log_debug(tt::LogMetalTrace, "Cleared LightMetalReplay BufferMap: {} entries", buffer_map_.size());
        buffer_map_.clear();
    }

    if (program_map_.size()) {
        log_debug(tt::LogMetalTrace, "Cleared LightMetalReplay ProgramMap: {} entries", program_map_.size());
        program_map_.clear();
    }

    if (kernel_handle_map_.size()) {
        log_debug(tt::LogMetalTrace, "Cleared LightMetalReplay KernelHandleMap: {} entries", kernel_handle_map_.size());
        kernel_handle_map_.clear();
    }

    if (kernel_map_.size()) {
        log_debug(tt::LogMetalTrace, "Cleared LightMetalReplay KernelMap: {} entries", kernel_map_.size());
        kernel_map_.clear();
    }

    if (cb_handle_map_.size()) {
        log_debug(tt::LogMetalTrace, "Cleared LightMetalReplay CBHandleMap: {} entries", cb_handle_map_.size());
        cb_handle_map_.clear();
    }
}

//////////////////////////////////////
// Executor                         //
//////////////////////////////////////

// execute a command by dispatching to appropriate handler based on type.
void LightMetalReplayImpl::execute(const tt::tt_metal::flatbuffer::Command* command) {
    switch (command->cmd_type()) {
        case ::tt::tt_metal::flatbuffer::CommandType::EnqueueTraceCommand: {
            execute(command->cmd_as_EnqueueTraceCommand());
            break;
        }
        case ::tt::tt_metal::flatbuffer::CommandType::ReplayTraceCommand: {
            execute(command->cmd_as_ReplayTraceCommand());
            break;
        }
        case ::tt::tt_metal::flatbuffer::CommandType::LoadTraceCommand: {
            execute(command->cmd_as_LoadTraceCommand());
            break;
        }
        case ::tt::tt_metal::flatbuffer::CommandType::ReleaseTraceCommand: {
            execute(command->cmd_as_ReleaseTraceCommand());
            break;
        }
        case ::tt::tt_metal::flatbuffer::CommandType::BufferCreateCommand: {
            execute(command->cmd_as_BufferCreateCommand());
            break;
        }
        case ::tt::tt_metal::flatbuffer::CommandType::BufferDeallocateCommand: {
            execute(command->cmd_as_BufferDeallocateCommand());
            break;
        }
        case ::tt::tt_metal::flatbuffer::CommandType::BufferDeleteCommand: {
            execute(command->cmd_as_BufferDeleteCommand());
            break;
        }
        case ::tt::tt_metal::flatbuffer::CommandType::EnqueueWriteBufferCommand: {
            execute(command->cmd_as_EnqueueWriteBufferCommand());
            break;
        }
        case ::tt::tt_metal::flatbuffer::CommandType::EnqueueReadBufferCommand: {
            execute(command->cmd_as_EnqueueReadBufferCommand());
            break;
        }
        case ::tt::tt_metal::flatbuffer::CommandType::FinishCommand: {
            execute(command->cmd_as_FinishCommand());
            break;
        }
        case ::tt::tt_metal::flatbuffer::CommandType::ProgramConstructorCommand: {
            execute(command->cmd_as_ProgramConstructorCommand());
            break;
        }
        case ::tt::tt_metal::flatbuffer::CommandType::EnqueueProgramCommand: {
            execute(command->cmd_as_EnqueueProgramCommand());
            break;
        }
        case ::tt::tt_metal::flatbuffer::CommandType::CreateKernelCommand: {
            execute(command->cmd_as_CreateKernelCommand());
            break;
        }
        case ::tt::tt_metal::flatbuffer::CommandType::SetRuntimeArgsUint32Command: {
            execute(command->cmd_as_SetRuntimeArgsUint32Command());
            break;
        }
        case ::tt::tt_metal::flatbuffer::CommandType::SetRuntimeArgsUint32VecPerCoreCommand: {
            execute(command->cmd_as_SetRuntimeArgsUint32VecPerCoreCommand());
            break;
        }
        case ::tt::tt_metal::flatbuffer::CommandType::SetRuntimeArgsCommand: {
            execute(command->cmd_as_SetRuntimeArgsCommand());
            break;
        }
        case ::tt::tt_metal::flatbuffer::CommandType::CreateCircularBufferCommand: {
            execute(command->cmd_as_CreateCircularBufferCommand());
            break;
        }
        case ::tt::tt_metal::flatbuffer::CommandType::LightMetalCompareCommand: {
            execute(command->cmd_as_LightMetalCompareCommand());
            break;
        }
        case ::tt::tt_metal::flatbuffer::CommandType::NONE:
            TT_THROW("LightMetalReplay execute encountered unsupported cmd type NONE");
            break;
    }
}

// Per API command handlers.
void LightMetalReplayImpl::execute(const tt::tt_metal::flatbuffer::EnqueueTraceCommand* cmd) {
    log_debug(
        tt::LogMetalTrace,
        "LightMetalReplay(EnqueueTrace) cq_id: {} tid: {} blocking: {}",
        cmd->cq_id(),
        cmd->tid(),
        cmd->blocking());
    CommandQueue& cq = this->device_->command_queue(cmd->cq_id());
    EnqueueTrace(cq, cmd->tid(), cmd->blocking());
}

void LightMetalReplayImpl::execute(const tt::tt_metal::flatbuffer::ReplayTraceCommand* cmd) {
    log_debug(
        tt::LogMetalTrace,
        "LightMetalReplay(ReplayTrace) cq_id: {} tid: {} blocking: {}",
        cmd->cq_id(),
        cmd->tid(),
        cmd->blocking());
    ReplayTrace(this->device_, cmd->cq_id(), cmd->tid(), cmd->blocking());
}

void LightMetalReplayImpl::execute(const tt::tt_metal::flatbuffer::LoadTraceCommand* cmd) {
    log_debug(tt::LogMetalTrace, "LightMetalReplay(LoadTrace) cq_id: {} tid: {}", cmd->cq_id(), cmd->tid());
    // Get the trace descriptor from flatbuffer and load it to device.
    auto trace_desc = get_trace_by_id(cmd->tid());
    LoadTrace(this->device_, cmd->cq_id(), cmd->tid(), trace_desc.value());
}

void LightMetalReplayImpl::execute(const tt::tt_metal::flatbuffer::ReleaseTraceCommand* cmd) {
    log_debug(tt::LogMetalTrace, "LightMetalReplay(ReleaseTrace) tid: {}", cmd->tid());
    ReleaseTrace(this->device_, cmd->tid());
}

void LightMetalReplayImpl::execute(const tt::tt_metal::flatbuffer::BufferCreateCommand* cmd) {
    log_debug(
        tt::LogMetalTrace,
        "LightMetalReplay(BufferCreate) global_id: {} size: {} page_size: {} layout: {} buffer_type: {}",
        cmd->global_id(),
        cmd->size(),
        cmd->page_size(),
        EnumNameTensorMemoryLayout(cmd->buffer_layout()),
        EnumNameBufferType(cmd->buffer_type()));

    // Handle optionals
    const auto shard_parameters = from_flatbuffer(cmd->shard_parameters());
    auto buffer_layout = static_cast<TensorMemoryLayout>(cmd->buffer_layout());
    const auto bottom_up = cmd->bottom_up() ? std::optional<bool>{cmd->bottom_up()->value()} : std::nullopt;
    const auto sub_device_id =
        cmd->sub_device_id() ? std::optional<SubDeviceId>{cmd->sub_device_id()->value()} : std::nullopt;

    // This API is overloaded with and without address field.
    if (cmd->address()) {
        auto buffer = Buffer::create(
            this->device_,
            cmd->address()->value(),
            cmd->size(),
            cmd->page_size(),
            from_flatbuffer(cmd->buffer_type()),
            BufferShardingArgs(shard_parameters, buffer_layout),
            bottom_up,
            sub_device_id);
        add_buffer_to_map(cmd->global_id(), buffer);

    } else {
        auto buffer = Buffer::create(
            this->device_,
            cmd->size(),
            cmd->page_size(),
            from_flatbuffer(cmd->buffer_type()),
            BufferShardingArgs(shard_parameters, buffer_layout),
            bottom_up,
            sub_device_id);
        add_buffer_to_map(cmd->global_id(), buffer);
    }
}

void LightMetalReplayImpl::execute(const tt::tt_metal::flatbuffer::BufferDeallocateCommand* cmd) {
    auto buffer = get_buffer_from_map(cmd->global_id());
    TT_FATAL(
        buffer,
        "Attempted to DeallocateBuffer() buffer w/ global_id: {} that was not previously created.",
        cmd->global_id());

    log_debug(tt::LogMetalTrace, "LightMetalReplay(BufferDeallocate) global_id: {}", cmd->global_id());
    DeallocateBuffer(*buffer);  // Buffer& expected.
}

void LightMetalReplayImpl::execute(const tt::tt_metal::flatbuffer::BufferDeleteCommand* cmd) {
    auto buffer = get_buffer_from_map(cmd->global_id());
    TT_FATAL(buffer, "Attempted to Delete buffer w/ global_id: {} that was not previously created.", cmd->global_id());
    log_debug(tt::LogMetalTrace, "LightMetalReplay(BufferDelete) global_id: {}", cmd->global_id());
    remove_bufer_from_map(cmd->global_id());
}

void LightMetalReplayImpl::execute(const tt::tt_metal::flatbuffer::EnqueueWriteBufferCommand* cmd) {
    auto buffer = get_buffer_from_map(cmd->buffer_global_id());
    TT_FATAL(
        buffer,
        "Attempted to EnqueueWriteBuffer() buffer w/ global_id: {} that was not previously created.",
        cmd->buffer_global_id());

    log_debug(
        tt::LogMetalTrace,
        "LightMetalReplay(EnqueueWriteBuffer) cq_global_id: {} buffer_global_id: {} addr: 0x{:x}",
        cmd->cq_global_id(),
        cmd->buffer_global_id(),
        buffer->address());

    // TODO (kmabee) - consider storing/getting CQ from global map instead.
    CommandQueue& cq = this->device_->command_queue(cmd->cq_global_id());
    EnqueueWriteBuffer(cq, buffer, cmd->src()->data(), cmd->blocking());
}

void LightMetalReplayImpl::execute(const tt::tt_metal::flatbuffer::EnqueueReadBufferCommand* cmd) {
    auto buffer = get_buffer_from_map(cmd->buffer_global_id());
    TT_FATAL(
        buffer,
        "Attempted to EnqueueReadBuffer() buffer w/ global_id: {} that was not previously created.",
        cmd->buffer_global_id());

    log_debug(
        tt::LogMetalTrace,
        "LightMetalReplay(EnqueueReadBuffer) cq_global_id: {} buffer_global_id: {} addr: 0x{:x} buf_size: {}",
        cmd->cq_global_id(),
        cmd->buffer_global_id(),
        buffer->address(),
        buffer->size());

    // TODO (kmabee) - consider storing/getting CQ from global map instead.
    CommandQueue& cq = this->device_->command_queue(cmd->cq_global_id());
    std::vector<uint32_t> readback_data(buffer->size() / sizeof(uint32_t), 0);
    EnqueueReadBuffer(cq, buffer, readback_data.data(), cmd->blocking());

    // TODO (kmabee) - TBD what to do with readback data. For now, optionally print.
    // One idea is to store in map by global_read_id that caller can access.
    if (show_reads_) {
        for (size_t i = 0; i < readback_data.size(); i++) {
            log_info(tt::LogMetalTrace, " rd_data i: {:3d} => data: {} ({:x})", i, readback_data[i], readback_data[i]);
        }
    }
}

void LightMetalReplayImpl::execute(const tt::tt_metal::flatbuffer::FinishCommand* cmd) {
    log_debug(tt::LogMetalTrace, "LightMetalReplay(Finish) cq_global_id: {}", cmd->cq_global_id());
    CommandQueue& cq = this->device_->command_queue(cmd->cq_global_id());
    auto sub_device_ids = from_flatbuffer(cmd->sub_device_ids());
    Finish(cq, sub_device_ids);
}

void LightMetalReplayImpl::execute(const tt::tt_metal::flatbuffer::ProgramConstructorCommand* cmd) {
    log_debug(tt::LogMetalTrace, "LightMetalReplay(ProgramConstructor) global_id: {} ", cmd->global_id());
    add_program_to_map(cmd->global_id(), std::make_shared<Program>());
}

void LightMetalReplayImpl::execute(const tt::tt_metal::flatbuffer::EnqueueProgramCommand* cmd) {
    auto program = get_program_from_map(cmd->program_global_id());
    TT_FATAL(
        program,
        "Attempted to EnqueueProgram() program w/ global_id: {} that was not previously created.",
        cmd->program_global_id());

    log_debug(
        tt::LogMetalTrace,
        "LightMetalReplay(EnqueueProgram) program_global_id: {} cq_global_id: {}",
        cmd->program_global_id(),
        cmd->cq_global_id());

    // TODO (kmabee) - consider storing/getting CQ from global map instead.
    CommandQueue& cq = this->device_->command_queue(cmd->cq_global_id());
    EnqueueProgram(cq, *program, cmd->blocking());
}

void LightMetalReplayImpl::execute(const tt::tt_metal::flatbuffer::CreateKernelCommand* cmd) {
    log_debug(
        tt::LogMetalTrace,
        "LightMetalReplay(CreateKernel) global_id: {} program_global_id: {}",
        cmd->global_id(),
        cmd->program_global_id());
    auto program = get_program_from_map(cmd->program_global_id());
    TT_FATAL(
        program,
        "Attempted to CreateKernel() using a program w/ global_id: {} that was not previously created.",
        cmd->program_global_id());

    auto core_spec = core_spec_from_flatbuffer(cmd);
    auto kernel_config = kernel_config_from_flatbuffer(cmd);
    auto kernel_id = CreateKernel(*program, cmd->file_name()->c_str(), core_spec, kernel_config);
    add_kernel_handle_to_map(cmd->global_id(), kernel_id);
    // Some APIs use Kernel, so convert to and store Kernel.
    std::shared_ptr<Kernel> kernel = program->get_kernel(kernel_id);
    add_kernel_to_map(cmd->global_id(), kernel);
}

void LightMetalReplayImpl::execute(const tt::tt_metal::flatbuffer::SetRuntimeArgsUint32Command* cmd) {
    log_debug(
        tt::LogMetalTrace,
        "LightMetalReplay(SetRuntimeArgs). program_global_id: {} kernel_global_id: {}",
        cmd->program_global_id(),
        cmd->kernel_global_id());
    auto program = get_program_from_map(cmd->program_global_id());
    auto kernel_id = get_kernel_handle_from_map(cmd->kernel_global_id());
    TT_FATAL(
        program,
        "Attempted to SetRuntimeArgs() using a program w/ global_id: {} that was not previously created.",
        cmd->program_global_id());
    TT_FATAL(
        kernel_id != UINT32_MAX,
        "Attempted to SetRuntimeArgs() using a kernel w/ global_id: {} that was not previously created.",
        cmd->kernel_global_id());

    // API expects a span so create from flatbuffer vector.
    stl::Span<const uint32_t> args_span(cmd->args()->data(), cmd->args()->size());
    auto core_spec = core_spec_from_flatbuffer(cmd);
    SetRuntimeArgs(*program, kernel_id, core_spec, args_span);
}

void LightMetalReplayImpl::execute(const tt::tt_metal::flatbuffer::SetRuntimeArgsUint32VecPerCoreCommand* cmd) {
    log_debug(
        tt::LogMetalTrace,
        "LightMetalReplay(SetRuntimeArgs). program_global_id: {} kernel_global_id: {}",
        cmd->program_global_id(),
        cmd->kernel_global_id());
    auto program = get_program_from_map(cmd->program_global_id());
    auto kernel_id = get_kernel_handle_from_map(cmd->kernel_global_id());
    TT_FATAL(
        program,
        "Attempted to SetRuntimeArgs() using a program w/ global_id: {} that was not previously created.",
        cmd->program_global_id());
    TT_FATAL(
        kernel_id != UINT32_MAX,
        "Attempted to SetRuntimeArgs() using a kernel w/ global_id: {} that was not previously created.",
        cmd->kernel_global_id());

    auto core_spec = from_flatbuffer(cmd->core_spec());
    auto runtime_args = from_flatbuffer(cmd->args());
    SetRuntimeArgs(*program, kernel_id, core_spec, runtime_args);
}

void LightMetalReplayImpl::execute(const tt::tt_metal::flatbuffer::SetRuntimeArgsCommand* cmd) {
    log_debug(
        tt::LogMetalTrace,
        "LightMetalReplay(SetRuntimeArgs). kernel_global_id: {} rt_args_size: {}",
        cmd->kernel_global_id(),
        cmd->args()->size());
    auto core_spec = core_spec_from_flatbuffer(cmd);
    auto runtime_args = rt_args_from_flatbuffer(cmd->args());
    auto kernel = get_kernel_from_map(cmd->kernel_global_id());
    TT_FATAL(
        kernel,
        "Attempted to SetRuntimeArgs() using a Kernel w/ global_id: {} that was not previously created.",
        cmd->kernel_global_id());
    SetRuntimeArgs(this->device_, kernel, core_spec, runtime_args);
}

void LightMetalReplayImpl::execute(const tt::tt_metal::flatbuffer::CreateCircularBufferCommand* cmd) {
    log_debug(
        tt::LogMetalTrace,
        "LightMetalReplay(CreateCircularBuffer) global_id: {} program_global_id: {}",
        cmd->global_id(),
        cmd->program_global_id());
    auto program = get_program_from_map(cmd->program_global_id());
    TT_FATAL(
        program,
        "Attempted to CreateCircularBuffer() using a Program w/ global_id: {} that was not previously created.",
        cmd->program_global_id());
    auto core_spec = core_spec_from_flatbuffer(cmd);

    // Convert global_id to optional Shadow Buffer here to keep from_flatbuffer standalone function.
    ::tt::tt_metal::Buffer* shadow_global_buffer = nullptr;
    auto shadow_buf_global_id = cmd->config()->shadow_buf_global_id();

    if (shadow_buf_global_id) {
        auto global_id = shadow_buf_global_id->value();
        auto shadow_buf = get_buffer_from_map(global_id);
        TT_FATAL(
            shadow_buf,
            "Attempted to CreateCircularBuffer() using a shadow Buffer w/ global_id: {} that was not previously "
            "created.",
            global_id);
        shadow_global_buffer = shadow_buf.get();  // Set the raw pointer
    }

    auto config = from_flatbuffer(cmd->config(), shadow_global_buffer);
    auto cb_handle = CreateCircularBuffer(*program, core_spec, config);
    add_cb_handle_to_map(cmd->global_id(), cb_handle);
}

// Verification command to compare readback of a buffer with golden from either capture or user expected values.
void LightMetalReplayImpl::execute(const ::tt::tt_metal::flatbuffer::LightMetalCompareCommand* cmd) {
    log_debug(
        tt::LogMetalTrace,
        "LightMetalReplay(LightMetalCompare) cq_global_id: {} buffer_global_id: {} is_user_data: {}",
        cmd->cq_global_id(),
        cmd->buffer_global_id(),
        cmd->is_user_data());

    auto buffer = get_buffer_from_map(cmd->buffer_global_id());
    TT_FATAL(
        buffer,
        "Attempted to run LightMetalCompareCommand using a Buffer w/ global_id: {} that was not previously created.",
        cmd->buffer_global_id());

    // TODO (kmabee) - consider storing/getting CQ from global map instead.
    CommandQueue& cq = this->device_->command_queue(cmd->cq_global_id());
    std::vector<uint32_t> rd_data(buffer->size() / sizeof(uint32_t), 0);
    EnqueueReadBuffer(cq, buffer, rd_data.data(), true);

    if (disable_checking_) {
        log_debug(
            tt::LogMetalTrace, "Skipping LightMetalCompareCommand for buffer_global_id: {}.", cmd->buffer_global_id());
    } else {
        if (rd_data.size() != cmd->golden_data()->size()) {
            TT_THROW(
                "Readback data size: {} does not match golden data size: {}",
                rd_data.size(),
                cmd->golden_data()->size());
        }

        // Optional debug to show verbose comparison
        if (show_reads_) {
            for (size_t i = 0; i < rd_data.size(); i++) {
                bool match = rd_data[i] == cmd->golden_data()->Get(i);
                log_info(
                    tt::LogMetalTrace,
                    "LightMetalCompare i: {:3d} match: {} RdData: {:x} Golden: {:x}",
                    i,
                    match,
                    rd_data[i],
                    cmd->golden_data()->Get(i));
            }
        }

        // Do simple equality comparison between two vectors
        if (!std::equal(rd_data.begin(), rd_data.end(), cmd->golden_data()->begin())) {
            TT_THROW("Golden vs rd_data mismatch for buffer_global_id: {}", cmd->buffer_global_id());
        }
    }
}

// Main entry point to execute a light metal binary blob, return true if pass.
bool LightMetalReplayImpl::run() {
    if (!fb_binary_) {
        std::cerr << "Cannot Replay empty/uninitialized Light Metal Binary." << std::endl;
        return false;
    }

    const bool replay_manages_device = device_ == nullptr;

    try {
        const auto* trace_descs = fb_binary_->trace_descriptors();
        const auto* commands = fb_binary_->commands();
        if (!commands) {
            std::cerr << "Nothing to run, no commands in binary." << std::endl;
            return false;
        }

        log_info(
            tt::LogMetalTrace,
            "Running LightMetal Binary with {} cmds, {} traces. ManageDevice: {}",
            commands->size(),
            trace_descs->size(),
            replay_manages_device);

        if (replay_manages_device) {
            setup_devices();
        }

        // Just loop over all commands, and execute. This is purposely kept simple for prototyping v0.
        // TODO (kmabee) - should expand to cover, multiple devices, cqs, etc.
        uint32_t idx = 1;
        for (const auto* cmd : *commands) {
            auto str_name = std::string(EnumNameCommandType(cmd->cmd_type()));
            log_trace(tt::LogMetalTrace, "Executing Binary CMD {}/{} (Type: {})", idx++, commands->size(), str_name);
            execute(cmd);
        }

        clear_object_maps();

        if (replay_manages_device) {
            close_devices();
        }

        return true;
    } catch (const std::exception& e) {
        log_fatal(tt::LogMetalTrace, "{}", e.what());
        clear_object_maps();
        if (replay_manages_device) {
            close_devices();
        }
        return false;
    }
}

}  // namespace detail
}  // namespace tt::tt_metal
