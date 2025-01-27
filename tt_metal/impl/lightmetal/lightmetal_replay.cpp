// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "lightmetal_replay.hpp"
#include <iostream>
#include "binary_generated.h"
#include "command_generated.h"
#include <trace_buffer.hpp>
#include <tt-metalium/logger.hpp>

#include <host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/command_queue.hpp>
#include <tt-metalium/device_impl.hpp>
#include "flatbuffer/base_types_from_flatbuffer.hpp"
#include "flatbuffer/program_types_from_flatbuffer.hpp"
#include "flatbuffer/buffer_types_from_flatbuffer.hpp"

namespace tt::tt_metal {
inline namespace v0 {

//////////////////////////////////////
// Helper Functions                 //
//////////////////////////////////////

detail::TraceDescriptor FromFlatbuffer(const tt::tt_metal::flatbuffer::TraceDescriptor* fb_desc) {
    if (!fb_desc) {
        std::cerr << "TraceDescriptor is null." << std::endl;
        return {};
    }

    detail::TraceDescriptor trace_desc;

    // Deserialize trace_data
    if (auto trace_data_fb = fb_desc->trace_data()) {
        trace_desc.data.assign(trace_data_fb->begin(), trace_data_fb->end());
    }

    // Deserialize sub_device_descriptors
    if (auto sub_device_descriptors_fb = fb_desc->sub_device_descriptors()) {
        for (const auto* mapping : *sub_device_descriptors_fb) {
            if (mapping) {
                detail::TraceDescriptor::Descriptor descriptor;
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

// Needs access to BufferMap, so part of LightMetalReplay class
std::shared_ptr<RuntimeArgs> LightMetalReplay::FromFlatbufferRtArgs(const FlatbufferRuntimeArgVector flatbuffer_args) {
    auto runtime_args = std::make_shared<RuntimeArgs>();

    for (const auto& flatbuffer_arg : *flatbuffer_args) {
        const auto* runtime_arg = flatbuffer_arg;
        if (!runtime_arg) {
            throw std::runtime_error("Null RuntimeArg in FlatBuffer vector");
        }

        // Determine the type of the RuntimeArg
        switch (runtime_arg->value_type()) {
            case tt::tt_metal::flatbuffer::RuntimeArgValue::UInt32Value: {
                // Extract UInt32Value
                const auto* uint32_value = runtime_arg->value_as_UInt32Value();
                if (!uint32_value) {
                    throw std::runtime_error("Failed to read UInt32Value");
                }
                runtime_args->emplace_back(uint32_value->value());
                break;
            }
            case tt::tt_metal::flatbuffer::RuntimeArgValue::BufferGlobalId: {
                // Extract BufferGlobalId
                const auto* buffer_global_id = runtime_arg->value_as_BufferGlobalId();
                if (!buffer_global_id) {
                    throw std::runtime_error("Failed to read BufferGlobalId");
                }
                uint32_t global_id = buffer_global_id->id();
                auto buffer = GetBufferFromMap(global_id);
                if (!buffer) {
                    throw std::runtime_error(
                        "Buffer w/ global_id: " + std::to_string(global_id) + " not previously created");
                }
                runtime_args->emplace_back(buffer.get());
                break;
            }
            default: throw std::runtime_error("Unknown RuntimeArgValue type in FlatBuffer");
        }
    }

    return runtime_args;
}

//////////////////////////////////////
// LightMetalReplay Class           //
//////////////////////////////////////

LightMetalReplay::LightMetalReplay(LightMetalBinary&& binary_blob) :
    binary_blob_(std::move(binary_blob)), fb_binary_(nullptr) {
    if (binary_blob_.IsEmpty()) {
        log_warning(tt::LogMetalTrace, "Empty LightMetalBinary provided to LightMetalReplay.");
    }

    show_reads_ = parse_env("TT_LIGHT_METAL_SHOW_READS", false);
    disable_checking_ = parse_env("TT_LIGHT_METAL_DISABLE_CHECKING", false);
    fb_binary_ = ParseFlatBufferBinary();  // Parse and store the FlatBuffer binary
}

const tt::tt_metal::flatbuffer::LightMetalBinary* LightMetalReplay::ParseFlatBufferBinary() {
    try {
        const uint8_t* data = binary_blob_.data.data();
        size_t size = binary_blob_.data.size();

        // Verify the FlatBuffer data.
        flatbuffers::Verifier verifier(data, size);
        if (!tt::tt_metal::flatbuffer::VerifyLightMetalBinaryBuffer(verifier)) {
            std::cerr << "Failed to verify FlatBuffer data." << std::endl;
            return nullptr;
        }

        // Parse and return the FlatBuffer object.
        return tt::tt_metal::flatbuffer::GetLightMetalBinary(data);
    } catch (const std::exception& e) {
        std::cerr << "Exception while parsing FlatBuffer binary: " << e.what() << std::endl;
        return nullptr;
    }
}

// Return a TraceDescriptor for a given trace_id from the FlatBuffer binary.
std::optional<detail::TraceDescriptor> LightMetalReplay::GetTraceByTraceId(uint32_t target_trace_id) {
    if (const auto* trace_descriptors = fb_binary_ ? fb_binary_->trace_descriptors() : nullptr) {
        if (const auto* fb_trace_desc_by_id = trace_descriptors->LookupByKey(target_trace_id)) {
            if (const auto* fb_desc = fb_trace_desc_by_id->desc()) {
                return FromFlatbuffer(fb_desc);
            }
        }
    }

    std::cerr << "Failed to find trace_id: " << target_trace_id << " in binary." << std::endl;
    return std::nullopt;
}

//////////////////////////////////////
// Object Map Public Accessors      //
//////////////////////////////////////

void LightMetalReplay::AddBufferToMap(uint32_t global_id, const std::shared_ptr<::tt::tt_metal::Buffer>& buffer) {
    if (buffer_map_.find(global_id) != buffer_map_.end()) {
        log_warning(tt::LogMetalTrace, "Buffer with global_id: {} already exists in map.", global_id);
    }
    buffer_map_[global_id] = buffer;  // Shared ownership
}

std::shared_ptr<::tt::tt_metal::Buffer> LightMetalReplay::GetBufferFromMap(uint32_t global_id) const {
    auto it = buffer_map_.find(global_id);
    if (it != buffer_map_.end()) {
        return it->second;  // Return shared_ptr
    }
    return nullptr;  // If not found
}

void LightMetalReplay::RemoveBufferFromMap(uint32_t global_id) { buffer_map_.erase(global_id); }

void LightMetalReplay::AddProgramToMap(uint32_t global_id, const std::shared_ptr<::tt::tt_metal::Program>& program) {
    if (program_map_.find(global_id) != program_map_.end()) {
        log_warning(tt::LogMetalTrace, "Program with global_id: {} already exists in map.", global_id);
    }
    program_map_[global_id] = program;  // Shared ownership
}

std::shared_ptr<::tt::tt_metal::Program> LightMetalReplay::GetProgramFromMap(uint32_t global_id) const {
    auto it = program_map_.find(global_id);
    if (it != program_map_.end()) {
        return it->second;  // Return shared_ptr
    }
    return nullptr;  // If not found
}

void LightMetalReplay::RemoveProgramFromMap(uint32_t global_id) { program_map_.erase(global_id); }

void LightMetalReplay::AddKernelHandleToMap(uint32_t global_id, ::tt::tt_metal::KernelHandle kernel_id) {
    if (kernel_handle_map_.find(global_id) != kernel_handle_map_.end()) {
        log_warning(tt::LogMetalTrace, "KernelHandle with global_id: {} already exists in map.", global_id);
    }
    kernel_handle_map_[global_id] = kernel_id;  // Shared ownership
}

::tt::tt_metal::KernelHandle LightMetalReplay::GetKernelHandleFromMap(uint32_t global_id) const {
    if (auto it = kernel_handle_map_.find(global_id); it != kernel_handle_map_.end()) {
        return it->second;  // Return KernelHandle.
    }
    throw std::runtime_error(fmt::format("KernelHandle with global_id: {} used but doesn't exist.", global_id));
}

void LightMetalReplay::RemoveKernelHandleFromMap(uint32_t global_id) { kernel_handle_map_.erase(global_id); }

void LightMetalReplay::AddKernelToMap(uint32_t global_id, const std::shared_ptr<::tt::tt_metal::Kernel>& kernel) {
    if (kernel_map_.find(global_id) != kernel_map_.end()) {
        log_warning(tt::LogMetalTrace, "Kernel with global_id: {} already exists in map.", global_id);
    }
    kernel_map_[global_id] = kernel;  // Shared ownership
}

std::shared_ptr<::tt::tt_metal::Kernel> LightMetalReplay::GetKernelFromMap(uint32_t global_id) const {
    if (auto it = kernel_map_.find(global_id); it != kernel_map_.end()) {
        return it->second;  // Return Kernel.
    }
    throw std::runtime_error(fmt::format("Kernel with global_id: {} used but doesn't exist.", global_id));
}

void LightMetalReplay::RemoveKernelFromMap(uint32_t global_id) { kernel_map_.erase(global_id); }

void LightMetalReplay::AddCBHandleToMap(uint32_t global_id, ::tt::tt_metal::CBHandle cb_handle) {
    if (cb_handle_map_.find(global_id) != cb_handle_map_.end()) {
        log_warning(tt::LogMetalTrace, "CBHandle with global_id: {} already exists in map.", global_id);
    }
    cb_handle_map_[global_id] = cb_handle;  // Shared ownership
}

::tt::tt_metal::CBHandle LightMetalReplay::GetCBHandleFromMap(uint32_t global_id) const {
    if (auto it = cb_handle_map_.find(global_id); it != cb_handle_map_.end()) {
        return it->second;  // Return CBHandle.
    }
    throw std::runtime_error(fmt::format("CBHandle with global_id: {} used but doesn't exist.", global_id));
}

void LightMetalReplay::RemoveCBHandleFromMap(uint32_t global_id) { cb_handle_map_.erase(global_id); }

//////////////////////////////////////
// Device Setup/Teardown            //
//////////////////////////////////////

// TODO (kmabee) - Hardcode for now, eventually capture/replay "systemdesc" from binary.
void LightMetalReplay::SetupDevices() {
    log_debug(tt::LogMetalTrace, "LightMetalReplay(SetupDevices) - Using single chip WH device as temp hack.");
    const size_t trace_region_size = 4096;  // Default is 0
    const int device_id = 0;
    const auto dispatch_core_type = tt_metal::DispatchCoreType::WORKER;
    const chip_id_t mmio_device_id = 0;
    auto devices_map = tt::tt_metal::detail::CreateDevices(
        {mmio_device_id}, 1, DEFAULT_L1_SMALL_SIZE, trace_region_size, dispatch_core_type);
    this->device_ = devices_map.at(mmio_device_id);
}

// TODO (kmabee) - Hardcode for now, eventually capture/replay "systemdesc" from binary or let user call.
void LightMetalReplay::CloseDevices() { CloseDevice(this->device_); }

//////////////////////////////////////
// Executor                         //
//////////////////////////////////////

// Execute a command by dispatching to appropriate handler based on type.
void LightMetalReplay::Execute(const tt::tt_metal::flatbuffer::Command* command) {
    switch (command->cmd_type()) {
        case ::tt::tt_metal::flatbuffer::CommandType::EnqueueTraceCommand: {
            Execute(command->cmd_as_EnqueueTraceCommand());
            break;
        }
        case ::tt::tt_metal::flatbuffer::CommandType::ReplayTraceCommand: {
            Execute(command->cmd_as_ReplayTraceCommand());
            break;
        }
        case ::tt::tt_metal::flatbuffer::CommandType::LoadTraceCommand: {
            Execute(command->cmd_as_LoadTraceCommand());
            break;
        }
        case ::tt::tt_metal::flatbuffer::CommandType::ReleaseTraceCommand: {
            Execute(command->cmd_as_ReleaseTraceCommand());
            break;
        }
        case ::tt::tt_metal::flatbuffer::CommandType::CreateBufferCommand: {
            Execute(command->cmd_as_CreateBufferCommand());
            break;
        }
        case ::tt::tt_metal::flatbuffer::CommandType::DeallocateBufferCommand: {
            Execute(command->cmd_as_DeallocateBufferCommand());
            break;
        }
        case ::tt::tt_metal::flatbuffer::CommandType::EnqueueWriteBufferCommand: {
            Execute(command->cmd_as_EnqueueWriteBufferCommand());
            break;
        }
        case ::tt::tt_metal::flatbuffer::CommandType::EnqueueReadBufferCommand: {
            Execute(command->cmd_as_EnqueueReadBufferCommand());
            break;
        }
        case ::tt::tt_metal::flatbuffer::CommandType::FinishCommand: {
            Execute(command->cmd_as_FinishCommand());
            break;
        }
        case ::tt::tt_metal::flatbuffer::CommandType::CreateProgramCommand: {
            Execute(command->cmd_as_CreateProgramCommand());
            break;
        }
        case ::tt::tt_metal::flatbuffer::CommandType::EnqueueProgramCommand: {
            Execute(command->cmd_as_EnqueueProgramCommand());
            break;
        }
        case ::tt::tt_metal::flatbuffer::CommandType::CreateKernelCommand: {
            Execute(command->cmd_as_CreateKernelCommand());
            break;
        }
        case ::tt::tt_metal::flatbuffer::CommandType::SetRuntimeArgsUint32Command: {
            Execute(command->cmd_as_SetRuntimeArgsUint32Command());
            break;
        }
        case ::tt::tt_metal::flatbuffer::CommandType::SetRuntimeArgsCommand: {
            Execute(command->cmd_as_SetRuntimeArgsCommand());
            break;
        }
        case ::tt::tt_metal::flatbuffer::CommandType::CreateCircularBufferCommand: {
            Execute(command->cmd_as_CreateCircularBufferCommand());
            break;
        }
        case ::tt::tt_metal::flatbuffer::CommandType::LightMetalCompareCommand: {
            Execute(command->cmd_as_LightMetalCompareCommand());
            break;
        }
        default:
            throw std::runtime_error("Unsupported type: " + std::string(EnumNameCommandType(command->cmd_type())));
            break;
    }
}

// Per API command handlers.
void LightMetalReplay::Execute(const tt::tt_metal::flatbuffer::EnqueueTraceCommand* cmd) {
    log_debug(
        tt::LogMetalTrace,
        "LightMetalReplay(EnqueueTrace) cq_id: {} tid: {} blocking: {}",
        cmd->cq_id(),
        cmd->tid(),
        cmd->blocking());
    CommandQueue& cq = this->device_->command_queue(cmd->cq_id());
    EnqueueTrace(cq, cmd->tid(), cmd->blocking());
}

void LightMetalReplay::Execute(const tt::tt_metal::flatbuffer::ReplayTraceCommand* cmd) {
    log_debug(
        tt::LogMetalTrace,
        "LightMetalReplay(ReplayTrace) cq_id: {} tid: {} blocking: {}",
        cmd->cq_id(),
        cmd->tid(),
        cmd->blocking());
    ReplayTrace(this->device_, cmd->cq_id(), cmd->tid(), cmd->blocking());
}

void LightMetalReplay::Execute(const tt::tt_metal::flatbuffer::LoadTraceCommand* cmd) {
    log_debug(tt::LogMetalTrace, "LightMetalReplay(LoadTrace) cq_id: {} tid: {}", cmd->cq_id(), cmd->tid());
    // Get the trace descriptor from flatbuffer and load it to device.
    auto trace_desc = GetTraceByTraceId(cmd->tid());
    LoadTrace(this->device_, cmd->cq_id(), cmd->tid(), trace_desc.value());
}

void LightMetalReplay::Execute(const tt::tt_metal::flatbuffer::ReleaseTraceCommand* cmd) {
    log_debug(tt::LogMetalTrace, "LightMetalReplay(ReleaseTrace) tid: {}", cmd->tid());
    ReleaseTrace(this->device_, cmd->tid());
}

void LightMetalReplay::Execute(const tt::tt_metal::flatbuffer::CreateBufferCommand* cmd) {
    log_debug(
        tt::LogMetalTrace,
        "LightMetalReplay(CreateBuffer) global_id: {} size: {} page_size: {} layout: {} buffer_type: {}",
        cmd->global_id(),
        cmd->config()->size(),
        cmd->config()->page_size(),
        EnumNameTensorMemoryLayout(cmd->config()->buffer_layout()),
        EnumNameBufferType(cmd->config()->buffer_type()));

    switch (cmd->config()->buffer_layout()) {
        case tt::tt_metal::flatbuffer::TensorMemoryLayout::Interleaved: {
            tt::tt_metal::InterleavedBufferConfig config{
                .device = this->device_,
                .size = cmd->config()->size(),
                .page_size = cmd->config()->page_size(),
                .buffer_type = FromFlatbuffer(cmd->config()->buffer_type())};

            auto buffer = CreateBuffer(config);
            AddBufferToMap(cmd->global_id(), buffer);
            break;
        }
        default:
            throw std::runtime_error(
                "Unsupported buffer_layout: " +
                std::string(EnumNameTensorMemoryLayout(cmd->config()->buffer_layout())));
    }
}

void LightMetalReplay::Execute(const tt::tt_metal::flatbuffer::DeallocateBufferCommand* cmd) {
    auto buffer = GetBufferFromMap(cmd->global_id());
    if (!buffer) {
        throw std::runtime_error(
            "Buffer w/ global_id: " + std::to_string(cmd->global_id()) + " not previously created");
    }

    log_debug(tt::LogMetalTrace, "LightMetalReplay(DeallocateBuffer) global_id: {}", cmd->global_id());
    DeallocateBuffer(*buffer);  // Buffer& expected.
    RemoveBufferFromMap(cmd->global_id());
}

void LightMetalReplay::Execute(const tt::tt_metal::flatbuffer::EnqueueWriteBufferCommand* cmd) {
    auto buffer = GetBufferFromMap(cmd->buffer_global_id());
    if (!buffer) {
        throw std::runtime_error(
            "Buffer w/ global_id: " + std::to_string(cmd->buffer_global_id()) + " not previously created");
    }

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

void LightMetalReplay::Execute(const tt::tt_metal::flatbuffer::EnqueueReadBufferCommand* cmd) {
    auto buffer = GetBufferFromMap(cmd->buffer_global_id());
    if (!buffer) {
        throw std::runtime_error(
            "Buffer w/ global_id: " + std::to_string(cmd->buffer_global_id()) + " not previously created");
    }

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

void LightMetalReplay::Execute(const tt::tt_metal::flatbuffer::FinishCommand* cmd) {
    log_debug(tt::LogMetalTrace, "LightMetalReplay(Finish) cq_global_id: {}", cmd->cq_global_id());
    CommandQueue& cq = this->device_->command_queue(cmd->cq_global_id());
    auto sub_device_ids = FromFlatBuffer(cmd->sub_device_ids());
    Finish(cq, sub_device_ids);
}

void LightMetalReplay::Execute(const tt::tt_metal::flatbuffer::CreateProgramCommand* cmd) {
    log_debug(tt::LogMetalTrace, "LightMetalReplay(CreateProgram) global_id: {} ", cmd->global_id());
    auto program = CreateProgram();
    AddProgramToMap(cmd->global_id(), std::make_shared<Program>(std::move(program)));
}

void LightMetalReplay::Execute(const tt::tt_metal::flatbuffer::EnqueueProgramCommand* cmd) {
    auto program = GetProgramFromMap(cmd->program_global_id());
    if (!program) {
        throw std::runtime_error(
            "Program with global_id: " + std::to_string(cmd->program_global_id()) + " not previously created");
    }
    log_debug(
        tt::LogMetalTrace,
        "LightMetalReplay(EnqueueProgram) program_global_id: {} cq_global_id: {}",
        cmd->program_global_id(),
        cmd->cq_global_id());

    // TODO (kmabee) - consider storing/getting CQ from global map instead.
    CommandQueue& cq = this->device_->command_queue(cmd->cq_global_id());
    EnqueueProgram(cq, *program, cmd->blocking());
}

void LightMetalReplay::Execute(const tt::tt_metal::flatbuffer::CreateKernelCommand* cmd) {
    log_debug(
        tt::LogMetalTrace,
        "LightMetalReplay(CreateKernel) global_id: {} program_global_id: {}",
        cmd->global_id(),
        cmd->program_global_id());
    auto program = GetProgramFromMap(cmd->program_global_id());
    if (!program) {
        throw std::runtime_error(
            "Program with global_id: " + std::to_string(cmd->program_global_id()) + " not previously created");
    }

    auto core_spec = FromFlatbuffer(cmd->core_spec_type(), cmd->core_spec());
    auto kernel_config = FromFlatbuffer(cmd->config_type(), cmd->config());
    auto kernel_id = CreateKernel(*program, cmd->file_name()->c_str(), core_spec, kernel_config);
    AddKernelHandleToMap(cmd->global_id(), kernel_id);
    // Some APIs use Kernel, so convert to and store Kernel.
    std::shared_ptr<Kernel> kernel = program->get_kernel(kernel_id);
    AddKernelToMap(cmd->global_id(), kernel);
}

void LightMetalReplay::Execute(const tt::tt_metal::flatbuffer::SetRuntimeArgsUint32Command* cmd) {
    log_debug(
        tt::LogMetalTrace,
        "LightMetalReplay(SetRuntimeArgs). program_global_id: {} kernel_global_id: {}",
        cmd->program_global_id(),
        cmd->kernel_global_id());
    auto program = GetProgramFromMap(cmd->program_global_id());
    auto kernel_id = GetKernelHandleFromMap(cmd->kernel_global_id());

    if (!program) {
        throw std::runtime_error(
            "Program with global_id: " + std::to_string(cmd->program_global_id()) + " not previously created");
    }

    // API expects a span so create from flatbuffer vector.
    stl::Span<const uint32_t> args_span(cmd->args()->data(), cmd->args()->size());
    auto core_spec = FromFlatbuffer(cmd->core_spec_type(), cmd->core_spec());
    SetRuntimeArgs(*program, kernel_id, core_spec, args_span);
}

void LightMetalReplay::Execute(const tt::tt_metal::flatbuffer::SetRuntimeArgsCommand* cmd) {
    log_debug(
        tt::LogMetalTrace,
        "LightMetalReplay(SetRuntimeArgs). kernel_global_id: {} rt_args_size: {}",
        cmd->kernel_global_id(),
        cmd->args()->size());
    auto core_spec = FromFlatbuffer(cmd->core_spec_type(), cmd->core_spec());
    auto runtime_args = FromFlatbufferRtArgs(cmd->args());
    auto kernel = GetKernelFromMap(cmd->kernel_global_id());
    if (!kernel) {
        throw std::runtime_error(
            "Kernel with global_id: " + std::to_string(cmd->kernel_global_id()) + " not previously created");
    }
    SetRuntimeArgs(this->device_, kernel, core_spec, runtime_args);
}

void LightMetalReplay::Execute(const tt::tt_metal::flatbuffer::CreateCircularBufferCommand* cmd) {
    log_debug(
        tt::LogMetalTrace,
        "LightMetalReplay(CreateCircularBuffer) global_id: {} program_global_id: {}",
        cmd->global_id(),
        cmd->program_global_id());
    auto program = GetProgramFromMap(cmd->program_global_id());
    if (!program) {
        throw std::runtime_error(
            "Program with global_id: " + std::to_string(cmd->program_global_id()) + " not previously created");
    }

    auto core_spec = FromFlatbuffer(cmd->core_spec_type(), cmd->core_spec());

    // Convert global_id to optional Shadow Buffer here to keep FromFlatbuffer standalone function.
    ::tt::tt_metal::Buffer* shadow_global_buffer = nullptr;
    auto shadow_buf_global_id = cmd->config()->shadow_buf_global_id();

    if (shadow_buf_global_id != 0) {
        auto shadow_buf = GetBufferFromMap(shadow_buf_global_id);
        if (!shadow_buf) {
            throw std::runtime_error(
                "Shadow Buffer w/ global_id: " + std::to_string(shadow_buf_global_id) + " not previously created");
        }
        shadow_global_buffer = shadow_buf.get();  // Set the raw pointer
    }

    auto config = FromFlatbuffer(cmd->config(), shadow_global_buffer);
    auto cb_handle = CreateCircularBuffer(*program, core_spec, config);
    AddCBHandleToMap(cmd->global_id(), cb_handle);
}

// Verification command to compare readback of a buffer with golden from either capture or user expected values.
void LightMetalReplay::Execute(const ::tt::tt_metal::flatbuffer::LightMetalCompareCommand* cmd) {
    log_debug(
        tt::LogMetalTrace,
        "LightMetalReplay(LightMetalCompare) cq_global_id: {} buffer_global_id: {} is_user_data: {}",
        cmd->cq_global_id(),
        cmd->buffer_global_id(),
        cmd->is_user_data());

    auto buffer = GetBufferFromMap(cmd->buffer_global_id());
    if (!buffer) {
        throw std::runtime_error(
            "Buffer w/ global_id: " + std::to_string(cmd->buffer_global_id()) + " not previously created");
    }

    // TODO (kmabee) - consider storing/getting CQ from global map instead.
    CommandQueue& cq = this->device_->command_queue(cmd->cq_global_id());
    std::vector<uint32_t> rd_data(buffer->size() / sizeof(uint32_t), 0);
    EnqueueReadBuffer(cq, buffer, rd_data.data(), true);

    if (disable_checking_) {
        log_debug(
            tt::LogMetalTrace, "Skipping LightMetalCompareCommand for buffer_global_id: {}.", cmd->buffer_global_id());
    } else {
        if (rd_data.size() != cmd->golden_data()->size()) {
            throw std::runtime_error(
                "Readback data size: " + std::to_string(rd_data.size()) +
                " does not match golden data size: " + std::to_string(cmd->golden_data()->size()));
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
            throw std::runtime_error(
                "Golden vs rd_data mismatch for buffer_global_id: " + std::to_string(cmd->buffer_global_id()));
        }
    }
}

// Main entry point to execute a light metal binary blob, return true if pass.
bool LightMetalReplay::ExecuteLightMetalBinary() {
    if (!fb_binary_) {
        std::cerr << "Cannot Replay empty/uninitialized Light Metal Binary." << std::endl;
        return false;
    }

    try {
        const auto* trace_descs = fb_binary_->trace_descriptors();
        const auto* commands = fb_binary_->commands();
        if (!commands) {
            std::cerr << "Nothing to run, no commands in binary." << std::endl;
            return false;
        }

        SetupDevices();
        log_info(
            tt::LogMetalTrace,
            "Running LightMetal Binary with {} cmds, {} traces.",
            commands->size(),
            trace_descs->size());

        // Just loop over all commands, and execute. This is purposely kept simple for prototyping v0.
        // TODO (kmabee) - should expand to cover, multiple devices, cqs, etc.
        uint32_t idx = 1;
        for (const auto* cmd : *commands) {
            auto str_name = std::string(EnumNameCommandType(cmd->cmd_type()));
            log_trace(tt::LogMetalTrace, "Executing Binary CMD {}/{} (Type: {})", idx++, commands->size(), str_name);
            Execute(cmd);
        }

        CloseDevices();

        return true;
    } catch (const std::exception& e) {
        log_fatal(e.what());
        return false;
    }
}

}  // namespace v0
}  // namespace tt::tt_metal
