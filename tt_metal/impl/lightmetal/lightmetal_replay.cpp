// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "lightmetal_replay.hpp"
#include <iostream>
#include "binary_generated.h"
#include "command_generated.h"
#include "tt_metal/impl/trace/trace_buffer.hpp"
#include "tt_metal/common/logger.hpp"

#include "tt_metal/host_api.hpp"
#include "tt_metal/detail/tt_metal.hpp"
#include "tt_metal/impl/dispatch/command_queue.hpp"
#include "tt_metal/impl/device/device.hpp"

namespace tt::tt_metal {
inline namespace v0 {

//////////////////////////////////////
// Helper Functions                 //
//////////////////////////////////////

// A convenience function - Read arbitrary binary blob from file.
void readBinaryBlobFromFile(const std::string& filename, std::vector<uint8_t>& blob) {
    std::ifstream file(filename, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file: " + filename);
    }

    std::streamsize size = file.tellg();
    if (size <= 0) {
        throw std::runtime_error("File is empty or invalid: " + filename);
    }

    blob.resize(static_cast<size_t>(size));

    file.seekg(0, std::ios::beg);
    if (!file.read(reinterpret_cast<char*>(blob.data()), size)) {
        throw std::runtime_error("Failed to read file: " + filename);
    }
}

detail::TraceDescriptor fromFlatBuffer(const tt::target::lightmetal::TraceDescriptor* fb_desc) {
    if (!fb_desc) {
        std::cerr << "TraceDescriptor is null." << std::endl;
        return {};
    }

    detail::TraceDescriptor traceDesc;
    if (auto trace_data_fb = fb_desc->trace_data()) {
        traceDesc.data.assign(trace_data_fb->begin(), trace_data_fb->end());
    }
    traceDesc.num_completion_worker_cores = fb_desc->num_completion_worker_cores();
    traceDesc.num_traced_programs_needing_go_signal_multicast = fb_desc->num_traced_programs_needing_go_signal_multicast();
    traceDesc.num_traced_programs_needing_go_signal_unicast = fb_desc->num_traced_programs_needing_go_signal_unicast();

    return traceDesc;
}

inline BufferType fromFlatbuffer(tt::target::BufferType type) {
    switch (type) {
        case tt::target::BufferType::DRAM: return BufferType::DRAM;
        case tt::target::BufferType::L1: return BufferType::L1;
        case tt::target::BufferType::SystemMemory: return BufferType::SYSTEM_MEMORY;
        case tt::target::BufferType::L1Small: return BufferType::L1_SMALL;
        case tt::target::BufferType::Trace: return BufferType::TRACE;
        default: throw std::invalid_argument("Unknown tt::target::BufferType value");
    }
}

//////////////////////////////////////
// LightMetalReplay Class           //
//////////////////////////////////////

LightMetalReplay::LightMetalReplay(std::vector<uint8_t> blob)
    : blob_(std::move(blob)), lm_binary_(nullptr) {
    lm_binary_ = parseFlatBufferBinary();  // Parse and store the FlatBuffer binary
    if (!lm_binary_) {
        throw std::runtime_error("Failed to parse FlatBuffer binary during initialization.");
    }
}

const target::lightmetal::LightMetalBinary* LightMetalReplay::parseFlatBufferBinary() {
    try {
        const uint8_t* data = blob_.data();
        size_t size = blob_.size();

        // Verify the FlatBuffer data.
        flatbuffers::Verifier verifier(data, size);
        if (!target::lightmetal::VerifyLightMetalBinaryBuffer(verifier)) {
            std::cerr << "Failed to verify FlatBuffer data." << std::endl;
            return nullptr;
        }

        // Parse and return the FlatBuffer object.
        return target::lightmetal::GetLightMetalBinary(data);
    } catch (const std::exception& e) {
        std::cerr << "Exception while parsing FlatBuffer binary: " << e.what() << std::endl;
        return nullptr;
    }
}

// Return a TraceDescriptor for a given trace_id from the FlatBuffer binary.
std::optional<detail::TraceDescriptor> LightMetalReplay::getTraceByTraceId(uint32_t target_trace_id) {
    if (const auto* trace_descriptors = lm_binary_ ? lm_binary_->trace_descriptors() : nullptr) {
        if (const auto* fb_trace_desc_by_id = trace_descriptors->LookupByKey(target_trace_id)) {
            if (const auto* fb_desc = fb_trace_desc_by_id->desc()) {
                return fromFlatBuffer(fb_desc);
            }
        }
    }

    std::cerr << "Failed to find trace_id: " << target_trace_id << " in binary." << std::endl;
    return std::nullopt;
}


// Object maps public accessors
void LightMetalReplay::addBufferToMap(uint32_t global_id, std::shared_ptr<::tt::tt_metal::Buffer> buffer) {
    if (bufferMap_.find(global_id) != bufferMap_.end()) {
        log_warning(tt::LogMetalTrace, "Buffer with global_id: {} already exists in map.", global_id);
    }
    bufferMap_[global_id] = buffer; // Shared ownership
}

std::shared_ptr<::tt::tt_metal::Buffer> LightMetalReplay::getBufferFromMap(uint32_t global_id) const {
    auto it = bufferMap_.find(global_id);
    if (it != bufferMap_.end()) {
        return it->second; // Return shared_ptr
    }
    return nullptr; // If not found
}

void LightMetalReplay::removeBufferFromMap(uint32_t global_id) {
    bufferMap_.erase(global_id);
}

void LightMetalReplay::setupDevices() {
    log_info(tt::LogMetalTrace, "Setting up system now...");

    // FIXME - Get these from lm_binary_ systemdesc once available. For now hardcode.
    const size_t buffer_size = 2048;
    this->arch_ = tt::ARCH::WORMHOLE_B0;
    const int device_id = 0;
    const auto dispatch_core_type = tt_metal::DispatchCoreType::WORKER;
    const chip_id_t mmio_device_id = 0;
    auto devices_map = tt::tt_metal::detail::CreateDevices({mmio_device_id}, 1, DEFAULT_L1_SMALL_SIZE, buffer_size, dispatch_core_type);
    this->device_ = devices_map.at(mmio_device_id);
}

//////////////////////////////////////
// Executor                         //
//////////////////////////////////////

// Some open questions...
// 1. How to pass Device* to replay functions? Can use a global variable for now.
// 2. How to pass other things like input tensors?
// 3. Can we fully encapsulate each host API command here.


// Execute a command by dispatching to appropriate handler based on type.
void LightMetalReplay::execute(tt::target::Command const *command) {
  switch (command->cmd_type()) {
  case ::tt::target::CommandType::EnqueueTraceCommand: {
    execute(command->cmd_as_EnqueueTraceCommand());
    break;
  }
  case ::tt::target::CommandType::ReplayTraceCommand: {
    execute(command->cmd_as_ReplayTraceCommand());
    break;
  }
  case ::tt::target::CommandType::LoadTraceCommand: {
    execute(command->cmd_as_LoadTraceCommand());
    break;
  }
  case ::tt::target::CommandType::ReleaseTraceCommand: {
    execute(command->cmd_as_ReleaseTraceCommand());
    break;
  }
  case ::tt::target::CommandType::CreateBufferCommand: {
    execute(command->cmd_as_CreateBufferCommand());
    break;
  }
  case ::tt::target::CommandType::DeallocateBufferCommand: {
    execute(command->cmd_as_DeallocateBufferCommand());
    break;
  }
  case ::tt::target::CommandType::EnqueueWriteBufferCommand: {
    execute(command->cmd_as_EnqueueWriteBufferCommand());
    break;
  }
  case ::tt::target::CommandType::EnqueueReadBufferCommand: {
    execute(command->cmd_as_EnqueueReadBufferCommand());
    break;
  }
  case ::tt::target::CommandType::FinishCommand: {
    execute(command->cmd_as_FinishCommand());
    break;
  }
  default:
    throw std::runtime_error("Unsupported type: " + std::string(EnumNameCommandType(command->cmd_type())));
    break;
  }
}

// Per API command handlers.
void LightMetalReplay::execute(tt::target::EnqueueTraceCommand const *cmd) {
    log_info(tt::LogMetalTrace, "LightMetalReplay EnqueueTrace(). cq_id: {} tid: {} blocking: {}", cmd->cq_id(), cmd->tid(), cmd->blocking());
    // FIXME - Needs some tweaking, since API takes CQ should binarize cq_id and device_id.
    CommandQueue &cq = this->device_->command_queue(cmd->cq_id());
    EnqueueTrace(cq, cmd->tid(), cmd->blocking());
}

void LightMetalReplay::execute(tt::target::ReplayTraceCommand const *cmd) {
    log_info(tt::LogMetalTrace, "LightMetalReplay ReplayTrace(). cq_id: {} tid: {} blocking: {}", cmd->cq_id(), cmd->tid(), cmd->blocking());
    ReplayTrace(this->device_, cmd->cq_id(), cmd->tid(), cmd->blocking());
}

void LightMetalReplay::execute(tt::target::LoadTraceCommand const *cmd) {
    log_info(tt::LogMetalTrace, "LightMetalReplay LoadTrace(). cq_id: {} tid: {}", cmd->cq_id(), cmd->tid());
    // Get the trace descriptor from flatbuffer and load it to device.
    auto trace_desc = getTraceByTraceId(cmd->tid());
    LoadTrace(this->device_, cmd->cq_id(), cmd->tid(), trace_desc.value());
}

void LightMetalReplay::execute(tt::target::ReleaseTraceCommand const *cmd) {
    log_info(tt::LogMetalTrace, "LightMetalReplay ReleaseTrace(). tid: {}", cmd->tid());
    ReleaseTrace(this->device_, cmd->tid());
}

void LightMetalReplay::execute(tt::target::CreateBufferCommand const *cmd) {
    log_info(tt::LogMetalTrace, "LightMetalReplay CreateBufferCommand(). global_id: {} size: {} page_size: {} layout: {} buffer_type: {}",
        cmd->global_id(), cmd->config()->size(), cmd->config()->page_size(),
        EnumNameTensorMemoryLayout(cmd->config()->buffer_layout()), EnumNameBufferType(cmd->config()->buffer_type()));

    switch (cmd->config()->buffer_layout()) {
    case tt::target::TensorMemoryLayout::Interleaved: {
        tt::tt_metal::InterleavedBufferConfig config{
            .device = this->device_,
            .size = cmd->config()->size(),
            .page_size = cmd->config()->page_size(),
            .buffer_type = fromFlatbuffer(cmd->config()->buffer_type())};

        auto buffer = CreateBuffer(config);
        addBufferToMap(cmd->global_id(), buffer);
        break;
    }
    default:
        throw std::runtime_error("Unsupported buffer_layout: " + std::string(EnumNameTensorMemoryLayout(cmd->config()->buffer_layout())));
    }
}

void LightMetalReplay::execute(tt::target::DeallocateBufferCommand const *cmd) {
    auto buffer = getBufferFromMap(cmd->global_id());
    if (!buffer) {
        throw std::runtime_error("Buffer w/ global_id: " + std::to_string(cmd->global_id()) + " not previously created");
    }
    DeallocateBuffer(*buffer); // Buffer& expected.
    removeBufferFromMap(cmd->global_id());
}

void LightMetalReplay::execute(tt::target::EnqueueWriteBufferCommand const *cmd) {
    auto buffer = getBufferFromMap(cmd->buffer_global_id());
    if (!buffer) {
        throw std::runtime_error("Buffer w/ global_id: " + std::to_string(cmd->buffer_global_id()) + " not previously created");
    }

    log_info(tt::LogMetalTrace, "LightMetalReplay EnqueueWriteBufferCommand(). cq_global_id: {} buffer_global_id: {} addr: 0x{:x}",
        cmd->cq_global_id(), cmd->buffer_global_id(), buffer->address());

    // FIXME - get cq object from global CQ map instead.
    CommandQueue &cq = this->device_->command_queue(cmd->cq_global_id());
    EnqueueWriteBuffer(cq, buffer, cmd->src()->data(), cmd->blocking());
}

void LightMetalReplay::execute(tt::target::EnqueueReadBufferCommand const *cmd) {
    auto buffer = getBufferFromMap(cmd->buffer_global_id());
    if (!buffer) {
        throw std::runtime_error("Buffer w/ global_id: " + std::to_string(cmd->buffer_global_id()) + " not previously created");
    }

    log_info(tt::LogMetalTrace, "LightMetalReplay EnqueueReadBufferCommand(). cq_global_id: {} buffer_global_id: {} addr: 0x{:x} buf_size: {}",
        cmd->cq_global_id(), cmd->buffer_global_id(), buffer->address(), buffer->size());

    // FIXME - get cq object from global CQ map instead.
    CommandQueue &cq = this->device_->command_queue(cmd->cq_global_id());
    std::vector<uint32_t> readback_data(buffer->size() / sizeof(uint32_t), 0);
    EnqueueReadBuffer(cq, buffer, readback_data.data(), cmd->blocking());

    // FIXME - What should we do with readback data? For not just print.
    // One idea is to store in map by global_read_id that caller can access.
    bool show_reads = std::getenv("SHOW_READS");
    if (show_reads) {
        for (size_t i = 0; i < readback_data.size(); i++) {
            log_info(tt::LogMetalTrace, " rd_data i: {:3d} => data: {}", i, readback_data[i]);
        }
    }
}

void LightMetalReplay::execute(tt::target::FinishCommand const *cmd) {
    log_info(tt::LogMetalTrace, "LightMetalReplay FinishCommand(). cq_global_id: {}", cmd->cq_global_id());
    CommandQueue &cq = this->device_->command_queue(cmd->cq_global_id());
    Finish(cq);
}

// Main entry point to execute a light metal binary blob, return true if pass.
bool LightMetalReplay::executeLightMetalBinary() {

    if (!lm_binary_) {
        std::cerr << "FlatBuffer binary not initialized." << std::endl;
        return false;
    }

    try {
        const auto* trace_descriptors = lm_binary_->trace_descriptors();
        const auto* commands = lm_binary_->commands();
        if (!commands) {
            std::cerr << "Nothing to run, no commands in binary." << std::endl;
            return false;
        }

        setupDevices();
        log_info(tt::LogMetalTrace, "Executing Binary w/ cmds: {} traces: {}", commands->size(), trace_descriptors->size());

        // Just loop over all commands, and execute. This is purposely kept simple for prototyping v0,
        // should expand to cover multiple program, devices, cqs, etc. FIXME
        uint32_t cmd_idx = 1; // Debug
        for (const auto* cmd : *commands) {
            log_info(tt::LogMetalTrace, "Executing Binary CMD {}/{} (Type: {})", cmd_idx++, commands->size(), std::string(EnumNameCommandType(cmd->cmd_type())));
            execute(cmd);
        }

        return true;
    } catch (const std::exception& e) {
        log_fatal(e.what());
        return false;
    }
}


}  // namespace v0
}  // namespace tt::tt_metal
