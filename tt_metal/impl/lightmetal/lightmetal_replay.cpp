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

// There are a bunch of things to do in this file and figure out
// 1. Executor: Open Flatbuffer binary, loop over contents, execute contents.
// 2. In order to do that, need deserialize/convert from flatbuffer representation
// 3. And have handlers to call Host API functions.

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



//////////////////////////////////////
// Debug Code                       //
//////////////////////////////////////

bool example_code() {
    int device_id = 0;
    tt_metal::Device* device = tt_metal::CreateDevice(device_id, 1, DEFAULT_L1_SMALL_SIZE, 900000000);
    tt_metal::CommandQueue& cq = device->command_queue();
    tt_metal::Program program = tt_metal::CreateProgram();
    bool pass = tt_metal::CloseDevice(device);
    return pass;
}

// Temporary debug function to print the contents of the FlatBuffer binary.
void LightMetalReplay::printLightMetalBinaryContents() {

    if (!lm_binary_) {
        std::cerr << "FlatBuffer binary not initialized." << std::endl;
        return;
    }

    const auto* trace_descriptors = lm_binary_->trace_descriptors();
    if (!trace_descriptors) {
        std::cout << "No trace descriptors found in the binary." << std::endl;
    } else {
        // Print all trace descriptors.
        std::cout << "Number of trace descriptors: " << trace_descriptors->size() << std::endl;
        for (const auto* descriptor_by_id : *trace_descriptors) {
            if (!descriptor_by_id) continue;

            uint32_t trace_id = descriptor_by_id->trace_id();
            const auto* trace_desc = descriptor_by_id->desc();

            if (!trace_desc) {
                std::cerr << "Descriptor is null for trace_id: " << trace_id << std::endl;
                continue;
            }

            // Print trace descriptor details.
            std::cout << "Trace ID: " << trace_id << std::endl;
            std::cout << "  Number of completion worker cores: "
                      << trace_desc->num_completion_worker_cores() << std::endl;
            std::cout << "  Number of programs needing multicast: "
                      << trace_desc->num_traced_programs_needing_go_signal_multicast() << std::endl;
            std::cout << "  Number of programs needing unicast: "
                      << trace_desc->num_traced_programs_needing_go_signal_unicast() << std::endl;

            // Print trace data.
            const auto* trace_data = trace_desc->trace_data();
            if (trace_data && trace_data->size() > 0) {
                std::cout << "  Trace Data (size: " << trace_data->size() << "): ";
                for (uint32_t value : *trace_data) {
                    std::cout << value << " ";
                }
                std::cout << std::endl;
            } else {
                std::cout << "  Trace Data: None" << std::endl;
            }
        }
    }

    // Print all commands.
    const auto* commands = lm_binary_->commands();
    if (!commands || commands->size() == 0) {
        std::cout << "No commands found in the binary." << std::endl;
    } else {
        std::cout << "Number of commands: " << commands->size() << std::endl;
        for (const auto* command : *commands) {
            if (!command) continue;

            auto cmd_type = command->cmd_type();
            switch (cmd_type) {
                case tt::target::CommandType::ReplayTraceCommand: {
                    const auto* cmd_variant = command->cmd_as_ReplayTraceCommand();
                    if (cmd_variant) {
                        std::cout << "ReplayTrace Command:" << std::endl;
                        std::cout << "  cq_id: " << cmd_variant->cq_id() << std::endl;
                        std::cout << "  tid: " << cmd_variant->tid() << std::endl;
                        std::cout << "  blocking: " << (cmd_variant->blocking() ? "true" : "false") << std::endl;
                    }
                    break;
                }
                case tt::target::CommandType::EnqueueTraceCommand: {
                    const auto* cmd_variant = command->cmd_as_EnqueueTraceCommand();
                    if (cmd_variant) {
                        std::cout << "EnqueueTrace Command:" << std::endl;
                        std::cout << "  cq_id: " << cmd_variant->cq_id() << std::endl;
                        std::cout << "  tid: " << cmd_variant->tid() << std::endl;
                        std::cout << "  blocking: " << (cmd_variant->blocking() ? "true" : "false") << std::endl;
                    }
                    break;
                }
                case tt::target::CommandType::LoadTraceCommand: {
                    const auto* cmd_variant = command->cmd_as_LoadTraceCommand();
                    if (cmd_variant) {
                        std::cout << "LoadTrace Command:" << std::endl;
                        std::cout << "  tid: " << cmd_variant->tid() << std::endl;
                        std::cout << "  cq_id: " << cmd_variant->cq_id() << std::endl;
                    }
                    break;
                }
                default:
                    std::cout << "Unsupported Command type: " << EnumNameCommandType(cmd_type) << std::endl;
                    break;
            }
        }
    }
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
  default:
    throw std::runtime_error("Unsupported type: " + std::string(EnumNameCommandType(command->cmd_type())));
    break;
  }
}

// Per API command handlers.
void LightMetalReplay::execute(tt::target::EnqueueTraceCommand const *cmd) {
    log_info(tt::LogMetalTrace, "KCM LightMetalReplay EnqueueTrace(). cq_id: {} tid: {} blocking: {}", cmd->cq_id(), cmd->tid(), cmd->blocking());
    // FIXME - Needs some tweaking, since API takes CQ should binarize cq_id and device_id.
    CommandQueue &cq = this->device_->command_queue(cmd->cq_id());
    EnqueueTrace(cq, cmd->tid(), cmd->blocking());
}

void LightMetalReplay::execute(tt::target::ReplayTraceCommand const *cmd) {
    log_info(tt::LogMetalTrace, "KCM LightMetalReplay ReplayTrace(). cq_id: {} tid: {} blocking: {}", cmd->cq_id(), cmd->tid(), cmd->blocking());
    ReplayTrace(this->device_, cmd->cq_id(), cmd->tid(), cmd->blocking());
}

void LightMetalReplay::execute(tt::target::LoadTraceCommand const *cmd) {
    log_info(tt::LogMetalTrace, "KCM LightMetalReplay LoadTrace(). cq_id: {} tid: {}", cmd->cq_id(), cmd->tid());
    // Get the trace descriptor from flatbuffer and load it to device.
    auto trace_desc = getTraceByTraceId(cmd->tid());
    LoadTrace(this->device_, cmd->cq_id(), cmd->tid(), trace_desc.value());
}

// Main entry point to execute a light metal binary blob, return true if pass.
bool LightMetalReplay::executeLightMetalBinary() {

    if (!lm_binary_) {
        std::cerr << "FlatBuffer binary not initialized." << std::endl;
        return false;
    }

    try {
        // example_code(); // Debug

        const auto* trace_descriptors = lm_binary_->trace_descriptors();
        const auto* commands = lm_binary_->commands();
        if (!commands) {
            std::cerr << "Nothing to run, no commands in binary." << std::endl;
            return false;
        }

        setupDevices();
        log_info(tt::LogMetalTrace, "KCM Executing Binary w/ cmds: {} traces: {}", commands->size(), trace_descriptors->size());

        // Just loop over all commands, and execute. This is purposely kept simple for prototyping v0,
        // should expand to cover multiple program, devices, cqs, etc. FIXME
        for (const auto* cmd : *commands) {
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
