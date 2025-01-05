// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
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

namespace tt::tt_metal {
inline namespace v0 {

//////////////////////////////////////
// Helper Functions                 //
//////////////////////////////////////

// A convenience function - Read arbitrary binary blob from file.
void ReadBinaryBlobFromFile(const std::string& filename, std::vector<uint8_t>& blob) {
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

detail::TraceDescriptor FromFlatbuffer(const tt::target::lightmetal::TraceDescriptor* fb_desc) {
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

//////////////////////////////////////
// LightMetalReplay Class           //
//////////////////////////////////////

LightMetalReplay::LightMetalReplay(std::vector<uint8_t>&& blob) : blob_(std::move(blob)), lm_binary_(nullptr) {
    lm_binary_ = ParseFlatBufferBinary();  // Parse and store the FlatBuffer binary
    if (!lm_binary_) {
        throw std::runtime_error("Failed to parse FlatBuffer binary during initialization.");
    }
}

const target::lightmetal::LightMetalBinary* LightMetalReplay::ParseFlatBufferBinary() {
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
std::optional<detail::TraceDescriptor> LightMetalReplay::GetTraceByTraceId(uint32_t target_trace_id) {
    if (const auto* trace_descriptors = lm_binary_ ? lm_binary_->trace_descriptors() : nullptr) {
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
// Device Setup/Teardown            //
//////////////////////////////////////

// TODO (kmabee) - Hardcode for now, eventually capture/replay "systemdesc" from binary.
void LightMetalReplay::SetupDevices() {
    log_debug(tt::LogMetalTrace, "LightMetalReplay(SetupDevices) - Using single chip WH device as temp hack.");
    const size_t trace_region_size = 2048;  // Default is 0
    this->arch_ = tt::ARCH::WORMHOLE_B0;
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
void LightMetalReplay::Execute(const tt::target::Command* command) {
    switch (command->cmd_type()) {
        case ::tt::target::CommandType::EnqueueTraceCommand: {
            Execute(command->cmd_as_EnqueueTraceCommand());
            break;
        }
        case ::tt::target::CommandType::ReplayTraceCommand: {
            Execute(command->cmd_as_ReplayTraceCommand());
            break;
        }
        case ::tt::target::CommandType::LoadTraceCommand: {
            Execute(command->cmd_as_LoadTraceCommand());
            break;
        }
        default:
            throw std::runtime_error("Unsupported type: " + std::string(EnumNameCommandType(command->cmd_type())));
            break;
    }
}

// Per API command handlers.
void LightMetalReplay::Execute(const tt::target::EnqueueTraceCommand* cmd) {
    log_info(
        tt::LogMetalTrace,
        "LightMetalReplay(EnqueueTrace) cq_id: {} tid: {} blocking: {}",
        cmd->cq_id(),
        cmd->tid(),
        cmd->blocking());
    CommandQueue& cq = this->device_->command_queue(cmd->cq_id());
    EnqueueTrace(cq, cmd->tid(), cmd->blocking());
}

void LightMetalReplay::Execute(const tt::target::ReplayTraceCommand* cmd) {
    log_debug(
        tt::LogMetalTrace,
        "LightMetalReplay(ReplayTrace) cq_id: {} tid: {} blocking: {}",
        cmd->cq_id(),
        cmd->tid(),
        cmd->blocking());
    ReplayTrace(this->device_, cmd->cq_id(), cmd->tid(), cmd->blocking());
}

void LightMetalReplay::Execute(const tt::target::LoadTraceCommand* cmd) {
    log_info(tt::LogMetalTrace, "LightMetalReplay(LoadTrace) cq_id: {} tid: {}", cmd->cq_id(), cmd->tid());
    // Get the trace descriptor from flatbuffer and load it to device.
    auto trace_desc = GetTraceByTraceId(cmd->tid());
    LoadTrace(this->device_, cmd->cq_id(), cmd->tid(), trace_desc.value());
}

// Main entry point to execute a light metal binary blob, return true if pass.
bool LightMetalReplay::ExecuteLightMetalBinary() {
    if (!lm_binary_) {
        std::cerr << "FlatBuffer binary not initialized." << std::endl;
        return false;
    }

    try {
        const auto* trace_descs = lm_binary_->trace_descriptors();
        const auto* commands = lm_binary_->commands();
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
