// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <flatbuffers/flatbuffers.h>
#include <string>
#include <vector>
#include <optional>

#include "impl/device/device.hpp"

// Forward Declaration to avoid trace_buffer.hpp
namespace tt::tt_metal::detail {
    class TraceDescriptor;
}

// Forward decl for command_generated.h
namespace tt::target {
    struct Command;
    struct ReplayTraceCommand;
    struct EnqueueTraceCommand;
    struct LoadTraceCommand;

    // Forward decl for binary_generated.h
    namespace lightmetal {
        struct TraceDescriptor;
        struct TraceDescriptorByTraceId;
        struct LightMetalBinary;
    }
}

namespace tt::tt_metal {
inline namespace v0 {

// General utility function to open a binary file and return contents as a binary blob
void readBinaryBlobFromFile(const std::string& filename, std::vector<uint8_t>& blob);

class LightMetalReplay {
public:
    // Constructor that initializes the class with a binary blob
    explicit LightMetalReplay(std::vector<uint8_t> blob);

    // Open a FlatBuffer binary from the stored blob
    const target::lightmetal::LightMetalBinary* openFlatBufferBinary();

    // Return the TraceDescriptor for a given trace_id from flatbuffer.
    std::optional<detail::TraceDescriptor> getTraceByTraceId(uint32_t target_trace_id);

    // Execute the stored LightMetal binary
    bool executeLightMetalBinary();

    // Executor functions for all traced host API calls
    void execute(tt::target::Command const *command);
    void execute(tt::target::EnqueueTraceCommand const *command);
    void execute(tt::target::ReplayTraceCommand const *command);
    void execute(tt::target::LoadTraceCommand const *command);

    // Temporary debug
    void printLightMetalBinaryContents();

private:

    // Workload related members --------------------
    const target::lightmetal::LightMetalBinary* parseFlatBufferBinary();

    std::vector<uint8_t> blob_;  // Stored binary blob
    const target::lightmetal::LightMetalBinary* lm_binary_;  // Parsed FlatBuffer binary

    // System related members ----------------------
    void setupDevices();

    tt::tt_metal::Device* device_;
    tt::ARCH arch_;

};

}  // namespace v0
}  // namespace tt::tt_metal
