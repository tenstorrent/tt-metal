// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <flatbuffers/flatbuffers.h>
#include <string>
#include <vector>
#include <optional>

#include <tt-metalium/device.hpp>

// Forward decl for trace_buffer.hpp
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
}  // namespace lightmetal
}  // namespace tt::target

namespace tt::tt_metal {
inline namespace v0 {

// General utility function to open a binary file and return contents as a binary blob
void ReadBinaryBlobFromFile(const std::string& filename, std::vector<uint8_t>& blob);

class LightMetalReplay {
public:
    // Constructor that initializes the class with a binary blob and transfers ownership of the blob.
    explicit LightMetalReplay(std::vector<uint8_t>&& blob);

    // Open a FlatBuffer binary from the stored blob
    const target::lightmetal::LightMetalBinary* OpenFlatBufferBinary();

    // Return the TraceDescriptor for a given trace_id from flatbuffer.
    std::optional<detail::TraceDescriptor> GetTraceByTraceId(uint32_t target_trace_id);

    // Execute the stored LightMetal binary
    bool ExecuteLightMetalBinary();

    // Executor functions for all traced host API calls
    void Execute(const tt::target::Command* command);
    void Execute(const tt::target::EnqueueTraceCommand* command);
    void Execute(const tt::target::ReplayTraceCommand* command);
    void Execute(const tt::target::LoadTraceCommand* command);

private:
    // Workload related members --------------------
    const target::lightmetal::LightMetalBinary* ParseFlatBufferBinary();

    std::vector<uint8_t> blob_;                              // Stored binary blob
    const target::lightmetal::LightMetalBinary* lm_binary_;  // Parsed FlatBuffer binary

    // System related members ----------------------
    void SetupDevices();
    void CloseDevices();

    tt::tt_metal::IDevice* device_;
    tt::ARCH arch_;
};

}  // namespace v0
}  // namespace tt::tt_metal
