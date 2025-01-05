// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <flatbuffers/flatbuffers.h>
#include <string>
#include <vector>
#include <optional>
#include "lightmetal_binary.hpp"

#include <tt-metalium/device.hpp>

// Forward decl for trace_buffer.hpp
namespace tt::tt_metal {
class TraceDescriptor;
}

// Forward decl for command_generated.h / binary_generated.h
namespace tt::tt_metal::flatbuffer {
struct Command;
struct ReplayTraceCommand;
struct EnqueueTraceCommand;
struct LoadTraceCommand;

struct TraceDescriptor;
struct TraceDescriptorByTraceId;
struct LightMetalBinary;
}  // namespace tt::tt_metal::flatbuffer

namespace tt::tt_metal {
inline namespace v0 {

class LightMetalReplay {
public:
    // Constructor that initializes the class with a binary blob and transfers ownership of the blob.
    explicit LightMetalReplay(LightMetalBinary&& binary);

    // Open a FlatBuffer binary from the stored blob
    const tt::tt_metal::flatbuffer::LightMetalBinary* OpenFlatBufferBinary();

    // Return the TraceDescriptor for a given trace_id from flatbuffer.
    std::optional<TraceDescriptor> GetTraceByTraceId(uint32_t target_trace_id);

    // Execute the stored LightMetal binary
    bool ExecuteLightMetalBinary();

    // Executor functions for all traced host API calls
    void Execute(const tt::tt_metal::flatbuffer::Command* command);
    void Execute(const tt::tt_metal::flatbuffer::EnqueueTraceCommand* command);
    void Execute(const tt::tt_metal::flatbuffer::ReplayTraceCommand* command);
    void Execute(const tt::tt_metal::flatbuffer::LoadTraceCommand* command);

private:
    // Workload related members --------------------
    const tt::tt_metal::flatbuffer::LightMetalBinary* ParseFlatBufferBinary();

    LightMetalBinary binary_;                                      // Stored binary blob
    const tt::tt_metal::flatbuffer::LightMetalBinary* fb_binary_;  // Parsed FlatBuffer binary

    // System related members ----------------------
    void SetupDevices();
    void CloseDevices();

    tt::tt_metal::IDevice* device_;
    tt::ARCH arch_;
};

}  // namespace v0
}  // namespace tt::tt_metal
