// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>
#include <memory>
#include <flatbuffers/flatbuffers.h>
#include "lightmetal_binary.hpp"

// Forward decl for command_generated.h
namespace tt::tt_metal::flatbuffer {
class Command;
}

// Forward decl for binary_generated.h
namespace tt::tt_metal::flatbuffer {
struct TraceDescriptor;
struct TraceDescriptorByTraceId;
}  // namespace tt::tt_metal::flatbuffer

// Forward decl for trace_buffer.hpp
namespace tt::tt_metal {
class TraceDescriptor;
}

namespace tt::tt_metal {
inline namespace v0 {

using TraceDescriptorByTraceIdOffset = flatbuffers::Offset<tt::tt_metal::flatbuffer::TraceDescriptorByTraceId>;
class LightMetalCaptureContext {
public:
    static LightMetalCaptureContext& Get();

    bool IsTracing() const;
    void SetTracing(bool tracing);

    flatbuffers::FlatBufferBuilder& GetBuilder();
    std::vector<flatbuffers::Offset<tt::tt_metal::flatbuffer::Command>>& GetCmdsVector();
    void CaptureTraceDescriptor(const TraceDescriptor& trace_desc, uint32_t tid);
    LightMetalBinary CreateLightMetalBinary();

    void Reset();

private:
    LightMetalCaptureContext();  // Private constructor

    bool is_tracing_;
    flatbuffers::FlatBufferBuilder builder_;
    std::vector<flatbuffers::Offset<tt::tt_metal::flatbuffer::Command>> cmds_vec_;
    std::vector<TraceDescriptorByTraceIdOffset> trace_descs_vec_;

    LightMetalCaptureContext(const LightMetalCaptureContext&) = delete;
    LightMetalCaptureContext& operator=(const LightMetalCaptureContext&) = delete;
};

TraceDescriptorByTraceIdOffset ToFlatbuffer(
    flatbuffers::FlatBufferBuilder& builder, const TraceDescriptor& trace_desc, uint32_t trace_id);

}  // namespace v0
}  // namespace tt::tt_metal
