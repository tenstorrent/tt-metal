// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>
#include <memory>
#include <flatbuffers/flatbuffers.h>

// Forward decl for command_generated.h
namespace tt::target {
class Command;
}

// Forward decl for binary_generated.h
namespace tt::target::lightmetal {
struct TraceDescriptor;
struct TraceDescriptorByTraceId;
}  // namespace tt::target::lightmetal

// Forward decl for trace_buffer.hpp
namespace tt::tt_metal::detail {
class TraceDescriptor;
}

namespace tt::tt_metal {
inline namespace v0 {

using TraceDescriptorByTraceIdOffset = flatbuffers::Offset<tt::target::lightmetal::TraceDescriptorByTraceId>;
class LightMetalCaptureContext {
public:
    static LightMetalCaptureContext& Get();

    bool IsTracing() const;
    void SetTracing(bool tracing);

    flatbuffers::FlatBufferBuilder& GetBuilder();
    std::vector<flatbuffers::Offset<tt::target::Command>>& GetCmdsVector();
    void CaptureTraceDescriptor(const detail::TraceDescriptor& trace_desc, const uint32_t tid);
    std::vector<uint8_t> CreateLightMetalBinary();

    void Reset();

private:
    LightMetalCaptureContext();  // Private constructor

    bool is_tracing_;
    flatbuffers::FlatBufferBuilder builder_;
    std::vector<flatbuffers::Offset<tt::target::Command>> cmds_vec_;
    std::vector<TraceDescriptorByTraceIdOffset> trace_descs_vec_;

    // Delete copy constructor and assignment operator
    LightMetalCaptureContext(const LightMetalCaptureContext&) = delete;
    LightMetalCaptureContext& operator=(const LightMetalCaptureContext&) = delete;
};

bool WriteBinaryBlobToFile(const std::string& filename, const std::vector<uint8_t>& blob);
TraceDescriptorByTraceIdOffset ToFlatbuffer(
    flatbuffers::FlatBufferBuilder& builder, const detail::TraceDescriptor& trace_desc, const uint32_t trace_id);

}  // namespace v0
}  // namespace tt::tt_metal
