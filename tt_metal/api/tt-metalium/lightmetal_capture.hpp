// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>
#include <memory>
#include <flatbuffers/flatbuffers.h>
#include "lightmetal_types.hpp"

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
namespace tt::tt_metal::detail {
class TraceDescriptor;
}

namespace tt::tt_metal {
inline namespace v0 {

class Buffer;
class Program;
class Kernel;
using CBHandle = uintptr_t;
using TraceDescriptorByTraceIdOffset = flatbuffers::Offset<tt::tt_metal::flatbuffer::TraceDescriptorByTraceId>;

class LightMetalCaptureContext {
public:
    static LightMetalCaptureContext& Get();

    bool IsTracing() const;
    void SetTracing(bool tracing);

    flatbuffers::FlatBufferBuilder& GetBuilder();
    std::vector<flatbuffers::Offset<tt::tt_metal::flatbuffer::Command>>& GetCmdsVector();
    void CaptureTraceDescriptor(const detail::TraceDescriptor& trace_desc, const uint32_t tid);
    LightMetalBinary CreateLightMetalBinary();
    void Reset();

    // Object Map Public Accessors
    bool IsInMap(const Buffer* obj);
    uint32_t AddToMap(const Buffer* obj);
    void RemoveFromMap(const Buffer* obj);
    uint32_t GetGlobalId(const Buffer* obj);
    bool IsInMap(const Program* obj);
    uint32_t AddToMap(const Program* obj);
    void RemoveFromMap(const Program* obj);
    uint32_t GetGlobalId(const Program* obj);
    bool IsInMap(const Kernel* obj);
    uint32_t AddToMap(const Kernel* obj);
    void RemoveFromMap(const Kernel* obj);
    uint32_t GetGlobalId(const Kernel* obj);
    bool IsInMap(const CBHandle handle);
    uint32_t AddToMap(const CBHandle handle);
    void RemoveFromMap(const CBHandle handle);
    uint32_t GetGlobalId(const CBHandle handle);

private:
    LightMetalCaptureContext();  // Private constructor

    bool is_tracing_;
    flatbuffers::FlatBufferBuilder builder_;
    std::vector<flatbuffers::Offset<tt::tt_metal::flatbuffer::Command>> cmds_vec_;
    std::vector<TraceDescriptorByTraceIdOffset> trace_descs_vec_;

    // Object maps for associating each object with a global_id
    uint32_t next_global_id_ = 0;  // Shared across all object types.
    std::unordered_map<const Buffer*, uint32_t> buffer_to_global_id_map_;
    std::unordered_map<const Program*, uint32_t> program_to_global_id_map_;
    std::unordered_map<const Kernel*, uint32_t> kernel_to_global_id_map_;
    std::unordered_map<CBHandle, uint32_t> cb_handle_to_global_id_map_;
    // TODO (kmabee) - consider adding map for CommandQueue object.

    // Delete copy constructor and assignment operator
    LightMetalCaptureContext(const LightMetalCaptureContext&) = delete;
    LightMetalCaptureContext& operator=(const LightMetalCaptureContext&) = delete;
};

TraceDescriptorByTraceIdOffset ToFlatbuffer(
    flatbuffers::FlatBufferBuilder& builder, const detail::TraceDescriptor& trace_desc, const uint32_t trace_id);

}  // namespace v0
}  // namespace tt::tt_metal
