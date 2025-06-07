// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>
#include <memory>
#include <flatbuffers/flatbuffers.h>
#include <lightmetal_binary.hpp>

// Forward decl for command_generated.h
namespace tt::tt_metal::flatbuffer {
class Command;
}

// Forward decl for light_metal_binary_generated.h
namespace tt::tt_metal::flatbuffer {
struct TraceDescriptorByTraceId;
}  // namespace tt::tt_metal::flatbuffer

// Forward decl for trace_buffer.hpp
namespace tt::tt_metal {
class TraceDescriptor;
}

namespace tt::tt_metal {

class Buffer;
class Program;
class Kernel;
using CBHandle = uintptr_t;
using TraceDescriptorByTraceIdOffset = flatbuffers::Offset<tt::tt_metal::flatbuffer::TraceDescriptorByTraceId>;

class LightMetalCaptureContext {
public:
    static LightMetalCaptureContext& get();

    LightMetalCaptureContext(const LightMetalCaptureContext&) = delete;
    LightMetalCaptureContext& operator=(const LightMetalCaptureContext&) = delete;

    bool is_tracing() const;
    void set_tracing(bool tracing);

    flatbuffers::FlatBufferBuilder& get_builder();
    std::vector<flatbuffers::Offset<tt::tt_metal::flatbuffer::Command>>& get_cmds_vector();
    void capture_trace_descriptor(const TraceDescriptor& trace_desc, uint32_t tid);
    LightMetalBinary create_light_metal_binary();
    void reset();

    // Object Map Public Accessors
    bool is_in_map(const Buffer* obj);
    uint32_t add_to_map(const Buffer* obj);
    void remove_from_map(const Buffer* obj);
    uint32_t get_global_id(const Buffer* obj);
    bool is_in_map(const Program* obj);
    uint32_t add_to_map(const Program* obj);
    void remove_from_map(const Program* obj);
    uint32_t get_global_id(const Program* obj);
    bool is_in_map(const Kernel* obj);
    uint32_t add_to_map(const Kernel* obj);
    void remove_from_map(const Kernel* obj);
    uint32_t get_global_id(const Kernel* obj);
    bool is_in_map(const CBHandle handle);
    uint32_t add_to_map(const CBHandle handle);
    void remove_from_map(const CBHandle handle);
    uint32_t get_global_id(const CBHandle handle);

private:
    LightMetalCaptureContext();  // Private constructor

    bool is_tracing_ = false;
    flatbuffers::FlatBufferBuilder builder_;
    std::vector<flatbuffers::Offset<tt::tt_metal::flatbuffer::Command>> cmds_vec_;
    std::vector<TraceDescriptorByTraceIdOffset> trace_descs_vec_;

    // Object maps for associating each object (or identifier) with a global_id
    // TODO (kmabee) - upgrade all global_id to be uint64_t for capture + replay.
    uint32_t next_global_id_ = 0;  // Shared across all object types.
    std::unordered_map<uint64_t, uint32_t> buffer_id_to_global_id_map_;
    std::unordered_map<uint64_t, uint32_t> program_id_to_global_id_map_;
    std::unordered_map<const Kernel*, uint32_t> kernel_to_global_id_map_;
    std::unordered_map<CBHandle, uint32_t> cb_handle_to_global_id_map_;
    // TODO (kmabee) - consider adding map for CommandQueue object.
};

TraceDescriptorByTraceIdOffset to_flatbuffer(
    flatbuffers::FlatBufferBuilder& builder, const TraceDescriptor& trace_desc, uint32_t trace_id);

}  // namespace tt::tt_metal
