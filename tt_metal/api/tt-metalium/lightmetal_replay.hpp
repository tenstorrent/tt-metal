// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <flatbuffers/flatbuffers.h>
#include <string>
#include <vector>
#include <optional>
#include "lightmetal_types.hpp"

#include <tt-metalium/device.hpp>

// Forward decl for trace_buffer.hpp
namespace tt::tt_metal::detail {
class TraceDescriptor;
}

// Forward decl for command_generated.h / binary_generated.h
namespace tt::tt_metal::flatbuffer {
struct Command;
struct ReplayTraceCommand;
struct EnqueueTraceCommand;
struct LoadTraceCommand;
struct ReleaseTraceCommand;
struct CreateBufferCommand;
struct DeallocateBufferCommand;
struct EnqueueWriteBufferCommand;
struct EnqueueReadBufferCommand;
struct FinishCommand;
struct CreateProgramCommand;
struct EnqueueProgramCommand;
struct CreateKernelCommand;
struct SetRuntimeArgsUint32Command;
struct SetRuntimeArgsCommand;
struct CreateCircularBufferCommand;
struct LightMetalCompareCommand;
struct RuntimeArg;

struct TraceDescriptor;
struct TraceDescriptorByTraceId;
struct LightMetalBinary;
}  // namespace tt::tt_metal::flatbuffer

using FlatbufferRuntimeArgVector =
    const flatbuffers::Vector<flatbuffers::Offset<tt::tt_metal::flatbuffer::RuntimeArg>>*;
using RuntimeArgs = std::vector<std::variant<Buffer*, uint32_t>>;

namespace tt::tt_metal {
inline namespace v0 {

class LightMetalReplay {
public:
    // Constructor that initializes the class with a binary blob and transfers ownership of the blob.
    explicit LightMetalReplay(LightMetalBinary&& binary_blob);

    // Open a FlatBuffer binary from the stored blob
    const tt::tt_metal::flatbuffer::LightMetalBinary* OpenFlatBufferBinary();

    // Return the TraceDescriptor for a given trace_id from flatbuffer.
    std::optional<detail::TraceDescriptor> GetTraceByTraceId(uint32_t target_trace_id);

    // fromFlatBuffer that need class state
    std::shared_ptr<RuntimeArgs> FromFlatbufferRtArgs(const FlatbufferRuntimeArgVector flatbuffer_args);

    // Execute the stored LightMetal binary
    bool ExecuteLightMetalBinary();

    // Executor functions for all traced host API calls
    void Execute(const tt::tt_metal::flatbuffer::Command* command);
    void Execute(const tt::tt_metal::flatbuffer::EnqueueTraceCommand* command);
    void Execute(const tt::tt_metal::flatbuffer::ReplayTraceCommand* command);
    void Execute(const tt::tt_metal::flatbuffer::LoadTraceCommand* command);
    void Execute(const tt::tt_metal::flatbuffer::ReleaseTraceCommand* command);
    void Execute(const tt::tt_metal::flatbuffer::CreateBufferCommand* command);
    void Execute(const tt::tt_metal::flatbuffer::DeallocateBufferCommand* command);
    void Execute(const tt::tt_metal::flatbuffer::EnqueueWriteBufferCommand* command);
    void Execute(const tt::tt_metal::flatbuffer::EnqueueReadBufferCommand* command);
    void Execute(const tt::tt_metal::flatbuffer::FinishCommand* command);
    void Execute(const tt::tt_metal::flatbuffer::CreateProgramCommand* command);
    void Execute(const tt::tt_metal::flatbuffer::EnqueueProgramCommand* command);
    void Execute(const tt::tt_metal::flatbuffer::CreateKernelCommand* command);
    void Execute(const tt::tt_metal::flatbuffer::SetRuntimeArgsUint32Command* command);
    void Execute(const tt::tt_metal::flatbuffer::SetRuntimeArgsCommand* command);
    void Execute(const tt::tt_metal::flatbuffer::CreateCircularBufferCommand* command);
    void Execute(const tt::tt_metal::flatbuffer::LightMetalCompareCommand* command);

    // Object maps public accessors
    void AddBufferToMap(uint32_t global_id, const std::shared_ptr<::tt::tt_metal::Buffer>& buffer);
    std::shared_ptr<::tt::tt_metal::Buffer> GetBufferFromMap(uint32_t global_id) const;
    void RemoveBufferFromMap(uint32_t global_id);

    void AddProgramToMap(uint32_t global_id, const std::shared_ptr<::tt::tt_metal::Program>& program);
    std::shared_ptr<::tt::tt_metal::Program> GetProgramFromMap(uint32_t global_id) const;
    void RemoveProgramFromMap(uint32_t global_id);

    void AddKernelHandleToMap(uint32_t global_id, ::tt::tt_metal::KernelHandle kernel_id);
    ::tt::tt_metal::KernelHandle GetKernelHandleFromMap(uint32_t global_id) const;
    void RemoveKernelHandleFromMap(uint32_t global_id);

    void AddKernelToMap(uint32_t global_id, const std::shared_ptr<::tt::tt_metal::Kernel>& kernel);
    std::shared_ptr<::tt::tt_metal::Kernel> GetKernelFromMap(uint32_t global_id) const;
    void RemoveKernelFromMap(uint32_t global_id);

    void AddCBHandleToMap(uint32_t global_id, ::tt::tt_metal::CBHandle cb_handle);
    ::tt::tt_metal::CBHandle GetCBHandleFromMap(uint32_t global_id) const;
    void RemoveCBHandleFromMap(uint32_t global_id);

private:
    // Workload related members --------------------
    const tt::tt_metal::flatbuffer::LightMetalBinary* ParseFlatBufferBinary();

    LightMetalBinary binary_blob_;                                 // Stored binary blob
    const tt::tt_metal::flatbuffer::LightMetalBinary* fb_binary_;  // Parsed FlatBuffer binary
    bool show_reads_ = false;                                      // Flag to show read buffer contents
    bool disable_checking_ = false;                                // Optionally disable equality checking in Compare command.

    // System related members ----------------------
    void SetupDevices();
    void CloseDevices();

    tt::tt_metal::IDevice* device_;

    // Object maps for storing objects by global_id
    std::unordered_map<uint32_t, std::shared_ptr<::tt::tt_metal::Buffer>> buffer_map_;
    std::unordered_map<uint32_t, std::shared_ptr<::tt::tt_metal::Program>> program_map_;
    std::unordered_map<uint32_t, tt::tt_metal::KernelHandle> kernel_handle_map_;
    std::unordered_map<uint32_t, std::shared_ptr<::tt::tt_metal::Kernel>> kernel_map_;
    std::unordered_map<uint32_t, tt::tt_metal::CBHandle> cb_handle_map_;
};

}  // namespace v0
}  // namespace tt::tt_metal
