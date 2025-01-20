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
struct RuntimeArg;

// Forward decl for binary_generated.h
namespace lightmetal {
struct TraceDescriptor;
struct TraceDescriptorByTraceId;
struct LightMetalBinary;
}  // namespace lightmetal
}  // namespace tt::target

using FlatbufferRuntimeArgVector = const flatbuffers::Vector<flatbuffers::Offset<tt::target::RuntimeArg>>*;
using RuntimeArgs = std::vector<std::variant<Buffer*, uint32_t>>;

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

    // fromFlatBuffer that need class state
    std::shared_ptr<RuntimeArgs> FromFlatbufferRtArgs(const FlatbufferRuntimeArgVector flatbuffer_args);

    // Execute the stored LightMetal binary
    bool ExecuteLightMetalBinary();

    // Executor functions for all traced host API calls
    void Execute(const tt::target::Command* command);
    void Execute(const tt::target::EnqueueTraceCommand* command);
    void Execute(const tt::target::ReplayTraceCommand* command);
    void Execute(const tt::target::LoadTraceCommand* command);
    void Execute(const tt::target::ReleaseTraceCommand* command);
    void Execute(const tt::target::CreateBufferCommand* command);
    void Execute(const tt::target::DeallocateBufferCommand* command);
    void Execute(const tt::target::EnqueueWriteBufferCommand* command);
    void Execute(const tt::target::EnqueueReadBufferCommand* command);
    void Execute(const tt::target::FinishCommand* command);
    void Execute(const tt::target::CreateProgramCommand* command);
    void Execute(const tt::target::EnqueueProgramCommand* command);
    void Execute(const tt::target::CreateKernelCommand* command);
    void Execute(const tt::target::SetRuntimeArgsUint32Command* command);
    void Execute(const tt::target::SetRuntimeArgsCommand* command);
    void Execute(const tt::target::CreateCircularBufferCommand* command);

    // Object maps public accessors
    void AddBufferToMap(uint32_t global_id, std::shared_ptr<::tt::tt_metal::Buffer> buffer);
    std::shared_ptr<::tt::tt_metal::Buffer> GetBufferFromMap(uint32_t global_id) const;
    void RemoveBufferFromMap(uint32_t global_id);

    void AddProgramToMap(uint32_t global_id, std::shared_ptr<::tt::tt_metal::Program> program);
    std::shared_ptr<::tt::tt_metal::Program> GetProgramFromMap(uint32_t global_id) const;
    void RemoveProgramFromMap(uint32_t global_id);

    void AddKernelHandleToMap(uint32_t global_id, ::tt::tt_metal::KernelHandle kernel_id);
    ::tt::tt_metal::KernelHandle GetKernelHandleFromMap(uint32_t global_id) const;
    void RemoveKernelHandleFromMap(uint32_t global_id);

    void AddKernelToMap(uint32_t global_id, std::shared_ptr<::tt::tt_metal::Kernel> kernel);
    std::shared_ptr<::tt::tt_metal::Kernel> GetKernelFromMap(uint32_t global_id) const;
    void RemoveKernelFromMap(uint32_t global_id);

    void AddCBHandleToMap(uint32_t global_id, ::tt::tt_metal::CBHandle cb_handle);
    ::tt::tt_metal::CBHandle GetCBHandleFromMap(uint32_t global_id) const;
    void RemoveCBHandleFromMap(uint32_t global_id);

private:
    // Workload related members --------------------
    const target::lightmetal::LightMetalBinary* ParseFlatBufferBinary();

    std::vector<uint8_t> blob_;                              // Stored binary blob
    const target::lightmetal::LightMetalBinary* lm_binary_;  // Parsed FlatBuffer binary
    bool show_reads_ = false;                                // Flag to show read buffer contents

    // System related members ----------------------
    void SetupDevices();
    void CloseDevices();

    tt::tt_metal::IDevice* device_;
    tt::ARCH arch_;

    // Object maps for storing objects by global_id
    std::unordered_map<uint32_t, std::shared_ptr<::tt::tt_metal::Buffer>> buffer_map_;
    std::unordered_map<uint32_t, std::shared_ptr<::tt::tt_metal::Program>> program_map_;
    std::unordered_map<uint32_t, tt::tt_metal::KernelHandle> kernel_handle_map_;
    std::unordered_map<uint32_t, std::shared_ptr<::tt::tt_metal::Kernel>> kernel_map_;
    std::unordered_map<uint32_t, tt::tt_metal::CBHandle> cb_handle_map_;
};

}  // namespace v0
}  // namespace tt::tt_metal
