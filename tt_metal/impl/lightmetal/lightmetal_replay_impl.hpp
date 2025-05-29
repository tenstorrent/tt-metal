// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <flatbuffers/flatbuffers.h>
#include <cstdint>
#include <string>
#include <vector>
#include <optional>
#include <tt-metalium/lightmetal_binary.hpp>

#include <tt-metalium/program.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/mesh_workload.hpp>
#include <tt-metalium/mesh_buffer.hpp>

namespace tt::tt_metal {
class TraceDescriptor;
}

// Forward decl for command_generated.h / light_metal_binary_generated.h
namespace tt::tt_metal::flatbuffer {
struct Command;
struct BeginTraceCaptureCommand;
struct EndTraceCaptureCommand;
struct ReplayTraceCommand;
struct EnqueueTraceCommand;
struct LoadTraceCommand;
struct ReleaseTraceCommand;
struct EnqueueRecordEventCommand;
struct EnqueueRecordEventToHostCommand;
struct EnqueueWaitForEventCommand;
struct EventSynchronizeCommand;
struct SynchronizeCommand;
struct BufferCreateCommand;
struct BufferDeallocateCommand;
struct BufferDeleteCommand;
struct EnqueueWriteBufferCommand;
struct EnqueueReadBufferCommand;
struct MeshBufferCreateCommand;
struct MeshWorkloadCreateCommand;
struct AddProgramToMeshWorkloadCommand;
struct EnqueueMeshWorkloadCommand;
struct EnqueueReadMeshBufferCommand;
struct EnqueueWriteMeshBufferCommand;
struct ReadShardCommand;
struct WriteShardCommand;
struct FinishCommand;
struct ProgramConstructorCommand;
struct EnqueueProgramCommand;
struct CreateKernelCommand;
struct SetRuntimeArgsUint32Command;
struct SetRuntimeArgsUint32VecPerCoreCommand;
struct SetRuntimeArgsCommand;
struct CreateCircularBufferCommand;
struct LightMetalCompareCommand;
struct RuntimeArg;

struct MeshEvent;

struct TraceDescriptor;
struct TraceDescriptorByTraceId;
struct LightMetalBinary;
}  // namespace tt::tt_metal::flatbuffer

using FlatbufferRuntimeArgVector =
    const flatbuffers::Vector<flatbuffers::Offset<tt::tt_metal::flatbuffer::RuntimeArg>>*;
using RuntimeArgs = std::vector<std::variant<tt::tt_metal::Buffer*, uint32_t>>;

namespace tt::tt_metal {

namespace detail {
class LightMetalReplayImpl {
public:
    // Constructor
    explicit LightMetalReplayImpl(LightMetalBinary&& binary, IDevice* device);

    // Core functionality
    bool run();

    // Executor functions for all traced host API calls (commands)
    void execute(const tt::tt_metal::flatbuffer::Command* command);
    void execute(const tt::tt_metal::flatbuffer::BeginTraceCaptureCommand* command);
    void execute(const tt::tt_metal::flatbuffer::EndTraceCaptureCommand* command);
    void execute(const tt::tt_metal::flatbuffer::ReplayTraceCommand* command);
    void execute(const tt::tt_metal::flatbuffer::EnqueueTraceCommand* command);
    void execute(const tt::tt_metal::flatbuffer::LoadTraceCommand* command);
    void execute(const tt::tt_metal::flatbuffer::ReleaseTraceCommand* command);
    void execute(const tt::tt_metal::flatbuffer::EnqueueRecordEventCommand* command);
    void execute(const tt::tt_metal::flatbuffer::EnqueueRecordEventToHostCommand* command);
    void execute(const tt::tt_metal::flatbuffer::EnqueueWaitForEventCommand* command);
    void execute(const tt::tt_metal::flatbuffer::EventSynchronizeCommand* command);
    void execute(const tt::tt_metal::flatbuffer::SynchronizeCommand* command);
    void execute(const tt::tt_metal::flatbuffer::BufferCreateCommand* command);
    void execute(const tt::tt_metal::flatbuffer::BufferDeallocateCommand* command);
    void execute(const tt::tt_metal::flatbuffer::BufferDeleteCommand* command);
    void execute(const tt::tt_metal::flatbuffer::EnqueueWriteBufferCommand* command);
    void execute(const tt::tt_metal::flatbuffer::EnqueueReadBufferCommand* command);
    void execute(const tt::tt_metal::flatbuffer::MeshBufferCreateCommand* command);
    void execute(const tt::tt_metal::flatbuffer::MeshWorkloadCreateCommand* command);
    void execute(const tt::tt_metal::flatbuffer::AddProgramToMeshWorkloadCommand* command);
    void execute(const tt::tt_metal::flatbuffer::EnqueueMeshWorkloadCommand* command);
    void execute(const tt::tt_metal::flatbuffer::EnqueueReadMeshBufferCommand* command);
    void execute(const tt::tt_metal::flatbuffer::EnqueueWriteMeshBufferCommand* command);
    void execute(const tt::tt_metal::flatbuffer::ReadShardCommand* command);
    void execute(const tt::tt_metal::flatbuffer::WriteShardCommand* command);
    void execute(const tt::tt_metal::flatbuffer::FinishCommand* command);
    void execute(const tt::tt_metal::flatbuffer::ProgramConstructorCommand* command);
    void execute(const tt::tt_metal::flatbuffer::EnqueueProgramCommand* command);
    void execute(const tt::tt_metal::flatbuffer::CreateKernelCommand* command);
    void execute(const tt::tt_metal::flatbuffer::SetRuntimeArgsUint32Command* command);
    void execute(const tt::tt_metal::flatbuffer::SetRuntimeArgsUint32VecPerCoreCommand* command);
    void execute(const tt::tt_metal::flatbuffer::SetRuntimeArgsCommand* command);
    void execute(const tt::tt_metal::flatbuffer::CreateCircularBufferCommand* command);
    void execute(const tt::tt_metal::flatbuffer::LightMetalCompareCommand* command);

    // Object maps public accessors
    void add_buffer_to_map(uint32_t global_id, const std::shared_ptr<::tt::tt_metal::Buffer>& buffer);
    std::shared_ptr<::tt::tt_metal::Buffer> get_buffer_from_map(uint32_t global_id) const;
    void remove_bufer_from_map(uint32_t global_id);

    void add_program_to_map(uint32_t global_id, const std::shared_ptr<::tt::tt_metal::Program>& program);
    std::shared_ptr<::tt::tt_metal::Program> get_program_from_map(uint32_t global_id) const;
    void remove_program_from_map(uint32_t global_id);

    void add_kernel_handle_to_map(uint32_t global_id, ::tt::tt_metal::KernelHandle kernel_id);
    ::tt::tt_metal::KernelHandle get_kernel_handle_from_map(uint32_t global_id) const;
    void remove_kernel_handle_from_map(uint32_t global_id);

    void add_kernel_to_map(uint32_t global_id, const std::shared_ptr<::tt::tt_metal::Kernel>& kernel);
    std::shared_ptr<::tt::tt_metal::Kernel> get_kernel_from_map(uint32_t global_id) const;
    void remove_kernel_from_map(uint32_t global_id);

    void add_cb_handle_to_map(uint32_t global_id, ::tt::tt_metal::CBHandle cb_handle);
    ::tt::tt_metal::CBHandle get_cb_handle_from_map(uint32_t global_id) const;
    void remove_cb_handle_from_map(uint32_t global_id);

    void add_mesh_workload_to_map(uint32_t global_id, const std::shared_ptr<distributed::MeshWorkload>& meshworkload);
    std::shared_ptr<distributed::MeshWorkload> get_mesh_workload_from_map(uint32_t global_id) const;
    void remove_mesh_workload_from_map(uint32_t global_id);

    // Return the TraceDescriptor for a given trace_id from flatbuffer.
    std::optional<TraceDescriptor> get_trace_by_id(uint32_t target_trace_id);

    // fromFlatBuffer that need class state
    std::shared_ptr<RuntimeArgs> rt_args_from_flatbuffer(const FlatbufferRuntimeArgVector flatbuffer_args);

    // Workload related members --------------------
    const tt::tt_metal::flatbuffer::LightMetalBinary* parse_flatbuffer_binary();

    void clear_object_maps();

    // System related members ----------------------
    void setup_devices();
    void close_devices();

private:
    // Workload related members
    LightMetalBinary binary_;
    const flatbuffer::LightMetalBinary* fb_binary_;
    bool show_reads_ = false;
    bool disable_checking_ = false;

    tt::tt_metal::IDevice* device_ = nullptr;

    // Object maps for storing objects by global_id
    // TODO: (jjiang) - Access these with the translation tables from capture, use them in impl
    std::unordered_map<uint32_t, tt::tt_metal::distributed::MeshTraceId> mesh_trace_ids_;
    std::unordered_map<uint32_t, std::shared_ptr<tt::tt_metal::distributed::MeshBuffer>> mesh_buffer_map_;
    std::unordered_map<uint32_t, std::shared_ptr<tt::tt_metal::distributed::MeshWorkload>> mesh_workload_map_;

    std::unordered_map<uint32_t, std::shared_ptr<::tt::tt_metal::Buffer>> buffer_map_;
    std::unordered_map<uint32_t, std::shared_ptr<::tt::tt_metal::Program>> program_map_;
    std::unordered_map<uint32_t, tt::tt_metal::KernelHandle> kernel_handle_map_;
    std::unordered_map<uint32_t, std::shared_ptr<::tt::tt_metal::Kernel>> kernel_map_;
    std::unordered_map<uint32_t, tt::tt_metal::CBHandle> cb_handle_map_;
};

}  // namespace detail
}  // namespace tt::tt_metal
