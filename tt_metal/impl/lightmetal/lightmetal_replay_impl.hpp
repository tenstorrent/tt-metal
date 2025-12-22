// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <flatbuffers/flatbuffers.h>
#include <string>
#include <vector>
#include <optional>
#include <tt-metalium/experimental/lightmetal/lightmetal_binary.hpp>

#include <tt-metalium/program.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/circular_buffer.hpp>

#include "impl/kernels/kernel.hpp"

namespace tt::tt_metal {
struct TraceDescriptor;
}

// Forward decl for command_generated.h / light_metal_binary_generated.h
namespace tt::tt_metal::flatbuffer {
class Command;
struct ReplayTraceCommand;
struct EnqueueTraceCommand;
struct LoadTraceCommand;
struct ReleaseTraceCommand;
struct BufferCreateCommand;
struct BufferDeallocateCommand;
struct BufferDeleteCommand;
struct EnqueueWriteBufferCommand;
struct EnqueueReadBufferCommand;
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

struct TraceDescriptorByTraceId;
struct LightMetalBinary;
}  // namespace tt::tt_metal::flatbuffer

using FlatbufferRuntimeArgVector =
    const flatbuffers::Vector<flatbuffers::Offset<tt::tt_metal::flatbuffer::RuntimeArg>>*;
using RuntimeArgs = std::vector<std::variant<tt::tt_metal::Buffer*, uint32_t>>;

namespace tt::tt_metal::experimental::lightmetal::detail {
class LightMetalReplayImpl {
public:
    // Constructor
    explicit LightMetalReplayImpl(LightMetalBinary&& binary, IDevice* device);

    // Core functionality
    bool run();

    // Executor functions for all traced host API calls (commands)
    // Trace APIs are no longer supported due to trace API deprecation. See Issue #24955
    void execute(const tt::tt_metal::flatbuffer::Command* command);
    void execute(const tt::tt_metal::flatbuffer::EnqueueTraceCommand* cmd);
    void execute(const tt::tt_metal::flatbuffer::ReplayTraceCommand* cmd);
    void execute(const tt::tt_metal::flatbuffer::LoadTraceCommand* cmd);
    void execute(const tt::tt_metal::flatbuffer::ReleaseTraceCommand* cmd);
    void execute(const tt::tt_metal::flatbuffer::BufferCreateCommand* cmd);
    void execute(const tt::tt_metal::flatbuffer::BufferDeallocateCommand* cmd);
    void execute(const tt::tt_metal::flatbuffer::BufferDeleteCommand* cmd);
    void execute(const tt::tt_metal::flatbuffer::EnqueueWriteBufferCommand* cmd);
    void execute(const tt::tt_metal::flatbuffer::EnqueueReadBufferCommand* cmd);
    void execute(const tt::tt_metal::flatbuffer::FinishCommand* cmd);
    void execute(const tt::tt_metal::flatbuffer::ProgramConstructorCommand* cmd);
    void execute(const tt::tt_metal::flatbuffer::EnqueueProgramCommand* command);
    void execute(const tt::tt_metal::flatbuffer::CreateKernelCommand* cmd);
    void execute(const tt::tt_metal::flatbuffer::SetRuntimeArgsUint32Command* cmd);
    void execute(const tt::tt_metal::flatbuffer::SetRuntimeArgsUint32VecPerCoreCommand* cmd);
    void execute(const tt::tt_metal::flatbuffer::CreateCircularBufferCommand* cmd);
    void execute(const tt::tt_metal::flatbuffer::LightMetalCompareCommand* cmd);

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

    // Return the TraceDescriptor for a given trace_id from flatbuffer.
    // No longer supported due to trace API deprecation. See Issue #24955
    std::optional<TraceDescriptor> get_trace_by_id(uint32_t target_trace_id);

    // fromFlatBuffer that need class state
    std::shared_ptr<RuntimeArgs> rt_args_from_flatbuffer(FlatbufferRuntimeArgVector flatbuffer_args);

    // Workload related members --------------------
    const tt::tt_metal::flatbuffer::LightMetalBinary* parse_flatbuffer_binary();

    void clear_object_maps();

    // System related members ----------------------
    void setup_devices();
    void close_devices();

private:
    // Workload related members
    LightMetalBinary binary_;
    const flatbuffer::LightMetalBinary* fb_binary_{nullptr};
    bool show_reads_ = false;
    bool disable_checking_ = false;

    tt::tt_metal::IDevice* device_ = nullptr;

    // Object maps for storing objects by global_id
    std::unordered_map<uint32_t, std::shared_ptr<::tt::tt_metal::Buffer>> buffer_map_;
    std::unordered_map<uint32_t, std::shared_ptr<::tt::tt_metal::Program>> program_map_;
    std::unordered_map<uint32_t, tt::tt_metal::KernelHandle> kernel_handle_map_;
    std::unordered_map<uint32_t, std::shared_ptr<::tt::tt_metal::Kernel>> kernel_map_;
    std::unordered_map<uint32_t, tt::tt_metal::CBHandle> cb_handle_map_;
};

}  // namespace tt::tt_metal::experimental::lightmetal::detail
