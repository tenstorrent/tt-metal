// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <flatbuffers/buffer.h>
#include <flatbuffers/flatbuffers.h>
#include <flatbuffers/vector.h>
#include <stdint.h>
#include <tt-metalium/device.hpp>
#include <tt-metalium/lightmetal_binary.hpp>
#include <tt-metalium/program_impl.hpp>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

#include "buffer.hpp"
#include "circular_buffer_types.hpp"
#include "kernel_types.hpp"

namespace tt {
namespace tt_metal {
class IDevice;
class Kernel;
}  // namespace tt_metal
}  // namespace tt

namespace tt::tt_metal {
class TraceDescriptor;
}

// Forward decl for command_generated.h / light_metal_binary_generated.h
namespace tt::tt_metal::flatbuffer {
struct BufferCreateCommand;
struct BufferDeallocateCommand;
struct BufferDeleteCommand;
struct Command;
struct CreateCircularBufferCommand;
struct CreateKernelCommand;
struct EnqueueProgramCommand;
struct EnqueueReadBufferCommand;
struct EnqueueTraceCommand;
struct EnqueueWriteBufferCommand;
struct FinishCommand;
struct LightMetalBinary;
struct LightMetalCompareCommand;
struct LoadTraceCommand;
struct ProgramConstructorCommand;
struct ReleaseTraceCommand;
struct ReplayTraceCommand;
struct RuntimeArg;
struct SetRuntimeArgsCommand;
struct SetRuntimeArgsUint32Command;
struct SetRuntimeArgsUint32VecPerCoreCommand;
struct TraceDescriptor;
struct TraceDescriptorByTraceId;
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
    void execute(const tt::tt_metal::flatbuffer::EnqueueTraceCommand* command);
    void execute(const tt::tt_metal::flatbuffer::ReplayTraceCommand* command);
    void execute(const tt::tt_metal::flatbuffer::LoadTraceCommand* command);
    void execute(const tt::tt_metal::flatbuffer::ReleaseTraceCommand* command);
    void execute(const tt::tt_metal::flatbuffer::BufferCreateCommand* cmd);
    void execute(const tt::tt_metal::flatbuffer::BufferDeallocateCommand* command);
    void execute(const tt::tt_metal::flatbuffer::BufferDeleteCommand* command);
    void execute(const tt::tt_metal::flatbuffer::EnqueueWriteBufferCommand* command);
    void execute(const tt::tt_metal::flatbuffer::EnqueueReadBufferCommand* command);
    void execute(const tt::tt_metal::flatbuffer::FinishCommand* command);
    void execute(const tt::tt_metal::flatbuffer::ProgramConstructorCommand* command);
    void execute(const tt::tt_metal::flatbuffer::EnqueueProgramCommand* command);
    void execute(const tt::tt_metal::flatbuffer::CreateKernelCommand* command);
    void execute(const tt::tt_metal::flatbuffer::SetRuntimeArgsUint32Command* command);
    void execute(const tt::tt_metal::flatbuffer::SetRuntimeArgsUint32VecPerCoreCommand* cmd);
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
    std::unordered_map<uint32_t, std::shared_ptr<::tt::tt_metal::Buffer>> buffer_map_;
    std::unordered_map<uint32_t, std::shared_ptr<::tt::tt_metal::Program>> program_map_;
    std::unordered_map<uint32_t, tt::tt_metal::KernelHandle> kernel_handle_map_;
    std::unordered_map<uint32_t, std::shared_ptr<::tt::tt_metal::Kernel>> kernel_map_;
    std::unordered_map<uint32_t, tt::tt_metal::CBHandle> cb_handle_map_;
};

}  // namespace detail
}  // namespace tt::tt_metal
