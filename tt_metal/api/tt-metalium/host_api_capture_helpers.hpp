// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "lightmetal_capture.hpp"
#include "command_generated.h"
#include <tt-metalium/logger.hpp>
#include "span.hpp"
#include "flatbuffer/base_types_to_flatbuffer.hpp"
#include "flatbuffer/program_types_to_flatbuffer.hpp"
#include "flatbuffer/buffer_types_to_flatbuffer.hpp"

//////////////////////////////////////////////////////////////
// TRACE GUARD & TRACE MACRO                                //
//////////////////////////////////////////////////////////////

namespace tt::tt_metal {

// This struct will disable further tracing in current scope, and re-enable
// when scope ends. Prevents recursive tracing of host APIs.
struct TraceScope {
    // Provide an inline definition in the header
    static inline thread_local int depth = 0;
    // Increment depth on entering scope, decrement on exiting
    TraceScope() { ++depth; }
    ~TraceScope() { --depth; }
};

}  // namespace tt::tt_metal

#if defined(TT_ENABLE_LIGHT_METAL_TRACE) && (TT_ENABLE_LIGHT_METAL_TRACE == 1)

// What should we name this? Another idea is TRACE_FUNCTION_THIS_SCOPE
#define TRACE_FUNCTION_ENTRY() tt::tt_metal::TraceScope __traceScopeGuard

#define TRACE_FUNCTION_CALL(capture_func, ...)                                                     \
    do {                                                                                           \
        log_trace(                                                                                 \
            tt::LogMetalTrace,                                                                     \
            "TRACE_FUNCTION_CALL: {} via {} istracing: {} depth: {}",                              \
            #capture_func,                                                                         \
            __FUNCTION__,                                                                          \
            LightMetalCaptureContext::Get().IsTracing(),                                           \
            tt::tt_metal::TraceScope::depth);                                                      \
        if (LightMetalCaptureContext::Get().IsTracing() && tt::tt_metal::TraceScope::depth == 1) { \
            capture_func(__VA_ARGS__);                                                             \
        }                                                                                          \
    } while (0)
#else

#define TRACE_FUNCTION_ENTRY()
#define TRACE_FUNCTION_CALL(capture_func, ...) \
    do {                                       \
    } while (0)

#endif

namespace tt::tt_metal {

//////////////////////////////////////////////////////////////
// Debug Code                                               //
//////////////////////////////////////////////////////////////

inline void PrintHostDataType(const HostDataType& data) {
    std::visit(
        [](const auto& value) {
            using T = std::decay_t<decltype(value)>;
            if constexpr (std::is_same_v<T, const std::shared_ptr<std::vector<uint8_t>>>) {
                log_info(tt::LogMetalTrace, "HostDataType contains: std::shared_ptr<std::vector<uint8_t>>");
            } else if constexpr (std::is_same_v<T, const std::shared_ptr<std::vector<uint16_t>>>) {
                log_info(tt::LogMetalTrace, "HostDataType contains: std::shared_ptr<std::vector<uint16_t>>");
            } else if constexpr (std::is_same_v<T, const std::shared_ptr<std::vector<int32_t>>>) {
                log_info(tt::LogMetalTrace, "HostDataType contains: std::shared_ptr<std::vector<int32_t>>");
            } else if constexpr (std::is_same_v<T, const std::shared_ptr<std::vector<uint32_t>>>) {
                log_info(tt::LogMetalTrace, "HostDataType contains: std::shared_ptr<std::vector<uint32_t>>");
            } else if constexpr (std::is_same_v<T, const std::shared_ptr<std::vector<float>>>) {
                log_info(tt::LogMetalTrace, "HostDataType contains: std::shared_ptr<std::vector<float>>");
            } else if constexpr (std::is_same_v<T, const std::shared_ptr<std::vector<bfloat16>>>) {
                log_info(tt::LogMetalTrace, "HostDataType contains: std::shared_ptr<std::vector<bfloat16>>");
            } else if constexpr (std::is_same_v<T, const void*>) {
                log_info(tt::LogMetalTrace, "HostDataType contains: const void*");
            } else {
                log_info(tt::LogMetalTrace, "HostDataType contains: Unknown type");
            }
        },
        data);
}

//////////////////////////////////////////////////////////////
// Host API tracing helper functions                        //
//////////////////////////////////////////////////////////////

// Generic helper to build command and add to vector of cmds (CQ)
inline void CaptureCommand(tt::tt_metal::flatbuffer::CommandType cmd_type, ::flatbuffers::Offset<void> fb_offset) {
    auto& ctx = LightMetalCaptureContext::Get();
    ctx.GetCmdsVector().push_back(tt::tt_metal::flatbuffer::CreateCommand(ctx.GetBuilder(), cmd_type, fb_offset));
}

inline void CaptureReplayTrace(IDevice* device, uint8_t cq_id, uint32_t tid, bool blocking) {
    auto& ctx = LightMetalCaptureContext::Get();
    log_debug(tt::LogMetalTrace, "{}: cq_id: {}, tid: {}, blocking: {}", __FUNCTION__, cq_id, tid, blocking);
    auto cmd = tt::tt_metal::flatbuffer::CreateReplayTraceCommand(ctx.GetBuilder(), cq_id, tid, blocking);
    CaptureCommand(tt::tt_metal::flatbuffer::CommandType::ReplayTraceCommand, cmd.Union());
}

inline void CaptureEnqueueTrace(CommandQueue& cq, uint32_t tid, bool blocking) {
    auto& ctx = LightMetalCaptureContext::Get();
    log_debug(tt::LogMetalTrace, "{}: cq_id: {}, tid: {}, blocking: {}", __FUNCTION__, cq.id(), tid, blocking);
    auto cmd = tt::tt_metal::flatbuffer::CreateEnqueueTraceCommand(ctx.GetBuilder(), cq.id(), tid, blocking);
    CaptureCommand(tt::tt_metal::flatbuffer::CommandType::EnqueueTraceCommand, cmd.Union());
}

inline void CaptureLoadTrace(IDevice* device, const uint8_t cq_id, const uint32_t tid) {
    auto& ctx = LightMetalCaptureContext::Get();
    log_debug(tt::LogMetalTrace, "{}: cq_id: {}, tid: {}", __FUNCTION__, cq_id, tid);
    auto cmd = tt::tt_metal::flatbuffer::CreateLoadTraceCommand(ctx.GetBuilder(), tid, cq_id);
    CaptureCommand(tt::tt_metal::flatbuffer::CommandType::LoadTraceCommand, cmd.Union());
}

inline void CaptureReleaseTrace(IDevice* device, uint32_t tid) {
    auto& ctx = LightMetalCaptureContext::Get();
    log_debug(tt::LogMetalTrace, "{}: tid: {}", __FUNCTION__, tid);
    auto cmd = tt::tt_metal::flatbuffer::CreateReleaseTraceCommand(ctx.GetBuilder(), tid);
    CaptureCommand(tt::tt_metal::flatbuffer::CommandType::ReleaseTraceCommand, cmd.Union());
}

inline void CaptureCreateBuffer(std::shared_ptr<Buffer> buffer, const InterleavedBufferConfig& config) {
    auto& ctx = LightMetalCaptureContext::Get();

    uint32_t buffer_global_id = ctx.AddToMap(buffer.get());
    log_debug(
        tt::LogMetalTrace,
        "{}: size: {} page_size: {} buffer_type: {} buffer_layout: {} buffer_global_id: {}",
        __FUNCTION__,
        config.size,
        config.page_size,
        config.buffer_type,
        config.buffer_layout,
        buffer_global_id);

    assert(config.device->id() == 0 && "multichip not supported yet");
    auto buffer_config_offset = tt::tt_metal::flatbuffer::CreateInterleavedBufferConfig(
        ctx.GetBuilder(),
        config.device->id(),
        config.size,
        config.page_size,
        ToFlatbuffer(config.buffer_type),
        ToFlatbuffer(config.buffer_layout));
    auto cmd =
        tt::tt_metal::flatbuffer::CreateCreateBufferCommand(ctx.GetBuilder(), buffer_global_id, buffer_config_offset);
    CaptureCommand(tt::tt_metal::flatbuffer::CommandType::CreateBufferCommand, cmd.Union());
}

inline void CaptureDeallocateBuffer(Buffer* buffer) {
    auto& ctx = LightMetalCaptureContext::Get();

    // Kind of a workaround, but Program Binaries buffer is created via Buffer::create() but can be
    // deallocated on Program destruction while capturing is still enabled depending on test structure (scope)
    // so let's just not capture these DeallocateBuffer() calls since they will occur on playback naturally.
    if (!ctx.IsInMap(buffer)) {
        log_debug(tt::LogMetalTrace, "Cannot capture DeallocateBuffer() without CreateBuffer() - ignoring.");
        return;
    }

    auto buffer_global_id = ctx.GetGlobalId(buffer);

    log_debug(
        tt::LogMetalTrace,
        "{}: buffer_global_id: {} size: {} address: {}",
        __FUNCTION__,
        buffer_global_id,
        buffer->size(),
        buffer->address());

    auto cmd = tt::tt_metal::flatbuffer::CreateDeallocateBufferCommand(ctx.GetBuilder(), buffer_global_id);
    CaptureCommand(tt::tt_metal::flatbuffer::CommandType::DeallocateBufferCommand, cmd.Union());
}

inline void CaptureEnqueueWriteBuffer(
    CommandQueue& cq,
    std::variant<std::reference_wrapper<Buffer>, std::shared_ptr<Buffer>> buffer,
    HostDataType src,
    bool blocking) {
    auto& ctx = LightMetalCaptureContext::Get();

    // We don't want to use shared_ptr to extend lifetime of buffer when adding to global_id map.
    Buffer* buffer_ptr = std::holds_alternative<std::shared_ptr<Buffer>>(buffer)
                             ? std::get<std::shared_ptr<Buffer>>(buffer).get()
                             : &std::get<std::reference_wrapper<Buffer>>(buffer).get();

    uint32_t cq_global_id = cq.id();  // TODO (kmabee) - consider storing/getting CQ from global map instead.
    uint32_t buffer_global_id = ctx.GetGlobalId(buffer_ptr);

    log_debug(
        tt::LogMetalTrace, "{}: cq_global_id: {} buffer_global_id: {}", __FUNCTION__, cq_global_id, buffer_global_id);
    // PrintHostDataType(src); // Debug

    // TODO (kmabee) - Currently support limited data formats. Long term we might not store data in flatbuffer,
    // but have it provided at runtime so just do what's easiest here and support few types for now.
    ::flatbuffers::Offset<::flatbuffers::Vector<uint32_t>> src_vector;
    if (auto* uint32_vec = std::get_if<const std::shared_ptr<std::vector<uint32_t>>>(&src)) {
        src_vector = ctx.GetBuilder().CreateVector(**uint32_vec);
    } else if (auto* uint16_vec = std::get_if<const std::shared_ptr<std::vector<uint16_t>>>(&src)) {
        // Convert uint16_t to uint32_t before creating the FlatBuffers vector
        std::vector<uint32_t> converted(uint16_vec->get()->begin(), uint16_vec->get()->end());
        src_vector = ctx.GetBuilder().CreateVector(converted);
    } else if (auto* void_ptr = std::get_if<const void*>(&src)) {
        // Assuming the void* points to a buffer of uint32_t values. Infer size, cast to uint32_t.
        size_t num_elements = buffer_ptr->size() / sizeof(uint32_t);
        auto uint32_data = static_cast<const uint32_t*>(*void_ptr);
        src_vector = ctx.GetBuilder().CreateVector(uint32_data, num_elements);
    } else {
        throw std::runtime_error("Unsupported HostDataType for captureEnqueueWriteBuffer()");
    }

    auto cmd = tt::tt_metal::flatbuffer::CreateEnqueueWriteBufferCommand(
        ctx.GetBuilder(), cq_global_id, buffer_global_id, src_vector, blocking);
    CaptureCommand(tt::tt_metal::flatbuffer::CommandType::EnqueueWriteBufferCommand, cmd.Union());
}

inline void CaptureEnqueueReadBuffer(
    CommandQueue& cq,
    std::variant<std::reference_wrapper<Buffer>, std::shared_ptr<Buffer>> buffer,
    void* dst,
    bool blocking) {
    auto& ctx = LightMetalCaptureContext::Get();

    // We don't want to use shared_ptr to extend lifetime of buffer when adding to global_id map.
    Buffer* buffer_ptr = std::holds_alternative<std::shared_ptr<Buffer>>(buffer)
                             ? std::get<std::shared_ptr<Buffer>>(buffer).get()
                             : &std::get<std::reference_wrapper<Buffer>>(buffer).get();

    uint32_t cq_global_id = cq.id();  // TODO (kmabee) - consider storing/getting CQ from global map instead.
    uint32_t buffer_global_id = ctx.GetGlobalId(buffer_ptr);

    log_debug(
        tt::LogMetalTrace, "{}: cq_global_id: {} buffer_global_id: {}", __FUNCTION__, cq_global_id, buffer_global_id);

    // Idea store a read_global_id to keep track of read results.
    auto cmd = tt::tt_metal::flatbuffer::CreateEnqueueReadBufferCommand(
        ctx.GetBuilder(), cq_global_id, buffer_global_id, blocking);
    CaptureCommand(tt::tt_metal::flatbuffer::CommandType::EnqueueReadBufferCommand, cmd.Union());
}

inline void CaptureFinish(CommandQueue& cq, tt::stl::Span<const SubDeviceId> sub_device_ids) {
    auto& ctx = LightMetalCaptureContext::Get();
    uint32_t cq_global_id = cq.id();  // TODO (kmabee) - consider storing/getting CQ from global map instead.

    // Use ToFlatbuffer to convert SubDeviceIds to FlatBuffer vector
    auto fb_sub_device_ids = ToFlatbuffer(ctx.GetBuilder(), sub_device_ids);

    log_debug(
        tt::LogMetalTrace, "{}: cq_global_id: {} sub_devices: {}", __FUNCTION__, cq_global_id, sub_device_ids.size());
    auto cmd = tt::tt_metal::flatbuffer::CreateFinishCommand(ctx.GetBuilder(), cq_global_id, fb_sub_device_ids);
    CaptureCommand(tt::tt_metal::flatbuffer::CommandType::FinishCommand, cmd.Union());
}

inline void CaptureCreateProgram(Program& program) {
    auto& ctx = LightMetalCaptureContext::Get();
    uint32_t program_global_id = ctx.AddToMap(&program);
    log_debug(tt::LogMetalTrace, "{}: program_global_id: {}", __FUNCTION__, program_global_id);

    auto cmd = tt::tt_metal::flatbuffer::CreateCreateProgramCommand(ctx.GetBuilder(), program_global_id);
    CaptureCommand(tt::tt_metal::flatbuffer::CommandType::CreateProgramCommand, cmd.Union());
}

inline void CaptureEnqueueProgram(CommandQueue& cq, Program& program, bool blocking) {
    auto& ctx = LightMetalCaptureContext::Get();

    // When Metal Trace is enabled, skip EnqueueProgram capture (replaced with LoadTrace + ReplayTrace)
    if (cq.hw_command_queue().manager.get_bypass_mode()) {
        return;
    }

    uint32_t cq_global_id = cq.id();  // TODO (kmabee) - consider storing/getting CQ from global map instead.
    uint32_t program_global_id = ctx.GetGlobalId(&program);
    log_debug(
        tt::LogMetalTrace, "{}: cq_global_id: {} program_global_id: {}", __FUNCTION__, cq_global_id, program_global_id);

    auto cmd = tt::tt_metal::flatbuffer::CreateEnqueueProgramCommand(
        ctx.GetBuilder(), cq_global_id, program_global_id, blocking);
    CaptureCommand(tt::tt_metal::flatbuffer::CommandType::EnqueueProgramCommand, cmd.Union());
}

inline void CaptureCreateKernel(
    KernelHandle kernel_id,
    Program& program,
    const std::string& file_name,
    const std::variant<CoreCoord, CoreRange, CoreRangeSet>& core_spec,
    const std::variant<DataMovementConfig, ComputeConfig, EthernetConfig>& config) {
    auto& ctx = LightMetalCaptureContext::Get();

    std::shared_ptr<Kernel> kernel = program.get_kernel(kernel_id);
    uint32_t kernel_global_id = ctx.AddToMap(kernel.get());
    uint32_t program_global_id = ctx.GetGlobalId(&program);
    log_debug(
        tt::LogMetalTrace,
        "{}: file_name: {} kernel_global_id: {} (kernel_id: {}) program_global_id: {}",
        __FUNCTION__,
        file_name,
        kernel_global_id,
        kernel_id,
        program_global_id);

    auto& fbb = ctx.GetBuilder();
    auto filename_offset = fbb.CreateString(file_name);
    auto [core_spec_type, core_spec_offset] = ToFlatbuffer(fbb, core_spec);
    auto [config_type, config_offset] = ToFlatbuffer(fbb, config);

    auto cmd = tt::tt_metal::flatbuffer::CreateCreateKernelCommand(
        fbb,
        kernel_global_id,
        program_global_id,
        filename_offset,
        core_spec_type,
        core_spec_offset,
        config_type,
        config_offset);
    CaptureCommand(tt::tt_metal::flatbuffer::CommandType::CreateKernelCommand, cmd.Union());
}

inline void CaptureSetRuntimeArgsUint32(
    const Program& program,
    KernelHandle kernel_id,
    const std::variant<CoreCoord, CoreRange, CoreRangeSet>& core_spec,
    tt::stl::Span<const uint32_t> runtime_args) {
    auto& ctx = LightMetalCaptureContext::Get();

    std::shared_ptr<Kernel> kernel = program.get_kernel(kernel_id);
    uint32_t program_global_id = ctx.GetGlobalId(&program);
    uint32_t kernel_global_id = ctx.GetGlobalId(kernel.get());
    log_debug(
        tt::LogMetalTrace,
        "{}(uint32): kernel_global_id: {} program_global_id: {} rt_args: {}",
        __FUNCTION__,
        kernel_global_id,
        program_global_id,
        runtime_args.size());

    auto& fbb = ctx.GetBuilder();
    auto [core_spec_type, core_spec_offset] = ToFlatbuffer(fbb, core_spec);
    auto rt_args_offset = fbb.CreateVector(runtime_args.data(), runtime_args.size());

    auto cmd = tt::tt_metal::flatbuffer::CreateSetRuntimeArgsUint32Command(
        fbb, program_global_id, kernel_global_id, core_spec_type, core_spec_offset, rt_args_offset);
    CaptureCommand(tt::tt_metal::flatbuffer::CommandType::SetRuntimeArgsUint32Command, cmd.Union());
}

inline void CaptureSetRuntimeArgs(
    IDevice* device,
    const std::shared_ptr<Kernel> kernel,
    const std::variant<CoreCoord, CoreRange, CoreRangeSet>& core_spec,
    std::shared_ptr<RuntimeArgs> runtime_args) {
    auto& ctx = LightMetalCaptureContext::Get();
    auto& fbb = ctx.GetBuilder();
    uint32_t kernel_global_id = ctx.GetGlobalId(kernel.get());
    auto [core_spec_type, core_spec_offset] = ToFlatbuffer(fbb, core_spec);
    auto rt_args_offset = ToFlatbuffer(fbb, runtime_args);
    log_debug(
        tt::LogMetalTrace,
        "{}(RuntimeArgs): kernel_global_id: {} rt_args_size: {}",
        __FUNCTION__,
        kernel_global_id,
        runtime_args->size());

    auto cmd = tt::tt_metal::flatbuffer::CreateSetRuntimeArgsCommand(
        fbb, kernel_global_id, core_spec_type, core_spec_offset, rt_args_offset);
    CaptureCommand(tt::tt_metal::flatbuffer::CommandType::SetRuntimeArgsCommand, cmd.Union());
}

inline void CaptureCreateCircularBuffer(
    CBHandle& cb_handle,
    Program& program,
    const std::variant<CoreCoord, CoreRange, CoreRangeSet>& core_spec,
    const CircularBufferConfig& config) {
    auto& ctx = LightMetalCaptureContext::Get();
    auto& fbb = ctx.GetBuilder();
    uint32_t cb_global_id = ctx.AddToMap(cb_handle);
    uint32_t program_global_id = ctx.GetGlobalId(&program);
    auto [core_spec_type, core_spec_offset] = ToFlatbuffer(fbb, core_spec);
    auto cb_config_offset = ToFlatbuffer(config, fbb);
    log_debug(
        tt::LogMetalTrace,
        "{}: cb_global_id: {} program_global_id: {} ",
        __FUNCTION__,
        cb_global_id,
        program_global_id);

    auto cmd = tt::tt_metal::flatbuffer::CreateCreateCircularBufferCommand(
        fbb, cb_global_id, program_global_id, core_spec_type, core_spec_offset, cb_config_offset);
    CaptureCommand(tt::tt_metal::flatbuffer::CommandType::CreateCircularBufferCommand, cmd.Union());
}

inline void CaptureLightMetalCompare(
    CommandQueue& cq,
    std::variant<std::reference_wrapper<Buffer>, std::shared_ptr<Buffer>> buffer,
    void* golden_data,
    bool is_user_data) {
    auto& ctx = LightMetalCaptureContext::Get();

    // We don't want to use shared_ptr to extend lifetime of buffer when adding to global_id map.
    Buffer* buffer_ptr = std::holds_alternative<std::shared_ptr<Buffer>>(buffer)
                             ? std::get<std::shared_ptr<Buffer>>(buffer).get()
                             : &std::get<std::reference_wrapper<Buffer>>(buffer).get();

    uint32_t cq_global_id = cq.id();  // TODO (kmabee) - consider storing/getting CQ from global map instead.
    uint32_t buffer_global_id = ctx.GetGlobalId(buffer_ptr);

    // Calculate num uint32_t elements in buffer, and convert golden void* to vector
    size_t golden_data_len = buffer_ptr->size() / sizeof(uint32_t);
    const uint32_t* golden_data_uint32 = static_cast<const uint32_t*>(golden_data);
    std::vector<uint32_t> golden_data_vector(golden_data_uint32, golden_data_uint32 + golden_data_len);

    log_debug(
        tt::LogMetalTrace,
        "{}: buffer_global_id: {} is_user_data: {} golden_data_len: {}",
        __FUNCTION__,
        buffer_global_id,
        is_user_data,
        golden_data_len);

    // Serialize golden_data into FlatBuffer
    auto golden_data_fb = ctx.GetBuilder().CreateVector(golden_data_vector);

    auto cmd = tt::tt_metal::flatbuffer::CreateLightMetalCompareCommand(
        ctx.GetBuilder(), cq_global_id, buffer_global_id, golden_data_fb, is_user_data);
    CaptureCommand(tt::tt_metal::flatbuffer::CommandType::LightMetalCompareCommand, cmd.Union());
}

}  // namespace tt::tt_metal
