// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt_stl/overloaded.hpp>
#include <circular_buffer_config.hpp>
#include <tt-metalium/command_queue.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/program.hpp>
#include "dispatch/system_memory_manager.hpp"

#include <kernel_types.hpp>
#include "lightmetal/host_api_capture_helpers.hpp"
#include "command_generated.h"
#include "lightmetal/lightmetal_capture.hpp"
#include "flatbuffer/base_types_to_flatbuffer.hpp"
#include "flatbuffer/program_types_to_flatbuffer.hpp"
#include "flatbuffer/buffer_types_to_flatbuffer.hpp"

namespace tt::tt_metal {

//////////////////////////////////////////////////////////////
// Debug Code                                               //
//////////////////////////////////////////////////////////////

namespace {
// This can be useful for debug. Not all data types are currently supported, can use this during developmenmt.
void PrintHostDataType(const HostDataType& data) {
    std::visit(
        tt::stl::overloaded{
            [](const std::shared_ptr<std::vector<uint8_t>>& /*value*/) {
                log_info(tt::LogMetalTrace, "HostDataType contains: std::shared_ptr<std::vector<uint8_t>>");
            },
            [](const std::shared_ptr<std::vector<uint16_t>>& /*value*/) {
                log_info(tt::LogMetalTrace, "HostDataType contains: std::shared_ptr<std::vector<uint16_t>>");
            },
            [](const std::shared_ptr<std::vector<int32_t>>& /*value*/) {
                log_info(tt::LogMetalTrace, "HostDataType contains: std::shared_ptr<std::vector<int32_t>>");
            },
            [](const std::shared_ptr<std::vector<uint32_t>>& /*value*/) {
                log_info(tt::LogMetalTrace, "HostDataType contains: std::shared_ptr<std::vector<uint32_t>>");
            },
            [](const std::shared_ptr<std::vector<float>>& /*value*/) {
                log_info(tt::LogMetalTrace, "HostDataType contains: std::shared_ptr<std::vector<float>>");
            },
            [](const std::shared_ptr<std::vector<bfloat16>>& /*value*/) {
                log_info(tt::LogMetalTrace, "HostDataType contains: std::shared_ptr<std::vector<bfloat16>>");
            },
            [](const void* /*value*/) { log_info(tt::LogMetalTrace, "HostDataType contains: const void*"); },
            [](auto&&) { log_info(tt::LogMetalTrace, "HostDataType contains: Unknown type"); }},
        data);
}
}  // namespace

//////////////////////////////////////////////////////////////
// Host API tracing helper functions                        //
//////////////////////////////////////////////////////////////

// Generic helper to build command and add to vector of cmds (CQ) - no need to make public
namespace {
void CaptureCommand(tt::tt_metal::flatbuffer::CommandType cmd_type, ::flatbuffers::Offset<void> fb_offset) {
    auto& ctx = LightMetalCaptureContext::get();
    ctx.get_cmds_vector().push_back(tt::tt_metal::flatbuffer::CreateCommand(ctx.get_builder(), cmd_type, fb_offset));
}
}  // namespace

void CaptureReplayTrace(IDevice* /*device*/, uint8_t cq_id, uint32_t trace_id, bool blocking) {
    auto& ctx = LightMetalCaptureContext::get();
    log_debug(tt::LogMetalTrace, "{}: cq_id: {} trace_id: {} blocking: {}", __FUNCTION__, cq_id, trace_id, blocking);
    auto cmd = tt::tt_metal::flatbuffer::CreateReplayTraceCommand(ctx.get_builder(), cq_id, trace_id, blocking);
    CaptureCommand(tt::tt_metal::flatbuffer::CommandType::ReplayTraceCommand, cmd.Union());
}

void CaptureEnqueueTrace(CommandQueue& cq, uint32_t trace_id, bool blocking) {
    auto& ctx = LightMetalCaptureContext::get();
    log_debug(tt::LogMetalTrace, "{}: cq_id: {} trace_id: {} blocking: {}", __FUNCTION__, cq.id(), trace_id, blocking);
    auto cmd = tt::tt_metal::flatbuffer::CreateEnqueueTraceCommand(ctx.get_builder(), cq.id(), trace_id, blocking);
    CaptureCommand(tt::tt_metal::flatbuffer::CommandType::EnqueueTraceCommand, cmd.Union());
}

void CaptureLoadTrace(IDevice* /*device*/, uint8_t cq_id, uint32_t trace_id) {
    auto& ctx = LightMetalCaptureContext::get();
    log_debug(tt::LogMetalTrace, "{}: cq_id: {} trace_id: {}", __FUNCTION__, cq_id, trace_id);
    auto cmd = tt::tt_metal::flatbuffer::CreateLoadTraceCommand(ctx.get_builder(), trace_id, cq_id);
    CaptureCommand(tt::tt_metal::flatbuffer::CommandType::LoadTraceCommand, cmd.Union());
}

void CaptureReleaseTrace(IDevice* /*device*/, uint32_t trace_id) {
    auto& ctx = LightMetalCaptureContext::get();
    log_debug(tt::LogMetalTrace, "{}: trace_id: {}", __FUNCTION__, trace_id);
    auto cmd = tt::tt_metal::flatbuffer::CreateReleaseTraceCommand(ctx.get_builder(), trace_id);
    CaptureCommand(tt::tt_metal::flatbuffer::CommandType::ReleaseTraceCommand, cmd.Union());
}

void CaptureBufferCreate(
    const std::shared_ptr<Buffer>& buffer,
    IDevice* device,
    const std::optional<DeviceAddr> address,
    DeviceAddr size,
    DeviceAddr page_size,
    const BufferType buffer_type,
    const TensorMemoryLayout buffer_layout,
    const std::optional<std::variant<ShardSpecBuffer, BufferDistributionSpec>>& shard_parameters,
    const std::optional<bool> bottom_up,
    const std::optional<SubDeviceId> sub_device_id) {
    auto& ctx = LightMetalCaptureContext::get();
    auto& fbb = ctx.get_builder();

    uint32_t buffer_global_id = ctx.add_to_map(buffer.get());

    log_debug(
        tt::LogMetalTrace,
        "{}: size: {} page_size: {} buffer_type: {} buffer_layout: {} buffer_global_id: {}",
        __FUNCTION__,
        size,
        page_size,
        buffer_type,
        buffer_layout,
        buffer_global_id);

    // Convert the optional fields to flatbuffer offsets.
    // Address is not true optional for API, but Buffer::create() API has 2 flavors, one with address
    // and one without, so commonize via single capture function and schema and treat it as optional.
    auto address_offset = address.has_value() ? flatbuffer::CreateUint32Optional(fbb, address.value()) : 0;
    auto bottom_up_offset = bottom_up.has_value() ? flatbuffer::CreateBoolOptional(fbb, bottom_up.value()) : 0;
    auto sub_device_id_offset = sub_device_id.has_value() ? flatbuffer::CreateUint8Optional(fbb, **sub_device_id) : 0;
    std::optional<ShardSpecBuffer> shard_spec_buffer;
    if (shard_parameters && std::holds_alternative<ShardSpecBuffer>(*shard_parameters)) {
        shard_spec_buffer = std::get<ShardSpecBuffer>(*shard_parameters);
    }
    auto shard_parameters_offset = to_flatbuffer(shard_spec_buffer, fbb);

    auto cmd = tt::tt_metal::flatbuffer::CreateBufferCreateCommand(
        fbb,
        buffer_global_id,
        device->id(),
        address_offset,
        size,
        page_size,
        to_flatbuffer(buffer_type),
        to_flatbuffer(buffer_layout),
        shard_parameters_offset,
        bottom_up_offset,
        sub_device_id_offset);

    CaptureCommand(tt::tt_metal::flatbuffer::CommandType::BufferCreateCommand, cmd.Union());
}

void CaptureBufferDeallocate(const Buffer& buffer) {
    auto& ctx = LightMetalCaptureContext::get();

    // Kind of a workaround, but Program Binaries buffer is created via Buffer::create() but can be
    // deallocated on Program destruction while capturing is still enabled depending on test structure (scope)
    // so let's just not capture these DeallocateBuffer() calls since they will occur on playback naturally.
    if (!ctx.is_in_map(&buffer)) {
        log_debug(tt::LogMetalTrace, "Cannot capture DeallocateBuffer() without CreateBuffer() - ignoring.");
        return;
    }

    auto buffer_global_id = ctx.get_global_id(&buffer);

    log_debug(
        tt::LogMetalTrace,
        "{}: buffer_global_id: {} size: {} address: {}",
        __FUNCTION__,
        buffer_global_id,
        buffer.size(),
        buffer.address());

    auto cmd = tt::tt_metal::flatbuffer::CreateBufferDeallocateCommand(ctx.get_builder(), buffer_global_id);
    CaptureCommand(tt::tt_metal::flatbuffer::CommandType::BufferDeallocateCommand, cmd.Union());
}

void CaptureBufferDelete(const Buffer& buffer) {
    auto& ctx = LightMetalCaptureContext::get();

    // Kind of a workaround, but Program Binaries buffer is created via Buffer::create() but can be
    // deallocated on Program destruction while capturing is still enabled depending on test structure (scope)
    // so let's just not capture these DeallocateBuffer() calls since they will occur on playback naturally.
    if (!ctx.is_in_map(&buffer)) {
        log_debug(
            tt::LogMetalTrace,
            "Cannot capture Buffer Delete without CreateBuffer() - ignoring Buffer w/ addr: 0x{:x}",
            buffer.address());
        return;
    }

    auto buffer_global_id = ctx.get_global_id(&buffer);

    log_debug(
        tt::LogMetalTrace,
        "{}: buffer_global_id: {} size: {} address: {}",
        __FUNCTION__,
        buffer_global_id,
        buffer.size(),
        buffer.address());

    auto cmd = tt::tt_metal::flatbuffer::CreateBufferDeleteCommand(ctx.get_builder(), buffer_global_id);
    CaptureCommand(tt::tt_metal::flatbuffer::CommandType::BufferDeleteCommand, cmd.Union());
}

void CaptureEnqueueWriteBuffer(
    CommandQueue& cq,
    std::variant<std::reference_wrapper<Buffer>, std::shared_ptr<Buffer>> buffer,
    HostDataType src,
    bool blocking) {
    auto& ctx = LightMetalCaptureContext::get();

    // We don't want to use shared_ptr to extend lifetime of buffer when adding to global_id map.
    Buffer* buffer_ptr = std::holds_alternative<std::shared_ptr<Buffer>>(buffer)
                             ? std::get<std::shared_ptr<Buffer>>(buffer).get()
                             : &std::get<std::reference_wrapper<Buffer>>(buffer).get();

    uint32_t cq_global_id = cq.id();  // TODO (kmabee) - consider storing/getting CQ from global map instead.
    uint32_t buffer_global_id = ctx.get_global_id(buffer_ptr);

    log_debug(
        tt::LogMetalTrace, "{}: cq_global_id: {} buffer_global_id: {}", __FUNCTION__, cq_global_id, buffer_global_id);
    // PrintHostDataType(src); // Debug

    // TODO (kmabee) - Currently support limited data formats. Long term we might not store data in flatbuffer,
    // but have it provided at runtime so just do what's easiest here and support few types for now.
    ::flatbuffers::Offset<::flatbuffers::Vector<uint32_t>> src_vector;
    if (auto* uint32_vec = std::get_if<const std::shared_ptr<std::vector<uint32_t>>>(&src)) {
        src_vector = ctx.get_builder().CreateVector(**uint32_vec);
    } else if (auto* uint16_vec = std::get_if<const std::shared_ptr<std::vector<uint16_t>>>(&src)) {
        // Convert uint16_t to uint32_t before creating the FlatBuffers vector
        std::vector<uint32_t> converted(uint16_vec->get()->begin(), uint16_vec->get()->end());
        src_vector = ctx.get_builder().CreateVector(converted);
    } else if (auto* void_ptr = std::get_if<const void*>(&src)) {
        // Assuming the void* points to a buffer of uint32_t values. Infer size, cast to uint32_t.
        size_t num_elements = buffer_ptr->size() / sizeof(uint32_t);
        auto uint32_data = static_cast<const uint32_t*>(*void_ptr);
        src_vector = ctx.get_builder().CreateVector(uint32_data, num_elements);
    } else {
        TT_THROW("Unsupported HostDataType for captureEnqueueWriteBuffer()");
    }

    auto cmd = tt::tt_metal::flatbuffer::CreateEnqueueWriteBufferCommand(
        ctx.get_builder(), cq_global_id, buffer_global_id, src_vector, blocking);
    CaptureCommand(tt::tt_metal::flatbuffer::CommandType::EnqueueWriteBufferCommand, cmd.Union());
}

void CaptureEnqueueReadBuffer(
    CommandQueue& cq,
    std::variant<std::reference_wrapper<Buffer>, std::shared_ptr<Buffer>> buffer,
    void* /*dst*/,
    bool blocking) {
    auto& ctx = LightMetalCaptureContext::get();

    // We don't want to use shared_ptr to extend lifetime of buffer when adding to global_id map.
    Buffer* buffer_ptr = std::holds_alternative<std::shared_ptr<Buffer>>(buffer)
                             ? std::get<std::shared_ptr<Buffer>>(buffer).get()
                             : &std::get<std::reference_wrapper<Buffer>>(buffer).get();

    uint32_t cq_global_id = cq.id();  // TODO (kmabee) - consider storing/getting CQ from global map instead.
    uint32_t buffer_global_id = ctx.get_global_id(buffer_ptr);

    log_debug(
        tt::LogMetalTrace, "{}: cq_global_id: {} buffer_global_id: {}", __FUNCTION__, cq_global_id, buffer_global_id);

    // Idea store a read_global_id to keep track of read results.
    auto cmd = tt::tt_metal::flatbuffer::CreateEnqueueReadBufferCommand(
        ctx.get_builder(), cq_global_id, buffer_global_id, blocking);
    CaptureCommand(tt::tt_metal::flatbuffer::CommandType::EnqueueReadBufferCommand, cmd.Union());
}

void CaptureFinish(CommandQueue& cq, tt::stl::Span<const SubDeviceId> sub_device_ids) {
    auto& ctx = LightMetalCaptureContext::get();
    uint32_t cq_global_id = cq.id();  // TODO (kmabee) - consider storing/getting CQ from global map instead.

    // Use to_flatbuffer to convert SubDeviceIds to FlatBuffer vector
    auto fb_sub_device_ids = to_flatbuffer(ctx.get_builder(), sub_device_ids);

    log_debug(
        tt::LogMetalTrace, "{}: cq_global_id: {} sub_devices: {}", __FUNCTION__, cq_global_id, sub_device_ids.size());
    auto cmd = tt::tt_metal::flatbuffer::CreateFinishCommand(ctx.get_builder(), cq_global_id, fb_sub_device_ids);
    CaptureCommand(tt::tt_metal::flatbuffer::CommandType::FinishCommand, cmd.Union());
}

void CaptureProgramConstructor(Program& program) {
    auto& ctx = LightMetalCaptureContext::get();
    uint32_t program_global_id = ctx.add_to_map(&program);
    log_debug(tt::LogMetalTrace, "{}: program_global_id: {}", __FUNCTION__, program_global_id);

    auto cmd = tt::tt_metal::flatbuffer::CreateProgramConstructorCommand(ctx.get_builder(), program_global_id);
    CaptureCommand(tt::tt_metal::flatbuffer::CommandType::ProgramConstructorCommand, cmd.Union());
}

void CaptureEnqueueProgram(CommandQueue& cq, Program& program, bool blocking) {
    auto& ctx = LightMetalCaptureContext::get();

    // When Metal Trace is enabled, skip EnqueueProgram capture (replaced with LoadTrace + ReplayTrace)
    if (cq.sysmem_manager().get_bypass_mode()) {
        return;
    }

    uint32_t cq_global_id = cq.id();  // TODO (kmabee) - consider storing/getting CQ from global map instead.
    uint32_t program_global_id = ctx.get_global_id(&program);
    log_debug(
        tt::LogMetalTrace, "{}: cq_global_id: {} program_global_id: {}", __FUNCTION__, cq_global_id, program_global_id);

    auto cmd = tt::tt_metal::flatbuffer::CreateEnqueueProgramCommand(
        ctx.get_builder(), cq_global_id, program_global_id, blocking);
    CaptureCommand(tt::tt_metal::flatbuffer::CommandType::EnqueueProgramCommand, cmd.Union());
}

void CaptureCreateKernel(
    KernelHandle kernel_id,
    Program& program,
    const std::string& file_name,
    const std::variant<CoreCoord, CoreRange, CoreRangeSet>& core_spec,
    const std::variant<DataMovementConfig, ComputeConfig, EthernetConfig>& config) {
    auto& ctx = LightMetalCaptureContext::get();

    std::shared_ptr<Kernel> kernel = program.get_kernel(kernel_id);
    uint32_t kernel_global_id = ctx.add_to_map(kernel.get());
    uint32_t program_global_id = ctx.get_global_id(&program);
    log_debug(
        tt::LogMetalTrace,
        "{}: file_name: {} kernel_global_id: {} (kernel_id: {}) program_global_id: {}",
        __FUNCTION__,
        file_name,
        kernel_global_id,
        kernel_id,
        program_global_id);

    auto& fbb = ctx.get_builder();
    auto filename_offset = fbb.CreateString(file_name);
    auto [core_spec_type, core_spec_offset] = to_flatbuffer(fbb, core_spec);
    auto [kernel_config_type, kernel_config_offset] = to_flatbuffer(fbb, config);

    auto cmd = tt::tt_metal::flatbuffer::CreateCreateKernelCommand(
        fbb,
        kernel_global_id,
        program_global_id,
        filename_offset,
        core_spec_type,
        core_spec_offset,
        kernel_config_type,
        kernel_config_offset);
    CaptureCommand(tt::tt_metal::flatbuffer::CommandType::CreateKernelCommand, cmd.Union());
}

void CaptureSetRuntimeArgsUint32(
    const Program& program,
    KernelHandle kernel_id,
    const std::variant<CoreCoord, CoreRange, CoreRangeSet>& core_spec,
    tt::stl::Span<const uint32_t> runtime_args) {
    auto& ctx = LightMetalCaptureContext::get();

    std::shared_ptr<Kernel> kernel = program.get_kernel(kernel_id);
    uint32_t program_global_id = ctx.get_global_id(&program);
    uint32_t kernel_global_id = ctx.get_global_id(kernel.get());
    log_debug(
        tt::LogMetalTrace,
        "{}: kernel_global_id: {} program_global_id: {} rt_args: {}",
        __FUNCTION__,
        kernel_global_id,
        program_global_id,
        runtime_args.size());

    auto& fbb = ctx.get_builder();
    auto [core_spec_type, core_spec_offset] = to_flatbuffer(fbb, core_spec);
    auto rt_args_offset = fbb.CreateVector(runtime_args.data(), runtime_args.size());

    auto cmd = tt::tt_metal::flatbuffer::CreateSetRuntimeArgsUint32Command(
        fbb, program_global_id, kernel_global_id, core_spec_type, core_spec_offset, rt_args_offset);
    CaptureCommand(tt::tt_metal::flatbuffer::CommandType::SetRuntimeArgsUint32Command, cmd.Union());
}

void CaptureSetRuntimeArgsUint32VecPerCore(
    const Program& program,
    KernelHandle kernel_id,
    const std::vector<CoreCoord>& core_spec,
    const std::vector<std::vector<uint32_t>>& runtime_args) {
    auto& ctx = LightMetalCaptureContext::get();

    std::shared_ptr<Kernel> kernel = program.get_kernel(kernel_id);
    uint32_t program_global_id = ctx.get_global_id(&program);
    uint32_t kernel_global_id = ctx.get_global_id(kernel.get());
    log_debug(
        tt::LogMetalTrace,
        "{}: kernel_global_id: {} program_global_id: {} num_cores: {}",
        __FUNCTION__,
        kernel_global_id,
        program_global_id,
        core_spec.size());

    auto& fbb = ctx.get_builder();
    auto core_spec_offset = to_flatbuffer(fbb, core_spec);
    auto runtime_args_offset = to_flatbuffer(fbb, runtime_args);

    auto cmd = tt::tt_metal::flatbuffer::CreateSetRuntimeArgsUint32VecPerCoreCommand(
        fbb, program_global_id, kernel_global_id, core_spec_offset, runtime_args_offset);

    CaptureCommand(tt::tt_metal::flatbuffer::CommandType::SetRuntimeArgsUint32VecPerCoreCommand, cmd.Union());
}
void CaptureSetRuntimeArgs(
    IDevice* /*device*/,
    const std::shared_ptr<Kernel>& kernel,
    const std::variant<CoreCoord, CoreRange, CoreRangeSet>& core_spec,
    const std::shared_ptr<RuntimeArgs>& runtime_args) {
    auto& ctx = LightMetalCaptureContext::get();
    auto& fbb = ctx.get_builder();
    uint32_t kernel_global_id = ctx.get_global_id(kernel.get());
    auto [core_spec_type, core_spec_offset] = to_flatbuffer(fbb, core_spec);
    auto rt_args_offset = to_flatbuffer(fbb, runtime_args);
    log_debug(
        tt::LogMetalTrace,
        "{}: kernel_global_id: {} rt_args_size: {}",
        __FUNCTION__,
        kernel_global_id,
        runtime_args->size());

    auto cmd = tt::tt_metal::flatbuffer::CreateSetRuntimeArgsCommand(
        fbb, kernel_global_id, core_spec_type, core_spec_offset, rt_args_offset);
    CaptureCommand(tt::tt_metal::flatbuffer::CommandType::SetRuntimeArgsCommand, cmd.Union());
}

void CaptureCreateCircularBuffer(
    CBHandle& cb_handle,
    Program& program,
    const std::variant<CoreCoord, CoreRange, CoreRangeSet>& core_spec,
    const CircularBufferConfig& config) {
    auto& ctx = LightMetalCaptureContext::get();
    auto& fbb = ctx.get_builder();
    uint32_t cb_global_id = ctx.add_to_map(cb_handle);
    uint32_t program_global_id = ctx.get_global_id(&program);
    auto [core_spec_type, core_spec_offset] = to_flatbuffer(fbb, core_spec);
    auto cb_config_offset = to_flatbuffer(config, fbb);
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

void CaptureLightMetalCompare(
    CommandQueue& cq,
    std::variant<std::reference_wrapper<Buffer>, std::shared_ptr<Buffer>> buffer,
    void* golden_data,
    bool is_user_data) {
    auto& ctx = LightMetalCaptureContext::get();

    // We don't want to use shared_ptr to extend lifetime of buffer when adding to global_id map.
    Buffer* buffer_ptr = std::holds_alternative<std::shared_ptr<Buffer>>(buffer)
                             ? std::get<std::shared_ptr<Buffer>>(buffer).get()
                             : &std::get<std::reference_wrapper<Buffer>>(buffer).get();

    uint32_t cq_global_id = cq.id();  // TODO (kmabee) - consider storing/getting CQ from global map instead.
    uint32_t buffer_global_id = ctx.get_global_id(buffer_ptr);

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
    auto golden_data_fb = ctx.get_builder().CreateVector(golden_data_vector);

    auto cmd = tt::tt_metal::flatbuffer::CreateLightMetalCompareCommand(
        ctx.get_builder(), cq_global_id, buffer_global_id, golden_data_fb, is_user_data);
    CaptureCommand(tt::tt_metal::flatbuffer::CommandType::LightMetalCompareCommand, cmd.Union());
}

}  // namespace tt::tt_metal
