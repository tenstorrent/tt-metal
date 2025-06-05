// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "flatbuffers/flatbuffers.h"
#include "lightmetal/lightmetal_capture.hpp"
#include <tt-logger/tt-logger.hpp>
#include <tt_stl/span.hpp>
#include <tt-metalium/buffer.hpp>
#include <kernel_types.hpp>

namespace tt::tt_metal {

// Many forward decls and aliases to reduce includes.
class CommandQueue;
struct DataMovementConfig;
struct ComputeConfig;
struct EthernetConfig;

class IDevice;
struct BufferConfig;
struct CircularBufferConfig;
using RuntimeArgs = std::vector<std::variant<Buffer*, uint32_t>>;

//////////////////////////////////////////////////////////////
// TRACE GUARD & LIGHT METAL TRACE MACRO                    //
//////////////////////////////////////////////////////////////

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

#define LIGHT_METAL_TRACE_FUNCTION_ENTRY() tt::tt_metal::TraceScope __traceScopeGuard

#define LIGHT_METAL_TRACE_FUNCTION_CALL(capture_func, ...)                                          \
    do {                                                                                            \
        log_trace(                                                                                  \
            tt::LogMetalTrace,                                                                      \
            "LIGHT_METAL_TRACE_FUNCTION_CALL: {} via {} istracing: {} depth: {}",                   \
            #capture_func,                                                                          \
            __FUNCTION__,                                                                           \
            LightMetalCaptureContext::get().is_tracing(),                                           \
            tt::tt_metal::TraceScope::depth);                                                       \
        if (LightMetalCaptureContext::get().is_tracing() && tt::tt_metal::TraceScope::depth == 1) { \
            capture_func(__VA_ARGS__);                                                              \
        }                                                                                           \
    } while (0)
#else

#define LIGHT_METAL_TRACE_FUNCTION_ENTRY()
#define LIGHT_METAL_TRACE_FUNCTION_CALL(capture_func, ...) \
    do {                                                   \
    } while (0)

#endif

namespace tt::tt_metal {

// Per Command type capture helper functions
void CaptureReplayTrace(IDevice* device, uint8_t cq_id, uint32_t tid, bool blocking);

void CaptureEnqueueTrace(CommandQueue& cq, uint32_t tid, bool blocking);

void CaptureLoadTrace(IDevice* device, const uint8_t cq_id, const uint32_t tid);

void CaptureReleaseTrace(IDevice* device, uint32_t tid);

void CaptureBufferCreate(
    const std::shared_ptr<Buffer>& buffer,
    IDevice* device,
    const std::optional<DeviceAddr> address,  // Made optional to share with 2 variants.
    DeviceAddr size,
    DeviceAddr page_size,
    const BufferType buffer_type,
    const TensorMemoryLayout buffer_layout,
    const std::optional<std::variant<ShardSpecBuffer, BufferDistributionSpec>>& shard_parameters,
    const std::optional<bool> bottom_up,
    const std::optional<SubDeviceId> sub_device_id);

void CaptureBufferDeallocate(const Buffer& buffer);
void CaptureBufferDelete(const Buffer& buffer);

void CaptureEnqueueWriteBuffer(
    CommandQueue& cq,
    std::variant<std::reference_wrapper<Buffer>, std::shared_ptr<Buffer>> buffer,
    HostDataType src,
    bool blocking);

void CaptureEnqueueReadBuffer(
    CommandQueue& cq,
    std::variant<std::reference_wrapper<Buffer>, std::shared_ptr<Buffer>> buffer,
    void* dst,
    bool blocking);

void CaptureFinish(CommandQueue& cq, tt::stl::Span<const SubDeviceId> sub_device_ids);
void CaptureProgramConstructor(Program& program);
void CaptureEnqueueProgram(CommandQueue& cq, Program& program, bool blocking);

void CaptureCreateKernel(
    KernelHandle kernel_id,
    Program& program,
    const std::string& file_name,
    const std::variant<CoreCoord, CoreRange, CoreRangeSet>& core_spec,
    const std::variant<DataMovementConfig, ComputeConfig, EthernetConfig>& config);

void CaptureSetRuntimeArgsUint32(
    const Program& program,
    KernelHandle kernel_id,
    const std::variant<CoreCoord, CoreRange, CoreRangeSet>& core_spec,
    tt::stl::Span<const uint32_t> runtime_args);

void CaptureSetRuntimeArgsUint32VecPerCore(
    const Program& program,
    KernelHandle kernel_id,
    const std::vector<CoreCoord>& core_spec,
    const std::vector<std::vector<uint32_t>>& runtime_args);

void CaptureSetRuntimeArgs(
    IDevice* device,
    const std::shared_ptr<Kernel>& kernel,
    const std::variant<CoreCoord, CoreRange, CoreRangeSet>& core_spec,
    const std::shared_ptr<RuntimeArgs>& runtime_args);

void CaptureCreateCircularBuffer(
    CBHandle& cb_handle,
    Program& program,
    const std::variant<CoreCoord, CoreRange, CoreRangeSet>& core_spec,
    const CircularBufferConfig& config);

void CaptureLightMetalCompare(
    CommandQueue& cq,
    std::variant<std::reference_wrapper<Buffer>, std::shared_ptr<Buffer>> buffer,
    void* golden_data,
    bool is_user_data);

}  // namespace tt::tt_metal
