// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt_stl/fmt.hpp>
#include "data_collection.hpp"

#include <cstdint>
#include "impl/context/metal_context.hpp"
#include "impl/kernels/kernel.hpp"
#include "tt-metalium/program.hpp"
#include "data_collector.hpp"

using namespace tt;
using namespace tt::tt_metal;

using tt::tt_metal::detail::ProgramImpl;

namespace tt {

void RecordDispatchData(
    ContextId context_id,
    uint64_t program_id,
    data_collector_t type,
    uint32_t transaction_size,
    std::optional<HalProcessorIdentifier> processor) {
    MetalContext& metal_ctx = MetalContext::instance(context_id);
    // Do nothing if we're not enabling data collection.
    if (!metal_ctx.rtoptions().get_dispatch_data_collection_enabled()) {
        return;
    }
    metal_ctx.data_collector()->RecordData(program_id, type, transaction_size, processor);
}

void RecordKernelGroup(
    ContextId context_id, ProgramImpl& program, HalProgrammableCoreType core_type, const KernelGroup& kernel_group) {
    MetalContext& metal_ctx = MetalContext::instance(context_id);
    // Do nothing if we're not enabling data collection.
    if (!metal_ctx.rtoptions().get_dispatch_data_collection_enabled()) {
        return;
    }

    metal_ctx.data_collector()->RecordKernelGroup(program, core_type, kernel_group);
}

void RecordProgramRun(ContextId context_id, uint64_t program_id) {
    MetalContext& metal_ctx = MetalContext::instance(context_id);
    // Do nothing if we're not enabling data collection.
    if (!metal_ctx.rtoptions().get_dispatch_data_collection_enabled()) {
        return;
    }

    metal_ctx.data_collector()->RecordProgramRun(program_id);
}

void RecordProgramSubDevice(
    ContextId context_id,
    tt::ChipId device_id,
    uint64_t sub_device_manager_id,
    uint64_t runtime_id,
    SubDeviceId sub_device_id,
    uint32_t num_available_worker_cores) {
    MetalContext::instance(context_id)
        .data_collector()
        ->RecordProgramSubDevice(
            device_id, sub_device_manager_id, runtime_id, sub_device_id, num_available_worker_cores);
}

std::optional<ProgramSubDeviceInfo> GetProgramSubDevice(
    ContextId context_id, tt::ChipId device_id, uint64_t runtime_id) {
    return MetalContext::instance(context_id).data_collector()->GetProgramSubDevice(device_id, runtime_id);
}

void RecordProgramMetadata(ContextId context_id, ProgramImpl& program) {
    MetalContext::instance(context_id).data_collector()->RecordProgramMetadata(program);
}

std::span<const std::string_view> GetKernelSourcesForRuntimeId(ContextId context_id, uint16_t runtime_id) {
    return MetalContext::instance(context_id).data_collector()->GetKernelSourcesForRuntimeId(runtime_id);
}

ProgramRealtimeProfilerCallbackHandle RegisterProgramRealtimeProfilerCallback(
    ProgramRealtimeProfilerCallback callback) {
    return tt::tt_metal::MetalContext::instance().data_collector()->RegisterProgramRealtimeProfilerCallback(
        std::move(callback));
}

void UnregisterProgramRealtimeProfilerCallback(ProgramRealtimeProfilerCallbackHandle handle) {
    tt::tt_metal::MetalContext::instance().data_collector()->UnregisterProgramRealtimeProfilerCallback(handle);
}

bool IsProgramRealtimeProfilerActive() {
    return tt::tt_metal::MetalContext::instance().data_collector()->IsRealtimeProfilerActive();
}

void NotifyProgramRealtimeProfilerActivated(uint32_t chip_id) {
    tt::tt_metal::MetalContext::instance().data_collector()->NotifyRealtimeProfilerActivated(chip_id);
}

void NotifyProgramRealtimeProfilerDeactivated(uint32_t chip_id) {
    tt::tt_metal::MetalContext::instance().data_collector()->NotifyRealtimeProfilerDeactivated(chip_id);
}

}  // namespace tt

// Public experimental API — delegates to the internal tt:: functions.
namespace tt::tt_metal::experimental {

ProgramRealtimeProfilerCallbackHandle RegisterProgramRealtimeProfilerCallback(
    ProgramRealtimeProfilerCallback callback) {
    return tt::RegisterProgramRealtimeProfilerCallback(std::move(callback));
}

void UnregisterProgramRealtimeProfilerCallback(ProgramRealtimeProfilerCallbackHandle handle) {
    tt::UnregisterProgramRealtimeProfilerCallback(handle);
}

bool IsProgramRealtimeProfilerActive() { return tt::IsProgramRealtimeProfilerActive(); }

}  // namespace tt::tt_metal::experimental
