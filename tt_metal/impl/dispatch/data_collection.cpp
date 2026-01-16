// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

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
    uint64_t program_id,
    data_collector_t type,
    uint32_t transaction_size,
    std::optional<HalProcessorIdentifier> processor) {
    // Do nothing if we're not enabling data collection.
    if (!tt::tt_metal::MetalContext::instance().rtoptions().get_dispatch_data_collection_enabled()) {
        return;
    }
    tt::tt_metal::MetalContext::instance().data_collector()->RecordData(program_id, type, transaction_size, processor);
}

void RecordKernelGroup(ProgramImpl& program, HalProgrammableCoreType core_type, const KernelGroup& kernel_group) {
    // Do nothing if we're not enabling data collection.
    if (!tt::tt_metal::MetalContext::instance().rtoptions().get_dispatch_data_collection_enabled()) {
        return;
    }

    tt::tt_metal::MetalContext::instance().data_collector()->RecordKernelGroup(program, core_type, kernel_group);
}

void RecordProgramRun(uint64_t program_id) {
    // Do nothing if we're not enabling data collection.
    if (!tt::tt_metal::MetalContext::instance().rtoptions().get_dispatch_data_collection_enabled()) {
        return;
    }

    tt::tt_metal::MetalContext::instance().data_collector()->RecordProgramRun(program_id);
}

void RecordKernelSourceMap(ProgramImpl& program) {
    tt::tt_metal::MetalContext::instance().data_collector()->RecordKernelSourceMap(program);
}

std::string GetKernelSourcesForRuntimeId(uint64_t runtime_id) {
    return tt::tt_metal::MetalContext::instance().data_collector()->GetKernelSourcesForRuntimeId(runtime_id);
}

std::vector<std::string> GetKernelSourcesVecForRuntimeId(uint64_t runtime_id) {
    return tt::tt_metal::MetalContext::instance().data_collector()->GetKernelSourcesVecForRuntimeId(runtime_id);
}

ProgramPerfCallbackHandle RegisterProgramPerfCallback(ProgramPerfCallback callback) {
    return tt::tt_metal::MetalContext::instance().data_collector()->RegisterProgramPerfCallback(std::move(callback));
}

void UnregisterProgramPerfCallback(ProgramPerfCallbackHandle handle) {
    tt::tt_metal::MetalContext::instance().data_collector()->UnregisterProgramPerfCallback(handle);
}

void InvokeProgramPerfCallbacks(const ProgramPerfRecord& record) {
    tt::tt_metal::MetalContext::instance().data_collector()->InvokeProgramPerfCallbacks(record);
}

}  // namespace tt

// Public experimental API — delegates to the internal tt:: functions.
namespace tt::tt_metal::experimental {

ProgramPerfCallbackHandle RegisterProgramPerfCallback(ProgramPerfCallback callback) {
    return tt::RegisterProgramPerfCallback(std::move(callback));
}

void UnregisterProgramPerfCallback(ProgramPerfCallbackHandle handle) { tt::UnregisterProgramPerfCallback(handle); }

}  // namespace tt::tt_metal::experimental
