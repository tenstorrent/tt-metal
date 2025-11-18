// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "data_collection.hpp"

#include <cstdint>
#include "impl/context/metal_context.hpp"
#include "impl/kernels/kernel_impl.hpp"
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

}  // namespace tt
