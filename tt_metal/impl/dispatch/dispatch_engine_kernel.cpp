// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "impl/dispatch/dispatch_engine_cores.hpp"

#include <memory>
#include <string>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/hal_types.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt_stl/assert.hpp>

#include "host_api/temp_quasar_api.hpp"
#include "impl/kernels/kernel.hpp"
#include "impl/kernels/kernel_source.hpp"
#include "impl/program/program_impl.hpp"

namespace tt::tt_metal::detail {

namespace {

KernelHandle create_dispatch_engine_kernel(
    Program& program,
    const std::string& file_name,
    const CoreCoord& core,
    DataMovementProcessor dm_processor,
    const experimental::quasar::QuasarDataMovementConfig& config) {
    TT_FATAL(
        config.num_threads_per_cluster == 1,
        "CreateDispatchEngineKernel requires num_threads_per_cluster=1");
    const CoreRangeSet core_ranges = CoreRangeSet(core);
    const KernelSource kernel_src(file_name, KernelSource::FILE_PATH);
    std::shared_ptr<Kernel> kernel = std::make_shared<experimental::quasar::DispatchEngineKernel>(
        kernel_src, core_ranges, config, dm_processor);
    return program.impl().add_kernel(kernel, HalProgrammableCoreType::DISPATCH);
}

}  // namespace

KernelHandle CreateDispatchEngineKernel(
    Program& program,
    const std::string& file_name,
    const CoreCoord& core,
    const experimental::quasar::QuasarDataMovementConfig& config) {
    // Same policy as Quasar Tensix CreateKernel: reserve the lowest-numbered free DM(s).
    const CoreRangeSet core_ranges = CoreRangeSet(core);
    const auto free_dms = experimental::quasar::GetAvailableDataMovementProcessors(
        program, core_ranges, config.num_threads_per_cluster, HalProgrammableCoreType::DISPATCH);
    TT_FATAL(!free_dms.empty(), "No free data-movement processors on dispatch-engine core {}", core.str());
    return create_dispatch_engine_kernel(program, file_name, core, *free_dms.begin(), config);
}

KernelHandle CreateDispatchEngineKernel(
    Program& program,
    const std::string& file_name,
    const CoreCoord& core,
    DataMovementProcessor dm_processor,
    const experimental::quasar::QuasarDataMovementConfig& config) {
    return create_dispatch_engine_kernel(program, file_name, core, dm_processor, config);
}

}  // namespace tt::tt_metal::detail
