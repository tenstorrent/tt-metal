// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Shared builders for the Metal 2.0 GlobalSemaphoreParameter unit tests (mock device).

#include <string>
#include <vector>

#include <tt-metalium/experimental/metal2_host_api/global_semaphore_parameter.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/global_semaphore.hpp>

#include "metal2_host_api/test_helpers/test_helpers.hpp"

namespace tt::tt_metal::experimental::test_helpers {

inline void BindGlobalSemaphoreToKernel(KernelSpec& kernel, const std::string& param_name, std::string accessor_name) {
    kernel.semaphore_bindings.push_back(SemaphoreBinding{
        .semaphore_spec_name = GlobalSemaphoreParamName{param_name}, .accessor_name = std::move(accessor_name)});
}

// MakeMinimalGen1ValidProgramSpec (dm_kernel + compute_kernel) placed on `nodes`, with each name in
// `param_names` declared as a GlobalSemaphoreParameter and bound by dm_kernel under the same accessor name.
inline ProgramSpec MakeGlobalSemaphoreSpec(
    const std::vector<std::string>& param_names = {"gsem"}, const Nodes& nodes = NodeCoord{0, 0}) {
    ProgramSpec spec = MakeMinimalGen1ValidProgramSpec();
    spec.work_units = {MakeMinimalWorkUnit("work_unit_0", nodes, {"dm_kernel", "compute_kernel"})};
    for (const auto& name : param_names) {
        spec.global_semaphore_parameters.push_back(
            GlobalSemaphoreParameter{.unique_id = GlobalSemaphoreParamName{name}});
        BindGlobalSemaphoreToKernel(spec.kernels[0], name, name);
    }
    return spec;
}

inline ProgramRunArgs GlobalSemaphoreArgs(const std::vector<std::pair<std::string, const GlobalSemaphore*>>& sems) {
    ProgramRunArgs params;
    for (const auto& [name, sem] : sems) {
        params.global_semaphore_args.insert({GlobalSemaphoreParamName{name}, GlobalSemaphoreArgument{*sem}});
    }
    return params;
}

}  // namespace tt::tt_metal::experimental::test_helpers
