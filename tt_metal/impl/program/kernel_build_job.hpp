// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <impl/context/context_types.hpp>
#include <tt-umd/types/cluster_descriptor_types.hpp>  // ChipId

#include "jit_build/jit_build_options.hpp"

namespace tt::tt_metal {

class Kernel;
class BuildEnvManager;
struct DeviceBuildEnv;

namespace detail {

// One kernel's build, resolved from its program while the device was open: everything a worker needs to
// build it later without the device. The device's MetalContext must outlive the job.
struct KernelBuildJob {
    std::shared_ptr<Kernel> kernel;
    // The build env the job was prepared for, kept alive and unchanged; build_options refers into it.
    std::shared_ptr<const DeviceBuildEnv> build_env;
    BuildEnvManager* build_env_manager = nullptr;
    JitBuildOptions build_options;
    std::size_t kernel_hash = 0;
    ContextId context_id = DEFAULT_CONTEXT_ID;
    ChipId build_id = 0;
};

enum class KernelBuildResult {
    Built,              // built here (on-disk cache hits included)
    BuiltRemotely,      // built by the compile server
    AlreadyBuilt,       // this process had already built it
    BuildingElsewhere,  // another thread is building it right now
};

// Builds the job's kernel into the on-disk kernel cache, through the compile server when remote_endpoints is
// non-empty. Shares the in-process build-once table with ProgramImpl::compile, so the same kernel is never
// built twice at once; it never waits for another thread's build. Throws on failure.
KernelBuildResult run_kernel_build_job(KernelBuildJob& job, const std::vector<std::string>& remote_endpoints);

}  // namespace detail
}  // namespace tt::tt_metal
