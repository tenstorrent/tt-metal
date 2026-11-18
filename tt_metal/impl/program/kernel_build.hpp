// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <exception>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include <impl/context/context_types.hpp>
#include <tt_stl/span.hpp>
#include <tt-umd/types/cluster_descriptor_types.hpp>  // ChipId

#include "jit_build/jit_build_options.hpp"

namespace tt::tt_metal {

class Kernel;
class BuildEnvManager;
struct DeviceBuildEnv;

namespace detail {

// One kernel's build, resolved from its program on the compiling thread: everything building it needs, and nothing
// the program or the device owns, so the build can also finish after the program has changed or the device has closed.
// The device's MetalContext must outlive it.
struct KernelBuild {
    std::shared_ptr<Kernel> kernel;
    // The build env the kernel was prepared for, kept alive and unchanged; build_options refers into it.
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
    BuildingElsewhere,  // another thread is building it right now; its binaries may not be on disk yet
    Failed,             // see KernelBuildStatus::error
};

struct KernelBuildStatus {
    KernelBuildResult result = KernelBuildResult::Failed;
    std::exception_ptr error;
};

// Builds kernels into the on-disk kernel cache: the one build path of ProgramImpl::compile and
// experimental::CompileSession. All builds share one build env (one program compiled for one device).
//
// With remote_endpoints empty, each kernel builds as its own task on the JIT build threads. Otherwise the kernels go to
// the compile server as one batch, submitted together so that their requests overlap, then collected; a batch that
// fails fails every kernel it submitted. If the device was reopened with another build env since the kernels were
// prepared, they build locally, since the server is handed the device's current firmware.
//
// Never waits for a kernel another thread is building (BuildingElsewhere). Calls on_built for each kernel whose
// binaries this call put on disk or found there: on its build thread for a local build, on the calling thread after the
// batch for a remote one. Returns one status per build, in order, once nothing it started is still running; a failed
// build is reported in its status, not thrown.
std::vector<KernelBuildStatus> build_kernels(
    ttsl::Span<KernelBuild> builds,
    const std::vector<std::string>& remote_endpoints,
    const std::function<void(KernelBuild&)>& on_built = nullptr);

// Builds one kernel locally, first waiting out another thread's build of it; for a kernel build_kernels reported as
// BuildingElsewhere. Throws if the build fails.
void join_kernel_build(KernelBuild& build);

// Marks the calling thread as compiling a workload whose kernel binaries nobody will read: MeshWorkloadImpl::compile
// for a mock device with a build sink on, which stops before finalizing offsets, and whose EnqueueMeshWorkload loads no
// binaries. ProgramImpl::compile hands kernels to the build sink only inside this scope; every other compile on a mock
// device (a direct CompileProgram) builds normally, since its caller goes on to read the binaries.
class BuildSinkScope {
public:
    explicit BuildSinkScope(bool active);
    ~BuildSinkScope();
    BuildSinkScope(const BuildSinkScope&) = delete;
    BuildSinkScope& operator=(const BuildSinkScope&) = delete;
    static bool active();

private:
    bool previous_;
};

}  // namespace detail
}  // namespace tt::tt_metal
