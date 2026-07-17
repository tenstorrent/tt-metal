// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>

#include <tt-metalium/program.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/experimental/offline_kernel_compile.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>

namespace tt::tt_metal::experimental {

//------------------------------------------------
// Metal 2.0 offline (AOT) compile + load APIs
// (experimental namespace free functions)
//
// Requires an initialized MeshDevice: both APIs go through the real on-device CompileProgram
// path, so the producer and loader compute identical kernel hashes by construction
// (the loader recomputes the same hash at dispatch time). The device's build env (arch +
// build flags, folded into build_key) must match the device the AOT binaries will run on;
// a mock MeshDevice configured for the target arch is sufficient to produce them.
//------------------------------------------------

// Produce: compile every kernel in `spec` against `mesh_device` and emit AOT artifacts under
// `output_dir`. The on-disk layout matches the precompiled loader's search path, so a later
// SetProgramPrecompiledConfig(program, {output_dir, ...}) call will find these binaries.
//
// Existing files under `output_dir` are overwritten (copy_file with overwrite_existing); the
// directory is otherwise left intact. `output_dir` must be non-empty.
//
// `mesh_device` is taken by const ref (only get_devices()/build env are read). Only the first
// device's build env is used; a warning is logged for larger meshes.
//
// PRECONDITION: unsupported when the remote JIT compile server is enabled
// (jit_server::JitCompileRpcClient::enabled()); the call TT_FATALs in that case.
void CompileProgramSpecOffline(
    const distributed::MeshDevice& mesh_device, const ProgramSpec& spec, const std::string& output_dir);

// Load: mark every kernel in `program` to load precompiled binaries from config.precompiled_dir
// (with the given FallbackPolicy). Mirrors CreateKernelFromPrecompiled for Metal 2.0.
//
// CONTRACT: Must be called before the program is first compiled. ProgramImpl::compile
// short-circuits once a build key has been compiled, so a late call would silently be a no-op;
// this function TT_FATALs if the program is already compiled to make that loud instead of silent.
//
// PRECONDITION: unsupported when the remote JIT compile server is enabled
// (the remote compile path does not honor precompiled config); the call TT_FATALs in that case.
void SetProgramPrecompiledConfig(Program& program, const PrecompiledKernelConfig& config);

}  // namespace tt::tt_metal::experimental
