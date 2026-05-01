// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <core_coord.hpp>
#include <filesystem>
#include <string>
#include <vector>

namespace tt::tt_metal {
struct KernelSource;
}  // namespace tt::tt_metal

namespace tt::tt_metal {

class JitBuildEnv;
class JitBuildOptions;
class JitBuildSettings;

void jit_build_genfiles_kernel_include(
    const JitBuildEnv& env, const JitBuildSettings& settings, const KernelSource& kernel_src);
void jit_build_genfiles_triscs_src(
    const JitBuildEnv& env, const JitBuildSettings& settings, const KernelSource& kernel_src);

void jit_build_genfiles_descriptors(const JitBuildEnv& env, const JitBuildOptions& options);

// Metal 2.0 per-kernel auto-generated headers.  Exposed so the tt-emule JIT
// pipeline can emit them into its temp dir and add `-I<dir>` to the compile
// command.  Only meaningful when JitBuildSettings::is_metal2_kernel().
void write_kernel_bindings_generated_header(const std::filesystem::path& out_dir, const JitBuildSettings& settings);
void write_kernel_args_generated_header(const std::filesystem::path& out_dir, const JitBuildSettings& settings);

}  // namespace tt::tt_metal
