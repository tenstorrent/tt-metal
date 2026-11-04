// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

namespace tt::tt_metal {
struct KernelSource;
class JitBuildEnv;
class JitBuildOptions;
class JitBuildSettings;

void jit_build_genfiles_kernel_include(
    const JitBuildEnv& env, const JitBuildSettings& settings, const KernelSource& kernel_src);
void jit_build_genfiles_triscs_src(
    const JitBuildEnv& env, const JitBuildSettings& settings, const KernelSource& kernel_src);

void jit_build_genfiles_descriptors(const JitBuildEnv& env, const JitBuildOptions& options);

}  // namespace tt::tt_metal
