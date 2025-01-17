// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>
#include <vector>

#include <core_coord.hpp>
#include <kernel.hpp>

namespace tt::tt_metal {

class JitBuildEnv;
class JitBuildSettings;
class JitBuildOptions;

void jit_build_genfiles_kernel_include(
    const JitBuildEnv& env, const JitBuildSettings& settings, const KernelSource& kernel_src);
void jit_build_genfiles_triscs_src(
    const JitBuildEnv& env, const JitBuildSettings& settings, const KernelSource& kernel_src);

void jit_build_genfiles_descriptors(const JitBuildEnv& env, JitBuildOptions& options);

}  // namespace tt::tt_metal
