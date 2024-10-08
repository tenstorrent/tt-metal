// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>
#include <vector>

#include "common/core_coord.h"
#include "impl/kernels/kernel.hpp"

namespace tt::tt_metal {

class JitBuildEnv;
class JitBuildSettings;
class JitBuildOptions;

void jit_build_genfiles_kernel_include(
    const JitBuildEnv& env, const JitBuildSettings& settings, const KernelSource& kernel_src);
void jit_build_genfiles_triscs_src(
    const JitBuildEnv& env, const JitBuildSettings& settings, const KernelSource& kernel_src);

void jit_build_genfiles_bank_to_noc_coord_descriptor(
    const std::string& path,
    tt_xy_pair grid_size,
    std::vector<CoreCoord>& dram_bank_map,
    std::vector<int32_t>& dram_bank_offset_map,
    std::vector<CoreCoord>& l1_bank_map,
    std::vector<int32_t>& l1_bank_offset_map,
    uint32_t allocator_alignment);

void jit_build_genfiles_descriptors(const JitBuildEnv& env, JitBuildOptions& options);

}  // namespace tt::tt_metal
