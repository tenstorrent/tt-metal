// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>
#include <vector>

#include "common/core_coord.h"

namespace tt::tt_metal {

class JitBuildEnv;
class JitBuildSettings;
class JitBuildOptions;

void jit_build_genfiles_kernel_include(
    const JitBuildEnv& env, const JitBuildSettings& settings, const string& input_hlk_file_path);
void jit_build_genfiles_triscs_src(const JitBuildEnv& env,
                                   const JitBuildSettings& settings,
                                   const std::string& kernel_in_path);

void jit_build_genfiles_bank_to_noc_coord_descriptor(
    const std::string& path,
    tt_xy_pair grid_size,
    std::vector<CoreCoord>& dram_bank_map,
    std::vector<int32_t>& dram_bank_offset_map,
    std::vector<CoreCoord>& l1_bank_map,
    std::vector<int32_t>& l1_bank_offset_map,
    int core_count_per_dram,
    const std::map<CoreCoord, int32_t>& profiler_flat_id_map
);

void jit_build_genfiles_noc_addr_ranges_header(
    const std::string& path,
    uint64_t pcie_addr_base,
    uint64_t pcie_addr_size,
    uint64_t dram_addr_base,
    uint64_t dram_addr_size,
    const std::vector<CoreCoord>& pcie_cores,
    const std::vector<CoreCoord>& dram_cores,
    const std::vector<CoreCoord>& ethernet_cores,
    CoreCoord grid_size,
    const std::vector<uint32_t>& harvested_rows,
    bool has_pcie_cores);

void jit_build_genfiles_descriptors(const JitBuildEnv& env,
                                    JitBuildOptions& options);

} // namespace tt::tt_metal
