/*
 * SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once
#include <thread>
#include <boost/functional/hash.hpp>
#include <string>

#include "common/tt_backend_api_types.hpp"
#include "common/utils.hpp"
#include "common/core_coord.h"
#include "build_kernels_for_riscv/data_format.hpp"
#include "build_kernels_for_riscv/build_kernel_options.hpp"
#include "hostdevcommon/common_values.hpp"
#include "tt_metal/third_party/tracy/public/tracy/Tracy.hpp"

namespace tt::tt_metal {
enum RISCID { NC = 0, TR0 = 1, TR1 = 2, TR2 = 3, BR = 4, ER = 5};
static_assert(RISCID::TR1 == RISCID::TR0+1 && RISCID::TR2 == RISCID::TR1+1);

void generate_data_format_descriptors(
    tt::build_kernel_for_riscv_options_t* build_kernel_for_riscv_options,
    std::string out_dir_path, const tt::ARCH arch);

void generate_binary_for_risc(
    RISCID id,
    tt::build_kernel_for_riscv_options_t* build_kernel_for_riscv_options,
    const std::string &out_dir_path,
    const std::string& arch_name,
    const std::uint8_t noc_index=0,
    const std::vector<std::uint32_t>& kernel_compile_time_args = {});

struct generate_binaries_params_t {
    bool            skip_hlkc = false;
    bool            compile_trisc = true;
    bool            compile_ncrisc = true;
    bool            compile_brisc = true;
    // for some kernels there's a header dependency on generated header from hlk
    // so we can't enable this for all kernels by default yet (TODO(AP))
    // when this flag is false, 3 HLKs are compiled in parallel, then (BR+NC) in parallel
    bool            parallel_trncbr = false;
    bool            parallel_hlk = true;
    std::uint8_t    br_noc_index = 0;
    std::uint8_t    nc_noc_index = 1;

    std::vector<std::uint32_t> br_kernel_compile_time_args = {};
    std::vector<std::uint32_t> nc_kernel_compile_time_args = {};
    std::vector<std::uint32_t> er_kernel_compile_time_args = {};
    std::vector<std::uint32_t> compute_kernel_compile_time_args = {};
};

void generate_binaries_all_riscs(
    tt::build_kernel_for_riscv_options_t* build_kernel_for_riscv_options, const std::string& out_dir_path, const tt::ARCH arch,
    generate_binaries_params_t params);

inline void generate_binary_for_brisc(
    tt::build_kernel_for_riscv_options_t* topts,
    const std::string &dir,
    const std::string& arch_name,
    const std::uint8_t noc_index=0,
    const std::vector<std::uint32_t>& kernel_compile_time_args = {})
{
    ZoneScoped;
    const std::string tracyPrefix = "generate_binary_for_brisc_";
    ZoneName( (tracyPrefix + dir).c_str(), dir.length() + tracyPrefix.length());
    generate_binary_for_risc(RISCID::BR, topts, dir, arch_name, noc_index, kernel_compile_time_args);
}

inline void generate_binary_for_ncrisc(
    tt::build_kernel_for_riscv_options_t* topts,
    const std::string &dir,
    const std::string& arch_name,
    const std::uint8_t noc_index=1,
    const std::vector<std::uint32_t>& kernel_compile_time_args = {})
{
    ZoneScoped;
    const std::string tracyPrefix = "generate_binary_for_ncrisc_";
    ZoneName( (tracyPrefix + dir).c_str(), dir.length() + tracyPrefix.length());
    generate_binary_for_risc(RISCID::NC, topts, dir, arch_name, noc_index, kernel_compile_time_args);
}

inline void generate_binary_for_erisc(
    tt::build_kernel_for_riscv_options_t* topts,
    const std::string &dir,
    const std::string& arch_name,
    const std::uint8_t noc_index=0,
    const std::vector<std::uint32_t>& kernel_compile_time_args = {})
{
    ZoneScoped;
    const std::string tracyPrefix = "generate_binary_for_erisc_";
    ZoneName( (tracyPrefix + dir).c_str(), dir.length() + tracyPrefix.length());
    generate_binary_for_risc(RISCID::ER, topts, dir, arch_name, noc_index, kernel_compile_time_args);
}

void generate_src_for_triscs(
    tt::build_kernel_for_riscv_options_t* topts,
    const string &out_dir_path,
    const string& arch_name,
    std::vector<uint32_t> kernel_compile_time_args);

void generate_binaries_for_triscs(
    tt::build_kernel_for_riscv_options_t* topts,
    const std::string &dir,
    const std::string& arch_name,
    const std::vector<std::uint32_t>& kernel_compile_time_args = {});

void generate_bank_to_noc_coord_descriptor(
    tt::build_kernel_for_riscv_options_t* build_kernel_for_riscv_options,
    string out_dir_path,
    tt_xy_pair grid_size,
    std::vector<CoreCoord>& dram_bank_map,
    std::vector<int32_t>& dram_bank_offset_map,
    std::vector<CoreCoord>& l1_bank_map,
    std::vector<int32_t>& l1_bank_offset_map
);

void generate_noc_addr_ranges_header(
    build_kernel_for_riscv_options_t* build_kernel_for_riscv_options,
    tt::ARCH arch,
    string out_dir_path,
    uint64_t pcie_addr_base,
    uint64_t pcie_addr_size,
    uint64_t dram_addr_base,
    uint64_t dram_addr_size,
    const std::vector<CoreCoord>& pcie_cores,
    const std::vector<CoreCoord>& dram_cores,
    const std::vector<CoreCoord>& ethernet_cores,
    CoreCoord grid_size,
    const std::vector<uint32_t>& harvested_rows,
    const std::vector<CoreCoord>& dispatch_cores);

void generate_descriptors(
    tt::build_kernel_for_riscv_options_t* opts, const std::string &op_dir, const tt::ARCH arch);

std::string get_string_aliased_arch_lowercase(tt::ARCH arch);

}
