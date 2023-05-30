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

enum RISCID { NC = 0, TR0 = 1, TR1 = 2, TR2 = 3, BR = 4 };
static_assert(RISCID::TR1 == RISCID::TR0+1 && RISCID::TR2 == RISCID::TR1+1);

void generate_data_format_descriptors(
    tt::build_kernel_for_riscv_options_t* build_kernel_for_riscv_options,
    std::string out_dir_path);

void generate_binary_for_risc(
    RISCID id,
    tt::build_kernel_for_riscv_options_t* build_kernel_for_riscv_options,
    const std::string &out_dir_path,
    const std::string& arch_name,
    const std::uint8_t noc_index=0,
    const std::vector<std::uint32_t>& kernel_compile_time_args = {},
    bool profile_kernel = false);

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
    std::vector<std::uint32_t> compute_kernel_compile_time_args = {};
};

void generate_binaries_all_riscs(
    tt::build_kernel_for_riscv_options_t* build_kernel_for_riscv_options, const std::string& out_dir_path, const std::string& arch_name,
    generate_binaries_params_t params, bool profile_kernel = false);

inline void generate_binary_for_brisc(
    tt::build_kernel_for_riscv_options_t* topts,
    const std::string &dir,
    const std::string& arch_name,
    const std::uint8_t noc_index=0,
    const std::vector<std::uint32_t>& kernel_compile_time_args = {},
    bool profile_kernel = false)
{
    // PROF_BEGIN("CCGEN_BR")
    generate_binary_for_risc(RISCID::BR, topts, dir, arch_name, noc_index, kernel_compile_time_args, profile_kernel);
    // PROF_END("CCGEN_BR")
}

inline void generate_binary_for_ncrisc(
    tt::build_kernel_for_riscv_options_t* topts,
    const std::string &dir,
    const std::string& arch_name,
    const std::uint8_t noc_index=1,
    const std::vector<std::uint32_t>& kernel_compile_time_args = {},
    bool profile_kernel = false)
{
    // PROF_BEGIN("CCGEN_NC")
    generate_binary_for_risc(RISCID::NC, topts, dir, arch_name, noc_index, kernel_compile_time_args, profile_kernel);
    // PROF_END("CCGEN_NC")
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
    const std::vector<std::uint32_t>& kernel_compile_time_args = {},
    bool profile_kernel = false);

void generate_bank_to_noc_coord_descriptor(
    tt::build_kernel_for_riscv_options_t* build_kernel_for_riscv_options,
    string out_dir_path,
    std::vector<CoreCoord>& dram_bank_map,
    std::vector<CoreCoord>& l1_bank_map,
    std::vector<i32>& l1_bank_offset_map
);

void generate_descriptors(
    tt::build_kernel_for_riscv_options_t* opts, const std::string &op_dir);

std::string get_string_aliased_arch_lowercase(tt::ARCH arch);

namespace __internal {
void generate_default_bank_to_noc_coord_descriptor(
    tt::build_kernel_for_riscv_options_t* build_kernel_for_riscv_options,
    string out_dir_path,
    tt::ARCH arch
);
}
