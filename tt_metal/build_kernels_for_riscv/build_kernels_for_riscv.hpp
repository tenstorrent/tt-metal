#pragma once
#include <thread>
#include <boost/functional/hash.hpp>
#include <string>

#include "common/tt_backend_api_types.hpp"
#include "common/utils.hpp"
#include "build_kernels_for_riscv/data_format.hpp"
#include "build_kernels_for_riscv/build_kernel_options.hpp"

// external APIs to generate binaries

void generate_data_format_descriptors(tt::build_kernel_for_riscv_options_t* build_kernel_for_riscv_options, std::string out_dir_path);

void generate_binary_for_brisc(
        tt::build_kernel_for_riscv_options_t* build_kernel_for_riscv_options,
        const std::string &out_dir_path,
        const std::string& arch_name,
        const std::uint8_t noc_index=0,
        const std::vector<std::uint32_t>& kernel_compile_time_args = {},
        bool profile_kernel = false);

void generate_binary_for_ncrisc(
        tt::build_kernel_for_riscv_options_t* build_kernel_for_riscv_options,
        const std::string &out_dir_path,
        const std::string& arch_name,
        const std::uint8_t noc_index=1,
        const std::vector<std::uint32_t>& kernel_compile_time_args = {},
        bool profile_kernel = false);

void generate_binaries_for_triscs(
        tt::build_kernel_for_riscv_options_t* build_kernel_for_riscv_options,
        const std::string &out_dir_path,
        const std::string& arch_name,
        bool skip_hlkc=false,
        bool parallel = false,
        std::vector<uint32_t> kernel_compile_time_args = {});

void generate_binaries_for_triscs_new(
        tt::build_kernel_for_riscv_options_t* build_kernel_for_riscv_options,
        const std::string &out_dir_path,
        const std::string& arch_name,
        bool parallel = false,
        std::vector<uint32_t> kernel_compile_time_args = {});

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
    generate_binaries_params_t params);
