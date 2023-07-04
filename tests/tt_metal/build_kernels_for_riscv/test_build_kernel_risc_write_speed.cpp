#include <iostream>

#include "tt_cluster.hpp"
#include "build_kernels_for_riscv/build_kernels_for_riscv.hpp"
#include "tests/tt_metal/test_utils/env_vars.hpp"




int main(int argc, char** argv) {

    std::vector<std::string> input_args(argv, argv + argc);
    std::uint32_t profile;
    try {
        std::tie(profile, input_args) =
            test_args::get_command_option_uint32_and_remaining_args(input_args, "--profile", 0);
    } catch (const std::exception& e) {
        log_fatal(tt::LogTest, "Command line arguments found exception", e.what());
    }

    std::string root_dir = tt::utils::get_root_dir();
    std::string arch_name = tt::test_utils::get_env_arch_name();

    // Create and config an OP
    tt::build_kernel_for_riscv_options_t build_kernel_for_riscv_options("dummy_type","risc_write_speed");
    std::string out_dir_path = root_dir + "/built_kernels/" + build_kernel_for_riscv_options.name;

    log_info(tt::LogBuildKernels, "Compiling OP: {} to {}", build_kernel_for_riscv_options.name, out_dir_path);

    build_kernel_for_riscv_options.brisc_kernel_file_name = "tt_metal/kernels/dataflow/blank.cpp";
    build_kernel_for_riscv_options.ncrisc_kernel_file_name = "tt_metal/kernels/dataflow/risc_write_speed.cpp";
    bool profile_marker = false;
    if (profile) {
        log_info(tt::LogBuildKernels, "Profile on");
        profile_marker = true;
    }

    generate_binary_for_brisc(&build_kernel_for_riscv_options, out_dir_path, arch_name, 0, {}, profile_marker);
    generate_binary_for_ncrisc(&build_kernel_for_riscv_options, out_dir_path, arch_name, 1, {}, profile_marker);

    return 0;
}
