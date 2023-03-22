#include <iostream>

#include "build_kernels_for_riscv/build_kernels_for_riscv.hpp"




int main(int argc, char* argv[]) {

    std::string root_dir = tt::utils::get_root_dir();

    // Create and config an OP
    tt::build_kernel_for_riscv_options_t build_kernel_for_riscv_options("dummy_type","l1_to_dram_copy");
    std::string out_dir_path = root_dir + "/built_kernels/" + build_kernel_for_riscv_options.name;

    log_info(tt::LogBuildKernels, "Compiling OP: {} to {}", build_kernel_for_riscv_options.name, out_dir_path);

    build_kernel_for_riscv_options.ncrisc_kernel_file_name = "tt_metal/kernels/dataflow/l1_to_dram_copy.cpp";

    generate_binary_for_ncrisc(&build_kernel_for_riscv_options, out_dir_path, "grayskull");

    return 0;
}
