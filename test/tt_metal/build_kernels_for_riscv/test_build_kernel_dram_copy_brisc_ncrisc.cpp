#include <iostream>

#include "build_kernels_for_riscv/build_kernels_for_riscv.hpp"




int main() {

    std::string root_dir = tt::utils::get_root_dir();

    // Create and config an OP
    tt::build_kernel_for_riscv_options_t build_kernel_for_riscv_options("dummy_type","dram_copy_brisc_ncrisc");
    std::string out_dir_path = root_dir + "/built_kernels/" + build_kernel_for_riscv_options.name;

    log_info(tt::LogBuildKernels, "Compiling OP: {} to {}", build_kernel_for_riscv_options.name, out_dir_path);

    // dram copy kernel
    build_kernel_for_riscv_options.brisc_kernel_file_name = "tt_metal/kernels/dataflow/dram_copy.cpp";

    // same as the brisc version but with different arg addresses
    build_kernel_for_riscv_options.ncrisc_kernel_file_name = "tt_metal/kernels/dataflow/dram_copy.cpp";

    generate_binary_for_brisc(&build_kernel_for_riscv_options, out_dir_path, "grayskull");
    generate_binary_for_ncrisc(&build_kernel_for_riscv_options, out_dir_path, "grayskull");

    // WH doesn't work? FIXME: SFPU issues?
    //generate_all_fw(&build_kernel_for_riscv_options, "wormhole");

    return 0;
}
