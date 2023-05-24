#include <iostream>

#include "build_kernels_for_riscv/build_kernels_for_riscv.hpp"
#include "tests/tt_metal/test_utils/env_vars.hpp"




int main() {

    std::string arch_name = tt::test_utils::get_env_arch_name();

    // Create and config an OP
    tt::build_kernel_for_riscv_options_t build_kernel_for_riscv_options("dummy_type","risc_rw_speed_banked_dram");

    log_info(tt::LogBuildKernels, "Compiling OP: {}", build_kernel_for_riscv_options.name);

    build_kernel_for_riscv_options.brisc_kernel_file_name = "tt_metal/kernels/dataflow/risc_read_speed_banked_dram.cpp";
    build_kernel_for_riscv_options.ncrisc_kernel_file_name = "tt_metal/kernels/dataflow/risc_write_speed_banked_dram.cpp";

    __internal::generate_default_bank_to_noc_coord_descriptor (
        &build_kernel_for_riscv_options,
        build_kernel_for_riscv_options.name,
        tt::get_arch_from_string(arch_name)
    );
    generate_binary_for_brisc(&build_kernel_for_riscv_options, build_kernel_for_riscv_options.name, arch_name);
    generate_binary_for_ncrisc(&build_kernel_for_riscv_options, build_kernel_for_riscv_options.name, arch_name);

    return 0;
}
