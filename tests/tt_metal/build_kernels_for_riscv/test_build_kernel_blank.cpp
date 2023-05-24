#include <iostream>

#include "build_kernels_for_riscv/build_kernels_for_riscv.hpp"
#include "tests/tt_metal/test_utils/env_vars.hpp"


namespace blank {
//#include "hlks/eltwise_add"
// FIXME:copy pasted the args here from the kernel file,  we could refactor the HLK file


struct hlk_args_t {
    std::int32_t dummy;
};
}



int main(int argc, char* argv[]) {

    std::string arch_name = tt::test_utils::get_env_arch_name();

    // Create and config an OP
    tt::build_kernel_for_riscv_options_t build_kernel_for_riscv_options("dummy_type","blank_op");

    log_info(tt::LogBuildKernels, "Compiling OP: {}", build_kernel_for_riscv_options.name);

    void *hlk_args = new blank::hlk_args_t{
        .dummy = 0,
    };
    build_kernel_for_riscv_options.set_hlk_args_all_cores(hlk_args, sizeof(blank::hlk_args_t));
    build_kernel_for_riscv_options.set_hlk_file_name_all_cores("tt_metal/kernels/compute/blank.cpp");
    build_kernel_for_riscv_options.ncrisc_kernel_file_name = "tt_metal/kernels/dataflow/blank.cpp";
    build_kernel_for_riscv_options.brisc_kernel_file_name = "tt_metal/kernels/dataflow/blank.cpp";


    generate_binaries_params_t params;

    __internal::generate_default_bank_to_noc_coord_descriptor (
        &build_kernel_for_riscv_options,
        build_kernel_for_riscv_options.name,
        tt::get_arch_from_string(arch_name)
    );
    generate_binaries_all_riscs(&build_kernel_for_riscv_options, build_kernel_for_riscv_options.name, arch_name, params);

    return 0;
}
