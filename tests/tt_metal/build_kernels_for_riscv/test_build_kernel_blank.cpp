#include <iostream>

#include "build_kernels_for_riscv/build_kernels_for_riscv.hpp"


namespace blank {
//#include "hlks/eltwise_add"
// FIXME:copy pasted the args here from the kernel file,  we could refactor the HLK file


struct hlk_args_t {
    std::int32_t dummy;
};
}



int main(int argc, char* argv[]) {

    std::string root_dir = tt::utils::get_root_dir();

    // Create and config an OP
    tt::build_kernel_for_riscv_options_t build_kernel_for_riscv_options("dummy_type","blank_op");
    std::string out_dir_path = root_dir + "/built_kernels/" + build_kernel_for_riscv_options.name;

    log_info(tt::LogBuildKernels, "Compiling OP: {} to {}", build_kernel_for_riscv_options.name, out_dir_path);

    void *hlk_args = new blank::hlk_args_t{
        .dummy = 0,
    };
    build_kernel_for_riscv_options.set_hlk_args_all_cores(hlk_args, sizeof(blank::hlk_args_t));
    build_kernel_for_riscv_options.set_hlk_file_name_all_cores("tt_metal/kernels/compute/blank.cpp");
    build_kernel_for_riscv_options.ncrisc_kernel_file_name = "tt_metal/kernels/dataflow/blank.cpp";
    build_kernel_for_riscv_options.brisc_kernel_file_name = "tt_metal/kernels/dataflow/blank.cpp";

    bool skip_hlkc = false;
    if (argc > 1) {
        skip_hlkc = true;
    }
    generate_binaries_params_t params = {.skip_hlkc = skip_hlkc};
    generate_binaries_all_riscs(&build_kernel_for_riscv_options, out_dir_path, "grayskull", params);

    return 0;
}
