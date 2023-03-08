#include <iostream>

#include "build_kernels_for_riscv/build_kernels_for_riscv.hpp"


namespace unary_datacopy {
//#include "hlks/eltwise_copy.cpp"
// FIXME:copy pasted the args here from the kernel file,  we could refactor the HLK file
struct hlk_args_t {
    std::int32_t per_core_tile_cnt;
};
}




int main(int argc, char* argv[]) {

    std::string root_dir = tt::utils::get_root_dir();

    // Create and config an OP
    tt::build_kernel_for_riscv_options_t build_kernel_for_riscv_options("unary", "transpose_hc_op");
    std::string out_dir_path = root_dir + "/built_kernels/" + build_kernel_for_riscv_options.name;

    log_info(tt::LogBuildKernels, "Compiling OP: {} to {}", build_kernel_for_riscv_options.name, out_dir_path);

    void *hlk_args = new unary_datacopy::hlk_args_t{
        .per_core_tile_cnt = 96*4*5 // TODO(AP): this has to be in sync with tensor size in test_run_transpose_hc.cpp
    };
    build_kernel_for_riscv_options.set_hlk_args_all_cores(hlk_args, sizeof(unary_datacopy::hlk_args_t));
    build_kernel_for_riscv_options.set_hlk_file_name_all_cores("kernels/compute/eltwise_copy.cpp");

    build_kernel_for_riscv_options.set_hlk_operand_dataformat_all_cores(tt::HlkOperand::in0, tt::DataFormat::Float16_b);
    build_kernel_for_riscv_options.set_hlk_operand_dataformat_all_cores(tt::HlkOperand::out0, tt::DataFormat::Float16_b);
    // make sure to set this to false on GS (because no FP32 in dst), otherwise pack_src_format will be incorrect
    build_kernel_for_riscv_options.fp32_dest_acc_en = false;

    build_kernel_for_riscv_options.ncrisc_kernel_file_name = "kernels/dataflow/transpose_hc.cpp";
    build_kernel_for_riscv_options.brisc_kernel_file_name = "kernels/dataflow/writer_unary.cpp";

    bool skip_hlkc = false;
    if (argc > 1) {
        skip_hlkc = true;
    }
    generate_binaries_params_t params = {.skip_hlkc = skip_hlkc};
    generate_binaries_all_riscs(&build_kernel_for_riscv_options, out_dir_path, "grayskull", params);

    return 0;
}
