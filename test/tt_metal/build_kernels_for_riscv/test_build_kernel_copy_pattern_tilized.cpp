#include <iostream>

#include "build_kernels_for_riscv/build_kernels_for_riscv.hpp"


int main(int argc, char* argv[]) {

    std::string root_dir = tt::utils::get_root_dir();

    // Create and config an OP
    tt::build_kernel_for_riscv_options_t build_kernel_for_riscv_options("dummy_type","copy_pattern_tilized");
    std::string out_dir_path = root_dir + "/built_kernels/" + build_kernel_for_riscv_options.name;

    log_info(tt::LogBuildKernels, "Compiling OP: {} to {}", build_kernel_for_riscv_options.name, out_dir_path);

    std::vector<uint32_t> compute_kernel_args = {
        18
    };
    build_kernel_for_riscv_options.set_hlk_file_name_all_cores("kernels/compute/eltwise_copy.cpp");

    build_kernel_for_riscv_options.set_hlk_operand_dataformat_all_cores(tt::HlkOperand::in0, tt::DataFormat::Float16_b);
    build_kernel_for_riscv_options.set_hlk_operand_dataformat_all_cores(tt::HlkOperand::out0, tt::DataFormat::Float16_b);
    // make sure to set this to false on GS (because no FP32 in dst), otherwise pack_src_format will be incorrect
    build_kernel_for_riscv_options.fp32_dest_acc_en = false;

    build_kernel_for_riscv_options.ncrisc_kernel_file_name = "kernels/dataflow/copy_pattern_tilized.cpp";
    build_kernel_for_riscv_options.brisc_kernel_file_name = "kernels/dataflow/writer_unary.cpp";

    bool skip_hlkc = false;
    if (argc > 1) {
        skip_hlkc = true;
    }
    generate_binaries_params_t params = {.skip_hlkc = skip_hlkc, .compute_kernel_compile_time_args = compute_kernel_args};
    generate_binaries_all_riscs(&build_kernel_for_riscv_options, out_dir_path, "grayskull", params);

    return 0;
}
