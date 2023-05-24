#include <iostream>

#include "build_kernels_for_riscv/build_kernels_for_riscv.hpp"
#include "tests/tt_metal/test_utils/env_vars.hpp"

void compile_datacopy_kernel() {
    std::string arch_name = tt::test_utils::get_env_arch_name();
    // Create and config an OP
    tt::build_kernel_for_riscv_options_t build_kernel_for_riscv_options("unary","datacopy_op_dispatch");

    log_info(tt::LogBuildKernels, "Compiling OP: {}", build_kernel_for_riscv_options.name);

    std::vector<uint32_t> compute_kernel_args = {
        4
    };

    // HLK config
    build_kernel_for_riscv_options.set_hlk_file_name_all_cores("tt_metal/kernels/compute/eltwise_copy.cpp");

    // data-copy has one input operand and one output operand
    build_kernel_for_riscv_options.set_hlk_operand_dataformat_all_cores(tt::HlkOperand::in0, tt::DataFormat::Float16_b);
    build_kernel_for_riscv_options.set_hlk_operand_dataformat_all_cores(tt::HlkOperand::out0, tt::DataFormat::Float16_b);

    // make sure to set this to false on GS (because no FP32 in dst), otherwise pack_src_format will be incorrect
    build_kernel_for_riscv_options.fp32_dest_acc_en = false;

    // NCRISC / BRISC config
    build_kernel_for_riscv_options.ncrisc_kernel_file_name = "tt_metal/kernels/dataflow/reader_unary.cpp";
    build_kernel_for_riscv_options.brisc_kernel_file_name = "tt_metal/kernels/dataflow/writer_unary.cpp";

    // Defines
    std::map<string, string> dataflow_defines = {{"DEVICE_DISPATCH_MODE", ""}};
    build_kernel_for_riscv_options.brisc_defines = dataflow_defines;
    build_kernel_for_riscv_options.ncrisc_defines = dataflow_defines;


    // generate binaries
    __internal::generate_default_bank_to_noc_coord_descriptor (
        &build_kernel_for_riscv_options,
        build_kernel_for_riscv_options.name,
        tt::get_arch_from_string(arch_name)
    );
    generate_binaries_params_t params = {.compute_kernel_compile_time_args = compute_kernel_args};
    generate_binaries_all_riscs(&build_kernel_for_riscv_options, build_kernel_for_riscv_options.name, arch_name, params);
}

void compile_dispatch_kernel() {
    std::string arch_name = tt::test_utils::get_env_arch_name();
    // Create and config an OP
    tt::build_kernel_for_riscv_options_t build_kernel_for_riscv_options("unary", "dispatch");

    log_info(tt::LogBuildKernels, "Compiling OP: {} to {}", build_kernel_for_riscv_options.name, build_kernel_for_riscv_options.name);

    // make sure to set this to false on GS (because no FP32 in dst), otherwise pack_src_format will be incorrect
    build_kernel_for_riscv_options.fp32_dest_acc_en = false;

    // NCRISC / BRISC config
    build_kernel_for_riscv_options.brisc_kernel_file_name = "tt_metal/kernels/dataflow/dispatch/dispatch.cpp";

    std::map<string, string> brisc_defines = {{"IS_DISPATCH_KERNEL", ""}, {"DEVICE_DISPATCH_MODE", ""}};
    build_kernel_for_riscv_options.brisc_defines = brisc_defines;

    // generate binaries
    __internal::generate_default_bank_to_noc_coord_descriptor (
        &build_kernel_for_riscv_options,
        build_kernel_for_riscv_options.name,
        tt::get_arch_from_string(arch_name)
    );
    generate_binary_for_risc(RISCID::BR, &build_kernel_for_riscv_options, build_kernel_for_riscv_options.name, arch_name);
}

int main(int argc, char* argv[]) {

    compile_datacopy_kernel();
    compile_dispatch_kernel();

    return 0;
}
