#include <iostream>
#include <filesystem>

#include "build_kernels_for_riscv/build_kernels_for_riscv.hpp"
#include "build_kernels_for_riscv/build_kernel_options.hpp"
#include "tests/tt_metal/test_utils/env_vars.hpp"


namespace fs = std::filesystem;



int main(int argc, char* argv[]) {

    std::string arch_name = tt::test_utils::get_env_arch_name();

    // Create and config an OP
    tt::build_kernel_for_riscv_options_t build_kernel_for_riscv_options("test","dataflow_cb_test");
    std::string out_dir_path = tt::get_kernel_compile_outpath() + build_kernel_for_riscv_options.name;

    log_info(tt::LogBuildKernels, "Compiling OP: {}", build_kernel_for_riscv_options.name);

    build_kernel_for_riscv_options.set_cb_dataformat_all_cores(tt::CB::c_in0, tt::DataFormat::Float16_b);
    build_kernel_for_riscv_options.set_cb_dataformat_all_cores(tt::CB::c_out0, tt::DataFormat::Float16_b);
    build_kernel_for_riscv_options.set_cb_dataformat_all_cores(tt::CB::c_intermed0, tt::DataFormat::Float16_b);

    // not all of them may be used in the runtime test, this is to inspect (manually atm) that data formats get compiled correctly
    build_kernel_for_riscv_options.set_cb_dataformat_all_cores(tt::CB::dataflow0, tt::DataFormat::Float16_b);
    build_kernel_for_riscv_options.set_cb_dataformat_all_cores(tt::CB::dataflow1, tt::DataFormat::Float16_b);
    build_kernel_for_riscv_options.set_cb_dataformat_all_cores(tt::CB::dataflow2, tt::DataFormat::Float16_b);
    build_kernel_for_riscv_options.set_cb_dataformat_all_cores(tt::CB::dataflow3, tt::DataFormat::Float16_b);
    build_kernel_for_riscv_options.set_cb_dataformat_all_cores(tt::CB::dataflow4, tt::DataFormat::Float16_b);
    build_kernel_for_riscv_options.set_cb_dataformat_all_cores(tt::CB::dataflow5, tt::DataFormat::Float16_b);
    build_kernel_for_riscv_options.set_cb_dataformat_all_cores(tt::CB::dataflow6, tt::DataFormat::Float16_b);
    build_kernel_for_riscv_options.set_cb_dataformat_all_cores(tt::CB::dataflow7, tt::DataFormat::Float16_b);

    // NCRISC / BRISC config
    build_kernel_for_riscv_options.ncrisc_kernel_file_name = "tt_metal/kernels/dataflow/reader_cb_test.cpp";
    build_kernel_for_riscv_options.brisc_kernel_file_name = "tt_metal/kernels/dataflow/writer_cb_test.cpp";

    fs::create_directories(out_dir_path);
    generate_data_format_descriptors(&build_kernel_for_riscv_options, build_kernel_for_riscv_options.name);

    __internal::generate_default_bank_to_noc_coord_descriptor (
        &build_kernel_for_riscv_options,
        build_kernel_for_riscv_options.name,
        tt::get_arch_from_string(arch_name)
    );
    generate_binary_for_ncrisc(&build_kernel_for_riscv_options, build_kernel_for_riscv_options.name, arch_name, 1, {8, 4});
    generate_binary_for_brisc(&build_kernel_for_riscv_options, build_kernel_for_riscv_options.name, arch_name, 0, {8, 2});

    return 0;
}
