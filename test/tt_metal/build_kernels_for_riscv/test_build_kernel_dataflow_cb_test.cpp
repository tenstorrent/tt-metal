#include <iostream>
#include <filesystem>

#include "build_kernels_for_riscv/build_kernels_for_riscv.hpp"


namespace fs = std::filesystem;



int main(int argc, char* argv[]) {

    std::string root_dir = tt::utils::get_root_dir();

    // Create and config an OP
    tt::build_kernel_for_riscv_options_t build_kernel_for_riscv_options("test","dataflow_cb_test");
    std::string out_dir_path = root_dir + "/built_kernels/" + build_kernel_for_riscv_options.name;

    log_info(tt::LogBuildKernels, "Compiling OP: {} to {}", build_kernel_for_riscv_options.name, out_dir_path);

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
    generate_data_format_descriptors(&build_kernel_for_riscv_options, out_dir_path);

    generate_binary_for_ncrisc(&build_kernel_for_riscv_options, out_dir_path, "grayskull", 1, {8, 4});
    generate_binary_for_brisc(&build_kernel_for_riscv_options, out_dir_path, "grayskull", 0, {8, 2});

    return 0;
}
