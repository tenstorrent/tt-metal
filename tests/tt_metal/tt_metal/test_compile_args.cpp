#include <algorithm>
#include <functional>
#include <random>

#include <filesystem>

#include "tt_metal/host_api.hpp"
#include "common/bfloat16.hpp"
#include "llrt/tt_debug_print_server.hpp"

//////////////////////////////////////////////////////////////////////////////////////////
// TODO: explain what test does
//////////////////////////////////////////////////////////////////////////////////////////
using namespace tt;

bool test_compile_args(std::vector<uint32_t> compile_args_vec, int pci_express_slot) {
    bool pass = true;

    ////////////////////////////////////////////////////////////////////////////
    //                      Grayskull Device Setup
    ////////////////////////////////////////////////////////////////////////////
    tt_metal::Device *device =
        tt_metal::CreateDevice(tt::ARCH::GRAYSKULL, pci_express_slot);

    pass &= tt_metal::InitializeDevice(device);

    ////////////////////////////////////////////////////////////////////////////
    //                      Application Setup
    ////////////////////////////////////////////////////////////////////////////
    tt_metal::Program program = tt_metal::Program();

    CoreCoord core = {0, 0};

    tt_metal::DataMovementKernel *unary_reader_kernel = tt_metal::CreateDataMovementKernel(
        program,
        "tt_metal/kernels/dataflow/test_compile_args.cpp",
        core,
        compile_args_vec,
        tt_metal::DataMovementProcessor::RISCV_1,
        tt_metal::NOC::RISCV_1_default);


    tt_metal::DataMovementKernel *unary_writer_kernel = tt_metal::CreateDataMovementKernel(
        program, "tt_metal/kernels/dataflow/blank.cpp",
        core, tt_metal::DataMovementProcessor::RISCV_0, tt_metal::NOC::RISCV_0_default);

    vector<uint32_t> compute_args = {
        0 // dummy
    };

    bool fp32_dest_acc_en = false;
    bool math_approx_mode = false;
    auto eltwise_unary_kernel = tt_metal::CreateComputeKernel(
        program, "tt_metal/kernels/compute/blank.cpp",
        core, compute_args, MathFidelity::HiFi4, fp32_dest_acc_en, math_approx_mode);

    ////////////////////////////////////////////////////////////////////////////
    //                      Compile Application
    ////////////////////////////////////////////////////////////////////////////
    pass &= tt_metal::CompileProgram(device, program);

    return pass;
}

int main(int argc, char **argv) {
    bool pass = true;

    try {
        int pci_express_slot = 0;
        static const std::string kernel_name = "test_compile_args";
        auto binary_path_str = get_kernel_compile_outpath(pci_express_slot) + kernel_name;
        std::filesystem::remove_all(binary_path_str);

        pass &= test_compile_args({0, 68, 0, 124}, pci_express_slot);
        pass &= test_compile_args({1, 5, 0, 124}, pci_express_slot);

        log_assert(std::filesystem::exists(binary_path_str), "Expected kernel to be compiled!");

        std::filesystem::path binary_path{binary_path_str};
        auto num_built_kernels = std::distance(std::filesystem::directory_iterator(binary_path), std::filesystem::directory_iterator{});
        log_assert(num_built_kernels == 2, "Expected compute kernel test_compile_args to be compiled twice!");

    } catch (const std::exception &e) {
        pass = false;
        // Capture the exception error message
        log_error(LogTest, "{}", e.what());
        // Capture system call errors that may have returned from driver/kernel
        log_error(LogTest, "System error message: {}", std::strerror(errno));
    }

    if (pass) {
        log_info(LogTest, "Test Passed");
    } else {
        log_fatal(LogTest, "Test Failed");
    }

    TT_ASSERT(pass);

    return 0;
}
