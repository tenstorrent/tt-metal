#include <algorithm>
#include <functional>
#include <random>

#include "tt_metal/host_api.hpp"
#include "common/bfloat16.hpp"
#include "llrt/tt_debug_print_server.hpp"

//////////////////////////////////////////////////////////////////////////////////////////
// TODO: explain what test does
//////////////////////////////////////////////////////////////////////////////////////////
using namespace tt;

bool test_compile_args(std::vector<uint32_t> compile_args_vec) {
    bool pass = true;

    ////////////////////////////////////////////////////////////////////////////
    //                      Grayskull Device Setup
    ////////////////////////////////////////////////////////////////////////////
    int pci_express_slot = 0;
    tt_metal::Device *device =
        tt_metal::CreateDevice(tt::ARCH::GRAYSKULL, pci_express_slot);

    pass &= tt_metal::InitializeDevice(device);
    tt_start_debug_print_server(device->cluster(), {0}, {{1, 1}});
    std::cout<<"Host Compile Time Args:"<<std::endl;
    for(uint32_t i = 0; i < compile_args_vec.size(); i++) {
        std::cout<<compile_args_vec[i]<<std::endl;
    }

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

    ////////////////////////////////////////////////////////////////////////////
    //                      Execute Application
    ////////////////////////////////////////////////////////////////////////////

    pass &= tt_metal::ConfigureDeviceWithProgram(device, program);

    pass &= tt_metal::LaunchKernels(device, program);

    tt_stop_debug_print_server(device->cluster());

    pass &= tt_metal::CloseDevice(device);

    return pass;
}

int main(int argc, char **argv) {
    bool pass = true;

    try {
        pass &= test_compile_args({0, 68, 0, 124});
        pass &= test_compile_args({1, 5, 0, 124});

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
