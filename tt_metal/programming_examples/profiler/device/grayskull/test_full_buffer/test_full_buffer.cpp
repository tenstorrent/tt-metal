#include "tt_metal/host_api.hpp"

using namespace tt;

bool RunCustomCycle(tt_metal::Device *device, int loop_count, string run_name = " ")
{
    bool pass = true;

    tt_xy_pair start_core = {0, 0};
    tt_xy_pair end_core = {11, 9};;
    tt_metal::CoreRange all_cores(start_core, end_core);

    tt_metal::Program program = tt_metal::Program();

    tt_metal::DataMovementKernel *brisc_kernel = tt_metal::CreateDataMovementKernel(
        program, "tt_metal/programming_examples/profiler/device/grayskull/test_full_buffer/kernels/full_buffer.cpp",
        all_cores, tt_metal::DataMovementProcessor::RISCV_0, tt_metal::NOC::RISCV_0_default);

    tt_metal::DataMovementKernel *ncrisc_kernel = tt_metal::CreateDataMovementKernel(
        program, "tt_metal/programming_examples/profiler/device/grayskull/test_full_buffer/kernels/full_buffer.cpp",
        all_cores, tt_metal::DataMovementProcessor::RISCV_1, tt_metal::NOC::RISCV_1_default);

    vector<uint32_t> trisc_kernel_args = {};
    tt_metal::KernelArgs trisc_args = tt_metal::KernelArgs(all_cores, trisc_kernel_args);

    bool fp32_dest_acc_en = false;
    bool math_approx_mode = false;
    tt_metal::ComputeKernel *trisc_kernel = tt_metal::CreateComputeKernel(
        program, "tt_metal/programming_examples/profiler/device/grayskull/test_full_buffer/kernels/full_buffer_compute.cpp",
        all_cores,
        trisc_args,
        MathFidelity::HiFi4,
        fp32_dest_acc_en,
        math_approx_mode
    );

    constexpr int loop_size = 200;
    constexpr bool profile_device = true;
    brisc_kernel->add_define("LOOP_COUNT",loop_count);
    ncrisc_kernel->add_define("LOOP_COUNT",loop_count);
    trisc_kernel->add_define("LOOP_COUNT",loop_count);

    brisc_kernel->add_define("LOOP_SIZE",loop_size);
    ncrisc_kernel->add_define("LOOP_SIZE",loop_size);
    trisc_kernel->add_define("LOOP_SIZE",loop_size);

    pass &= tt_metal::CompileProgram(device, program, profile_device);
    pass &= tt_metal::ConfigureDeviceWithProgram(device, program);
    pass &= tt_metal::LaunchKernels(device, program);
    if (profile_device)
        tt_metal::DumpDeviceProfileResults(device, program);
    tt_metal::DumpHostProfileResults(run_name);

    return pass;
}

int main(int argc, char **argv) {
    bool pass = true;

    try {
        ////////////////////////////////////////////////////////////////////////////
        //                      Grayskull Device Setup
        ////////////////////////////////////////////////////////////////////////////
        int pci_express_slot = 0;
        tt_metal::Device *device =
            tt_metal::CreateDevice(tt::ARCH::GRAYSKULL, pci_express_slot);

        pass &= tt_metal::InitializeDevice(device);

        int loop_count = 20;
        pass &= RunCustomCycle(device, loop_count);

        pass &= tt_metal::CloseDevice(device);

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
