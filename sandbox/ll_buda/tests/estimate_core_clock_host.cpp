#include <algorithm>
#include <functional>
#include <random>

#include "ll_buda/host_api.hpp"

////////////////////////////////////////////////////////////////////////////
// Runs the add_two_ints kernel on BRISC to add two ints in L1
// Result is read from L1
////////////////////////////////////////////////////////////////////////////
using namespace tt;


int main(int argc, char **argv) {
    bool pass = true;

    try {
        ////////////////////////////////////////////////////////////////////////////
        //                      Grayskull Device Setup
        ////////////////////////////////////////////////////////////////////////////
        int pci_express_slot = 0;
        ll_buda::Device *device =
            ll_buda::CreateDevice(tt::ARCH::GRAYSKULL, pci_express_slot);

        pass &= ll_buda::InitializeDevice(device);

        ////////////////////////////////////////////////////////////////////////////
        //                      Application Setup
        ////////////////////////////////////////////////////////////////////////////
        ll_buda::Program *program = new ll_buda::Program();
        CoreCoord core = {0, 0};
        std::vector<uint32_t> first_runtime_args = {101, 202, 1};
        std::vector<uint32_t> second_runtime_args = {303, 606, 200000};

        ll_buda::DataMovementKernel *add_two_ints_kernel = ll_buda::CreateDataMovementKernel(
            program, "./sandbox/kernels/estimate_core_clock_device.cpp", core, ll_buda::DataMovementProcessor::RISCV_0, ll_buda::NOC::RISCV_0_default);

        ////////////////////////////////////////////////////////////////////////////
        //                      Compile Application
        ////////////////////////////////////////////////////////////////////////////
        constexpr bool profile_kernel = true;
        pass &= ll_buda::CompileProgram(device, program, profile_kernel);

        ////////////////////////////////////////////////////////////////////////////
        //                      Execute Application
        ////////////////////////////////////////////////////////////////////////////
        pass &= ll_buda::ConfigureDeviceWithProgram(device, program, profile_kernel);

        ll_buda::WriteRuntimeArgsToDevice(device, add_two_ints_kernel, core, first_runtime_args);

        pass &= ll_buda::LaunchKernels(device, program);

        std::vector<uint32_t> first_kernel_result;
        ll_buda::ReadFromDeviceL1(device, core, BRISC_L1_RESULT_BASE, first_kernel_result, sizeof(int));
        log_info(LogVerif, "first kernel result = {}", first_kernel_result[0]);

        ll_buda::dumpProfilerResults("Short");
        ////////////////////////////////////////////////////////////////////////////
        //                  Update Runtime Args and Re-run Application
        ////////////////////////////////////////////////////////////////////////////
        ll_buda::WriteRuntimeArgsToDevice(device, add_two_ints_kernel, core, second_runtime_args);

        pass &= ll_buda::LaunchKernels(device, program);

        std::vector<uint32_t> second_kernel_result;
        ll_buda::ReadFromDeviceL1(device, core, BRISC_L1_RESULT_BASE, second_kernel_result, sizeof(int));
        log_info(LogVerif, "second kernel result = {}", second_kernel_result[0]);

        ll_buda::dumpProfilerResults("Long");
        ll_buda::stopPrintfServer();
        ////////////////////////////////////////////////////////////////////////////
        //                      Validation & Teardown
        ////////////////////////////////////////////////////////////////////////////
        uint32_t first_expected_result = first_runtime_args[0] + first_runtime_args[1];
        uint32_t second_expected_result = second_runtime_args[0] + second_runtime_args[1];
        log_info(LogVerif, "first expected result = {} second expected result = {}", first_expected_result, second_expected_result);
        pass = first_kernel_result[0] == first_expected_result;
        pass = second_kernel_result[0] == second_expected_result;

        pass &= ll_buda::CloseDevice(device);;

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
