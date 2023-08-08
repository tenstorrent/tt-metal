#include <algorithm>
#include <functional>
#include <random>

#include "tt_metal/host_api.hpp"
#include "tools/profiler/profiler_state.hpp"
#include "tt_metal/detail/tt_metal.hpp"

////////////////////////////////////////////////////////////////////////////
// Runs the add_two_ints kernel on BRISC to add two ints in L1
// Result is read from L1
////////////////////////////////////////////////////////////////////////////
using namespace tt;

int main(int argc, char **argv) {
    bool pass = true;

    // Once this test is uplifted to use fast dispatch, this can be removed.
    char env[] = "TT_METAL_SLOW_DISPATCH_MODE=1";
    putenv(env);

    try {
        ////////////////////////////////////////////////////////////////////////////
        //                      Initial Runtime Args Parse
        ////////////////////////////////////////////////////////////////////////////
        std::vector<std::string> input_args(argv, argv + argc);
        string arch_name = "";
        try {
            std::tie(arch_name, input_args) =
                test_args::get_command_option_and_remaining_args(input_args, "--arch", "grayskull");
        } catch (const std::exception& e) {
            log_fatal(tt::LogTest, "Command line arguments found exception", e.what());
        }
        const tt::ARCH arch = tt::get_arch_from_string(arch_name);
        ////////////////////////////////////////////////////////////////////////////
        //                      Device Setup
        ////////////////////////////////////////////////////////////////////////////
        int pci_express_slot = 0;
        tt_metal::Device *device =
            tt_metal::CreateDevice(arch, pci_express_slot);

        pass &= tt_metal::InitializeDevice(device);

        ////////////////////////////////////////////////////////////////////////////
        //                      Application Setup
        ////////////////////////////////////////////////////////////////////////////
        tt_metal::Program program = tt_metal::Program();
        CoreCoord core = {0, 0};
        std::vector<uint32_t> first_runtime_args = {101, 202};
        std::vector<uint32_t> second_runtime_args = {303, 606};

        tt_metal::KernelID add_two_ints_kernel = tt_metal::CreateDataMovementKernel(
            program, "tt_metal/kernels/riscv_draft/add_two_ints.cpp", core,
            tt_metal::DataMovementConfig{.processor = tt_metal::DataMovementProcessor::RISCV_0, .noc = tt_metal::NOC::RISCV_0_default});

        ////////////////////////////////////////////////////////////////////////////
        //                      Compile Application
        ////////////////////////////////////////////////////////////////////////////
        pass &= tt_metal::CompileProgram(device, program);

        ////////////////////////////////////////////////////////////////////////////
        //                      Execute Application
        ////////////////////////////////////////////////////////////////////////////
        if (getDeviceProfilerState() == false){
            StartDebugPrintServer(device, {{1,1}});
        }
        tt_metal::SetRuntimeArgs(program, add_two_ints_kernel, core, first_runtime_args);

        pass &= tt_metal::ConfigureDeviceWithProgram(device, program);
        tt_metal::WriteRuntimeArgsToDevice(device, program);

        pass &= tt_metal::LaunchKernels(device, program);

        std::vector<uint32_t> first_kernel_result;
        tt_metal::detail::ReadFromDeviceL1(device, core, BRISC_L1_RESULT_BASE, sizeof(int), first_kernel_result);
        log_info(LogVerif, "first kernel result = {}", first_kernel_result[0]);

        ////////////////////////////////////////////////////////////////////////////
        //                  Update Runtime Args and Re-run Application
        ////////////////////////////////////////////////////////////////////////////
        tt_metal::SetRuntimeArgs(program, add_two_ints_kernel, core, second_runtime_args);

        tt_metal::WriteRuntimeArgsToDevice(device, program);

        pass &= tt_metal::LaunchKernels(device, program);

        std::vector<uint32_t> second_kernel_result;
        tt_metal::detail::ReadFromDeviceL1(device, core, BRISC_L1_RESULT_BASE, sizeof(int), second_kernel_result);
        log_info(LogVerif, "second kernel result = {}", second_kernel_result[0]);

        ////////////////////////////////////////////////////////////////////////////
        //                      Validation & Teardown
        ////////////////////////////////////////////////////////////////////////////
        uint32_t first_expected_result = first_runtime_args[0] + first_runtime_args[1];
        uint32_t second_expected_result = second_runtime_args[0] + second_runtime_args[1];
        log_info(LogVerif, "first expected result = {} second expected result = {}", first_expected_result, second_expected_result);
        pass = first_kernel_result[0] == first_expected_result;
        pass = second_kernel_result[0] == second_expected_result;

        pass &= tt_metal::CloseDevice(device);;

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
