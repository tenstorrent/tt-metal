#include "ll_buda/host_api.hpp"

using namespace tt::ll_buda;

int main(int argc, char **argv) {
    bool pass = true;

    try {
        constexpr int pci_express_slot = 0;
        Device *device =
            CreateDevice(tt::ARCH::GRAYSKULL, pci_express_slot);

        pass &= InitializeDevice(device);

        Program *program = new Program();

        constexpr bool skip_hlkc = false;
        pass &= CompileProgram(device, program, skip_hlkc);

        pass &= ConfigureDeviceWithProgram(device, program);

        pass &= LaunchKernels(device, program);

        pass &= CloseDevice(device);

    } catch (const std::exception &e) {
        tt::log_error(tt::LogTest, "Test failed with exception!");
        tt::log_error(tt::LogTest, "{}", e.what());

        throw;
    }

    if (pass) {
        tt::log_info(tt::LogTest, "Test Passed");
    } else {
        tt::log_fatal(tt::LogTest, "Test Failed");
    }

    return 0;
}
