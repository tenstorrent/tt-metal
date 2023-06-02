#include "tt_metal/host_api.hpp"
#include "tensor/tensor.hpp"
#include "tt_dnn/op_library/transpose/transpose_op.hpp"

#include <algorithm>
#include <functional>
#include <random>

using namespace tt;
using namespace tt_metal;
using namespace constants;

//////////////////////////////////////////////////////////////////////////////////////////
// TODO: explain what test does
//////////////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv) {
    bool pass = true;

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
        tt_metal::Host *host = tt_metal::GetHost();

        pass &= InitializeDevice(device);

        ////////////////////////////////////////////////////////////////////////////
        //                      Application Setup
        ////////////////////////////////////////////////////////////////////////////
        std::array<uint32_t, 4> shape = {1, 1, TILE_HEIGHT, TILE_WIDTH};
        // Allocates a DRAM buffer on device populated with values specified by initialize
        Tensor a = Tensor(shape, Initialize::RANDOM, DataType::BFLOAT16, Layout::TILE, device);

        tt_metal::Tensor c = tt_metal::transpose(a);

        tt_metal::Tensor d = c.to(host);

        ////////////////////////////////////////////////////////////////////////////
        //                      Validation & Teardown
        ////////////////////////////////////////////////////////////////////////////
        tt_metal::Tensor host_a = a.to(host); // Move tensor a to host to validate
        //pass &= (host_a.data() == d.data()); // src1 is all 0's

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
