#include "tt_metal/host_api.hpp"
#include "tensor/tensor.hpp"
#include "tt_dnn/op_library/bmm/bmm_op.hpp"
#include "constants.hpp"

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

        pass &= tt_metal::InitializeDevice(device);

        ////////////////////////////////////////////////////////////////////////////
        //                      Application Setup
        ////////////////////////////////////////////////////////////////////////////
        // Mt, Nt, Kt = num tiles, B = batch
        uint32_t Mt = 3;
        uint32_t Kt = 2;
        uint32_t Nt = 4;
        uint32_t B = 5;
        std::array<uint32_t, 4> shapea = {B, 1, Mt*TILE_HEIGHT, Kt*TILE_WIDTH};
        std::array<uint32_t, 4> shapeb = {B, 1, Kt*TILE_HEIGHT, Nt*TILE_WIDTH};
        std::array<uint32_t, 4> shapeb1 = {1, 1, Kt*TILE_HEIGHT, Nt*TILE_WIDTH};

        // Allocates a DRAM buffer on device populated with values specified by initialize
        Tensor a = Tensor(shapea, Initialize::RANDOM, DataType::BFLOAT16, Layout::TILE, device);
        Tensor b = Tensor(shapeb, Initialize::ZEROS, DataType::BFLOAT16, Layout::TILE, device);
        Tensor b1 = Tensor(shapeb1, Initialize::ZEROS, DataType::BFLOAT16, Layout::TILE, device);

        Tensor mm = bmm(a, b).to(host);
        Tensor mm1 = matmul(a, b1).to(host);

        ////////////////////////////////////////////////////////////////////////////
        //                      Validation & Teardown
        ////////////////////////////////////////////////////////////////////////////
        Tensor host_a = a.to(host); // Move tensor a to host to validate

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
