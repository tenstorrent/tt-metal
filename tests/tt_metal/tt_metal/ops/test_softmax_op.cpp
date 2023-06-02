#include "tt_metal/host_api.hpp"
#include "libs/tensor/tensor.hpp"
#include "libs/tt_dnn/op_library/softmax/softmax_op.hpp"

#include <algorithm>
#include <functional>
#include <random>

#include "tt_metal/llrt/tt_debug_print_server.hpp"

using namespace tt;
using namespace tt::tt_metal;
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
        Host *host = GetHost();
        pass &= InitializeDevice(device);
        tt_start_debug_print_server(device->cluster(), {0}, {{1, 1}});
        std::array<uint32_t, 4> shape = {1, 1, TILE_HEIGHT, TILE_WIDTH};
        Tensor a = Tensor(shape, Initialize::RANDOM, DataType::BFLOAT16, Layout::TILE, device);
        Tensor c = softmax_in_place(a);
        Tensor d = c.to(host);
        Tensor host_a = a.to(host); // Move tensor a to host to validate
        pass &= CloseDevice(device);
    } catch (const std::exception &e) {
        pass = false;
        log_error(LogTest, "{}", e.what());
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
