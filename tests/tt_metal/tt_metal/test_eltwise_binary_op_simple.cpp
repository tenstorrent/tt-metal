#include <algorithm>
#include <functional>
#include <random>
#include <iostream>

#include "tt_metal/host_api.hpp"
#include "tensor/tensor.hpp"
#include "tt_dnn/op_library/eltwise_binary/add.hpp"

using namespace tt;
using namespace std;
using namespace tt::tt_metal;

int main(int argc, char **argv) {

    cout << "\nRunning Test\n" << endl;

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
    bool pass = InitializeDevice(device);

    Tensor a = Tensor({1, 1, 32, 32}, Initialize::RANDOM, Layout::TILES, device);
    Tensor b = Tensor({1, 1, 32, 32}, Initialize::ZEROS,  Layout::TILES, device);
    Tensor c = add(a, b);
    Tensor d = c.to(Location::HOST);

    pass &= CloseDevice(device);
    cout << "\nTest Complete\n" << endl;
    return 0;
}
