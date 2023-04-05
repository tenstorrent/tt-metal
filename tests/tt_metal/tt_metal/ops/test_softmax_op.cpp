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
        int pci_express_slot = 0;
        Device *device = CreateDevice(tt::ARCH::GRAYSKULL, pci_express_slot);
        Host *host = GetHost();
        pass &= InitializeDevice(device);
        tt_start_debug_print_server(device->cluster(), {0}, {{1, 1}});
        std::array<uint32_t, 4> shape = {1, 1, TILE_HEIGHT, TILE_WIDTH};
        Tensor a = Tensor(shape, Initialize::RANDOM, DataType::BFLOAT16, Layout::TILE, device);
        Tensor c = softmax(a);
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
