#include "tt_metal/host_api.hpp"
#include "tt_metal/tensor/tensor.hpp"
#include "tt_metal/op_library/eltwise_binary/eltwise_binary_op.hpp"
#include "constants.hpp"

#include <algorithm>
#include <functional>
#include <random>

using namespace tt;
using namespace tt_metal;
using namespace constants;


bool test_single_tile_single_dram_bank_loopback(Device *device, Host *host) {
    bool pass = true;
    std::array<uint32_t, 4> single_tile_shape = {1, 1, TILE_HEIGHT, TILE_WIDTH};

    Tensor host_a = Tensor(single_tile_shape, Initialize::RANDOM, DataType::BFLOAT16, Layout::TILE);
    Tensor device_a = host_a.to(device);
    Tensor loopbacked_a = device_a.to(host);
    auto host_a_data = *reinterpret_cast<std::vector<bfloat16>*>(host_a.data_ptr());
    auto loopbacked_a_data = *reinterpret_cast<std::vector<bfloat16>*>(loopbacked_a.data_ptr());
    pass &=  host_a_data == loopbacked_a_data;

    return pass;
}

bool test_multi_tile_multi_dram_bank_loopback(Device *device, Host *host) {
    bool pass = true;
    std::array<uint32_t, 4> multi_tile_shape = {1, 1, 4*TILE_HEIGHT, 3*TILE_WIDTH};

    Tensor host_a = Tensor(multi_tile_shape, Initialize::RANDOM, DataType::BFLOAT16, Layout::TILE);
    Tensor device_a = host_a.to(device);
    Tensor loopbacked_a = device_a.to(host);
    auto host_a_data = *reinterpret_cast<std::vector<bfloat16>*>(host_a.data_ptr());
    auto loopbacked_a_data = *reinterpret_cast<std::vector<bfloat16>*>(loopbacked_a.data_ptr());
    pass &= host_a_data == loopbacked_a_data;

    return pass;
}

int main(int argc, char **argv) {
    bool pass = true;

    try {
        int pci_express_slot = 0;
        tt_metal::Device *device =
            tt_metal::CreateDevice(tt::ARCH::GRAYSKULL, pci_express_slot);
        tt_metal::Host *host = tt_metal::GetHost();

        pass &= tt_metal::InitializeDevice(device);

        pass &= test_single_tile_single_dram_bank_loopback(device, host);

        pass &= test_multi_tile_multi_dram_bank_loopback(device, host);

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
