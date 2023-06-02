#include "tt_metal/host_api.hpp"
#include "tensor/tensor.hpp"
#include "common/bfloat16.hpp"
#include "constants.hpp"

#include <algorithm>
#include <functional>
#include <random>

using namespace tt;
using namespace tt_metal;
using namespace constants;

bool test_tensor_copy_semantics(Device *device, Host *host) {
    bool pass = true;
    std::array<uint32_t, 4> single_tile_shape = {1, 1, TILE_HEIGHT, TILE_WIDTH};

    // host tensor to host tensor copy constructor
    Tensor host_a = Tensor(single_tile_shape, Initialize::RANDOM, DataType::BFLOAT16, Layout::TILE);
    Tensor host_a_copy = host_a;
    auto host_a_data = *reinterpret_cast<std::vector<bfloat16>*>(host_a.data_ptr());
    auto host_a_copy_data = *reinterpret_cast<std::vector<bfloat16>*>(host_a_copy.data_ptr());
    pass &= host_a_data == host_a_copy_data;

    // dev tensor to dev tensor copy constructor
    Tensor dev_a = Tensor(single_tile_shape, Initialize::RANDOM, DataType::BFLOAT16, Layout::TILE, device);
    Tensor dev_a_copy = dev_a;
    auto dev_a_data = *reinterpret_cast<std::vector<bfloat16>*>(dev_a.to(host).data_ptr());
    auto dev_a_copy_data = *reinterpret_cast<std::vector<bfloat16>*>(dev_a_copy.to(host).data_ptr());
    pass &= dev_a_data == dev_a_copy_data;

    // host tensor updated with host tensor copy assignment
    Tensor host_c = Tensor(single_tile_shape, Initialize::INCREMENT, DataType::BFLOAT16, Layout::TILE);
    Tensor host_c_copy = Tensor(single_tile_shape, Initialize::RANDOM, DataType::BFLOAT16, Layout::TILE);
    host_c_copy = host_c;
    auto host_c_data = *reinterpret_cast<std::vector<bfloat16>*>(host_c.data_ptr());
    auto host_c_copy_data = *reinterpret_cast<std::vector<bfloat16>*>(host_c_copy.data_ptr());
    pass &= host_c_data == host_c_copy_data;

    // host tensor updated with dev tensor copy assignment
    Tensor host_d_copy = Tensor(single_tile_shape, Initialize::RANDOM, DataType::BFLOAT16, Layout::TILE);
    host_d_copy = dev_a;
    pass &= (not host_d_copy.on_host());
    auto host_d_copy_data = *reinterpret_cast<std::vector<bfloat16>*>(host_d_copy.to(host).data_ptr());
    pass &= dev_a_data == host_d_copy_data;

    // dev tensor updated with host tensor copy assignment
    Tensor host_e = Tensor(single_tile_shape, Initialize::ONES, DataType::BFLOAT16, Layout::TILE);
    Tensor dev_e_copy = Tensor(single_tile_shape, Initialize::RANDOM, DataType::BFLOAT16, Layout::TILE, device);
    dev_e_copy = host_e;
    pass &= (dev_e_copy.on_host());
    auto host_e_data = *reinterpret_cast<std::vector<bfloat16>*>(host_e.data_ptr());
    auto dev_e_copy_data = *reinterpret_cast<std::vector<bfloat16>*>(dev_e_copy.data_ptr());
    pass &= host_e_data == dev_e_copy_data;

    // dev tensor updated with dev tensor copy assignment
    Tensor dev_b = Tensor(single_tile_shape, Initialize::ONES, DataType::BFLOAT16, Layout::TILE, device);
    Tensor dev_b_copy = Tensor(single_tile_shape, Initialize::ZEROS, DataType::BFLOAT16, Layout::TILE, device);
    dev_b_copy = dev_b;
    pass &= (not dev_b_copy.on_host());
    auto dev_b_data = *reinterpret_cast<std::vector<bfloat16>*>(dev_b.to(host).data_ptr());
    auto dev_b_copy_data = *reinterpret_cast<std::vector<bfloat16>*>(dev_b_copy.to(host).data_ptr());
    pass &= dev_b_data == dev_b_copy_data;

    return pass;
}

bool test_tensor_move_semantics(Device *device, Host *host) {
    bool pass = true;
    std::array<uint32_t, 4> single_tile_shape = {1, 1, TILE_HEIGHT, TILE_WIDTH};

    auto uint32_data = create_random_vector_of_bfloat16(TILE_HEIGHT * TILE_WIDTH * sizeof(bfloat16), 100, std::chrono::system_clock::now().time_since_epoch().count());
    auto bfloat_data = unpack_uint32_vec_into_bfloat16_vec(uint32_data);
    auto bfloat_data_copy = bfloat_data;

    // host tensor to host tensor move constructor
    Tensor host_a = Tensor(bfloat_data, single_tile_shape, DataType::BFLOAT16, Layout::TILE);
    Tensor host_a_copy = std::move(host_a);
    auto host_a_copy_data = *reinterpret_cast<std::vector<bfloat16>*>(host_a_copy.data_ptr());
    pass &= host_a_copy_data == bfloat_data_copy;

    // dev tensor to dev tensor move constructor
    auto bfloat_data_two = unpack_uint32_vec_into_bfloat16_vec(uint32_data);
    Tensor dev_a = Tensor(bfloat_data_two, single_tile_shape, DataType::BFLOAT16, Layout::TILE, device);
    auto og_buffer_a = dev_a.buffer();
    Tensor dev_a_copy = std::move(dev_a);
    pass &= (dev_a.buffer() == nullptr and dev_a_copy.buffer() == og_buffer_a);
    auto dev_a_copy_data = *reinterpret_cast<std::vector<bfloat16>*>(dev_a_copy.to(host).data_ptr());
    pass &= dev_a_copy_data == bfloat_data_two;

    // host tensor updated with host tensor move assignment
    auto uint32_data_two = create_random_vector_of_bfloat16(TILE_HEIGHT * TILE_WIDTH * sizeof(bfloat16), 100, std::chrono::system_clock::now().time_since_epoch().count());
    auto bfloat_data_three = unpack_uint32_vec_into_bfloat16_vec(uint32_data_two);
    auto bfloat_data_three_copy = bfloat_data_three;
    Tensor host_c = Tensor(bfloat_data_three, single_tile_shape, DataType::BFLOAT16, Layout::TILE);
    Tensor host_c_copy = Tensor(dev_a_copy_data, single_tile_shape, DataType::BFLOAT16, Layout::TILE);
    host_c_copy = std::move(host_c);
    auto host_c_copy_data = *reinterpret_cast<std::vector<bfloat16>*>(host_c_copy.data_ptr());
    pass &= host_c_copy_data == bfloat_data_three_copy;

    // host tensor updated with dev tensor move assignment
    Tensor host_d_copy = Tensor(host_c_copy_data, single_tile_shape, DataType::BFLOAT16, Layout::TILE);
    host_d_copy = std::move(dev_a_copy);
    pass &= (not host_d_copy.on_host());
    auto host_d_copy_data = *reinterpret_cast<std::vector<bfloat16>*>(host_d_copy.to(host).data_ptr());
    pass &= host_d_copy_data == bfloat_data_two;

    // dev tensor updated with host tensor copy assignment
    auto uint32_data_three = create_random_vector_of_bfloat16(TILE_HEIGHT * TILE_WIDTH * sizeof(bfloat16), 100, std::chrono::system_clock::now().time_since_epoch().count());
    auto bfloat_data_four = unpack_uint32_vec_into_bfloat16_vec(uint32_data_two);
    auto bfloat_data_four_copy = bfloat_data_four;
    Tensor host_e = Tensor(bfloat_data_four, single_tile_shape, DataType::BFLOAT16, Layout::TILE);
    Tensor dev_e_copy = Tensor(host_d_copy_data, single_tile_shape, DataType::BFLOAT16, Layout::TILE, device);
    dev_e_copy = std::move(host_e);
    pass &= (dev_e_copy.on_host());
    auto dev_e_copy_data = *reinterpret_cast<std::vector<bfloat16>*>(dev_e_copy.data_ptr());
    pass &= dev_e_copy_data == bfloat_data_four_copy;

    // dev tensor updated with dev tensor copy assignment
    auto uint32_data_four = create_random_vector_of_bfloat16(TILE_HEIGHT * TILE_WIDTH * sizeof(bfloat16), 100, std::chrono::system_clock::now().time_since_epoch().count());
    auto bfloat_data_five = unpack_uint32_vec_into_bfloat16_vec(uint32_data_two);
    Tensor dev_b = Tensor(bfloat_data_five, single_tile_shape, DataType::BFLOAT16, Layout::TILE, device);
    Tensor dev_b_copy = Tensor(dev_e_copy_data, single_tile_shape, DataType::BFLOAT16, Layout::TILE, device);
    dev_b_copy = std::move(dev_b);
    pass &= (not dev_b_copy.on_host());
    auto dev_b_copy_data = *reinterpret_cast<std::vector<bfloat16>*>(dev_b_copy.to(host).data_ptr());
    pass &= dev_b_copy_data == bfloat_data_five;

    return pass;
}

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
        pass &= tt_metal::InitializeDevice(device);
        tt_metal::Host *host = tt_metal::GetHost();

        pass &= test_tensor_copy_semantics(device, host);

        pass &= test_tensor_move_semantics(device, host);

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
