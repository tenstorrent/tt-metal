// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "common/bfloat16.hpp"
#include "common/constants.hpp"
#include "ttnn/cpp/ttnn/operations/creation.hpp"
#include "ttnn/tensor/host_buffer/functions.hpp"
#include "ttnn/tensor/host_buffer/types.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/tensor_impl.hpp"
#include "tt_metal/host_api.hpp"
#include "ttnn/operations/functions.hpp"
#include "ttnn/cpp/ttnn/operations/experimental/reshape/reshape.hpp"

using namespace tt;
using namespace tt_metal;
using namespace constants;

bool test_tensor_copy_semantics(Device* device) {
    bool pass = true;
    tt::tt_metal::LegacyShape single_tile_shape = {1, 1, TILE_HEIGHT, TILE_WIDTH};

    // host tensor to host tensor copy constructor
    Tensor host_a = ttnn::random::random(single_tile_shape).to(Layout::TILE);
    Tensor host_a_copy = host_a;
    auto host_a_data = owned_buffer::get_as<bfloat16>(host_a);
    auto host_a_copy_data = owned_buffer::get_as<bfloat16>(host_a_copy);
    pass &= host_a_data == host_a_copy_data;

    // dev tensor to dev tensor copy constructor
    Tensor dev_a = ttnn::random::random(single_tile_shape).to(Layout::TILE).to(device);
    Tensor dev_a_copy = dev_a;
    auto dev_a_on_host = dev_a.cpu();
    auto dev_a_copy_on_host = dev_a_copy.cpu();
    auto dev_a_data = owned_buffer::get_as<bfloat16>(dev_a_on_host);
    auto dev_a_copy_data = owned_buffer::get_as<bfloat16>(dev_a_copy_on_host);
    pass &= dev_a_data == dev_a_copy_data;

    // host tensor updated with host tensor copy assignment
    Tensor host_c = ttnn::experimental::view(
                        ttnn::arange(/*start=*/0, /*stop=*/tt_metal::compute_volume(single_tile_shape), /*step=*/1),
                        single_tile_shape)
                        .to(Layout::TILE);
    Tensor host_c_copy = ttnn::random::random(single_tile_shape).to(Layout::TILE);
    host_c_copy = host_c;
    auto host_c_data = owned_buffer::get_as<bfloat16>(host_c);
    auto host_c_copy_data = owned_buffer::get_as<bfloat16>(host_c_copy);
    pass &= host_c_data == host_c_copy_data;

    // host tensor updated with dev tensor copy assignment
    Tensor host_d_copy = ttnn::random::random(single_tile_shape).to(Layout::TILE);
    host_d_copy = dev_a;
    pass &= (host_d_copy.storage_type() == StorageType::DEVICE);
    auto host_d_copy_on_host = host_d_copy.cpu();
    auto host_d_copy_data = owned_buffer::get_as<bfloat16>(host_d_copy_on_host);
    pass &= dev_a_data == host_d_copy_data;

    // dev tensor updated with host tensor copy assignment
    Tensor host_e = ttnn::ones(single_tile_shape, DataType::BFLOAT16, Layout::TILE);
    Tensor dev_e_copy = ttnn::random::random(single_tile_shape).to(Layout::TILE).to(device);
    dev_e_copy = host_e;
    pass &= (dev_e_copy.storage_type() == StorageType::OWNED);
    auto host_e_data = owned_buffer::get_as<bfloat16>(host_e);
    auto dev_e_copy_data = owned_buffer::get_as<bfloat16>(dev_e_copy);
    pass &= host_e_data == dev_e_copy_data;

    // dev tensor updated with dev tensor copy assignment
    Tensor dev_b = ttnn::ones(single_tile_shape, DataType::BFLOAT16, Layout::TILE, *device);
    Tensor dev_b_copy = ttnn::zeros(single_tile_shape, DataType::BFLOAT16, Layout::TILE, *device);
    dev_b_copy = dev_b;
    pass &= (dev_b_copy.storage_type() == StorageType::DEVICE);
    auto dev_b_on_host = dev_b.cpu();
    auto dev_b_copy_on_host = dev_b_copy.cpu();
    auto dev_b_data = owned_buffer::get_as<bfloat16>(dev_b_on_host);
    auto dev_b_copy_data = owned_buffer::get_as<bfloat16>(dev_b_copy_on_host);
    pass &= dev_b_data == dev_b_copy_data;

    return pass;
}

bool test_tensor_move_semantics(Device* device) {
    bool pass = true;
    tt::tt_metal::LegacyShape single_tile_shape = {1, 1, TILE_HEIGHT, TILE_WIDTH};

    auto random_tensor = ttnn::random::uniform(bfloat16(-1.0f), bfloat16(1.0f), single_tile_shape);
    auto bfloat_data = owned_buffer::get_as<bfloat16>(random_tensor);

    // host tensor to host tensor move constructor
    Tensor host_a = Tensor(OwnedStorage{bfloat_data}, single_tile_shape, DataType::BFLOAT16, Layout::TILE);
    Tensor host_a_copy = std::move(host_a);
    auto host_a_copy_data = owned_buffer::get_as<bfloat16>(host_a_copy);
    pass &= host_a_copy_data == bfloat_data;

    // dev tensor to dev tensor move constructor
    Tensor dev_a = Tensor(OwnedStorage{bfloat_data}, single_tile_shape, DataType::BFLOAT16, Layout::TILE).to(device);
    auto og_buffer_a = dev_a.buffer();
    Tensor dev_a_copy = std::move(dev_a);
    pass &= dev_a_copy.buffer() == og_buffer_a;
    auto dev_a_copy_on_host = dev_a_copy.cpu();
    auto dev_a_copy_data = owned_buffer::get_as<bfloat16>(dev_a_copy_on_host);
    pass &= dev_a_copy_data == bfloat_data;

    // host tensor updated with host tensor move assignment
    auto random_tensor_three = ttnn::random::uniform(bfloat16(-1.0f), bfloat16(1.0f), single_tile_shape);
    auto bfloat_data_three = owned_buffer::get_as<bfloat16>(random_tensor_three);
    Tensor host_c = Tensor(OwnedStorage{bfloat_data_three}, single_tile_shape, DataType::BFLOAT16, Layout::TILE);
    Tensor host_c_copy = Tensor(dev_a_copy_on_host.get_storage(), single_tile_shape, DataType::BFLOAT16, Layout::TILE);
    host_c_copy = std::move(host_c);
    auto host_c_copy_data = owned_buffer::get_as<bfloat16>(host_c_copy);
    pass &= host_c_copy_data == bfloat_data_three;

    // host tensor updated with dev tensor move assignment
    Tensor host_d_copy = Tensor(host_c_copy.get_storage(), single_tile_shape, DataType::BFLOAT16, Layout::TILE);
    host_d_copy = std::move(dev_a_copy);
    pass &= (host_d_copy.storage_type() == StorageType::DEVICE);
    auto host_d_copy_on_host = host_d_copy.cpu();
    auto host_d_copy_data = owned_buffer::get_as<bfloat16>(host_d_copy_on_host);
    pass &= host_d_copy_data == bfloat_data;

    // dev tensor updated with host tensor copy assignment
    auto random_tensor_four = ttnn::random::uniform(bfloat16(-1.0f), bfloat16(1.0f), single_tile_shape);
    auto bfloat_data_four = owned_buffer::get_as<bfloat16>(random_tensor_four);
    Tensor host_e = Tensor(random_tensor_four.get_storage(), single_tile_shape, DataType::BFLOAT16, Layout::TILE);
    Tensor dev_e_copy =
        Tensor(host_c_copy.get_storage(), single_tile_shape, DataType::BFLOAT16, Layout::TILE).to(device);
    dev_e_copy = std::move(host_e);
    pass &= (dev_e_copy.storage_type() == StorageType::OWNED);
    auto dev_e_copy_data = owned_buffer::get_as<bfloat16>(dev_e_copy);
    pass &= dev_e_copy_data == bfloat_data_four;

    // dev tensor updated with dev tensor copy assignment
    auto random_tensor_five = ttnn::random::uniform(bfloat16(-1.0f), bfloat16(1.0f), single_tile_shape);
    auto bfloat_data_five = owned_buffer::get_as<bfloat16>(random_tensor_five);
    Tensor dev_b =
        Tensor(random_tensor_four.get_storage(), single_tile_shape, DataType::BFLOAT16, Layout::TILE).to(device);
    Tensor dev_b_copy =
        Tensor(dev_e_copy.get_storage(), single_tile_shape, DataType::BFLOAT16, Layout::TILE).to(device);
    dev_b_copy = std::move(dev_b);
    pass &= (dev_b_copy.storage_type() == StorageType::DEVICE);
    auto dev_b_copy_on_host = dev_b_copy.cpu();
    auto dev_b_copy_data = owned_buffer::get_as<bfloat16>(dev_b_copy_on_host);
    pass &= dev_b_copy_data == bfloat_data_five;

    return pass;
}

bool test_tensor_deallocate_semantics(Device* device) {
    bool pass = true;
    tt::tt_metal::LegacyShape single_tile_shape = {1, 1, TILE_HEIGHT, TILE_WIDTH};

    MemoryConfig dram_mem_config =
        MemoryConfig{.memory_layout = TensorMemoryLayout::INTERLEAVED, .buffer_type = BufferType::DRAM};
    MemoryConfig l1_mem_config =
        MemoryConfig{.memory_layout = TensorMemoryLayout::INTERLEAVED, .buffer_type = BufferType::L1};

    // dev tensor allocate, deallocate, reallocate same address DRAM
    Tensor dev_a = ttnn::random::random(single_tile_shape).to(Layout::TILE).to(device, dram_mem_config);
    uint32_t address_a = dev_a.buffer()->address();
    dev_a.deallocate();
    Tensor dev_b = ttnn::random::random(single_tile_shape).to(Layout::TILE).to(device, dram_mem_config);
    uint32_t address_b = dev_b.buffer()->address();
    pass &= address_a == address_b;

    // dev tensor allocate, allocate, deallocate, reallocate same address DRAM
    Tensor dev_c = ttnn::random::random(single_tile_shape).to(Layout::TILE).to(device, dram_mem_config);
    dev_b.deallocate();
    Tensor dev_d = ttnn::random::random(single_tile_shape).to(Layout::TILE).to(device, dram_mem_config);
    uint32_t address_d = dev_d.buffer()->address();
    pass &= address_b == address_d;

    // dev tensor allocate, deallocate, reallocate same address L1
    Tensor dev_e = ttnn::random::random(single_tile_shape).to(Layout::TILE).to(device, l1_mem_config);
    uint32_t address_e = dev_e.buffer()->address();
    dev_e.deallocate();
    Tensor dev_f = ttnn::random::random(single_tile_shape).to(Layout::TILE).to(device, l1_mem_config);
    uint32_t address_f = dev_f.buffer()->address();
    pass &= address_e == address_f;

    // dev tensor allocate, allocate, deallocate, reallocate same address DRAM
    Tensor dev_g = ttnn::random::random(single_tile_shape).to(Layout::TILE).to(device, l1_mem_config);
    dev_f.deallocate();
    Tensor dev_h = ttnn::random::random(single_tile_shape).to(Layout::TILE).to(device, l1_mem_config);
    uint32_t address_h = dev_h.buffer()->address();
    pass &= address_f == address_h;

    return pass;
}

bool test_tensor_deallocate_and_close_device(Device* device) {
    bool pass = true;
    tt::tt_metal::LegacyShape single_tile_shape = {1, 1, TILE_HEIGHT, TILE_WIDTH};

    MemoryConfig dram_mem_config =
        MemoryConfig{.memory_layout = TensorMemoryLayout::INTERLEAVED, .buffer_type = BufferType::DRAM};
    MemoryConfig l1_mem_config =
        MemoryConfig{.memory_layout = TensorMemoryLayout::INTERLEAVED, .buffer_type = BufferType::L1};

    // dev tensor allocate, deallocate, reallocate same address DRAM
    Tensor dev_a = ttnn::random::random(single_tile_shape).to(Layout::TILE).to(device, dram_mem_config);
    uint32_t address_a = dev_a.buffer()->address();
    pass &= tt_metal::CloseDevice(device);
    dev_a.deallocate();

    return pass;
}

int main(int argc, char** argv) {
    bool pass = true;

    try {
        ////////////////////////////////////////////////////////////////////////////
        //                      Device Setup
        ////////////////////////////////////////////////////////////////////////////
        int device_id = 0;
        tt_metal::Device* device = tt_metal::CreateDevice(device_id);

        pass &= test_tensor_copy_semantics(device);

        pass &= test_tensor_move_semantics(device);

        pass &= test_tensor_deallocate_semantics(device);

        pass &= test_tensor_deallocate_and_close_device(device);

    } catch (const std::exception& e) {
        pass = false;
        // Capture the exception error message
        log_error(LogTest, "{}", e.what());
        // Capture system call errors that may have returned from driver/kernel
        log_error(LogTest, "System error message: {}", std::strerror(errno));
    }

    if (pass) {
        log_info(LogTest, "Test Passed");
    } else {
        TT_THROW("Test Failed");
    }

    TT_FATAL(pass, "Error");

    return 0;
}
