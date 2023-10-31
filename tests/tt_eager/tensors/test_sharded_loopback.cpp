// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/host_api.hpp"
#include "tensor/tensor.hpp"
#include "tensor/owned_buffer.hpp"
#include "tensor/owned_buffer_functions.hpp"
#include "tt_dnn/op_library/eltwise_binary/eltwise_binary_op.hpp"
#include "common/constants.hpp"
#include "tt_numpy/functions.hpp"

#include <algorithm>
#include <functional>
#include <random>
#include <optional>

using namespace tt;
using namespace tt_metal;
using namespace constants;


bool test_sharded_loopback(Device *device) {
    bool pass = true;
    int num_cores = 8;
    Shape single_tile_shape = {1, (u32) num_cores, TILE_HEIGHT, TILE_WIDTH};
    ShardSpec shard_spec = {.shard_grid = CoreRangeSet(
                                            {CoreRange(
                                                CoreCoord(0, 0), CoreCoord(0, num_cores - 1))
                                            }),
                            .shard_shape = {TILE_HEIGHT, TILE_WIDTH},
                            .shard_orientation = ShardOrientation::ROW_MAJOR
                                            };
    std::optional<ShardSpec> shard_spec_opt = std::make_optional<ShardSpec>(shard_spec);
    //Tensor host_a = tt::numpy::random::random(single_tile_shape, DataType::UINT32).to(Layout::TILE);
    Tensor host_a = tt::numpy::arange<uint32_t>((const int64_t)0, (const int64_t)1, (const Shape) single_tile_shape, (const Layout) Layout::TILE,  (Device *)nullptr, (const MemoryConfig) MemoryConfig{.memory_layout=tt::tt_metal::TensorMemoryLayout::HEIGHT_SHARDED});
    //arange(int64, int64_t stop, int64_t step, const Layout layout = Layout::ROW_MAJOR, Device * device = nullptr, const MemoryConfig& output_mem_config = MemoryConfig{.memory_layout=tt::tt_metal::TensorMemoryLayout::INTERLEAVED})
    MemoryConfig mem_config = {
            .memory_layout=tt::tt_metal::TensorMemoryLayout::HEIGHT_SHARDED,
            .buffer_storage=tt::tt_metal::BufferStorage::L1
    };
    if(mem_config.is_sharded()){
        TT_ASSERT(shard_spec_opt != std::nullopt);
    }

    Tensor device_a = host_a.to(device, mem_config, shard_spec_opt);
    Tensor loopbacked_a = device_a.cpu();
    auto host_a_data = owned_buffer::get_as<uint32_t>(host_a);
    auto loopbacked_a_data = owned_buffer::get_as<uint32_t>(loopbacked_a);

    //std::cout << "Host A Tensor " << std::endl;
    //host_a.print(Layout::TILE, false);
    //std::cout << "Device Tensor " << std::endl;
    //device_a.print(Layout::TILE, false);
    //std::cout << "Loopbacked Tensor " << std::endl;
    //loopbacked_a.print(Layout::TILE, false);
    pass &= host_a_data == loopbacked_a_data;

    return pass;
}


int main(int argc, char **argv) {
    bool pass = true;

    try {

        ////////////////////////////////////////////////////////////////////////////
        //                      Device Setup
        ////////////////////////////////////////////////////////////////////////////
        int device_id = 0;
        tt_metal::Device *device = tt_metal::CreateDevice(device_id);



        pass &= test_sharded_loopback(device);


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
