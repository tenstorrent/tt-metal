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
#include "tt_metal/llrt/llrt.hpp"
#include <algorithm>
#include <functional>
#include <random>
#include <optional>

using namespace tt;
using namespace tt_metal;
using namespace constants;


class test_config {
    public:
     test_config(
         uint32_t num_cores_height_,
         uint32_t num_cores_width_,
         uint32_t num_pages_per_core_height_,
         uint32_t num_pages_per_core_width_,
         MemoryConfig mem_config_) :
         num_cores_height(num_cores_height_),
         num_cores_width(num_cores_width_),
         num_pages_per_core_height(num_pages_per_core_height_),
         num_pages_per_core_width(num_pages_per_core_width_),
         mem_config(mem_config_),
         layout(Layout::TILE),
         page_width(TILE_WIDTH),
         page_height(TILE_HEIGHT) {
         mem_config.shard_spec = get_shard_spec();
     }

     test_config(
         uint32_t num_cores_height_,
         uint32_t num_cores_width_,
         uint32_t num_pages_per_core_height_,
         uint32_t num_pages_per_core_width_,
         MemoryConfig mem_config_,
         uint32_t page_height_,
         uint32_t page_width_) :
         num_cores_height(num_cores_height_),
         num_cores_width(num_cores_width_),
         num_pages_per_core_height(num_pages_per_core_height_),
         num_pages_per_core_width(num_pages_per_core_width_),
         mem_config(mem_config_),
         layout(Layout::ROW_MAJOR),
         page_width(page_width_),
         page_height(page_height_) {
         mem_config.shard_spec = get_shard_spec();
     }

        Shape get_legacy_shape(){
            Shape shape = {1, (uint32_t) num_cores_height, (uint32_t)page_height * num_pages_per_core_height, num_cores_width*num_pages_per_core_width * (uint32_t)page_width};
            return shape;
        }
        ShardSpec get_shard_spec(){
            ShardSpec shard_spec(
                CoreRangeSet({CoreRange(CoreCoord(0, 0), CoreCoord(0, num_cores_height * num_cores_width - 1))}),
                {page_height * num_pages_per_core_height, num_pages_per_core_width * page_width},
                ShardOrientation::ROW_MAJOR);
            return shard_spec;
        }
    private:
        public:
        const uint32_t num_cores_height;
        const uint32_t num_cores_width;
        const uint32_t num_pages_per_core_height;
        const uint32_t num_pages_per_core_width;
        MemoryConfig mem_config;
        const Layout layout;
        const uint32_t page_width;
        const uint32_t page_height;

};


bool test_sharded_loopback(Device *device, Layout layout, const Shape & shape, const MemoryConfig & mem_config) {
    bool pass = true;
    Tensor host_a = tt::numpy::random::random(shape);

    if(layout == Layout::TILE)
        host_a  = host_a.to(Layout::TILE);

    Tensor device_a = host_a.to(device, mem_config);
    Tensor loopbacked_a = device_a.cpu();
    auto host_a_data = owned_buffer::get_as<bfloat16>(host_a);
    auto loopbacked_a_data = owned_buffer::get_as<bfloat16>(loopbacked_a);

    pass &= host_a_data == loopbacked_a_data;

    return pass;
}


bool test_create_sharded_tensor(Device *device, DataType data_type , Layout layout, const Shape & shape, const MemoryConfig & mem_config) {
    bool pass = true;
    Tensor host_a = tt::numpy::random::random(shape);

    if(layout == Layout::TILE)
        host_a  = host_a.to(Layout::TILE);

    auto tensor = create_sharded_device_tensor(shape, data_type , layout, device, mem_config);
    return tensor.is_sharded();
}


int main(int argc, char **argv) {
    bool pass = true;

    try {

        ////////////////////////////////////////////////////////////////////////////
        //                      Device Setup
        ////////////////////////////////////////////////////////////////////////////
        int device_id = 0;
        tt_metal::Device *device = tt_metal::CreateDevice(device_id);


        // 8x2 TILES
        // 4 cores
        // 2x2 Block sharded
        {
            test_config config(
                4,
                1,
                2,
                2,
                MemoryConfig{.memory_layout = TensorMemoryLayout::BLOCK_SHARDED, .buffer_type = BufferType::L1});
            pass &= test_sharded_loopback(device, config.layout, config.get_legacy_shape(), config.mem_config);
        }

        // 4x4 TILES
        // 4 cores
        // 2x2 Block sharded
        {
            test_config config(
                2,
                2,
                2,
                2,
                MemoryConfig{.memory_layout = TensorMemoryLayout::BLOCK_SHARDED, .buffer_type = BufferType::L1});

            pass &= test_sharded_loopback(device, config.layout, config.get_legacy_shape(), config.mem_config);

        }

        // 2x8 TILES
        // 4 cores
        // Width Sharded
        {
            test_config config(
                1,
                4,
                2,
                2,
                MemoryConfig{.memory_layout = TensorMemoryLayout::WIDTH_SHARDED, .buffer_type = BufferType::L1});
            pass &= test_sharded_loopback(device, config.layout, config.get_legacy_shape(), config.mem_config);
        }

        // 8x2 TILES
        // 4 cores
        // Height Sharded
        {
            test_config config(
                4,
                1,
                2,
                2,
                MemoryConfig{.memory_layout = TensorMemoryLayout::HEIGHT_SHARDED, .buffer_type = BufferType::L1});
            pass &= test_sharded_loopback(device, config.layout, config.get_legacy_shape(), config.mem_config);
        }




        {
            Shape shape = {1, 1, 2304, 256};
            ShardSpec shard_spec(
                CoreRangeSet({CoreRange(CoreCoord(0, 0), CoreCoord(7, 7))}), {72, 128}, ShardOrientation::ROW_MAJOR);
            Layout layout = Layout::ROW_MAJOR;
            MemoryConfig mem_config = {.memory_layout=TensorMemoryLayout::BLOCK_SHARDED,
                                    .buffer_type=BufferType::L1, .shard_spec = shard_spec} ;
            pass &= test_sharded_loopback(device, layout, shape, mem_config);


        }


        {
            Shape shape = {1, 1, 800, 512};
            ShardSpec shard_spec(
                CoreRangeSet({CoreRange(CoreCoord(0, 0), CoreCoord(8, 7))}), {96, 64}, ShardOrientation::ROW_MAJOR);
            Layout layout = Layout::TILE;
            MemoryConfig mem_config = {.memory_layout=TensorMemoryLayout::BLOCK_SHARDED,
                                    .buffer_type=BufferType::L1, .shard_spec = shard_spec} ;
            DataType data_type = DataType::BFLOAT16;
            pass &= test_create_sharded_tensor(device, data_type , layout,  shape,  mem_config);


        }

        {
            Shape shape = {1, 1, 800, 512};
            ShardSpec shard_spec(
                CoreRangeSet({CoreRange(CoreCoord(0, 0), CoreCoord(8, 7))}), {96, 64}, ShardOrientation::ROW_MAJOR);
            Layout layout = Layout::TILE;
            MemoryConfig mem_config = {.memory_layout=TensorMemoryLayout::BLOCK_SHARDED,
                                    .buffer_type=BufferType::L1} ;
            DataType data_type = DataType::BFLOAT8_B;
            pass &= test_create_sharded_tensor(device, data_type , layout,  shape,  mem_config);


        }




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
    }

    TT_ASSERT(pass);

    return 0;
}
