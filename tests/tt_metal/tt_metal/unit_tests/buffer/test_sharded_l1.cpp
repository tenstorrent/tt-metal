// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "device_fixture.hpp"
#include "gtest/gtest.h"
#include "tt_metal/common/logger.hpp"
#include "tt_metal/common/math.hpp"
#include "tt_metal/detail/tt_metal.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_metal/test_utils/comparison.hpp"
#include "tt_metal/test_utils/df/df.hpp"
#include "tt_metal/test_utils/print_helpers.hpp"
#include "tt_metal/test_utils/stimulus.hpp"
#include "tt_metal/common/constants.hpp"
#include <optional>

using namespace tt::tt_metal;


namespace basic_tests::l1::sharded {

struct L1Config {
    uint32_t num_cores_height = 2;
    uint32_t num_cores_width = 1;
    uint32_t num_tiles_per_core_height = 2;
    uint32_t num_tiles_per_core_width = 2;
    uint32_t element_size = 2;
    uint32_t size_bytes = 1 * num_cores_height * num_tiles_per_core_height
                        * tt::constants::TILE_HEIGHT * num_cores_width
                        * num_tiles_per_core_width * tt::constants::TILE_WIDTH
                        * element_size;
    uint32_t page_size_bytes = tt::constants::TILE_HW * element_size;
    tt::DataFormat l1_data_format = tt::DataFormat::Float16_b;
    TensorMemoryLayout buffer_layout = TensorMemoryLayout::HEIGHT_SHARDED;

    bool sharded = true;
    ShardSpecBuffer shard_spec() const{
        std::array<uint32_t, 2> tensor_shape;
        uint32_t tensor_width_size;
        if (buffer_layout == TensorMemoryLayout::HEIGHT_SHARDED) {
            tensor_shape = {num_cores_height * num_cores_width * num_tiles_per_core_height * tt::constants::TILE_HEIGHT,
                            num_tiles_per_core_width * tt::constants::TILE_WIDTH};
            tensor_width_size = num_tiles_per_core_width * page_size_bytes;
        } else if (buffer_layout == TensorMemoryLayout::WIDTH_SHARDED) {
            tensor_shape = {num_tiles_per_core_height * tt::constants::TILE_HEIGHT,
                            num_cores_height * num_cores_width * num_tiles_per_core_width * tt::constants::TILE_WIDTH};
            tensor_width_size = num_cores_height * num_cores_width * num_tiles_per_core_width * page_size_bytes;
        } else {
            TT_THROW("Unsupported buffer layout");
        }
        return ShardSpecBuffer(
                        CoreRangeSet(std::set<CoreRange>(
                            { CoreRange(CoreCoord(0,0), CoreCoord(0, num_cores_height*num_cores_width - 1))}
                            )),
                        {(uint32_t)num_tiles_per_core_height * tt::constants::TILE_HEIGHT,
                            (uint32_t)num_tiles_per_core_width * tt::constants::TILE_WIDTH},
                        ShardOrientation::ROW_MAJOR,
                        false,
                        buffer_layout,
                        {tt::constants::TILE_HEIGHT, tt::constants::TILE_WIDTH},
                        tensor_shape,
                        tensor_width_size);
    }
};

namespace local_test_functions {



/// @brief does host -> L1 -> host and makes sure its the same data
/// @param device
/// @param test_config - Configuration of the test -- see struct
/// @return
bool l1_buffer_read_write(Device* device, const L1Config& test_config) {
    bool pass = true;

    auto buffer = test_config.sharded ? CreateBuffer(tt::tt_metal::ShardedBufferConfig{
                                            .device = device,
                                            .size = test_config.size_bytes,
                                            .page_size = test_config.page_size_bytes,
                                            .buffer_layout = test_config.buffer_layout,
                                            .shard_parameters = test_config.shard_spec()})
                                      : CreateBuffer(tt::tt_metal::BufferConfig{
                                            .device = device,
                                            .size = test_config.size_bytes,
                                            .page_size = test_config.page_size_bytes,
                                            .buffer_layout = test_config.buffer_layout});

    auto input = tt::test_utils::generate_uniform_random_vector<uint32_t>(0, 100, test_config.size_bytes / sizeof(uint32_t));

    tt::tt_metal::detail::WriteToBuffer(buffer, input);


    tt::Cluster::instance().l1_barrier(device->id());
    std::vector<uint32_t> output;
    tt::tt_metal::detail::ReadFromBuffer(buffer, output);
    pass &= (output == input);

    if(!pass){
        if(input.size() != output.size()){
            std::cout << "Different size of input and output, input.size() = " << input.size() << " output.size() " << output.size() << std::endl;
        }
        int smaller_size = std::min<int>(input.size(), output.size());
        auto entries_per_page = test_config.page_size_bytes/(sizeof(uint32_t));
        for(int i = 0; i<smaller_size; i++){
            if(input[i] != output[i]){
                std::cout << "mismatch on page: " << i/entries_per_page << " entry index: " << i % entries_per_page
                        << " with input being " << std::hex << input[i] << " and output being " << output[i] << std::dec << std::endl;
            }
        }
    }

    return pass;
}

}   // end namespace local_test_functions

TEST_F(DeviceFixture, TestInterleavedReadWrite)
{
    for (unsigned int id = 0; id < num_devices_; id++) {
        L1Config test_config;
        test_config.buffer_layout = TensorMemoryLayout::INTERLEAVED;
        test_config.sharded = false;
        EXPECT_TRUE(local_test_functions::l1_buffer_read_write(this->devices_.at(id), test_config));
    }

}

TEST_F(DeviceFixture, TestHeightShardReadWrite)
{
    for (unsigned int id = 0; id < num_devices_; id++) {
        L1Config test_config;
        EXPECT_TRUE(local_test_functions::l1_buffer_read_write(this->devices_.at(id), test_config));
    }

}

TEST_F(DeviceFixture, TestWidthShardReadWrite)
{
    for (unsigned int id = 0; id < num_devices_; id++) {
        L1Config test_config;
        test_config.buffer_layout = TensorMemoryLayout::WIDTH_SHARDED;
        EXPECT_TRUE(local_test_functions::l1_buffer_read_write(this->devices_.at(id), test_config));
    }

}

}   // end namespace basic_tests::l1::sharded
