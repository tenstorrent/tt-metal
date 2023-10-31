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

using namespace tt::tt_metal;


namespace basic_tests::l1::sharded {

struct L1Config {
    size_t num_cores = 4;
    size_t num_tiles_per_core = 2;
    size_t size_bytes = 1 * num_cores * num_tiles_per_core * tt::constants::TILE_HW;
    size_t page_size_bytes = tt::constants::TILE_HW;
    tt::DataFormat l1_data_format = tt::DataFormat::Float16_b;
    TensorMemoryLayout buffer_layout = TensorMemoryLayout::HEIGHT_SHARDED;
    std::optional<ShardSpec> shard_parameters = ShardSpec{
                            .shard_grid = CoreRangeSet(
                                            {CoreRange(
                                                CoreCoord(0, 0), CoreCoord(0, num_cores - 1))
                                            }),
                            .shard_shape = {(u32)num_tiles_per_core, (u32)1},
                            .shard_orientation=ShardOrientation::ROW_MAJOR
                            };
};

namespace local_test_functions {



/// @brief does host -> L1 -> host and makes sure its the same data
/// @param device
/// @param test_config - Configuration of the test -- see struct
/// @return
bool l1_buffer_read_write(Device* device, const L1Config& test_config) {
    bool pass = true;

    Buffer buffer = CreateBuffer(device,
                                test_config.size_bytes,
                                test_config.page_size_bytes,
                                BufferStorage::L1,
                                test_config.buffer_layout,
                                test_config.shard_parameters);

    auto input = tt::test_utils::generate_uniform_random_vector<uint32_t>(0, 100, test_config.size_bytes / sizeof(uint32_t));

    WriteToBuffer(buffer, input);

    tt::Cluster::instance().l1_barrier(device->id());
    std::vector<uint32_t> output;
    ReadFromBuffer(buffer, output);
    pass &= (output == input);

    return pass;
}

}   // end namespace local_test_functions

TEST_F(DeviceFixture, TestInterleavedReadWrite)
{
    for (unsigned int id = 0; id < num_devices_; id++) {
        L1Config test_config;
        test_config.buffer_layout = TensorMemoryLayout::INTERLEAVED;
        test_config.shard_parameters = std::nullopt;
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
