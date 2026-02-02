// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "common/device_fixture.hpp"

#include <chrono>
#include <cerrno>
#include <fmt/base.h>
#include <cstdlib>
#include <sys/types.h>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <exception>
#include <map>
#include <array>
#include <set>
#include <memory>
#include <utility>
#include <variant>
#include <vector>

#include <tt_stl/assert.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/data_types.hpp>
#include <tt-logger/tt-logger.hpp>
#include <tt_stl/span.hpp>

//////////////////////////////////////////////////////////////////////////////////////////
// This test verifies that the slow dispatch path can perform device reads and writes when the page size is not a
// multiple of the DMA alignment requirement.
//////////////////////////////////////////////////////////////////////////////////////////
using std::vector;
using namespace tt;
using namespace tt::tt_metal;

TEST_F(MeshDeviceSingleCardFixture, UnalignedReadWriteCore) {
    IDevice* dev = devices_[0]->get_devices()[0];
    bool pass = true;

    try {
        ////////////////////////////////////////////////////////////////////////////
        //                      Write Read Unaligned DRAM Interleaved Buffer
        ////////////////////////////////////////////////////////////////////////////
        uint32_t single_tile_size = 2 * 1024;
        uint32_t num_tiles = 2048;
        uint32_t dram_buffer_size =
            (single_tile_size * num_tiles) +
            2; /*** Non 4-byte aligned buffer size for BFLOAT16s. This is the point of this unaligned test. ***/

        tt_metal::InterleavedBufferConfig dram_interleaved_buffer_config{
            .device = dev,
            .size = dram_buffer_size,
            .page_size = dram_buffer_size,
            .buffer_type = tt_metal::BufferType::DRAM};
        auto device_dram_interleaved_buffer = CreateBuffer(dram_interleaved_buffer_config);

        std::vector<uint8_t> src_vec_dram_interleaved_case(dram_buffer_size);
        for (auto& v : src_vec_dram_interleaved_case) {
            v = static_cast<uint8_t>(std::rand() % 256);
        }
        tt_metal::detail::WriteToBuffer(device_dram_interleaved_buffer, src_vec_dram_interleaved_case);
        std::vector<uint8_t> result_vec_dram_interleaved_case;
        tt_metal::detail::ReadFromBuffer(device_dram_interleaved_buffer, result_vec_dram_interleaved_case);
        pass &= (src_vec_dram_interleaved_case == result_vec_dram_interleaved_case);
        TT_FATAL(pass, "Error");
        log_info(LogTest, "Passed Non-4-byte-aligned Read Write DRAM Interleaved Buffer Test");

        ////////////////////////////////////////////////////////////////////////////
        //                      Write Read Unaligned DRAM Sharded Buffer
        ////////////////////////////////////////////////////////////////////////////
        CoreRangeSet shard_grid(std::set<CoreRange>({CoreRange(CoreCoord(0, 0), CoreCoord(0, 0))}));
        auto shard_spec = tt_metal::ShardSpecBuffer(
            shard_grid,
            {1, dram_buffer_size / sizeof(uint16_t)},
            tt_metal::ShardOrientation::ROW_MAJOR,
            {1, dram_buffer_size / sizeof(uint16_t)},
            {1, 1});
        auto device_dram_sharded_buffer = CreateBuffer(tt_metal::ShardedBufferConfig{
            .device = dev,
            .size = dram_buffer_size,
            .page_size = dram_buffer_size,
            .buffer_type = tt_metal::BufferType::DRAM,
            .buffer_layout = tt_metal::TensorMemoryLayout::HEIGHT_SHARDED,
            .shard_parameters = shard_spec});

        std::vector<uint8_t> src_vec_dram_sharded_case(dram_buffer_size);
        for (auto& v : src_vec_dram_sharded_case) {
            v = static_cast<uint8_t>(std::rand() % 256);
        }

        tt_metal::detail::WriteToBuffer(device_dram_sharded_buffer, src_vec_dram_sharded_case);

        std::vector<uint8_t> result_vec_dram_sharded_case;
        tt_metal::detail::ReadFromBuffer(device_dram_sharded_buffer, result_vec_dram_sharded_case);
        pass &= (src_vec_dram_sharded_case == result_vec_dram_sharded_case);
        TT_FATAL(pass, "Error");
        log_info(LogTest, "Passed Non-4-byte-aligned Read Write DRAM Sharded Buffer Test");

        ////////////////////////////////////////////////////////////////////////////
        //                      Write Read Unaligned L1 Interleaved Buffer
        ////////////////////////////////////////////////////////////////////////////
        uint32_t l1_buffer_size =
            (single_tile_size * 4) +
            2; /*** Non 4-byte aligned buffer size for BFLOAT16s. This is the point of this unaligned test. ***/
        tt_metal::InterleavedBufferConfig l1_interleaved_buffer_config{
            .device = dev,
            .size = l1_buffer_size,
            .page_size = l1_buffer_size,
            .buffer_type = tt_metal::BufferType::L1};
        auto device_l1_interleaved_buffer = CreateBuffer(l1_interleaved_buffer_config);

        std::vector<uint8_t> src_vec_l1_interleaved_case(l1_buffer_size);
        for (auto& v : src_vec_l1_interleaved_case) {
            v = static_cast<uint8_t>(std::rand() % 256);
        }

        tt_metal::detail::WriteToBuffer(device_l1_interleaved_buffer, src_vec_l1_interleaved_case);

        std::vector<uint8_t> result_vec_l1_interleaved_case;
        tt_metal::detail::ReadFromBuffer(device_l1_interleaved_buffer, result_vec_l1_interleaved_case);
        pass &= (src_vec_l1_interleaved_case == result_vec_l1_interleaved_case);
        TT_FATAL(pass, "Error");
        log_info(LogTest, "Passed Non-4-byte-aligned Read Write L1 Interleaved Buffer Test");

        ////////////////////////////////////////////////////////////////////////////
        //                      Write Read Unaligned L1 Sharded Buffer
        ////////////////////////////////////////////////////////////////////////////
        CoreRangeSet l1_shard_grid(std::set<CoreRange>({CoreRange(CoreCoord(0, 0), CoreCoord(0, 0))}));
        auto l1_shard_spec = tt_metal::ShardSpecBuffer(
            l1_shard_grid,
            {1, l1_buffer_size / sizeof(uint16_t)},
            tt_metal::ShardOrientation::ROW_MAJOR,
            {1, l1_buffer_size / sizeof(uint16_t)},
            {1, 1});
        auto device_l1_sharded_buffer = CreateBuffer(tt_metal::ShardedBufferConfig{
            .device = dev,
            .size = l1_buffer_size,
            .page_size = l1_buffer_size,
            .buffer_type = tt_metal::BufferType::L1,
            .buffer_layout = tt_metal::TensorMemoryLayout::HEIGHT_SHARDED,
            .shard_parameters = l1_shard_spec});

        std::vector<uint8_t> src_vec_l1_sharded_case(l1_buffer_size);
        for (auto& v : src_vec_l1_sharded_case) {
            v = static_cast<uint8_t>(std::rand() % 256);
        }

        tt_metal::detail::WriteToBuffer(device_l1_sharded_buffer, src_vec_l1_sharded_case);

        std::vector<uint8_t> result_vec_l1_sharded_case;
        tt_metal::detail::ReadFromBuffer(device_l1_sharded_buffer, result_vec_l1_sharded_case);
        pass &= (src_vec_l1_sharded_case == result_vec_l1_sharded_case);
        TT_FATAL(pass, "Error");
        log_info(LogTest, "Passed Non-4-byte-aligned Read Write L1 Sharded Buffer Test");

    } catch (const std::exception& e) {
        pass = false;
        // Capture the exception error message
        log_error(LogTest, "{}", e.what());
        // Capture system call errors that may have returned from driver/kernel
        log_error(LogTest, "System error message: {}", std::strerror(errno));
    }

    ASSERT_TRUE(pass);
}
