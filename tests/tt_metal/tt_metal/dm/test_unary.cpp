// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "device_fixture.hpp"
#include "tt_metal/test_utils/comparison.hpp"
#include "tt_metal/test_utils/stimulus.hpp"
#include "tt_metal/test_utils/print_helpers.hpp"

namespace tt::tt_metal {

using std::vector;
using namespace tt;
using namespace tt::test_utils;

namespace unit_tests::dm {

struct DmConfig {
    size_t num_tiles = 0;
    size_t tile_byte_size = 0;
    DataFormat l1_data_format = DataFormat::Invalid;
    CoreRangeSet cores = CoreRangeSet();
};

/// @brief Does Dram --> Reader --> CB --> Writer --> Dram.
/// @param device
/// @param test_config - Configuration of the test -- see struct
/// @return
bool run_dm(IDevice* device, const DmConfig& test_config) {
    const size_t byte_size = test_config.num_tiles * test_config.tile_byte_size;
    Program program = CreateProgram();
    InterleavedBufferConfig dram_config{
        .device = device, .size = byte_size, .page_size = byte_size, .buffer_type = BufferType::DRAM};

    auto input_dram_buffer = CreateBuffer(dram_config);
    uint32_t input_dram_byte_address = input_dram_buffer->address();
    auto output_dram_buffer = CreateBuffer(dram_config);
    uint32_t output_dram_byte_address = output_dram_buffer->address();

    // Input
    vector<uint32_t> packed_input = generate_packed_uniform_random_vector<uint32_t, bfloat16>(
        -100.0f, 100.0f, byte_size / bfloat16::SIZEOF, std::chrono::system_clock::now().time_since_epoch().count());

    // Golden output
    vector<uint32_t> packed_golden = packed_input;

    // Same runtime args for every core
    vector<uint32_t> reader_rt_args = {
        (uint32_t)input_dram_byte_address,
        (uint32_t)0,
        (uint32_t)test_config.num_tiles,
    };

    vector<uint32_t> writer_rt_args = {
        (uint32_t)output_dram_byte_address,
        (uint32_t)0,
        (uint32_t)test_config.num_tiles,
    };

    for (const CoreRange& core_range : test_config.cores.ranges()) {
        CircularBufferConfig l1_cb_config =
            CircularBufferConfig(byte_size, {{CBIndex::c_0, test_config.l1_data_format}})
                .set_page_size(CBIndex::c_0, test_config.tile_byte_size);

        auto l1_cb = CreateCircularBuffer(program, core_range, l1_cb_config);

        auto reader_kernel = CreateKernel(
            program,
            "tests/tt_metal/tt_metal/dm/reader_unary.cpp",
            test_config.cores,
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});

        auto writer_kernel = CreateKernel(
            program,
            "tests/tt_metal/tt_metal/dm/writer_unary.cpp",
            test_config.cores,
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});

        for (const CoreCoord& core_coord : core_range) {
            SetRuntimeArgs(program, writer_kernel, core_coord, writer_rt_args);
            SetRuntimeArgs(program, reader_kernel, core_coord, reader_rt_args);
        }
    }

    vector<uint32_t> dest_buffer_data;
    detail::WriteToBuffer(input_dram_buffer, packed_input);
    detail::LaunchProgram(device, program);
    detail::ReadFromBuffer(output_dram_buffer, dest_buffer_data);

    // Print output and golden vectors
    log_info("Golden vector");
    print_vector<uint32_t>(packed_golden);
    log_info("Output vector");
    print_vector<uint32_t>(dest_buffer_data);

    return is_close_packed_vectors<bfloat16, uint32_t>(
        dest_buffer_data, packed_golden, [&](const bfloat16& a, const bfloat16& b) {
            return is_close(a, b);
        });  // TODO: do we want a different rtol and atol
}

}  // namespace unit_tests::dm

TEST_F(DeviceFixture, TensixDataMovement) {
    size_t num_tiles = 1;

    CoreRange core_range({0, 0}, {0, 0});
    CoreRangeSet core_range_set({core_range});
    unit_tests::dm::DmConfig test_config = {
        .num_tiles = num_tiles,
        .tile_byte_size = 2 * 32 * 32,
        .l1_data_format = DataFormat::Float16_b,
        .cores = core_range_set};
    for (unsigned int id = 0; id < num_devices_; id++) {
        EXPECT_TRUE(run_dm(devices_.at(id), test_config));
    }
}

}  // namespace tt::tt_metal
