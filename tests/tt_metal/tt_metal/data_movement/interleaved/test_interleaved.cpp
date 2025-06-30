// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "device_fixture.hpp"
#include "tt_metal/test_utils/comparison.hpp"
#include "tt_metal/test_utils/stimulus.hpp"
#include "tt_metal/test_utils/print_helpers.hpp"
#include "dm_common.hpp"

namespace tt::tt_metal {

using namespace std;
using namespace tt;
using namespace tt::test_utils;

namespace unit_tests::dm::dram {
// Test config, i.e. test parameters
struct InterleavedConfig {
    uint32_t test_id = 0;
    uint32_t num_tiles = 0;
    uint32_t tile_size_bytes = 0;
    DataFormat l1_data_format = DataFormat::Invalid;
    CoreRangeSet cores = CoreRangeSet();
    bool is_dram = true;  // else is L1
    // add flags for what kernels to run if L1 vs Dram - only test 1 of read or write at a time for L1
};

/// @brief Does Dram --> Reader --> L1 CB --> Writer --> Dram.
/// @param device
/// @param test_config - Configuration of the test -- see struct
/// @return
bool run_dm(IDevice* device, const InterleavedConfig& test_config) {
    // Program
    Program program = CreateProgram();

    // DRAM Buffers
    const size_t total_size_bytes = test_config.num_tiles * test_config.tile_size_bytes;

    InterleavedBufferConfig interleaved_buffer_config{
        .device = device,
        .size = total_size_bytes,
        .page_size = test_config.tile_size_bytes,
        .buffer_type = test_config.is_dram ? BufferType::DRAM : BufferType::L1};
    std::shared_ptr<Buffer> input_buffer;
    input_buffer = CreateBuffer(interleaved_buffer_config);

    uint32_t input_byte_address = input_buffer->address();  // careful here for testing interleaved L1 buffer

    auto output_buffer = CreateBuffer(interleaved_buffer_config);
    uint32_t output_byte_address = output_buffer->address();

    // Input
    // vector<uint32_t> packed_input = generate_packed_uniform_random_vector<uint32_t, bfloat16>(
    //     -100.0f, 100.0f, total_size_bytes / bfloat16::SIZEOF,
    //     chrono::system_clock::now().time_since_epoch().count());
    vector<uint32_t> packed_input = create_arange_vector_of_bfloat16(total_size_bytes, false);  // num_bytes, print bool

    // Golden output
    // NOLINTNEXTLINE(performance-unnecessary-copy-initialization)
    vector<uint32_t> packed_golden = packed_input;

    uint8_t l1_cb_index = CBIndex::c_0;

    // Compile-time arguments for kernels
    vector<uint32_t> reader_compile_args = {
        (uint32_t)test_config.num_tiles,
        (uint32_t)test_config.tile_size_bytes,
        (uint32_t)l1_cb_index,
        (uint32_t)test_config.test_id,
        (uint32_t)test_config.is_dram};

    vector<uint32_t> writer_compile_args = {
        (uint32_t)test_config.num_tiles,
        (uint32_t)test_config.tile_size_bytes,
        (uint32_t)l1_cb_index,
        (uint32_t)test_config.test_id,
        (uint32_t)test_config.is_dram};

    // Create circular buffers
    CircularBufferConfig l1_cb_config =
        CircularBufferConfig(total_size_bytes, {{l1_cb_index, test_config.l1_data_format}})
            .set_page_size(l1_cb_index, test_config.tile_size_bytes);
    auto l1_cb = CreateCircularBuffer(program, test_config.cores, l1_cb_config);

    // Kernels
    auto reader_kernel = CreateKernel(
        program,
        "tests/tt_metal/tt_metal/data_movement/interleaved/kernels/interleaved_tile_read.cpp",
        test_config.cores,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1,
            .noc = NOC::RISCV_1_default,
            .compile_args = reader_compile_args});

    std::vector<uint32_t> reader_run_time_args = {input_byte_address};
    tt::tt_metal::SetRuntimeArgs(program, reader_kernel, test_config.cores, reader_run_time_args);

    auto writer_kernel = CreateKernel(
        program,
        "tests/tt_metal/tt_metal/data_movement/interleaved/kernels/interleaved_tile_write.cpp",
        test_config.cores,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = writer_compile_args});

    std::vector<uint32_t> writer_run_time_args = {output_byte_address};
    tt::tt_metal::SetRuntimeArgs(program, writer_kernel, test_config.cores, writer_run_time_args);

    // Assign unique id
    log_info(tt::LogTest, "Running Test ID: {}, Run ID: {}", test_config.test_id, unit_tests::dm::runtime_host_id);
    program.set_runtime_id(unit_tests::dm::runtime_host_id++);

    // Launch program and record outputs
    vector<uint32_t> packed_output;
    detail::WriteToBuffer(input_buffer, packed_input);
    MetalContext::instance().get_cluster().dram_barrier(device->id());
    detail::LaunchProgram(device, program);
    detail::ReadFromBuffer(output_buffer, packed_output);

    // Results comparison
    bool pcc = is_close_packed_vectors<bfloat16, uint32_t>(
        packed_output, packed_golden, [&](const bfloat16& a, const bfloat16& b) { return is_close(a, b); });

    if (!pcc) {
        log_error(tt::LogTest, "PCC Check failed");
        log_info(tt::LogTest, "Golden vector");
        print_vector<uint32_t>(packed_golden);
        log_info(tt::LogTest, "Output vector");
        print_vector<uint32_t>(packed_output);
    }

    return pcc;
}
}  // namespace unit_tests::dm::dram

/* ========== Test case for varying number of tiles; Test id = 60 ========== */
TEST_F(DeviceFixture, TensixDataMovementDRAMInterleavedTileNumbers) {
    // Physical Constraints
    auto [flit_size_bytes, max_transmittable_bytes, max_transmittable_flits] =
        tt::tt_metal::unit_tests::dm::compute_physical_constraints(arch_, devices_.at(0));

    // Parameters
    uint32_t max_num_tiles = 512;            // Bound for testing different transaction sizes
    uint32_t tile_size_bytes = 32 * 32 * 2;  // = tile size, since bfloat16 is 2 bytes

    // Cores
    CoreRange core_range({0, 0}, {0, 0});
    CoreRangeSet core_range_set({core_range});

    for (uint32_t num_tiles = 1; num_tiles <= max_num_tiles; num_tiles *= 2) {
        if (num_tiles * tile_size_bytes > max_transmittable_bytes) {
            continue;
        }

        // Test config
        unit_tests::dm::dram::InterleavedConfig test_config = {
            .test_id = 60,
            .num_tiles = num_tiles,
            .tile_size_bytes = tile_size_bytes,
            .l1_data_format = DataFormat::Float16_b,
            .cores = core_range_set};

        // Run
        for (unsigned int id = 0; id < num_devices_; id++) {
            EXPECT_TRUE(run_dm(devices_.at(id), test_config));
        }
    }
}

/* ========== Test case for varying core location; Test id = 61 ========== */
TEST_F(DeviceFixture, TensixDataMovementDRAMInterleavedTileCoreLocations) {
    // Parameters
    uint32_t num_tiles = 256;
    uint32_t tile_size_bytes = 32 * 32 * 2;  // = tile size, since bfloat16 is 2 bytes

    for (unsigned int id = 0; id < num_devices_; id++) {
        // Cores
        auto grid_size = devices_.at(id)->compute_with_storage_grid_size();
        log_info(tt::LogTest, "Grid size x: {}, y: {}", grid_size.x, grid_size.y);
        for (unsigned int x = 0; x < grid_size.x; x++) {
            for (unsigned int y = 0; y < grid_size.y; y++) {
                CoreRangeSet core_range_set({CoreRange({x, y}, {x, y})});
                log_info(tt::LogTest, "Core Location x: {}, y: {}", x, y);
                // Test config
                unit_tests::dm::dram::InterleavedConfig test_config = {
                    .test_id = 61,
                    .num_tiles = num_tiles,
                    .tile_size_bytes = tile_size_bytes,
                    .l1_data_format = DataFormat::Float16_b,
                    .cores = core_range_set};

                // Run
                EXPECT_TRUE(run_dm(devices_.at(id), test_config));
            }
        }
    }
}

/* ========== Directed Ideal Test Case; Test id = 62 ========== */
TEST_F(DeviceFixture, TensixDataMovementDRAMInterleavedTileDirectedIdeal) {
    // Physical Constraints
    auto [flit_size_bytes, max_transmittable_bytes, max_transmittable_flits] =
        tt::tt_metal::unit_tests::dm::compute_physical_constraints(arch_, devices_.at(0));

    // Parameters
    uint32_t num_tiles = 512;                // Bound for testing different transaction sizes
    uint32_t tile_size_bytes = 32 * 32 * 2;  // = tile size, since bfloat16 is 2 bytes

    // Cores
    CoreRange core_range({0, 0}, {0, 0});
    CoreRangeSet core_range_set({core_range});

    // Test config
    unit_tests::dm::dram::InterleavedConfig test_config = {
        .test_id = 62,
        .num_tiles = num_tiles,
        .tile_size_bytes = tile_size_bytes,
        .l1_data_format = DataFormat::Float16_b,
        .cores = core_range_set};

    // Run
    for (unsigned int id = 0; id < num_devices_; id++) {
        EXPECT_TRUE(run_dm(devices_.at(id), test_config));
    }
}

}  // namespace tt::tt_metal
