// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "device_fixture.hpp"
#include "tt_metal.hpp"
#include "dm_common.hpp"

namespace tt::tt_metal {

using namespace std;
using namespace tt;
using namespace tt::test_utils;

namespace unit_tests::dm::interleaved_to_sharded_hardcoded {

constexpr uint32_t START_ID = 202;

// Test config, i.e. test parameters
struct TestConfig {
    uint32_t test_id = 0;
    std::vector<uint32_t> compile_args;
    std::vector<uint32_t> runtime_args;
    CoreCoord master_core_coord = {0, 0};
    NOC noc_id = NOC::NOC_0;
    tt::DataFormat input_data_format = tt::DataFormat::Float32;
};

//=================================================================
// Test 1: DRAM Sharded Row Major Writer Hardcoded
//=================================================================
namespace test1_writer_sharded_dram_row_major {
/// @brief Does L1 Sender Core --> L1 Receiver Core
/// @param device
/// @param test_config - Configuration of the test -- see struct
/// @return
bool run_dm(const std::shared_ptr<distributed::MeshDevice>& mesh_device, const TestConfig& test_config) {
    // Program
    distributed::MeshWorkload workload;
    auto zero_coord = distributed::MeshCoordinate(0, 0);
    auto device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);
    Program program = CreateProgram();
    auto& cq = mesh_device->mesh_command_queue();

    CoreRangeSet master_core_set({CoreRange(test_config.master_core_coord)});
    auto writer_kernel = CreateKernel(
        program,
        "tests/tt_metal/tt_metal/data_movement/interleaved_to_sharded_hardcoded/kernels/"
        "writer_unary_sharded_stick_layout_start_id.cpp",
        master_core_set,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = test_config.compile_args});

    // Runtime Arguments
    SetRuntimeArgs(program, writer_kernel, master_core_set, test_config.runtime_args);

    // Assign unique id
    log_info(tt::LogTest, "Running Test ID: {}, Run ID: {}", test_config.test_id, unit_tests::dm::runtime_host_id);
    program.set_runtime_id(unit_tests::dm::runtime_host_id++);

    tt::DataFormat input_cb_data_format = test_config.input_data_format;
    uint32_t output_page_size, num_input_units;
    uint32_t input_cb_index = tt::CBIndex::c_0;
    uint32_t out_cb_index = input_cb_index;
    output_page_size = 256;
    num_input_units = 128;
    DataFormat output_cb_data_format = input_cb_data_format;

    tt::tt_metal::CircularBufferConfig output_cb_out_config =
        tt::tt_metal::CircularBufferConfig(num_input_units * output_page_size, {{out_cb_index, output_cb_data_format}})
            .set_page_size(out_cb_index, output_page_size);
    tt::tt_metal::CreateCircularBuffer(program, test_config.master_core_coord, output_cb_out_config);

    distributed::AddProgramToMeshWorkload(workload, std::move(program), device_range);
    distributed::EnqueueMeshWorkload(cq, workload, true);

    return true;
}
}  // namespace test1_writer_sharded_dram_row_major

//=================================================================
// Test 2: DRAM Sharded Tile Writer Hardcoded
//=================================================================
namespace test2_writer_sharded_dram_tile {
/// @brief Does L1 Sender Core --> L1 Receiver Core
/// @param device
/// @param test_config - Configuration of the test -- see struct
/// @return
bool run_dm(const std::shared_ptr<distributed::MeshDevice>& mesh_device, const TestConfig& test_config) {
    // Program
    distributed::MeshWorkload workload;
    auto zero_coord = distributed::MeshCoordinate(0, 0);
    auto device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);
    Program program = CreateProgram();
    auto& cq = mesh_device->mesh_command_queue();

    CoreRangeSet master_core_set({CoreRange(test_config.master_core_coord)});
    auto writer_kernel = CreateKernel(
        program,
        "tests/tt_metal/tt_metal/data_movement/interleaved_to_sharded_hardcoded/kernels/"
        "writer_unary_sharded_blocks_start_id.cpp",
        master_core_set,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = test_config.compile_args});

    // Runtime Arguments
    SetRuntimeArgs(program, writer_kernel, master_core_set, test_config.runtime_args);

    // Assign unique id
    log_info(tt::LogTest, "Running Test ID: {}, Run ID: {}", test_config.test_id, unit_tests::dm::runtime_host_id);
    program.set_runtime_id(unit_tests::dm::runtime_host_id++);

    tt::DataFormat input_cb_data_format = test_config.input_data_format;
    uint32_t output_page_size, num_input_units;
    uint32_t input_cb_index = tt::CBIndex::c_0;
    uint32_t out_cb_index = input_cb_index;
    output_page_size = 4096;
    num_input_units = 8;
    DataFormat output_cb_data_format = input_cb_data_format;

    tt::tt_metal::CircularBufferConfig output_cb_out_config =
        tt::tt_metal::CircularBufferConfig(num_input_units * output_page_size, {{out_cb_index, output_cb_data_format}})
            .set_page_size(out_cb_index, output_page_size);
    tt::tt_metal::CreateCircularBuffer(program, test_config.master_core_coord, output_cb_out_config);

    distributed::AddProgramToMeshWorkload(workload, std::move(program), device_range);
    distributed::EnqueueMeshWorkload(cq, workload, true);

    return true;
}
}  // namespace test2_writer_sharded_dram_tile

//=================================================================
// Test 3: DRAM Interleaved Tile Reader Hardcoded
//=================================================================
namespace test3_interleaved_reader_tile_dram {
/// @brief Does L1 Sender Core --> L1 Receiver Core
/// @param device
/// @param test_config - Configuration of the test -- see struct
/// @return
bool run_dm(const std::shared_ptr<distributed::MeshDevice>& mesh_device, const TestConfig& test_config) {
    // Program
    distributed::MeshWorkload workload;
    auto zero_coord = distributed::MeshCoordinate(0, 0);
    auto device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);
    Program program = CreateProgram();
    auto& cq = mesh_device->mesh_command_queue();

    CoreRangeSet master_core_set({CoreRange(test_config.master_core_coord)});
    auto reader_kernel = CreateKernel(
        program,
        "tests/tt_metal/tt_metal/data_movement/interleaved_to_sharded_hardcoded/kernels/"
        "reader_unary_sharded_blocks_interleaved_start_id.cpp",
        master_core_set,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1,
            .noc = NOC::RISCV_1_default,
            .compile_args = test_config.compile_args});

    // Runtime Arguments
    SetRuntimeArgs(program, reader_kernel, master_core_set, test_config.runtime_args);

    // Assign unique id
    log_info(tt::LogTest, "Running Test ID: {}, Run ID: {}", test_config.test_id, unit_tests::dm::runtime_host_id);
    program.set_runtime_id(unit_tests::dm::runtime_host_id++);

    tt::DataFormat input_cb_data_format = test_config.input_data_format;
    uint32_t input_cb_index = tt::CBIndex::c_0;
    uint32_t out_cb_index = input_cb_index;
    uint32_t num_input_units = 4;
    uint32_t output_unit_size = tt::tt_metal::detail::TileSize(input_cb_data_format);
    uint32_t output_page_size = tt::align(output_unit_size, 4);
    tt::tt_metal::CircularBufferConfig output_cb_out_config =
        tt::tt_metal::CircularBufferConfig(num_input_units * output_page_size, {{out_cb_index, input_cb_data_format}})
            .set_page_size(out_cb_index, output_page_size);
    auto all_cores = CoreRangeSet({CoreRange(test_config.master_core_coord)});
    tt::tt_metal::CreateCircularBuffer(program, all_cores, output_cb_out_config);

    distributed::AddProgramToMeshWorkload(workload, std::move(program), device_range);

    distributed::EnqueueMeshWorkload(cq, workload, true);

    return true;
}
}  // namespace test3_interleaved_reader_tile_dram

//=================================================================
// Test 4: L1 Interleaved Tile Reader Hardcoded
//=================================================================
namespace test4_interleaved_reader_tile_l1 {
/// @brief Does L1 Sender Core --> L1 Receiver Core
/// @param device
/// @param test_config - Configuration of the test -- see struct
/// @return
bool run_dm(const std::shared_ptr<distributed::MeshDevice>& mesh_device, const TestConfig& test_config) {
    // Program
    distributed::MeshWorkload workload;
    auto zero_coord = distributed::MeshCoordinate(0, 0);
    auto device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);
    Program program = CreateProgram();
    auto& cq = mesh_device->mesh_command_queue();

    CoreRangeSet master_core_set({CoreRange(test_config.master_core_coord)});
    auto reader_kernel = CreateKernel(
        program,
        "tests/tt_metal/tt_metal/data_movement/interleaved_to_sharded_hardcoded/kernels/"
        "reader_unary_sharded_blocks_interleaved_start_id.cpp",
        master_core_set,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1,
            .noc = NOC::RISCV_1_default,
            .compile_args = test_config.compile_args});

    // Runtime Arguments
    SetRuntimeArgs(program, reader_kernel, master_core_set, test_config.runtime_args);

    // Assign unique id
    log_info(tt::LogTest, "Running Test ID: {}, Run ID: {}", test_config.test_id, unit_tests::dm::runtime_host_id);
    program.set_runtime_id(unit_tests::dm::runtime_host_id++);

    tt::DataFormat input_cb_data_format = test_config.input_data_format;
    uint32_t input_cb_index = tt::CBIndex::c_0;
    uint32_t out_cb_index = input_cb_index;
    uint32_t num_input_units = 4;
    uint32_t output_unit_size = tt::tt_metal::detail::TileSize(input_cb_data_format);
    ;
    uint32_t output_page_size = tt::align(output_unit_size, 4);
    tt::tt_metal::CircularBufferConfig output_cb_out_config =
        tt::tt_metal::CircularBufferConfig(num_input_units * output_page_size, {{out_cb_index, input_cb_data_format}})
            .set_page_size(out_cb_index, output_page_size);
    auto all_cores = CoreRangeSet({CoreRange(test_config.master_core_coord)});
    tt::tt_metal::CreateCircularBuffer(program, all_cores, output_cb_out_config);

    distributed::AddProgramToMeshWorkload(workload, std::move(program), device_range);
    distributed::EnqueueMeshWorkload(cq, workload, true);

    return true;
}
}  // namespace test4_interleaved_reader_tile_l1

//=================================================================
// Test 5: DRAM Interleaved Row Major Reader Hardcoded
//=================================================================
namespace test5_interleaved_reader_row_major_dram {
/// @brief Does L1 Sender Core --> L1 Receiver Core
/// @param device
/// @param test_config - Configuration of the test -- see struct
/// @return
bool run_dm(const std::shared_ptr<distributed::MeshDevice>& mesh_device, const TestConfig& test_config) {
    // Program
    distributed::MeshWorkload workload;
    auto zero_coord = distributed::MeshCoordinate(0, 0);
    auto device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);
    Program program = CreateProgram();
    auto& cq = mesh_device->mesh_command_queue();

    CoreRangeSet master_core_set({CoreRange(test_config.master_core_coord)});
    auto reader_kernel = CreateKernel(
        program,
        "tests/tt_metal/tt_metal/data_movement/interleaved_to_sharded_hardcoded/kernels/"
        "reader_unary_stick_layout_sharded_blocks_interleaved_start_id.cpp",
        master_core_set,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1,
            .noc = NOC::RISCV_1_default,
            .compile_args = test_config.compile_args});

    // Runtime Arguments
    SetRuntimeArgs(program, reader_kernel, master_core_set, test_config.runtime_args);

    // Assign unique id
    log_info(tt::LogTest, "Running Test ID: {}, Run ID: {}", test_config.test_id, unit_tests::dm::runtime_host_id);

    program.set_runtime_id(unit_tests::dm::runtime_host_id++);
    tt::DataFormat input_cb_data_format = test_config.input_data_format;
    uint32_t output_page_size, num_input_units;
    uint32_t input_cb_index = tt::CBIndex::c_0;
    uint32_t out_cb_index = input_cb_index;
    output_page_size = 512;
    num_input_units = 128;
    DataFormat output_cb_data_format = input_cb_data_format;

    tt::tt_metal::CircularBufferConfig output_cb_out_config =
        tt::tt_metal::CircularBufferConfig(num_input_units * output_page_size, {{out_cb_index, output_cb_data_format}})
            .set_page_size(out_cb_index, output_page_size);
    tt::tt_metal::CreateCircularBuffer(program, test_config.master_core_coord, output_cb_out_config);

    distributed::AddProgramToMeshWorkload(workload, std::move(program), device_range);

    distributed::EnqueueMeshWorkload(cq, workload, true);

    return true;
}
}  // namespace test5_interleaved_reader_row_major_dram

//=================================================================
// Test 6: L1 Interleaved Row Major Reader Hardcoded
//=================================================================
namespace test6_interleaved_reader_row_major_l1 {
/// @brief Does L1 Sender Core --> L1 Receiver Core
/// @param device
/// @param test_config - Configuration of the test -- see struct
/// @return
bool run_dm(const std::shared_ptr<distributed::MeshDevice>& mesh_device, const TestConfig& test_config) {
    // Program
    distributed::MeshWorkload workload;
    auto zero_coord = distributed::MeshCoordinate(0, 0);
    auto device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);
    Program program = CreateProgram();
    auto& cq = mesh_device->mesh_command_queue();

    CoreRangeSet master_core_set({CoreRange(test_config.master_core_coord)});
    auto reader_kernel = CreateKernel(
        program,
        "tests/tt_metal/tt_metal/data_movement/interleaved_to_sharded_hardcoded/kernels/"
        "reader_unary_stick_layout_sharded_blocks_interleaved_start_id.cpp",
        master_core_set,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1,
            .noc = NOC::RISCV_1_default,
            .compile_args = test_config.compile_args});

    // Runtime Arguments
    SetRuntimeArgs(program, reader_kernel, master_core_set, test_config.runtime_args);

    // Assign unique id
    log_info(tt::LogTest, "Running Test ID: {}, Run ID: {}", test_config.test_id, unit_tests::dm::runtime_host_id);

    program.set_runtime_id(unit_tests::dm::runtime_host_id++);
    tt::DataFormat input_cb_data_format = test_config.input_data_format;
    uint32_t output_page_size, num_input_units;
    uint32_t input_cb_index = tt::CBIndex::c_0;
    uint32_t out_cb_index = input_cb_index;
    output_page_size = 512;
    num_input_units = 128;
    DataFormat output_cb_data_format = input_cb_data_format;

    tt::tt_metal::CircularBufferConfig output_cb_out_config =
        tt::tt_metal::CircularBufferConfig(num_input_units * output_page_size, {{out_cb_index, output_cb_data_format}})
            .set_page_size(out_cb_index, output_page_size);
    tt::tt_metal::CreateCircularBuffer(program, test_config.master_core_coord, output_cb_out_config);

    distributed::AddProgramToMeshWorkload(workload, std::move(program), device_range);

    distributed::EnqueueMeshWorkload(cq, workload, true);

    return true;
}
}  // namespace test6_interleaved_reader_row_major_l1

}  // namespace unit_tests::dm::interleaved_to_sharded_hardcoded

//=================================================================
// TEST CASES
//=================================================================

// Test 1: Writer Sharded DRAM Row Major
TEST_F(MeshDeviceFixture, TensixDataMovementI2SWriterShardedDramRowMajor) {
    if (arch_ != tt::ARCH::WORMHOLE_B0) {
        GTEST_SKIP() << "Skipping test for non-WH architecture";
    }

    // Parameters
    uint32_t test_id = unit_tests::dm::interleaved_to_sharded_hardcoded::START_ID + 0;
    std::vector<uint32_t> compile_args;
    std::vector<uint32_t> runtime_args;
    CoreCoord master_core_coord = {0, 0};

    compile_args.push_back(0);    // cb_id_out0
    compile_args.push_back(2);    // Memory layout
    compile_args.push_back(4);    // The number of sharding cores
    compile_args.push_back(256);  // The page size we offset each write to
    compile_args.push_back(1);    // The number of pages in each sharding row not including padding pages
    compile_args.push_back(4);    // This defines times when contiguous pages can't be calculated
    compile_args.push_back(1);    // pages_per_shard_x
    compile_args.push_back(128);  // pages_per_shard_y
    compile_args.push_back(test_id);

    runtime_args.push_back(11040);     // dst_addr
    runtime_args.push_back(128);       // block_height
    runtime_args.push_back(256);       // block_width_bytes
    runtime_args.push_back(256);       // padded_block_width_bytes
    runtime_args.push_back(0);         // start_id
    runtime_args.push_back(1);         // output_width_in_pages
    runtime_args.push_back(256);       // ->mapping table info
    runtime_args.push_back(33555200);  // --^

    // Test config
    unit_tests::dm::interleaved_to_sharded_hardcoded::TestConfig test_config = {
        .test_id = test_id,
        .compile_args = compile_args,
        .runtime_args = runtime_args,
        .master_core_coord = master_core_coord};

    // Run
    for (unsigned int id = 0; id < num_devices_; id++) {
        log_info(tt::LogTest, "Running test on device {}", id);
        EXPECT_TRUE(unit_tests::dm::interleaved_to_sharded_hardcoded::test1_writer_sharded_dram_row_major::run_dm(
            devices_.at(id), test_config));
    }
}

// Test 2: Writer Sharded DRAM Tile
TEST_F(MeshDeviceFixture, TensixDataMovementI2SWriterShardedDramTile) {
    if (arch_ != tt::ARCH::WORMHOLE_B0) {
        GTEST_SKIP() << "Skipping test for non-WH architecture";
    }

    // Parameters
    uint32_t test_id = unit_tests::dm::interleaved_to_sharded_hardcoded::START_ID + 1;
    std::vector<uint32_t> compile_args;
    std::vector<uint32_t> runtime_args;
    CoreCoord master_core_coord = {0, 0};

    compile_args.push_back(0);        // cb_id_out
    compile_args.push_back(2);        // Memory layout
    compile_args.push_back(4);        // The number of sharding cores
    compile_args.push_back(4096);     // The page size we offset each write to
    compile_args.push_back(2);        // The number of pages in each sharding row not including padding pages
    compile_args.push_back(4);        // This defines times when contiguous pages can't be calculated
    compile_args.push_back(2);        // pages_per_shard_x
    compile_args.push_back(4);        // pages_per_shard_y
    compile_args.push_back(test_id);  // test_id

    runtime_args.push_back(2572320);   // dst_addr
    runtime_args.push_back(4);         // block_height_tiles
    runtime_args.push_back(2);         // block_width_tiles
    runtime_args.push_back(0);         // padded_offset
    runtime_args.push_back(8);         // block_width_padded_num_tiles
    runtime_args.push_back(2);         // output_width_tiles
    runtime_args.push_back(0);         // start_id_offset
    runtime_args.push_back(0);         // start_id_base
    runtime_args.push_back(256);       // ->mapping table info
    runtime_args.push_back(33555200);  // --^

    // Test config
    unit_tests::dm::interleaved_to_sharded_hardcoded::TestConfig test_config = {
        .test_id = test_id,
        .compile_args = compile_args,
        .runtime_args = runtime_args,
        .master_core_coord = master_core_coord};

    // Run
    for (unsigned int id = 0; id < num_devices_; id++) {
        EXPECT_TRUE(unit_tests::dm::interleaved_to_sharded_hardcoded::test2_writer_sharded_dram_tile::run_dm(
            devices_.at(id), test_config));
    }
}

// Test 3: Interleaved Reader Tile (DRAM)
TEST_F(MeshDeviceFixture, TensixDataMovementI2SDRAMInterleavedReaderTile) {
    if (arch_ != tt::ARCH::WORMHOLE_B0) {
        GTEST_SKIP() << "Skipping test for non-WH architecture";
    }

    // Parameters
    uint32_t test_id = unit_tests::dm::interleaved_to_sharded_hardcoded::START_ID + 2;
    std::vector<uint32_t> compile_args;
    std::vector<uint32_t> runtime_args;
    CoreCoord master_core_coord = {0, 0};

    compile_args.push_back(0);        // cb_id_in0
    compile_args.push_back(1);        // num_readers
    compile_args.push_back(2);        // isDram = true
    compile_args.push_back(test_id);  // test_id

    runtime_args.push_back(1024 * 1024);  // src_addr
    runtime_args.push_back(2);            // block_height_tiles
    runtime_args.push_back(2);            // block_width_tiles
    runtime_args.push_back(0);            // padded_offset_bytes
    runtime_args.push_back(4);            // input_width_offset_tiles
    runtime_args.push_back(4);            // block_num_tiles
    runtime_args.push_back(0);            // start_id_offset
    runtime_args.push_back(0);            // start_id_base

    // Test config
    unit_tests::dm::interleaved_to_sharded_hardcoded::TestConfig test_config = {
        .test_id = test_id,
        .compile_args = compile_args,
        .runtime_args = runtime_args,
        .master_core_coord = master_core_coord,
        .input_data_format = tt::DataFormat::Float16_b};

    // Run
    for (unsigned int id = 0; id < num_devices_; id++) {
        EXPECT_TRUE(unit_tests::dm::interleaved_to_sharded_hardcoded::test3_interleaved_reader_tile_dram::run_dm(
            devices_.at(id), test_config));
    }
}

// Test 4: Interleaved Reader Tile (L1)
TEST_F(MeshDeviceFixture, TensixDataMovementI2SL1InterleavedReaderTile) {
    if (arch_ != tt::ARCH::WORMHOLE_B0) {
        GTEST_SKIP() << "Skipping test for non-WH architecture";
    }

    // Parameters
    uint32_t test_id = unit_tests::dm::interleaved_to_sharded_hardcoded::START_ID + 3;
    std::vector<uint32_t> compile_args;
    std::vector<uint32_t> runtime_args;
    CoreCoord master_core_coord = {0, 0};

    compile_args.push_back(0);        // cb_id_in0
    compile_args.push_back(1);        // num_readers
    compile_args.push_back(0);        // isDram = false
    compile_args.push_back(test_id);  // test_id

    runtime_args.push_back(1024);  // src_addr
    runtime_args.push_back(2);     // block_height_tiles
    runtime_args.push_back(2);     // block_width_tiles
    runtime_args.push_back(0);     // padded_offset_bytes
    runtime_args.push_back(4);     // input_width_offset_tiles
    runtime_args.push_back(4);     // block_num_tiles
    runtime_args.push_back(0);     // start_id_offset
    runtime_args.push_back(0);     // start_id_base

    // Test config
    unit_tests::dm::interleaved_to_sharded_hardcoded::TestConfig test_config = {
        .test_id = test_id,
        .compile_args = compile_args,
        .runtime_args = runtime_args,
        .master_core_coord = master_core_coord,
        .input_data_format = tt::DataFormat::Float16_b};

    // Run
    for (unsigned int id = 0; id < num_devices_; id++) {
        EXPECT_TRUE(unit_tests::dm::interleaved_to_sharded_hardcoded::test4_interleaved_reader_tile_l1::run_dm(
            devices_.at(id), test_config));
    }
}

// Test 5: Interleaved Reader Row Major (DRAM)
TEST_F(MeshDeviceFixture, TensixDataMovementI2SDRAMInterleavedReaderRowMajor) {
    if (arch_ != tt::ARCH::WORMHOLE_B0) {
        GTEST_SKIP() << "Skipping test for non-WH architecture";
    }

    // Parameters
    uint32_t test_id = unit_tests::dm::interleaved_to_sharded_hardcoded::START_ID + 4;
    std::vector<uint32_t> compile_args;
    std::vector<uint32_t> runtime_args;
    CoreCoord master_core_coord = {0, 0};

    compile_args.push_back(0);        // input_cb_index
    compile_args.push_back(1);        // scratch_cb_index
    compile_args.push_back(2048);     // num_units_per_row
    compile_args.push_back(2);        // isDram = true
    compile_args.push_back(test_id);  // test_id

    runtime_args.push_back(2560032);  // src_buffer->address(),
    runtime_args.push_back(2048);     // num_units_per_row,
    runtime_args.push_back(128);      // block_height,
    runtime_args.push_back(512);      // block_width_bytes,
    runtime_args.push_back(512);      // padded_block_width_bytes,
    runtime_args.push_back(1);        // static_cast<uint32_t>(aligned),
    runtime_args.push_back(0);        // aligned_input_width_offset_bytes,
    runtime_args.push_back(512);      // aligned_block_width_bytes,
    runtime_args.push_back(0);        // aligned_offset,
    runtime_args.push_back(0);        // start_id

    // Test config
    unit_tests::dm::interleaved_to_sharded_hardcoded::TestConfig test_config = {
        .test_id = test_id,
        .compile_args = compile_args,
        .runtime_args = runtime_args,
        .master_core_coord = master_core_coord};

    // Run
    for (unsigned int id = 0; id < num_devices_; id++) {
        EXPECT_TRUE(unit_tests::dm::interleaved_to_sharded_hardcoded::test5_interleaved_reader_row_major_dram::run_dm(
            devices_.at(id), test_config));
    }
}

// Test 6: Interleaved Reader Row Major (L1)
TEST_F(MeshDeviceFixture, TensixDataMovementI2SL1InterleavedReaderRowMajor) {
    if (arch_ != tt::ARCH::WORMHOLE_B0) {
        GTEST_SKIP() << "Skipping test for non-WH architecture";
    }

    // Parameters
    uint32_t test_id = unit_tests::dm::interleaved_to_sharded_hardcoded::START_ID + 5;
    std::vector<uint32_t> compile_args;
    std::vector<uint32_t> runtime_args;
    CoreCoord master_core_coord = {0, 0};

    compile_args.push_back(0);        // input_cb_index
    compile_args.push_back(1);        // scratch_cb_index
    compile_args.push_back(2048);     // num_units_per_row
    compile_args.push_back(0);        // isDram = false
    compile_args.push_back(test_id);  // test_id

    runtime_args.push_back(1024);  // src_buffer->address(),
    runtime_args.push_back(2048);  // num_units_per_row,
    runtime_args.push_back(128);   // block_height,
    runtime_args.push_back(512);   // block_width_bytes,
    runtime_args.push_back(512);   // padded_block_width_bytes,
    runtime_args.push_back(1);     // static_cast<uint32_t>(aligned),
    runtime_args.push_back(0);     // aligned_input_width_offset_bytes,
    runtime_args.push_back(512);   // aligned_block_width_bytes,
    runtime_args.push_back(0);     // aligned_offset,
    runtime_args.push_back(0);     // start_id

    // Test config
    unit_tests::dm::interleaved_to_sharded_hardcoded::TestConfig test_config = {
        .test_id = test_id,
        .compile_args = compile_args,
        .runtime_args = runtime_args,
        .master_core_coord = master_core_coord};

    // Run
    for (unsigned int id = 0; id < num_devices_; id++) {
        EXPECT_TRUE(unit_tests::dm::interleaved_to_sharded_hardcoded::test6_interleaved_reader_row_major_l1::run_dm(
            devices_.at(id), test_config));
    }
}

}  // namespace tt::tt_metal
