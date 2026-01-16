// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "multi_device_fixture.hpp"
#include "tt_metal/test_utils/comparison.hpp"
#include "tt_metal/test_utils/stimulus.hpp"
#include "tt_metal/test_utils/print_helpers.hpp"
#include "dm_common.hpp"
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/mesh_buffer.hpp>
#include <distributed/mesh_device_impl.hpp>

namespace tt::tt_metal {

using namespace std;
using namespace tt;
using namespace tt::test_utils;

namespace unit_tests::dm::dram_sharded {
// Test config, i.e. test parameters
struct DramShardedConfig {
    uint32_t test_id = 0;
    uint32_t num_of_transactions = 0;
    uint32_t num_banks = 0;
    uint32_t pages_per_bank = 0;
    uint32_t page_size_bytes = 0;
    DataFormat l1_data_format = DataFormat::Invalid;
    CoreRangeSet cores;
    bool use_trid = false;
    uint32_t num_of_trids = 0;
};

/// @brief Reads from Sharded DRAM to L1 using stateful API
/// @param mesh_device - MeshDevice to run the test on
/// @param test_config - Configuration of the test -- see struct
/// @return
bool run_dm(const shared_ptr<distributed::MeshDevice>& mesh_device, const DramShardedConfig& test_config) {
    // Get the actual device for this single-device test
    IDevice* device = mesh_device->impl().get_device(0);

    // Program
    Program program = CreateProgram();

    uint32_t num_pages = test_config.num_banks * test_config.pages_per_bank;
    const size_t total_size_bytes = num_pages * test_config.page_size_bytes;

    // DRAM coords are 1D but use the same CoreCoord structs. y value is 0 and x range describes which banks to use
    CoreRange dram_bank_range({0, 0}, {test_config.num_banks - 1, 0});

    BufferDistributionSpec shard_spec = BufferDistributionSpec(
        Shape{1, num_pages},                   // tensor shape in pages
        Shape{1, test_config.pages_per_bank},  // shard shape in pages
        corerange_to_cores(dram_bank_range));

    uint32_t single_tile_size = test_config.page_size_bytes;
    distributed::DeviceLocalBufferConfig per_device_buffer_config{
        .page_size = single_tile_size,
        .buffer_type = BufferType::DRAM,
        .sharding_args = BufferShardingArgs(shard_spec)};

    Shape2D global_shape = {32, 32 * num_pages};
    distributed::ShardedBufferConfig sharded_buffer_config{
        .global_size = total_size_bytes,
        .global_buffer_shape = global_shape,
        .shard_shape = global_shape,  // since test run on only one chip
        .shard_orientation = ShardOrientation::ROW_MAJOR,
    };

    auto mesh_buffer =
        distributed::MeshBuffer::create(sharded_buffer_config, per_device_buffer_config, mesh_device.get());
    uint32_t input_buffer_address = mesh_buffer->address();

    // Input
    vector<uint32_t> packed_input = generate_packed_uniform_random_vector<uint32_t, bfloat16>(
        -100.0f, 100.0f, total_size_bytes / sizeof(bfloat16), chrono::system_clock::now().time_since_epoch().count());

    // Golden output
    // NOLINTNEXTLINE(performance-unnecessary-copy-initialization)
    vector<uint32_t> packed_golden = packed_input;

    // Compile-time arguments for kernel
    vector<uint32_t> reader_compile_args = {
        (uint32_t)test_config.num_of_transactions,
        (uint32_t)test_config.num_banks,
        (uint32_t)test_config.pages_per_bank,
        (uint32_t)test_config.page_size_bytes,
        (uint32_t)test_config.test_id};

    string kernel_path = "tests/tt_metal/tt_metal/data_movement/dram_sharded/kernels/dram_sharded_read";
    if (test_config.use_trid) {
        kernel_path += "_trid";
        reader_compile_args.push_back((uint32_t)test_config.num_of_trids);
    }
    kernel_path += ".cpp";

    // Kernels
    auto reader_kernel = CreateKernel(
        program,
        kernel_path,
        test_config.cores,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = reader_compile_args});

    uint32_t l1_addr = get_l1_address_and_size(mesh_device, corerange_to_cores(test_config.cores)[0]).base_address;
    std::vector<uint32_t> reader_run_time_args = {input_buffer_address, l1_addr};
    tt::tt_metal::SetRuntimeArgs(program, reader_kernel, test_config.cores, reader_run_time_args);

    // Assign unique id
    log_info(tt::LogTest, "Running Test ID: {}, Run ID: {}", test_config.test_id, unit_tests::dm::runtime_host_id);
    program.set_runtime_id(unit_tests::dm::runtime_host_id++);

    // Launch program and record outputs
    vector<uint32_t> packed_output;

    auto& cq = mesh_device->mesh_command_queue();
    distributed::EnqueueWriteMeshBuffer(cq, mesh_buffer, packed_input);

    auto mesh_workload = distributed::MeshWorkload();
    vector<uint32_t> coord_data = {0, 0};
    auto target_devices =
        distributed::MeshCoordinateRange(distributed::MeshCoordinate(coord_data));  // Single device at (0,0)
    mesh_workload.add_program(target_devices, std::move(program));

    distributed::EnqueueMeshWorkload(cq, mesh_workload, false);
    Finish(cq);

    detail::ReadFromDeviceL1(
        device, corerange_to_cores(test_config.cores)[0], l1_addr, total_size_bytes, packed_output);

    // Results comparison
    bool is_equal = (packed_output == packed_golden);

    if (!is_equal) {
        log_error(tt::LogTest, "Equality Check failed");
        log_info(tt::LogTest, "Golden vector");
        print_vector(unpack_vector<bfloat16, uint32_t>(packed_golden));
        log_info(tt::LogTest, "Output vector");
        print_vector(unpack_vector<bfloat16, uint32_t>(packed_output));
    }

    return is_equal;
}
}  // namespace unit_tests::dm::dram_sharded

/* ========== Directed Ideal Test Case; Test id = 84 ========== */
TEST_F(GenericMeshDeviceFixture, TensixDataMovementDRAMShardedReadDirectedIdeal) {
    auto mesh_device = get_mesh_device();

    // Parameters
    DataFormat l1_data_format = DataFormat::Float16_b;
    uint32_t page_size_bytes = tt::tile_size(l1_data_format);
    uint32_t num_of_transactions = 256;

    // Cores
    CoreRange core_range({0, 0}, {0, 0});
    CoreRangeSet core_range_set({core_range});

    // Test config
    unit_tests::dm::dram_sharded::DramShardedConfig test_config = {
        .test_id = 84,
        .num_of_transactions = num_of_transactions,
        .num_banks = mesh_device->num_dram_channels(),
        .pages_per_bank = 32,
        .page_size_bytes = page_size_bytes,
        .l1_data_format = l1_data_format,
        .cores = core_range_set};

    // Run
    EXPECT_TRUE(run_dm(mesh_device, test_config));
}

/* ========== Sweep over varying number of tiles per DRAM bank; Test id = 85 ========== */
TEST_F(GenericMeshDeviceFixture, TensixDataMovementDRAMShardedReadTileNumbers) {
    auto mesh_device = get_mesh_device();

    // Parameters
    DataFormat l1_data_format = DataFormat::Float16_b;
    uint32_t page_size_bytes = tt::tile_size(l1_data_format);
    uint32_t num_banks = mesh_device->num_dram_channels();
    uint32_t max_num_pages = 32;
    uint32_t max_transactions = 256;

    // Cores
    CoreRange core_range({0, 0}, {0, 0});
    CoreRangeSet core_range_set({core_range});

    for (uint32_t num_of_transactions = 1; num_of_transactions <= max_transactions; num_of_transactions *= 4) {
        for (uint32_t num_pages = 1; num_pages <= max_num_pages; num_pages *= 2) {
            // Test config
            unit_tests::dm::dram_sharded::DramShardedConfig test_config = {
                .test_id = 85,
                .num_of_transactions = num_of_transactions,
                .num_banks = num_banks,
                .pages_per_bank = num_pages,
                .page_size_bytes = page_size_bytes,
                .l1_data_format = l1_data_format,
                .cores = core_range_set};

            // Run
            EXPECT_TRUE(run_dm(mesh_device, test_config));
        }
    }
}

/* ========== Sweep over varying number of DRAM banks; Test id = 86 ========== */
TEST_F(GenericMeshDeviceFixture, TensixDataMovementDRAMShardedReadBankNumbers) {
    auto mesh_device = get_mesh_device();

    // Parameters
    DataFormat l1_data_format = DataFormat::Float16_b;
    uint32_t page_size_bytes = tt::tile_size(l1_data_format);
    uint32_t max_num_banks = mesh_device->num_dram_channels();
    uint32_t num_pages = 32;
    uint32_t max_transactions = 256;

    // Cores
    CoreRange core_range({0, 0}, {0, 0});
    CoreRangeSet core_range_set({core_range});

    for (uint32_t num_of_transactions = 1; num_of_transactions <= max_transactions; num_of_transactions *= 4) {
        for (uint32_t num_banks = 1; num_banks <= max_num_banks; num_banks++) {
            // Test config
            unit_tests::dm::dram_sharded::DramShardedConfig test_config = {
                .test_id = 86,
                .num_of_transactions = num_of_transactions,
                .num_banks = num_banks,
                .pages_per_bank = num_pages,
                .page_size_bytes = page_size_bytes,
                .l1_data_format = l1_data_format,
                .cores = core_range_set};

            // Run
            EXPECT_TRUE(run_dm(mesh_device, test_config));
        }
    }
}

/* ========== Directed Ideal Test Case with Transaction IDs; Test id = 87 ========== */
TEST_F(GenericMeshDeviceFixture, TensixDataMovementDRAMShardedReadTridDirectedIdeal) {
    auto mesh_device = get_mesh_device();

    // Parameters
    DataFormat l1_data_format = DataFormat::Float16_b;
    uint32_t page_size_bytes = tt::tile_size(l1_data_format);
    uint32_t num_of_transactions = 256;

    // Cores
    CoreRange core_range({0, 0}, {0, 0});
    CoreRangeSet core_range_set({core_range});

    // Test config
    unit_tests::dm::dram_sharded::DramShardedConfig test_config = {
        .test_id = 87,
        .num_of_transactions = num_of_transactions,
        .num_banks = mesh_device->num_dram_channels(),
        .pages_per_bank = 32,
        .page_size_bytes = page_size_bytes,
        .l1_data_format = l1_data_format,
        .cores = core_range_set,
        .use_trid = true,
        .num_of_trids = 16};

    // Run
    EXPECT_TRUE(run_dm(mesh_device, test_config));
}

}  // namespace tt::tt_metal
