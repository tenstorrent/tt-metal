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

namespace tt::tt_metal {

using namespace std;
using namespace tt;
using namespace tt::test_utils;

namespace unit_tests::dm::dram_sharded {
// Test config, i.e. test parameters
struct DramShardedConfig {
    uint32_t test_id = 0;
    uint32_t num_of_transactions = 0;
    uint32_t num_pages = 0;
    uint32_t page_size_bytes = 0;
    DataFormat l1_data_format = DataFormat::Invalid;
    CoreRangeSet cores = CoreRangeSet();
};

/// @brief Reads from Sharded DRAM to L1 using stateful API
/// @param mesh_device - MeshDevice to run the test on
/// @param test_config - Configuration of the test -- see struct
/// @return
bool run_dm(const shared_ptr<distributed::MeshDevice>& mesh_device, const DramShardedConfig& test_config) {
    // Get the actual device for this single-device test
    IDevice* device = mesh_device->get_device(0);

    // Program
    Program program = CreateProgram();

    // Shape tensor_pages_shape = {2, 2};
    // Shape shard_pages_shape = {1, 1};
    CoreRange dram_core_range({0, 0}, {5, 0});
    BufferDistributionSpec shard_spec = BufferDistributionSpec(
        // Shape{1, test_config.num_pages},  // tensor shape in pages
        Shape{1, 24},  // tensor shape in pages
        Shape{1, 4},   // shard shape in pages
        corerange_to_cores(dram_core_range));

    // uint32_t single_tile_size = tt::tile_size(test_config.l1_data_format);
    uint32_t single_tile_size = test_config.page_size_bytes;
    distributed::DeviceLocalBufferConfig per_device_buffer_config{
        .page_size = single_tile_size,
        .buffer_type = BufferType::DRAM,
        .sharding_args = BufferShardingArgs(shard_spec),
        .bottom_up = std::nullopt};  // idk what bottom up does
    // const size_t total_size_bytes = test_config.num_pages * test_config.page_size_bytes;
    // Shape2D global_shape = {32, 32 * test_config.num_pages};
    Shape2D global_shape = {32, 32 * 24};
    // Shape2D shard_shape = {32, 32};
    const size_t total_size_bytes = global_shape.height() * global_shape.width() * sizeof(bfloat16);
    distributed::ShardedBufferConfig sharded_buffer_config{
        .global_size = total_size_bytes,
        .global_buffer_shape = global_shape,
        .shard_shape = global_shape,  // equal to global buffer shape since only one chip?
        .shard_orientation = ShardOrientation::ROW_MAJOR,
    };

    auto mesh_buffer =
        distributed::MeshBuffer::create(sharded_buffer_config, per_device_buffer_config, mesh_device.get());
    uint32_t input_buffer_address = mesh_buffer->address();
    // ShardSpecBuffer(
    //     const CoreRangeSet& core_sets_,
    //     const std::array<uint32_t, 2>& shard_shape_,
    //     const ShardOrientation& shard_orientation_,
    //     const std::array<uint32_t, 2>& page_shape,
    //     const std::array<uint32_t, 2>& tensor2d_shape_in_pages)

    // struct ShardedBufferConfig {
    //     IDevice* device{};
    //     DeviceAddr size{};       // Size in bytes
    //     DeviceAddr page_size{};  // Size of unit being interleaved. For non-interleaved buffers: size == page_size
    //     BufferType buffer_type = BufferType::L1;
    //     TensorMemoryLayout buffer_layout = TensorMemoryLayout::HEIGHT_SHARDED;
    //     ShardSpecBuffer shard_parameters;
    // };

    // ShardSpecBuffer shard_spec = ShardSpecBuffer(
    //     test_config.cores,
    //     {total_size_bytes / test_config.page_size_bytes, 1},
    //     ShardOrientation::ROW_MAJOR,
    //     {test_config.page_size_bytes, 1},
    //     {total_size_bytes / test_config.page_size_bytes, 1});

    // InterleavedBufferConfig interleaved_buffer_config{
    //     .device = device,
    //     .size = total_size_bytes,
    //     .page_size = test_config.page_size_bytes,
    //     .buffer_type = test_config.is_dram ? BufferType::DRAM : BufferType::L1};
    // std::shared_ptr<Buffer> input_buffer;
    // input_buffer = CreateBuffer(interleaved_buffer_config);
    // uint32_t input_buffer_address = input_buffer->address();

    // auto output_buffer = CreateBuffer(interleaved_buffer_config);
    // uint32_t output_buffer_address = output_buffer->address();

    // Input
    // vector<uint32_t> packed_input = create_arange_vector_of_bfloat16(total_size_bytes, true);
    vector<uint32_t> packed_input = generate_packed_uniform_random_vector<uint32_t, bfloat16>(
        -100.0f, 100.0f, total_size_bytes / sizeof(bfloat16), chrono::system_clock::now().time_since_epoch().count());

    // Golden output
    // NOLINTNEXTLINE(performance-unnecessary-copy-initialization)
    vector<uint32_t> packed_golden = packed_input;

    // Compile-time arguments for kernels
    vector<uint32_t> reader_compile_args = {
        (uint32_t)test_config.num_of_transactions,
        (uint32_t)test_config.num_pages,
        (uint32_t)test_config.page_size_bytes,
        (uint32_t)test_config.test_id};
    // tt::tt_metal::TensorAccessorArgs(input_buffer).append_to(reader_compile_args);

    // Kernels
    auto reader_kernel = CreateKernel(
        program,
        "tests/tt_metal/tt_metal/data_movement/dram_sharded/kernels/dram_sharded_read.cpp",
        test_config.cores,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = reader_compile_args});

    uint32_t l1_addr = get_l1_address_and_size(mesh_device, corerange_to_cores(test_config.cores)[0]).base_address;
    std::vector<uint32_t> reader_run_time_args = {input_buffer_address, l1_addr};
    tt::tt_metal::SetRuntimeArgs(program, reader_kernel, test_config.cores, reader_run_time_args);

    // log_info(tt::LogTest, "Input buffer addr: {}, Output buffer addr: {}", input_buffer_address,
    // output_buffer_address);

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
    bool pcc = is_close_packed_vectors<bfloat16, uint32_t>(
        packed_output, packed_golden, [&](const bfloat16& a, const bfloat16& b) { return is_close(a, b); });

    if (!pcc) {
        log_error(tt::LogTest, "PCC Check failed");
        log_info(tt::LogTest, "Golden vector");
        print_vector(unpack_vector<bfloat16, uint32_t>(packed_golden));
        log_info(tt::LogTest, "Output vector");
        print_vector(unpack_vector<bfloat16, uint32_t>(packed_output));
    }

    return pcc;
}
}  // namespace unit_tests::dm::dram_sharded

/* ========== INTERLEAVED DRAM TESTS ========== */

/* ========== Directed Ideal Test Case; Test id = 65 ========== */
TEST_F(GenericMeshDeviceFixture, TensixDataMovementDRAMShardedReadBaseCase) {
    auto mesh_device = get_mesh_device();
    // Physical Constraints
    // auto [flit_size_bytes, max_transmittable_bytes, max_transmittable_flits] =
    //     tt::tt_metal::unit_tests::dm::compute_physical_constraints(mesh_device);
    // Parameters
    uint32_t page_size_bytes = tt::tile_size(DataFormat::Float16_b);
    uint32_t num_pages = 24;
    uint32_t num_of_transactions = 1;

    // Cores
    CoreRange core_range({0, 0}, {0, 0});
    CoreRangeSet core_range_set({core_range});

    // Test config
    unit_tests::dm::dram_sharded::DramShardedConfig test_config = {
        .test_id = 1000,
        .num_of_transactions = num_of_transactions,
        .num_pages = num_pages,
        .page_size_bytes = page_size_bytes,
        .l1_data_format = DataFormat::Float16_b,
        .cores = core_range_set};

    // Run
    EXPECT_TRUE(run_dm(mesh_device, test_config));
}

/* ========== Directed Ideal Test Case; Test id = 65 ========== */
TEST_F(GenericMeshDeviceFixture, TensixDataMovementDRAMShardedReadTileNumbers) {
    auto mesh_device = get_mesh_device();
    // Physical Constraints
    // auto [flit_size_bytes, max_transmittable_bytes, max_transmittable_flits] =
    //     tt::tt_metal::unit_tests::dm::compute_physical_constraints(mesh_device);
    // Parameters
    uint32_t page_size_bytes = tt::tile_size(DataFormat::Float16_b);
    uint32_t max_num_pages = 32;
    uint32_t max_transactions = 256;

    // Cores
    CoreRange core_range({0, 0}, {0, 0});
    CoreRangeSet core_range_set({core_range});

    for (uint32_t num_of_transactions = 1; num_of_transactions <= max_transactions; num_of_transactions *= 4) {
        for (uint32_t num_pages = 1; num_pages <= max_num_pages; num_pages *= 2) {
            // Test config
            unit_tests::dm::dram_sharded::DramShardedConfig test_config = {
                .test_id = 1001,
                .num_of_transactions = num_of_transactions,
                .num_pages = num_pages,
                .page_size_bytes = page_size_bytes,
                .l1_data_format = DataFormat::Float16_b,
                .cores = core_range_set};

            // Run
            EXPECT_TRUE(run_dm(mesh_device, test_config));
        }
    }
}

}  // namespace tt::tt_metal
