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
#include <distributed/mesh_device_impl.hpp>

namespace tt::tt_metal {

using namespace std;
using namespace tt;
using namespace tt::test_utils;

namespace unit_tests::dm::interleaved_page {
// Test config, i.e. test parameters
struct InterleavedConfig {
    uint32_t test_id = 0;
    uint32_t num_of_transactions = 0;
    uint32_t num_pages = 0;
    uint32_t page_size_bytes = 0;
    DataFormat l1_data_format = DataFormat::Invalid;
    CoreRangeSet cores;
    bool is_dram = true;  // else is L1
    bool read_kernel = true;
    bool write_kernel = true;
    bool default_noc = true;
};

/// @brief Does Interleaved buffer --> Reader --> L1 --> Writer --> Interleaved buffer
/// @param mesh_device - MeshDevice to run the test on
/// @param test_config - Configuration of the test -- see struct
/// @return
bool run_dm(const shared_ptr<distributed::MeshDevice>& mesh_device, const InterleavedConfig& test_config) {
    // Get the actual device for this single-device test
    IDevice* device = mesh_device->impl().get_device(0);

    // Program
    Program program = CreateProgram();

    const size_t total_size_bytes = test_config.num_pages * test_config.page_size_bytes;

    InterleavedBufferConfig interleaved_buffer_config{
        .device = device,
        .size = total_size_bytes,
        .page_size = test_config.page_size_bytes,
        .buffer_type = test_config.is_dram ? BufferType::DRAM : BufferType::L1};
    std::shared_ptr<Buffer> input_buffer;
    input_buffer = CreateBuffer(interleaved_buffer_config);
    uint32_t input_buffer_address = input_buffer->address();

    auto output_buffer = CreateBuffer(interleaved_buffer_config);
    uint32_t output_buffer_address = output_buffer->address();

    TT_FATAL(input_buffer_address != output_buffer_address, "Input and output buffer addresses must be different");
    TT_FATAL(test_config.read_kernel || test_config.write_kernel, "At least one kernel must run");

    // Input
    // vector<uint32_t> packed_input = create_arange_vector_of_bfloat16(total_size_bytes, false);
    vector<uint32_t> packed_input = generate_packed_uniform_random_vector<uint32_t, bfloat16>(
        -100.0f, 100.0f, total_size_bytes / sizeof(bfloat16), chrono::system_clock::now().time_since_epoch().count());

    // Golden output
    // NOLINTNEXTLINE(performance-unnecessary-copy-initialization)
    vector<uint32_t> packed_golden = packed_input;

    uint8_t l1_cb_index = CBIndex::c_0;
    bool sync = test_config.read_kernel == test_config.write_kernel;

    // Compile-time arguments for kernels
    vector<uint32_t> reader_compile_args = {
        (uint32_t)test_config.num_of_transactions,
        (uint32_t)test_config.num_pages,
        (uint32_t)test_config.page_size_bytes,
        (uint32_t)l1_cb_index,
        (uint32_t)test_config.test_id,
        (uint32_t)sync,
        (uint32_t)test_config.default_noc};
    tt::tt_metal::TensorAccessorArgs(input_buffer).append_to(reader_compile_args);

    vector<uint32_t> writer_compile_args = {
        (uint32_t)test_config.num_of_transactions,
        (uint32_t)test_config.num_pages,
        (uint32_t)test_config.page_size_bytes,
        (uint32_t)l1_cb_index,
        (uint32_t)test_config.test_id,
        (uint32_t)sync,
        (uint32_t)test_config.default_noc};
    tt::tt_metal::TensorAccessorArgs(output_buffer).append_to(writer_compile_args);

    if (sync) {
        // Create circular buffers
        CircularBufferConfig l1_cb_config =
            CircularBufferConfig(total_size_bytes, {{l1_cb_index, test_config.l1_data_format}})
                .set_page_size(l1_cb_index, test_config.page_size_bytes);
        CreateCircularBuffer(program, test_config.cores, l1_cb_config);
    }

    uint32_t l1_addr = get_l1_address_and_size(mesh_device, corerange_to_cores(test_config.cores)[0]).base_address;
    // log_info(tt::LogTest, "l1 addr: {}, bytes: {}, input buffer addr: {}, output buffer addr: {}", l1_addr,
    // total_size_bytes, input_buffer_address, output_buffer_address);
    if (!test_config.is_dram) {
        TT_FATAL(
            (l1_addr + total_size_bytes < input_buffer_address) || (input_buffer_address + total_size_bytes < l1_addr),
            "L1 buffer overlaps with input buffer");
        TT_FATAL(
            (l1_addr + total_size_bytes < output_buffer_address) ||
                (output_buffer_address + total_size_bytes < l1_addr),
            "L1 buffer overlaps with output buffer");
    }

    // Kernels
    if (test_config.read_kernel) {
        auto reader_kernel = CreateKernel(
            program,
            "tests/tt_metal/tt_metal/data_movement/interleaved/kernels/interleaved_page_read.cpp",
            test_config.cores,
            DataMovementConfig{
                .processor = test_config.default_noc ? DataMovementProcessor::RISCV_1 : DataMovementProcessor::RISCV_0,
                .noc = test_config.default_noc ? NOC::RISCV_1_default : NOC::RISCV_0_default,
                .compile_args = reader_compile_args});

        std::vector<uint32_t> reader_run_time_args = {input_buffer_address, l1_addr};
        tt::tt_metal::SetRuntimeArgs(program, reader_kernel, test_config.cores, reader_run_time_args);
    }

    if (test_config.write_kernel) {
        auto writer_kernel = CreateKernel(
            program,
            "tests/tt_metal/tt_metal/data_movement/interleaved/kernels/interleaved_page_write.cpp",
            test_config.cores,
            DataMovementConfig{
                .processor = test_config.default_noc ? DataMovementProcessor::RISCV_0 : DataMovementProcessor::RISCV_1,
                .noc = test_config.default_noc ? NOC::RISCV_0_default : NOC::RISCV_1_default,
                .compile_args = writer_compile_args});

        std::vector<uint32_t> writer_run_time_args = {output_buffer_address, l1_addr};
        tt::tt_metal::SetRuntimeArgs(program, writer_kernel, test_config.cores, writer_run_time_args);
    }

    // log_info(tt::LogTest, "Input buffer addr: {}, Output buffer addr: {}", input_buffer_address,
    // output_buffer_address);

    // Assign unique id
    log_info(tt::LogTest, "Running Test ID: {}, Run ID: {}", test_config.test_id, unit_tests::dm::runtime_host_id);
    program.set_runtime_id(unit_tests::dm::runtime_host_id++);

    // Launch program and record outputs
    vector<uint32_t> packed_output;

    if (test_config.read_kernel) {
        detail::WriteToBuffer(input_buffer, packed_input);
        if (test_config.is_dram) {
            MetalContext::instance().get_cluster().dram_barrier(device->id());
        } else {
            MetalContext::instance().get_cluster().l1_barrier(device->id());
        }
    } else {
        detail::WriteToDeviceL1(device, corerange_to_cores(test_config.cores)[0], l1_addr, packed_input);
        MetalContext::instance().get_cluster().l1_barrier(device->id());
    }

    auto mesh_workload = distributed::MeshWorkload();
    vector<uint32_t> coord_data = {0, 0};
    auto target_devices =
        distributed::MeshCoordinateRange(distributed::MeshCoordinate(coord_data));  // Single device at (0,0)
    mesh_workload.add_program(target_devices, std::move(program));

    auto& cq = mesh_device->mesh_command_queue();
    distributed::EnqueueMeshWorkload(cq, mesh_workload, false);
    Finish(cq);

    if (test_config.write_kernel) {
        detail::ReadFromBuffer(output_buffer, packed_output);
    } else {
        detail::ReadFromDeviceL1(
            device, corerange_to_cores(test_config.cores)[0], l1_addr, total_size_bytes, packed_output);
    }

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
}  // namespace unit_tests::dm::interleaved_page

/* ========== INTERLEAVED DRAM TESTS ========== */

/* ========== Test case for varying number of pages; Test id = 61 ========== */
TEST_F(GenericMeshDeviceFixture, TensixDataMovementDRAMInterleavedPageNumbers) {
    auto mesh_device = get_mesh_device();

    // Physical Constraints
    auto [flit_size_bytes, max_transmittable_bytes, max_transmittable_flits] =
        tt::tt_metal::unit_tests::dm::compute_physical_constraints(mesh_device);
    // Parameters
    uint32_t max_page_size_bytes = 256 * flit_size_bytes;  // 1 packet = 16 kB for BH, 8 kB for WH
    uint32_t max_num_pages = 256;
    uint32_t num_of_transactions = 1;
    uint32_t num_pages;

    // Cores
    CoreRange core_range({0, 0}, {0, 0});
    CoreRangeSet core_range_set({core_range});

    for (uint32_t pages = 1; pages <= max_num_pages; pages *= 4) {
        if (pages > 16) {
            // avoid writing too large of a memory block at once, prefer to overwrite multiple times
            num_of_transactions = pages / 16;
            num_pages = 16;
        } else {
            num_pages = pages;
        }
        for (uint32_t page_size_bytes = flit_size_bytes; page_size_bytes <= max_page_size_bytes; page_size_bytes *= 2) {
            // Test config
            unit_tests::dm::interleaved_page::InterleavedConfig test_config = {
                .test_id = 61,
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

/* ========== Test case for varying core location; Test id = 62 ========== */
TEST_F(GenericMeshDeviceFixture, TensixDataMovementDRAMInterleavedPageCoreLocations) {
    GTEST_SKIP() << "Skipping test";

    // Parameters
    uint32_t num_pages = 16;
    uint32_t page_size_bytes = 32 * 32 * 2;  // = tile
    uint32_t num_of_transactions = 16;

    // Cores
    auto mesh_device = get_mesh_device();
    auto* device = mesh_device->impl().get_device(0);
    auto grid_size = device->compute_with_storage_grid_size();
    log_info(tt::LogTest, "Grid size x: {}, y: {}", grid_size.x, grid_size.y);
    for (unsigned int x = 0; x < grid_size.x; x++) {
        for (unsigned int y = 0; y < grid_size.y; y++) {
            CoreRangeSet core_range_set({CoreRange({x, y}, {x, y})});
            log_info(tt::LogTest, "Core Location x: {}, y: {}", x, y);
            // Test config
            unit_tests::dm::interleaved_page::InterleavedConfig test_config = {
                .test_id = 62,
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

/* ========== Test noc_async_read_page kernel only; Test id = 63 ========== */
TEST_F(GenericMeshDeviceFixture, TensixDataMovementDRAMInterleavedPageReadNumbers) {
    auto mesh_device = get_mesh_device();
    // Physical Constraints
    auto [flit_size_bytes, max_transmittable_bytes, max_transmittable_flits] =
        tt::tt_metal::unit_tests::dm::compute_physical_constraints(mesh_device);
    // Parameters
    uint32_t max_page_size_bytes = 256 * flit_size_bytes;  // 1 packet = 16 kB for BH, 8 kB for WH
    uint32_t max_num_pages = 256;
    uint32_t num_of_transactions = 1;
    uint32_t num_pages;

    // Cores
    CoreRange core_range({0, 0}, {0, 0});
    CoreRangeSet core_range_set({core_range});

    for (uint32_t pages = 1; pages <= max_num_pages; pages *= 4) {
        if (pages > 16) {
            // avoid writing too large of a memory block at once, prefer to overwrite multiple times
            num_of_transactions = pages / 16;
            num_pages = 16;
        } else {
            num_pages = pages;
        }
        for (uint32_t page_size_bytes = flit_size_bytes; page_size_bytes <= max_page_size_bytes; page_size_bytes *= 2) {
            // Test config
            unit_tests::dm::interleaved_page::InterleavedConfig test_config = {
                .test_id = 63,
                .num_of_transactions = num_of_transactions,
                .num_pages = num_pages,
                .page_size_bytes = page_size_bytes,
                .l1_data_format = DataFormat::Float16_b,
                .cores = core_range_set,
                .is_dram = true,
                .read_kernel = true,
                .write_kernel = false};

            // Run
            EXPECT_TRUE(run_dm(mesh_device, test_config));
        }
    }
}

/* ========== Test noc_async_write_page kernel only; Test id = 64 ========== */
TEST_F(GenericMeshDeviceFixture, TensixDataMovementDRAMInterleavedPageWriteNumbers) {
    auto mesh_device = get_mesh_device();
    // Physical Constraints
    auto [flit_size_bytes, max_transmittable_bytes, max_transmittable_flits] =
        tt::tt_metal::unit_tests::dm::compute_physical_constraints(mesh_device);
    // Parameters
    uint32_t max_page_size_bytes = 256 * flit_size_bytes;  // 1 packet = 16 kB for BH, 8 kB for WH
    uint32_t max_num_pages = 256;
    uint32_t num_of_transactions = 1;
    uint32_t num_pages;

    // Cores
    CoreRange core_range({0, 0}, {0, 0});
    CoreRangeSet core_range_set({core_range});

    for (uint32_t pages = 1; pages <= max_num_pages; pages *= 4) {
        if (pages > 16) {
            // avoid writing too large of a memory block at once, prefer to overwrite multiple times
            num_of_transactions = pages / 16;
            num_pages = 16;
        } else {
            num_pages = pages;
        }
        for (uint32_t page_size_bytes = flit_size_bytes; page_size_bytes <= max_page_size_bytes; page_size_bytes *= 2) {
            // Test config
            unit_tests::dm::interleaved_page::InterleavedConfig test_config = {
                .test_id = 64,
                .num_of_transactions = num_of_transactions,
                .num_pages = num_pages,
                .page_size_bytes = page_size_bytes,
                .l1_data_format = DataFormat::Float16_b,
                .cores = core_range_set,
                .is_dram = true,
                .read_kernel = false,
                .write_kernel = true};

            // Run
            EXPECT_TRUE(run_dm(mesh_device, test_config));
        }
    }
}

/* ========== Directed Ideal Test Case; Test id = 65 ========== */
TEST_F(GenericMeshDeviceFixture, TensixDataMovementDRAMInterleavedPageDirectedIdeal) {
    auto mesh_device = get_mesh_device();
    // Physical Constraints
    auto [flit_size_bytes, max_transmittable_bytes, max_transmittable_flits] =
        tt::tt_metal::unit_tests::dm::compute_physical_constraints(mesh_device);
    // Parameters
    uint32_t page_size_bytes = 256 * flit_size_bytes;  // 1 packet = 16 kB for BH, 8 kB for WH
    uint32_t num_pages = 16;
    uint32_t num_of_transactions = 16;

    // Cores
    CoreRange core_range({0, 0}, {0, 0});
    CoreRangeSet core_range_set({core_range});

    // Test config
    unit_tests::dm::interleaved_page::InterleavedConfig test_config = {
        .test_id = 65,
        .num_of_transactions = num_of_transactions,
        .num_pages = num_pages,
        .page_size_bytes = page_size_bytes,
        .l1_data_format = DataFormat::Float16_b,
        .cores = core_range_set};

    // Run
    EXPECT_TRUE(run_dm(mesh_device, test_config));
}

/* ========== Test noc_async_read_page kernel only with swapped noc; Test id = 72 ========== */
TEST_F(GenericMeshDeviceFixture, TensixDataMovementDRAMInterleavedPageReadNocSwap) {
    GTEST_SKIP() << "Skipping test";

    auto mesh_device = get_mesh_device();
    // Physical Constraints
    auto [flit_size_bytes, max_transmittable_bytes, max_transmittable_flits] =
        tt::tt_metal::unit_tests::dm::compute_physical_constraints(mesh_device);
    // Parameters
    uint32_t max_page_size_bytes = 256 * flit_size_bytes;  // 1 packet = 16 kB for BH, 8 kB for WH
    uint32_t max_num_pages = 256;
    uint32_t num_of_transactions = 1;
    uint32_t num_pages;

    // Cores
    CoreRange core_range({0, 0}, {0, 0});
    CoreRangeSet core_range_set({core_range});

    for (uint32_t pages = 1; pages <= max_num_pages; pages *= 4) {
        if (pages > 16) {
            // avoid writing too large of a memory block at once, prefer to overwrite multiple times
            num_of_transactions = pages / 16;
            num_pages = 16;
        } else {
            num_pages = pages;
        }
        for (uint32_t page_size_bytes = flit_size_bytes; page_size_bytes <= max_page_size_bytes; page_size_bytes *= 2) {
            // Test config
            unit_tests::dm::interleaved_page::InterleavedConfig test_config = {
                .test_id = 72,
                .num_of_transactions = num_of_transactions,
                .num_pages = num_pages,
                .page_size_bytes = page_size_bytes,
                .l1_data_format = DataFormat::Float16_b,
                .cores = core_range_set,
                .is_dram = true,
                .read_kernel = true,
                .write_kernel = false,
                .default_noc = false};

            // Run
            EXPECT_TRUE(run_dm(mesh_device, test_config));
        }
    }
}

/* ========== Test noc_async_write_page kernel only with swapped noc; Test id = 73 ========== */
TEST_F(GenericMeshDeviceFixture, TensixDataMovementDRAMInterleavedPageWriteNocSwap) {
    GTEST_SKIP() << "Skipping test";

    auto mesh_device = get_mesh_device();
    // Physical Constraints
    auto [flit_size_bytes, max_transmittable_bytes, max_transmittable_flits] =
        tt::tt_metal::unit_tests::dm::compute_physical_constraints(mesh_device);
    // Parameters
    uint32_t max_page_size_bytes = 256 * flit_size_bytes;  // 1 packet = 16 kB for BH, 8 kB for WH
    uint32_t max_num_pages = 256;
    uint32_t num_of_transactions = 1;
    uint32_t num_pages;

    // Cores
    CoreRange core_range({0, 0}, {0, 0});
    CoreRangeSet core_range_set({core_range});

    for (uint32_t pages = 1; pages <= max_num_pages; pages *= 4) {
        if (pages > 16) {
            // avoid writing too large of a memory block at once, prefer to overwrite multiple times
            num_of_transactions = pages / 16;
            num_pages = 16;
        } else {
            num_pages = pages;
        }
        for (uint32_t page_size_bytes = flit_size_bytes; page_size_bytes <= max_page_size_bytes; page_size_bytes *= 2) {
            // Test config
            unit_tests::dm::interleaved_page::InterleavedConfig test_config = {
                .test_id = 73,
                .num_of_transactions = num_of_transactions,
                .num_pages = num_pages,
                .page_size_bytes = page_size_bytes,
                .l1_data_format = DataFormat::Float16_b,
                .cores = core_range_set,
                .is_dram = true,
                .read_kernel = false,
                .write_kernel = true,
                .default_noc = false};

            // Run
            EXPECT_TRUE(run_dm(mesh_device, test_config));
        }
    }
}

/* ========== INTERLEAVED L1 TESTS ========== */

/* ========== Test case for varying number of pages using interleaved L1; Test id = 66 ========== */
TEST_F(GenericMeshDeviceFixture, TensixDataMovementL1InterleavedPageNumbers) {
    auto mesh_device = get_mesh_device();
    // Physical Constraints
    auto [flit_size_bytes, max_transmittable_bytes, max_transmittable_flits] =
        tt::tt_metal::unit_tests::dm::compute_physical_constraints(mesh_device);
    // Parameters
    uint32_t max_page_size_bytes = 256 * flit_size_bytes;  // 1 packet = 16 kB for BH, 8 kB for WH
    uint32_t max_num_pages = 256;
    uint32_t num_of_transactions = 1;
    uint32_t num_pages;

    // Cores
    CoreRange core_range({0, 0}, {0, 0});
    CoreRangeSet core_range_set({core_range});

    for (uint32_t pages = 1; pages <= max_num_pages; pages *= 4) {
        if (pages > 16) {
            // avoid writing too large of a memory block at once, prefer to overwrite multiple times
            num_of_transactions = pages / 16;
            num_pages = 16;
        } else {
            num_pages = pages;
        }
        for (uint32_t page_size_bytes = flit_size_bytes; page_size_bytes <= max_page_size_bytes; page_size_bytes *= 2) {
            // Test config
            unit_tests::dm::interleaved_page::InterleavedConfig test_config = {
                .test_id = 66,
                .num_of_transactions = num_of_transactions,
                .num_pages = num_pages,
                .page_size_bytes = page_size_bytes,
                .l1_data_format = DataFormat::Float16_b,
                .cores = core_range_set,
                .is_dram = false};

            // Run
            EXPECT_TRUE(run_dm(mesh_device, test_config));
        }
    }
}

/* ========== Test case for varying core location; Test id = 67 ========== */
TEST_F(GenericMeshDeviceFixture, TensixDataMovementL1InterleavedPageCoreLocations) {
    GTEST_SKIP() << "Skipping test";

    // Parameters
    uint32_t num_pages = 16;
    uint32_t page_size_bytes = 32 * 32 * 2;  // = tile

    // Cores
    auto mesh_device = get_mesh_device();
    auto* device = mesh_device->impl().get_device(0);
    auto grid_size = device->compute_with_storage_grid_size();
    log_info(tt::LogTest, "Grid size x: {}, y: {}", grid_size.x, grid_size.y);
    for (unsigned int x = 0; x < grid_size.x; x++) {
        for (unsigned int y = 0; y < grid_size.y; y++) {
            CoreRangeSet core_range_set({CoreRange({x, y}, {x, y})});
            log_info(tt::LogTest, "Core Location x: {}, y: {}", x, y);
            // Test config
            unit_tests::dm::interleaved_page::InterleavedConfig test_config = {
                .test_id = 67,
                .num_of_transactions = 16,
                .num_pages = num_pages,
                .page_size_bytes = page_size_bytes,
                .l1_data_format = DataFormat::Float16_b,
                .cores = core_range_set,
                .is_dram = false};

            // Run
            EXPECT_TRUE(run_dm(mesh_device, test_config));
        }
    }
}

/* ========== Test noc_async_read_page only; Test id = 68 ========== */
TEST_F(GenericMeshDeviceFixture, TensixDataMovementL1InterleavedPageReadNumbers) {
    auto mesh_device = get_mesh_device();
    // Physical Constraints
    auto [flit_size_bytes, max_transmittable_bytes, max_transmittable_flits] =
        tt::tt_metal::unit_tests::dm::compute_physical_constraints(mesh_device);
    // Parameters
    uint32_t max_page_size_bytes = 256 * flit_size_bytes;  // 1 packet = 16 kB for BH, 8 kB for WH
    uint32_t max_num_pages = 256;
    uint32_t num_of_transactions = 1;
    uint32_t num_pages;

    // Cores
    CoreRange core_range({0, 0}, {0, 0});
    CoreRangeSet core_range_set({core_range});

    for (uint32_t pages = 1; pages <= max_num_pages; pages *= 4) {
        if (pages > 16) {
            // avoid writing too large of a memory block at once, prefer to overwrite multiple times
            num_of_transactions = pages / 16;
            num_pages = 16;
        } else {
            num_pages = pages;
        }
        for (uint32_t page_size_bytes = flit_size_bytes; page_size_bytes <= max_page_size_bytes; page_size_bytes *= 2) {
            // Test config
            unit_tests::dm::interleaved_page::InterleavedConfig test_config = {
                .test_id = 68,
                .num_of_transactions = num_of_transactions,
                .num_pages = num_pages,
                .page_size_bytes = page_size_bytes,
                .l1_data_format = DataFormat::Float16_b,
                .cores = core_range_set,
                .is_dram = false,
                .read_kernel = true,
                .write_kernel = false};

            // Run
            EXPECT_TRUE(run_dm(mesh_device, test_config));
        }
    }
}
/* ========== Test noc_async_write_page only; Test id = 69 ========== */
TEST_F(GenericMeshDeviceFixture, TensixDataMovementL1InterleavedPageWriteNumbers) {
    auto mesh_device = get_mesh_device();
    // Physical Constraints
    auto [flit_size_bytes, max_transmittable_bytes, max_transmittable_flits] =
        tt::tt_metal::unit_tests::dm::compute_physical_constraints(mesh_device);
    // Parameters
    uint32_t max_page_size_bytes = 256 * flit_size_bytes;  // 1 packet = 16 kB for BH, 8 kB for WH
    uint32_t max_num_pages = 256;
    uint32_t num_of_transactions = 1;
    uint32_t num_pages;

    // Cores
    CoreRange core_range({0, 0}, {0, 0});
    CoreRangeSet core_range_set({core_range});

    for (uint32_t pages = 1; pages <= max_num_pages; pages *= 4) {
        if (pages > 16) {
            // avoid writing too large of a memory block at once, prefer to overwrite multiple times
            num_of_transactions = pages / 16;
            num_pages = 16;
        } else {
            num_pages = pages;
        }
        for (uint32_t page_size_bytes = flit_size_bytes; page_size_bytes <= max_page_size_bytes; page_size_bytes *= 2) {
            // Test config
            unit_tests::dm::interleaved_page::InterleavedConfig test_config = {
                .test_id = 69,
                .num_of_transactions = num_of_transactions,
                .num_pages = num_pages,
                .page_size_bytes = page_size_bytes,
                .l1_data_format = DataFormat::Float16_b,
                .cores = core_range_set,
                .is_dram = false,
                .read_kernel = false,
                .write_kernel = true};

            // Run
            EXPECT_TRUE(run_dm(mesh_device, test_config));
        }
    }
}

/* ========== Directed Ideal Test Case; Test id = 71 ========== */
TEST_F(GenericMeshDeviceFixture, TensixDataMovementL1InterleavedPageDirectedIdeal) {
    auto mesh_device = get_mesh_device();
    // Physical Constraints
    auto [flit_size_bytes, max_transmittable_bytes, max_transmittable_flits] =
        tt::tt_metal::unit_tests::dm::compute_physical_constraints(mesh_device);
    // Parameters
    uint32_t page_size_bytes = 256 * flit_size_bytes;  // 1 packet = 16 kB for BH, 8 kB for WH
    uint32_t num_pages = 16;
    uint32_t num_of_transactions = 16;

    // Cores
    CoreRange core_range({0, 0}, {0, 0});
    CoreRangeSet core_range_set({core_range});

    // Test config
    unit_tests::dm::interleaved_page::InterleavedConfig test_config = {
        .test_id = 71,
        .num_of_transactions = num_of_transactions,
        .num_pages = num_pages,
        .page_size_bytes = page_size_bytes,
        .l1_data_format = DataFormat::Float16_b,
        .cores = core_range_set,
        .is_dram = false};

    // Run
    EXPECT_TRUE(run_dm(mesh_device, test_config));
}

/* ========== Test noc_async_read_page only with swapped noc; Test id = 74 ========== */
TEST_F(GenericMeshDeviceFixture, TensixDataMovementL1InterleavedPageReadNocSwap) {
    GTEST_SKIP() << "Skipping test";

    auto mesh_device = get_mesh_device();
    // Physical Constraints
    auto [flit_size_bytes, max_transmittable_bytes, max_transmittable_flits] =
        tt::tt_metal::unit_tests::dm::compute_physical_constraints(mesh_device);
    // Parameters
    uint32_t max_page_size_bytes = 256 * flit_size_bytes;  // 1 packet = 16 kB for BH, 8 kB for WH
    uint32_t max_num_pages = 256;
    uint32_t num_of_transactions = 1;
    uint32_t num_pages;

    // Cores
    CoreRange core_range({0, 0}, {0, 0});
    CoreRangeSet core_range_set({core_range});

    for (uint32_t pages = 1; pages <= max_num_pages; pages *= 4) {
        if (pages > 16) {
            // avoid writing too large of a memory block at once, prefer to overwrite multiple times
            num_of_transactions = pages / 16;
            num_pages = 16;
        } else {
            num_pages = pages;
        }
        for (uint32_t page_size_bytes = flit_size_bytes; page_size_bytes <= max_page_size_bytes; page_size_bytes *= 2) {
            // Test config
            unit_tests::dm::interleaved_page::InterleavedConfig test_config = {
                .test_id = 74,
                .num_of_transactions = num_of_transactions,
                .num_pages = num_pages,
                .page_size_bytes = page_size_bytes,
                .l1_data_format = DataFormat::Float16_b,
                .cores = core_range_set,
                .is_dram = false,
                .read_kernel = true,
                .write_kernel = false,
                .default_noc = false};

            // Run
            EXPECT_TRUE(run_dm(mesh_device, test_config));
        }
    }
}
/* ========== Test noc_async_write_page only; Test id = 75 ========== */
TEST_F(GenericMeshDeviceFixture, TensixDataMovementL1InterleavedPageWriteNocSwap) {
    GTEST_SKIP() << "Skipping test";

    auto mesh_device = get_mesh_device();
    // Physical Constraints
    auto [flit_size_bytes, max_transmittable_bytes, max_transmittable_flits] =
        tt::tt_metal::unit_tests::dm::compute_physical_constraints(mesh_device);
    // Parameters
    uint32_t max_page_size_bytes = 256 * flit_size_bytes;  // 1 packet = 16 kB for BH, 8 kB for WH
    uint32_t max_num_pages = 256;
    uint32_t num_of_transactions = 1;
    uint32_t num_pages;

    // Cores
    CoreRange core_range({0, 0}, {0, 0});
    CoreRangeSet core_range_set({core_range});

    for (uint32_t pages = 1; pages <= max_num_pages; pages *= 4) {
        if (pages > 16) {
            // avoid writing too large of a memory block at once, prefer to overwrite multiple times
            num_of_transactions = pages / 16;
            num_pages = 16;
        } else {
            num_pages = pages;
        }
        for (uint32_t page_size_bytes = flit_size_bytes; page_size_bytes <= max_page_size_bytes; page_size_bytes *= 2) {
            // Test config
            unit_tests::dm::interleaved_page::InterleavedConfig test_config = {
                .test_id = 75,
                .num_of_transactions = num_of_transactions,
                .num_pages = num_pages,
                .page_size_bytes = page_size_bytes,
                .l1_data_format = DataFormat::Float16_b,
                .cores = core_range_set,
                .is_dram = false,
                .read_kernel = false,
                .write_kernel = true,
                .default_noc = false};

            // Run
            EXPECT_TRUE(run_dm(mesh_device, test_config));
        }
    }
}

}  // namespace tt::tt_metal
