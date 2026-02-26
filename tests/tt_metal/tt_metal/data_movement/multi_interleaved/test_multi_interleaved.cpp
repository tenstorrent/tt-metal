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

namespace unit_tests::dm::multi_interleaved {
// Test config, i.e. test parameters
struct MultiInterleavedConfig {
    uint32_t test_id = 0;
    uint32_t num_of_transactions = 0;
    uint32_t num_pages = 0;
    uint32_t page_size_bytes = 0;
    DataFormat l1_data_format = DataFormat::Invalid;
    CoreRangeSet cores;
    bool read_kernel = true;
    bool write_kernel = true;
};

/// @brief Does Interleaved buffer --> Reader --> L1 --> Writer --> Interleaved buffer
/// @param mesh_device
/// @param test_config - Configuration of the test -- see struct
/// @return
bool run_dm(const shared_ptr<distributed::MeshDevice>& mesh_device, const MultiInterleavedConfig& test_config) {
    log_info(
        tt::LogTest,
        "num transaction {}, num pages: {}, page size bytes: {}",
        test_config.num_of_transactions,
        test_config.num_pages,
        test_config.page_size_bytes);

    // Get the actual device for this single-device test
    IDevice* device = mesh_device->impl().get_device(0);

    // Program
    Program program = CreateProgram();

    const uint32_t num_cores = test_config.cores.num_cores();
    const size_t per_core_size_bytes = test_config.num_pages * test_config.page_size_bytes;
    const size_t total_buffer_size_bytes = num_cores * per_core_size_bytes;

    InterleavedBufferConfig interleaved_buffer_config{
        .device = device,
        .size = total_buffer_size_bytes,
        .page_size = test_config.page_size_bytes,
        .buffer_type = BufferType::DRAM};
    std::shared_ptr<Buffer> input_buffer;
    input_buffer = CreateBuffer(interleaved_buffer_config);
    uint32_t input_buffer_address = input_buffer->address();

    auto output_buffer = CreateBuffer(interleaved_buffer_config);
    uint32_t output_buffer_address = output_buffer->address();

    TT_FATAL(input_buffer_address != output_buffer_address, "Input and output buffer addresses must be different");
    TT_FATAL(test_config.read_kernel || test_config.write_kernel, "At least one kernel must run");

    // Input - generate data for all cores
    vector<uint32_t> packed_input = generate_packed_uniform_random_vector<uint32_t, bfloat16>(
        -100.0f,
        100.0f,
        total_buffer_size_bytes / sizeof(bfloat16),
        chrono::system_clock::now().time_since_epoch().count());

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
        (uint32_t)sync};

    vector<uint32_t> writer_compile_args = {
        (uint32_t)test_config.num_of_transactions,
        (uint32_t)test_config.num_pages,
        (uint32_t)test_config.page_size_bytes,
        (uint32_t)l1_cb_index,
        (uint32_t)test_config.test_id,
        (uint32_t)sync};

    if (sync) {
        // Create circular buffers - each core only needs space for its own data
        CircularBufferConfig l1_cb_config =
            CircularBufferConfig(per_core_size_bytes, {{l1_cb_index, test_config.l1_data_format}})
                .set_page_size(l1_cb_index, test_config.page_size_bytes);
        CreateCircularBuffer(program, test_config.cores, l1_cb_config);
    }

    std::vector<uint32_t> l1_addrs;
    std::vector<CoreCoord> core_list = corerange_to_cores(test_config.cores);
    constexpr uint32_t noc_l1_alignment = 16u;
    constexpr uint32_t barrier_scratch_bytes = 2u * noc_l1_alignment;
    const uint32_t required_l1_bytes = per_core_size_bytes + barrier_scratch_bytes;
    for (auto& core : core_list) {
        auto [l1_addr, l1_size] = get_l1_address_and_size(mesh_device, core);
        TT_FATAL(
            l1_size >= required_l1_bytes,
            "L1 size {} must be >= per_core_size_bytes {} + barrier scratch {} (total {})",
            l1_size,
            per_core_size_bytes,
            barrier_scratch_bytes,
            required_l1_bytes);
        l1_addrs.push_back(l1_addr);
    }

    // ===== Barrier synchronization setup =====
    // CreateSemaphore allocates semaphores on all specified cores (same ID maps to same L1 offset).
    // We only use the coordinator's semaphore - all cores increment it via NOC and poll until num_cores.
    // Creating on all cores ensures get_semaphore(id) works correctly on every core.
    CoreCoord coordinator_core = core_list[0];
    CoreCoord coordinator_phys = device->worker_core_from_logical_core(coordinator_core);

    uint32_t reader_barrier_sem_id = 0;
    uint32_t writer_barrier_sem_id = 0;

    if (test_config.read_kernel) {
        reader_barrier_sem_id = CreateSemaphore(program, test_config.cores, 0);
    }
    if (test_config.write_kernel) {
        writer_barrier_sem_id = CreateSemaphore(program, test_config.cores, 0);
    }

    // Kernels
    if (test_config.read_kernel) {
        TensorAccessorArgs(*input_buffer).append_to(reader_compile_args);
        auto reader_kernel = CreateKernel(
            program,
            "tests/tt_metal/tt_metal/data_movement/multi_interleaved/kernels/multi_interleaved_read.cpp",
            test_config.cores,
            DataMovementConfig{
                .processor = DataMovementProcessor::RISCV_0,
                .noc = NOC::RISCV_0_default,  // NOC0 should be used when reading from DRAM into Tensix core L1
                .compile_args = reader_compile_args});

        for (size_t i = 0; i < num_cores; ++i) {
            // Each core reads from different pages to distribute across DRAM banks
            uint32_t page_offset = i * test_config.num_pages;
            // Use the end of L1 data buffer as scratch space for polling
            uint32_t local_barrier_addr = l1_addrs[i] + per_core_size_bytes;

            std::vector<uint32_t> reader_run_time_args = {
                input_buffer_address,
                l1_addrs[i],
                page_offset,
                reader_barrier_sem_id,  // Semaphore ID, kernel will call get_semaphore() to get address
                coordinator_phys.x,
                coordinator_phys.y,
                num_cores,
                local_barrier_addr};
            tt::tt_metal::SetRuntimeArgs(program, reader_kernel, core_list[i], reader_run_time_args);
        }
    }

    if (test_config.write_kernel) {
        TensorAccessorArgs(*output_buffer).append_to(writer_compile_args);
        auto writer_kernel = CreateKernel(
            program,
            "tests/tt_metal/tt_metal/data_movement/multi_interleaved/kernels/multi_interleaved_write.cpp",
            test_config.cores,
            DataMovementConfig{
                .processor = DataMovementProcessor::RISCV_1,
                .noc = NOC::RISCV_1_default,  // NOC1 should be used when writing from Tensix core L1 into DRAM
                .compile_args = writer_compile_args});

        for (size_t i = 0; i < num_cores; ++i) {
            // Each core writes to different pages to distribute across DRAM banks
            uint32_t page_offset = i * test_config.num_pages;
            // Use the end of L1 data buffer as scratch space for polling
            uint32_t local_barrier_addr = l1_addrs[i] + per_core_size_bytes + noc_l1_alignment;

            std::vector<uint32_t> writer_run_time_args = {
                output_buffer_address,
                l1_addrs[i],
                page_offset,
                writer_barrier_sem_id,  // Semaphore ID, kernel will call get_semaphore() to get address
                coordinator_phys.x,
                coordinator_phys.y,
                num_cores,
                local_barrier_addr};
            tt::tt_metal::SetRuntimeArgs(program, writer_kernel, core_list[i], writer_run_time_args);
        }
    }

    // Assign unique id
    log_info(tt::LogTest, "Running Test ID: {}, Run ID: {}", test_config.test_id, unit_tests::dm::runtime_host_id);
    program.set_runtime_id(unit_tests::dm::runtime_host_id++);

    // Launch program and record outputs

    if (test_config.read_kernel) {
        detail::WriteToBuffer(input_buffer, packed_input);
        MetalContext::instance().get_cluster().dram_barrier(device->id());
    } else {
        // If not reading, write each core's slice to L1 directly
        const size_t per_core_words = per_core_size_bytes / sizeof(uint32_t);
        for (size_t i = 0; i < num_cores; ++i) {
            vector<uint32_t> core_input(
                packed_input.begin() + i * per_core_words, packed_input.begin() + (i + 1) * per_core_words);
            detail::WriteToDeviceL1(device, core_list[i], l1_addrs[i], core_input);
        }
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

    vector<uint32_t> packed_output;
    bool is_equal = false;

    if (test_config.write_kernel) {
        detail::ReadFromBuffer(output_buffer, packed_output);
        is_equal = (packed_output == packed_golden);
        if (!is_equal) {
            log_error(tt::LogTest, "Equality Check failed");
            log_info(tt::LogTest, "Golden vector");
            print_vector<uint32_t>(packed_golden);
            log_info(tt::LogTest, "Output vector");
            print_vector<uint32_t>(packed_output);
        }
    } else {
        // Each core reads different pages, verify each core's L1 against its slice
        const size_t per_core_words = per_core_size_bytes / sizeof(uint32_t);
        for (size_t i = 0; i < num_cores; ++i) {
            detail::ReadFromDeviceL1(device, core_list[i], l1_addrs[i], per_core_size_bytes, packed_output);
            vector<uint32_t> core_golden(
                packed_golden.begin() + i * per_core_words, packed_golden.begin() + (i + 1) * per_core_words);
            is_equal = (packed_output == core_golden);
            if (!is_equal) {
                log_error(tt::LogTest, "Equality Check failed for core {}", i);
                log_info(tt::LogTest, "Golden vector for core {}", i);
                print_vector<uint32_t>(core_golden);
                log_info(tt::LogTest, "Output vector");
                print_vector<uint32_t>(packed_output);
                return is_equal;
            }
        }
    }
    return is_equal;
}

void directed_ideal_test(
    const shared_ptr<distributed::MeshDevice>& mesh_device,
    uint32_t test_case_id,
    CoreCoord mst_start_coord,
    CoreCoord mst_grid_size,
    bool read,
    bool write) {
    // Physical Constraints
    auto [flit_size_bytes, max_transmittable_bytes, max_transmittable_flits] =
        tt::tt_metal::unit_tests::dm::compute_physical_constraints(mesh_device);

    // Parameters
    uint32_t page_size_bytes = 256 * flit_size_bytes;  // 1 packet = 16 kB for BH, 8 kB for WH
    uint32_t num_pages = 16;
    uint32_t num_of_transactions = 16;

    // Cores
    CoreCoord mst_end_coord =
        CoreCoord(mst_start_coord.x + mst_grid_size.x - 1, mst_start_coord.y + mst_grid_size.y - 1);
    CoreRangeSet core_range_set({CoreRange(mst_start_coord, mst_end_coord)});

    // Test config
    unit_tests::dm::multi_interleaved::MultiInterleavedConfig test_config = {
        .test_id = test_case_id,
        .num_of_transactions = num_of_transactions,
        .num_pages = num_pages,
        .page_size_bytes = page_size_bytes,
        .l1_data_format = DataFormat::Float16_b,
        .cores = core_range_set,
        .read_kernel = read,
        .write_kernel = write};

    // Run
    EXPECT_TRUE(run_dm(mesh_device, test_config));
}

void packet_sizes_test(
    const shared_ptr<distributed::MeshDevice>& mesh_device,
    uint32_t test_case_id,
    CoreCoord mst_start_coord,
    CoreCoord mst_grid_size,
    bool read,
    bool write) {
    // Physical Constraints
    auto [flit_size_bytes, max_transmittable_bytes, max_transmittable_flits] =
        tt::tt_metal::unit_tests::dm::compute_physical_constraints(mesh_device);

    // Parameters
    uint32_t max_page_size_bytes = 256 * flit_size_bytes;  // 1 packet = 16 kB for BH, 8 kB for WH
    uint32_t max_num_pages = 256;
    uint32_t num_of_transactions = 1;
    uint32_t num_pages;

    // Cores
    CoreCoord mst_end_coord =
        CoreCoord(mst_start_coord.x + mst_grid_size.x - 1, mst_start_coord.y + mst_grid_size.y - 1);
    CoreRangeSet core_range_set({CoreRange(mst_start_coord, mst_end_coord)});

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
            unit_tests::dm::multi_interleaved::MultiInterleavedConfig test_config = {
                .test_id = test_case_id,
                .num_of_transactions = num_of_transactions,
                .num_pages = num_pages,
                .page_size_bytes = page_size_bytes,
                .l1_data_format = DataFormat::Float16_b,
                .cores = core_range_set,
                .read_kernel = read,
                .write_kernel = write};

            // Run
            EXPECT_TRUE(run_dm(mesh_device, test_config));
        }
    }
}

}  // namespace unit_tests::dm::multi_interleaved

/* ========== Full grid directed ideal ========== */
TEST_F(GenericMeshDeviceFixture, TensixDataMovementMultiInterleavedDirectedIdeal) {
    auto mesh_device = get_mesh_device();
    auto* device = mesh_device->impl().get_device(0);

    uint32_t test_case_id = 110;
    CoreCoord mst_start_coord = {0, 0};
    CoreCoord mst_grid_size = {device->compute_with_storage_grid_size().x, device->compute_with_storage_grid_size().y};
    unit_tests::dm::multi_interleaved::directed_ideal_test(
        mesh_device, test_case_id, mst_start_coord, mst_grid_size, true, true);
}

/* ========== Full grid packet sizes sweep ========== */
TEST_F(GenericMeshDeviceFixture, TensixDataMovementMultiInterleavedSizes) {
    auto mesh_device = get_mesh_device();
    auto* device = mesh_device->impl().get_device(0);

    uint32_t test_case_id = 111;
    CoreCoord mst_start_coord = {0, 0};
    CoreCoord mst_grid_size = {device->compute_with_storage_grid_size().x, device->compute_with_storage_grid_size().y};
    unit_tests::dm::multi_interleaved::packet_sizes_test(
        mesh_device, test_case_id, mst_start_coord, mst_grid_size, true, true);
}

/* ========== Full grid read kernel directed ideal ========== */
TEST_F(GenericMeshDeviceFixture, TensixDataMovementMultiInterleavedReadDirectedIdeal) {
    auto mesh_device = get_mesh_device();
    auto* device = mesh_device->impl().get_device(0);

    uint32_t test_case_id = 112;
    CoreCoord mst_start_coord = {0, 0};
    CoreCoord mst_grid_size = {device->compute_with_storage_grid_size().x, device->compute_with_storage_grid_size().y};
    unit_tests::dm::multi_interleaved::directed_ideal_test(
        mesh_device, test_case_id, mst_start_coord, mst_grid_size, true, false);
}

/* ========== Full grid read kernel packet sizes sweep ========== */
TEST_F(GenericMeshDeviceFixture, TensixDataMovementMultiInterleavedReadSizes) {
    auto mesh_device = get_mesh_device();
    auto* device = mesh_device->impl().get_device(0);

    uint32_t test_case_id = 113;
    CoreCoord mst_start_coord = {0, 0};
    CoreCoord mst_grid_size = {device->compute_with_storage_grid_size().x, device->compute_with_storage_grid_size().y};
    unit_tests::dm::multi_interleaved::packet_sizes_test(
        mesh_device, test_case_id, mst_start_coord, mst_grid_size, true, false);
}

/* ========== Full grid write kernel directed ideal ========== */
TEST_F(GenericMeshDeviceFixture, TensixDataMovementMultiInterleavedWriteDirectedIdeal) {
    auto mesh_device = get_mesh_device();
    auto* device = mesh_device->impl().get_device(0);

    uint32_t test_case_id = 114;
    CoreCoord mst_start_coord = {0, 0};
    CoreCoord mst_grid_size = {device->compute_with_storage_grid_size().x, device->compute_with_storage_grid_size().y};
    unit_tests::dm::multi_interleaved::directed_ideal_test(
        mesh_device, test_case_id, mst_start_coord, mst_grid_size, false, true);
}

/* ========== Full grid write kernel packet sizes sweep ========== */
TEST_F(GenericMeshDeviceFixture, TensixDataMovementMultiInterleavedWriteSizes) {
    auto mesh_device = get_mesh_device();
    auto* device = mesh_device->impl().get_device(0);

    uint32_t test_case_id = 115;
    CoreCoord mst_start_coord = {0, 0};
    CoreCoord mst_grid_size = {device->compute_with_storage_grid_size().x, device->compute_with_storage_grid_size().y};
    unit_tests::dm::multi_interleaved::packet_sizes_test(
        mesh_device, test_case_id, mst_start_coord, mst_grid_size, false, true);
}

/* ========== 2x2 CORE TESTS ========== */

TEST_F(GenericMeshDeviceFixture, TensixDataMovement2x2MultiInterleavedDirectedIdeal) {
    uint32_t test_case_id = 116;
    CoreCoord mst_start_coord = {0, 0};
    CoreCoord mst_grid_size = {2, 2};
    unit_tests::dm::multi_interleaved::directed_ideal_test(
        get_mesh_device(), test_case_id, mst_start_coord, mst_grid_size, true, true);
}

TEST_F(GenericMeshDeviceFixture, TensixDataMovement2x2MultiInterleavedSizes) {
    uint32_t test_case_id = 117;
    CoreCoord mst_start_coord = {0, 0};
    CoreCoord mst_grid_size = {2, 2};
    unit_tests::dm::multi_interleaved::packet_sizes_test(
        get_mesh_device(), test_case_id, mst_start_coord, mst_grid_size, true, true);
}

TEST_F(GenericMeshDeviceFixture, TensixDataMovement2x2MultiInterleavedReadDirectedIdeal) {
    uint32_t test_case_id = 118;
    CoreCoord mst_start_coord = {0, 0};
    CoreCoord mst_grid_size = {2, 2};
    unit_tests::dm::multi_interleaved::directed_ideal_test(
        get_mesh_device(), test_case_id, mst_start_coord, mst_grid_size, true, false);
}

TEST_F(GenericMeshDeviceFixture, TensixDataMovement2x2MultiInterleavedReadSizes) {
    uint32_t test_case_id = 119;
    CoreCoord mst_start_coord = {0, 0};
    CoreCoord mst_grid_size = {2, 2};
    unit_tests::dm::multi_interleaved::packet_sizes_test(
        get_mesh_device(), test_case_id, mst_start_coord, mst_grid_size, true, false);
}

TEST_F(GenericMeshDeviceFixture, TensixDataMovement2x2MultiInterleavedWriteDirectedIdeal) {
    uint32_t test_case_id = 120;
    CoreCoord mst_start_coord = {0, 0};
    CoreCoord mst_grid_size = {2, 2};
    unit_tests::dm::multi_interleaved::directed_ideal_test(
        get_mesh_device(), test_case_id, mst_start_coord, mst_grid_size, false, true);
}

TEST_F(GenericMeshDeviceFixture, TensixDataMovement2x2MultiInterleavedWriteSizes) {
    uint32_t test_case_id = 121;
    CoreCoord mst_start_coord = {0, 0};
    CoreCoord mst_grid_size = {2, 2};
    unit_tests::dm::multi_interleaved::packet_sizes_test(
        get_mesh_device(), test_case_id, mst_start_coord, mst_grid_size, false, true);
}

/* ========== 6x6 CORE TESTS ========== */

TEST_F(GenericMeshDeviceFixture, TensixDataMovement6x6MultiInterleavedDirectedIdeal) {
    uint32_t test_case_id = 122;
    CoreCoord mst_start_coord = {0, 0};
    CoreCoord mst_grid_size = {6, 6};
    unit_tests::dm::multi_interleaved::directed_ideal_test(
        get_mesh_device(), test_case_id, mst_start_coord, mst_grid_size, true, true);
}

TEST_F(GenericMeshDeviceFixture, TensixDataMovement6x6MultiInterleavedSizes) {
    uint32_t test_case_id = 123;
    CoreCoord mst_start_coord = {0, 0};
    CoreCoord mst_grid_size = {6, 6};
    unit_tests::dm::multi_interleaved::packet_sizes_test(
        get_mesh_device(), test_case_id, mst_start_coord, mst_grid_size, true, true);
}

TEST_F(GenericMeshDeviceFixture, TensixDataMovement6x6MultiInterleavedReadDirectedIdeal) {
    uint32_t test_case_id = 124;
    CoreCoord mst_start_coord = {0, 0};
    CoreCoord mst_grid_size = {6, 6};
    unit_tests::dm::multi_interleaved::directed_ideal_test(
        get_mesh_device(), test_case_id, mst_start_coord, mst_grid_size, true, false);
}

TEST_F(GenericMeshDeviceFixture, TensixDataMovement6x6MultiInterleavedReadSizes) {
    uint32_t test_case_id = 125;
    CoreCoord mst_start_coord = {0, 0};
    CoreCoord mst_grid_size = {6, 6};
    unit_tests::dm::multi_interleaved::packet_sizes_test(
        get_mesh_device(), test_case_id, mst_start_coord, mst_grid_size, true, false);
}

TEST_F(GenericMeshDeviceFixture, TensixDataMovement6x6MultiInterleavedWriteDirectedIdeal) {
    uint32_t test_case_id = 126;
    CoreCoord mst_start_coord = {0, 0};
    CoreCoord mst_grid_size = {6, 6};
    unit_tests::dm::multi_interleaved::directed_ideal_test(
        get_mesh_device(), test_case_id, mst_start_coord, mst_grid_size, false, true);
}

TEST_F(GenericMeshDeviceFixture, TensixDataMovement6x6MultiInterleavedWriteSizes) {
    uint32_t test_case_id = 127;
    CoreCoord mst_start_coord = {0, 0};
    CoreCoord mst_grid_size = {6, 6};
    unit_tests::dm::multi_interleaved::packet_sizes_test(
        get_mesh_device(), test_case_id, mst_start_coord, mst_grid_size, false, true);
}

}  // namespace tt::tt_metal
