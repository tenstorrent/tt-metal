// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "multi_device_fixture.hpp"
#include "device_fixture.hpp"
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/experimental/metal2_host_api/program.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/kernel_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/node_coord.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include <tt-metalium/experimental/metal2_host_api/semaphore_spec.hpp>
#include "tt_metal/test_utils/comparison.hpp"
#include "tt_metal/test_utils/stimulus.hpp"
#include "tt_metal/test_utils/print_helpers.hpp"
#include "dm_common.hpp"
#include <distributed/mesh_device_impl.hpp>

namespace tt::tt_metal {

using namespace std;
using namespace tt;
using namespace test_utils;

namespace unit_tests::dm::dram {
// Test config, i.e. test parameters
struct DramConfig {
    uint32_t test_id = 0;
    uint32_t num_of_transactions = 0;
    uint32_t pages_per_transaction = 0;
    uint32_t bytes_per_page = 0;
    DataFormat l1_data_format = DataFormat::Invalid;
    CoreCoord core_coord = {0, 0};
    uint32_t dram_channel = 0;
    uint32_t virtual_channel = 0;
};

/// @brief Does Dram --> Reader --> L1 CB --> Writer --> Dram.
/// @param device
/// @param test_config - Configuration of the test -- see struct
/// @param fixture - DispatchFixture pointer for dispatch-aware operations
/// @return
bool run_dm(const shared_ptr<distributed::MeshDevice>& mesh_device, const DramConfig& test_config) {
    IDevice* device = mesh_device->impl().get_device(0);
    // SETUP

    const size_t total_size_bytes = test_config.pages_per_transaction * test_config.bytes_per_page;

    // DRAM Address
    DramAddressInfo dram_info = unit_tests::dm::get_dram_address_and_size();

    uint32_t input_dram_address = dram_info.base_address;
    uint32_t output_dram_address = input_dram_address + total_size_bytes;

    // L1 Address
    L1AddressInfo l1_info = unit_tests::dm::get_l1_address_and_size(mesh_device, test_config.core_coord);

    uint32_t l1_address = l1_info.base_address;

    // Redundant but as an extra measure add a check to ensure both addresses are within DRAM bounds
    // Checks also needed for L1 maybe

    CoreRangeSet core_range_set = CoreRangeSet({CoreRange(test_config.core_coord)});

    const std::string reader_kernel_path =
        "tests/tt_metal/tt_metal/data_movement/dram_unary/kernels/reader_unary_2_0.cpp";
    const std::string writer_kernel_path =
        "tests/tt_metal/tt_metal/data_movement/dram_unary/kernels/writer_unary_2_0.cpp";

    using namespace tt::tt_metal::experimental;

    SemaphoreSpec rw_sync_sem{
        .unique_id = SemaphoreSpecName{"dram_unary_rw_sync"},
        .target_nodes = core_range_set,
    };

    KernelSpec::CompileTimeArgs reader_cta = {
        {"test_id", (uint32_t)test_config.test_id},
        {"bytes_per_page", (uint32_t)test_config.bytes_per_page},
        {"dram_addr", (uint32_t)input_dram_address},
        {"dram_channel", (uint32_t)test_config.dram_channel},
        {"l1_addr", (uint32_t)l1_address},
    };

    KernelSpec::CompileTimeArgs writer_cta = {
        {"test_id", (uint32_t)test_config.test_id},
        {"bytes_per_page", (uint32_t)test_config.bytes_per_page},
        {"dram_channel", (uint32_t)test_config.dram_channel},
        {"l1_addr", (uint32_t)l1_address},
        {"vc", (uint32_t)test_config.virtual_channel},
    };

    DataMovementHardwareConfig reader_hw_config;
    if (device->arch() == tt::ARCH::QUASAR) {
        reader_hw_config = DataMovementGen2Config{};
    } else {
        reader_hw_config = DataMovementGen1Config{
            .processor = DataMovementProcessor::RISCV_1,
            .noc = NOC::RISCV_1_default,
        };
    }
    KernelSpec reader_spec{
        .unique_id = KernelSpecName{"reader"},
        .source = reader_kernel_path,
        .num_threads = 1,
        .semaphore_bindings = {KernelSpec::SemaphoreBinding{
            .semaphore_spec_name = rw_sync_sem.unique_id, .accessor_name = "dram_sync"}},
        .compile_time_args = reader_cta,
        .runtime_arg_schema =
            {
                .runtime_arg_names = {"num_of_transactions", "pages_per_transaction"},
            },
        .hw_config = reader_hw_config,
    };

    DataMovementHardwareConfig writer_hw_config;
    if (device->arch() == tt::ARCH::QUASAR) {
        writer_hw_config = DataMovementGen2Config{};
    } else {
        writer_hw_config = DataMovementGen1Config{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
        };
    }
    KernelSpec writer_spec{
        .unique_id = KernelSpecName{"writer"},
        .source = writer_kernel_path,
        .num_threads = 1,
        .semaphore_bindings = {KernelSpec::SemaphoreBinding{
            .semaphore_spec_name = rw_sync_sem.unique_id, .accessor_name = "dram_sync"}},
        .compile_time_args = writer_cta,
        .runtime_arg_schema =
            {
                .runtime_arg_names = {"num_of_transactions", "pages_per_transaction", "dram_addr"},
            },
        .hw_config = writer_hw_config,
    };

    ProgramSpec spec{
        .name = "dram_unary_test",
        .kernels = {reader_spec, writer_spec},
        .semaphores = {rw_sync_sem},
        .work_units = {WorkUnitSpec{
            .name = "work_unit",
            .kernels = {reader_spec.unique_id, writer_spec.unique_id},
            .target_nodes = core_range_set,
        }},
    };

    Program program = MakeProgramFromSpec(*mesh_device, spec);

    ProgramRunArgs run_params;
    ProgramRunArgs::KernelRunArgs reader_run{.kernel = reader_spec.unique_id};
    AddRuntimeArgsForNode(
        reader_run.runtime_arg_values,
        test_config.core_coord,
        {
            {"num_of_transactions", (uint32_t)test_config.num_of_transactions},
            {"pages_per_transaction", (uint32_t)test_config.pages_per_transaction},
        });
    run_params.kernel_run_args.push_back(reader_run);

    ProgramRunArgs::KernelRunArgs writer_run{.kernel = writer_spec.unique_id};
    AddRuntimeArgsForNode(
        writer_run.runtime_arg_values,
        test_config.core_coord,
        {
            {"num_of_transactions", (uint32_t)test_config.num_of_transactions},
            {"pages_per_transaction", (uint32_t)test_config.pages_per_transaction},
            {"dram_addr", (uint32_t)output_dram_address},
        });
    run_params.kernel_run_args.push_back(writer_run);
    SetProgramRunArgs(program, run_params);

    // Assign unique id
    log_info(LogTest, "Running Test ID: {}, Run ID: {}", test_config.test_id, unit_tests::dm::runtime_host_id);
    program.set_runtime_id(unit_tests::dm::runtime_host_id++);

    // RUNNING THE PROGRAM

    // Setup Input and Golden Output
    vector<uint32_t> packed_input = generate_packed_uniform_random_vector<uint32_t, bfloat16>(
        -100.0f, 100.0f, total_size_bytes / sizeof(bfloat16), chrono::system_clock::now().time_since_epoch().count());

    // NOLINTNEXTLINE(performance-unnecessary-copy-initialization)
    vector<uint32_t> packed_golden = packed_input;

    // Write Input to DRAM
    detail::WriteToDeviceDRAMChannel(device, test_config.dram_channel, input_dram_address, packed_input);
    MetalContext::instance().get_cluster().dram_barrier(device->id());

    // LAUNCH PROGRAM - Use mesh workload approach
    auto mesh_workload = distributed::MeshWorkload();
    vector<uint32_t> coord_data = {0, 0};
    auto target_devices = distributed::MeshCoordinateRange(distributed::MeshCoordinate(coord_data));
    mesh_workload.add_program(target_devices, std::move(program));

    auto& cq = mesh_device->mesh_command_queue();
    distributed::EnqueueMeshWorkload(cq, mesh_workload, false);
    Finish(cq);

    // Read Intermediate Output from L1 (for debugging purposes)
    vector<uint32_t> packed_intermediate_output;
    detail::ReadFromDeviceL1(device, test_config.core_coord, l1_address, total_size_bytes, packed_intermediate_output);

    // Read Output from DRAM
    vector<uint32_t> packed_output;
    detail::ReadFromDeviceDRAMChannel(
        device, test_config.dram_channel, output_dram_address, total_size_bytes, packed_output);

    // Results comparison
    bool is_equal = (packed_output == packed_golden);

    if (!is_equal) {
        log_error(LogTest, "Equality Check failed");
        log_info(LogTest, "Golden vector");
        print_vector<uint32_t>(packed_golden);
        log_info(LogTest, "Output vector");
        print_vector<uint32_t>(packed_output);
    }

    return is_equal;
}

void directed_ideal_test(
    const shared_ptr<distributed::MeshDevice>& mesh_device,
    uint32_t test_case_id,
    CoreCoord core_coord = {0, 0},
    uint32_t dram_channel = 0,
    uint32_t virtual_channel = 0) {
    // Physical Constraints
    auto [bytes_per_page, max_transmittable_bytes, max_transmittable_pages] =
        unit_tests::dm::compute_physical_constraints(mesh_device);

    // Parameters
    uint32_t num_of_transactions = 256;
    uint32_t pages_per_transaction = max_transmittable_pages;
    if (mesh_device->impl().get_device(0)->arch() == ARCH::QUASAR) {
        num_of_transactions = 4;
        pages_per_transaction = 4;
    }

    // Test config
    unit_tests::dm::dram::DramConfig test_config = {
        .test_id = test_case_id,
        .num_of_transactions = num_of_transactions,
        .pages_per_transaction = pages_per_transaction,
        .bytes_per_page = bytes_per_page,
        .l1_data_format = DataFormat::Float16_b,
        .core_coord = core_coord,
        .dram_channel = dram_channel,
        .virtual_channel = virtual_channel,
    };

    // Run
    EXPECT_TRUE(run_dm(mesh_device, test_config));
}

void packet_sizes_test(
    const shared_ptr<distributed::MeshDevice>& mesh_device,
    uint32_t test_case_id,
    CoreCoord core_coord = {0, 0},
    uint32_t dram_channel = 0) {
    auto [bytes_per_page, max_reservable_bytes, max_reservable_pages] =
        unit_tests::dm::compute_physical_constraints(mesh_device);

    // Parameters
    uint32_t max_transactions = 256;
    uint32_t max_pages_per_transaction = 256;

    for (uint32_t num_of_transactions = 1; num_of_transactions <= max_transactions; num_of_transactions *= 4) {
        for (uint32_t pages_per_transaction = 1; pages_per_transaction <= max_pages_per_transaction;
             pages_per_transaction *= 2) {
            if (num_of_transactions * pages_per_transaction * bytes_per_page >= max_reservable_bytes) {
                continue;
            }

            // Test config
            unit_tests::dm::dram::DramConfig test_config = {
                .test_id = test_case_id,
                .num_of_transactions = num_of_transactions,
                .pages_per_transaction = pages_per_transaction,
                .bytes_per_page = bytes_per_page,
                .l1_data_format = DataFormat::Float16_b,
                .core_coord = core_coord,
                .dram_channel = dram_channel,
                .virtual_channel = 0,
            };

            // Run
            EXPECT_TRUE(run_dm(mesh_device, test_config));
        }
    }
}

}  // namespace unit_tests::dm::dram

/* ========== Test case for varying transaction numbers and sizes; Test id = 0 ========== */
TEST_F(GenericMeshDeviceFixture, TensixDataMovementDRAMPacketSizes) {
    unit_tests::dm::dram::packet_sizes_test(
        get_mesh_device(),
        0,      // Test case ID
        {0, 0}  // Core coordinates (default)
    );
}

/* ========== Test case for varying core locations; Test id = 1 ========== */
TEST_F(GenericMeshDeviceFixture, TensixDataMovementDRAMCoreLocations) {
    uint32_t test_case_id = 1;

    auto mesh_device = get_mesh_device();
    auto* device = mesh_device->impl().get_device(0);

    CoreCoord core_coord;
    uint32_t dram_channel = 0;

    // Cores
    auto grid_size = device->compute_with_storage_grid_size();
    log_info(LogTest, "Grid size x: {}, y: {}", grid_size.x, grid_size.y);

    for (unsigned int x = 0; x < grid_size.x; x++) {
        for (unsigned int y = 0; y < grid_size.y; y++) {
            core_coord = {x, y};

            unit_tests::dm::dram::directed_ideal_test(mesh_device, test_case_id, core_coord, dram_channel);
        }
    }
}

// DRAM channels

/* ========== Test case for varying DRAM channels; Test id = 2 ========== */
TEST_F(GenericMeshDeviceFixture, TensixDataMovementDRAMChannels) {
    uint32_t test_case_id = 2;

    auto mesh_device = get_mesh_device();
    auto* device = mesh_device->impl().get_device(0);

    CoreCoord core_coord = {0, 0};

    for (unsigned int dram_channel = 0; dram_channel < device->num_dram_channels(); dram_channel++) {
        for (unsigned int vc = 0; vc < 4; vc++) {
            unit_tests::dm::dram::directed_ideal_test(mesh_device, test_case_id, core_coord, dram_channel, vc);
        }
    }
}

/* ========== Directed ideal test case; Test id = 3 ========== */
TEST_F(GenericMeshDeviceFixture, TensixDataMovementDRAMDirectedIdeal) {
    auto mesh_device = get_mesh_device();
    if (mesh_device->impl().get_device(0)->arch() == ARCH::QUASAR) {
        auto [bytes_per_page, max_transmittable_bytes, max_transmittable_pages] =
            unit_tests::dm::compute_physical_constraints(mesh_device);
        unit_tests::dm::dram::DramConfig test_config = {
            .test_id = 3,
            .num_of_transactions = 4,
            .pages_per_transaction = 4,
            .bytes_per_page = bytes_per_page,
            .l1_data_format = DataFormat::Float16_b,
            .core_coord = {0, 0},
            .dram_channel = 0};
        EXPECT_TRUE(run_dm(mesh_device, test_config));
        return;
    }
    // Test ID (Arbitrary)
    uint32_t test_id = 3;
    unit_tests::dm::dram::directed_ideal_test(mesh_device, test_id);
}

/* ========== Test case for varying transaction numbers and sizes with 2.0 API; Test id = 40 ========== */
TEST_F(GenericMeshDeviceFixture, TensixDataMovementDRAMPacketSizes2_0) {
    auto mesh_device = get_mesh_device();
    if (mesh_device->impl().get_device(0)->arch() == ARCH::QUASAR) {
        // Quasar emulator: full sweep is too slow (same as legacy
        // TensixDataMovementDRAMPacketSizes timed out). Run a single small config to
        // exercise the Metal 2.0 host path + DRAM read/write+ semaphore handshake.
        auto [bytes_per_page, max_transmittable_bytes, max_transmittable_pages] =
            unit_tests::dm::compute_physical_constraints(mesh_device);
        unit_tests::dm::dram::DramConfig test_config = {
            .test_id = 40,
            .num_of_transactions = 4,
            .pages_per_transaction = 4,
            .bytes_per_page = bytes_per_page,
            .l1_data_format = DataFormat::Float16_b,
            .core_coord = {0, 0},
            .dram_channel = 0,
            .virtual_channel = 0,
        };
        EXPECT_TRUE(unit_tests::dm::dram::run_dm(mesh_device, test_config));
        return;
    }
    unit_tests::dm::dram::packet_sizes_test(mesh_device, 40, {0, 0}, 0);
}

TEST_F(GenericMeshDeviceFixture, TensixDataMovementDRAMDirectedIdeal2_0) {
    auto mesh_device = get_mesh_device();
    if (mesh_device->impl().get_device(0)->arch() == ARCH::QUASAR) {
        auto [bytes_per_page, max_transmittable_bytes, max_transmittable_pages] =
            unit_tests::dm::compute_physical_constraints(mesh_device);
        unit_tests::dm::dram::DramConfig test_config = {
            .test_id = 41,
            .num_of_transactions = 4,
            .pages_per_transaction = 4,
            .bytes_per_page = bytes_per_page,
            .l1_data_format = DataFormat::Float16_b,
            .core_coord = {0, 0},
            .dram_channel = 0,
            .virtual_channel = 0,
        };
        EXPECT_TRUE(unit_tests::dm::dram::run_dm(mesh_device, test_config));
        return;
    }
    unit_tests::dm::dram::directed_ideal_test(mesh_device, 41, {0, 0}, 0, 0);
}

TEST_F(GenericMeshDeviceFixture, TensixDataMovementDRAMCoreLocations2_0) {
    auto mesh_device = get_mesh_device();
    auto* device = mesh_device->impl().get_device(0);

    if (device->arch() == ARCH::QUASAR) {
        unit_tests::dm::dram::directed_ideal_test(mesh_device, 42, {0, 0}, 0, 0);
        return;
    }

    auto grid_size = device->compute_with_storage_grid_size();
    for (unsigned int x = 0; x < grid_size.x; x++) {
        for (unsigned int y = 0; y < grid_size.y; y++) {
            unit_tests::dm::dram::directed_ideal_test(mesh_device, 42, {x, y}, 0, 0);
        }
    }
}

TEST_F(GenericMeshDeviceFixture, TensixDataMovementDRAMChannels2_0) {
    auto mesh_device = get_mesh_device();
    auto* device = mesh_device->impl().get_device(0);

    if (device->arch() == ARCH::QUASAR) {
        unit_tests::dm::dram::directed_ideal_test(mesh_device, 43, {0, 0}, 0, 0);
        return;
    }

    for (unsigned int dram_channel = 0; dram_channel < device->num_dram_channels(); dram_channel++) {
        for (unsigned int vc = 0; vc < 4; vc++) {
            unit_tests::dm::dram::directed_ideal_test(mesh_device, 43, {0, 0}, dram_channel, vc);
        }
    }
}

}  // namespace tt::tt_metal
