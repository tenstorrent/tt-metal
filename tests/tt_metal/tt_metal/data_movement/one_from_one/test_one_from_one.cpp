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
#include "tt_metal/test_utils/comparison.hpp"
#include "tt_metal/test_utils/stimulus.hpp"
#include "tt_metal/test_utils/print_helpers.hpp"
#include "dm_common.hpp"
#include <distributed/mesh_device_impl.hpp>

namespace tt::tt_metal {

using namespace std;
using namespace tt;
using namespace test_utils;

namespace unit_tests::dm::core_from_core {
// Test config, i.e. test parameters
struct OneFromOneConfig {
    uint32_t test_id = 0;
    CoreCoord master_core_coord = {0, 0};
    CoreCoord subordinate_core_coord = {0, 0};
    uint32_t num_of_transactions = 0;
    uint32_t transaction_size_pages = 0;
    uint32_t page_size_bytes = 0;
    DataFormat l1_data_format = DataFormat::Invalid;
    NOC noc_id = NOC::RISCV_1_default;
    uint32_t num_virtual_channels = 1;
};

/// @brief Does Requestor Core --> L1 Responder Core --> L1 Requestor Core
/// @param device
/// @param test_config - Configuration of the test -- see struct
/// @param fixture - DispatchFixture pointer for dispatch-aware operations
/// @return
bool run_dm(const shared_ptr<distributed::MeshDevice>& mesh_device, const OneFromOneConfig& test_config) {
    IDevice* device = mesh_device->impl().get_device(0);
    const size_t transaction_size_bytes = test_config.transaction_size_pages * test_config.page_size_bytes;

    // (Logical) Core Coordinates and ranges
    CoreRangeSet master_core_set({CoreRange(test_config.master_core_coord)});

    // Obtain L1 Address for Storing Data
    // NOTE: We don't know if the whole block of memory is actually available.
    //       This is something that could probably be checked
    L1AddressInfo master_l1_info = unit_tests::dm::get_l1_address_and_size(mesh_device, test_config.master_core_coord);
    L1AddressInfo subordinate_l1_info =
        unit_tests::dm::get_l1_address_and_size(mesh_device, test_config.subordinate_core_coord);
    // Checks that both master and subordinate cores have the same L1 base address and size
    if (master_l1_info.base_address != subordinate_l1_info.base_address ||
        master_l1_info.size != subordinate_l1_info.size) {
        log_error(LogTest, "Mismatch in L1 address or size between master and subordinate cores");
        return false;
    }
    // Check if the L1 size is sufficient for the test configuration
    if (master_l1_info.size < transaction_size_bytes) {
        log_error(LogTest, "Insufficient L1 size for the test configuration");
        return false;
    }
    // Assigns a "safe" L1 local address for the master and subordinate cores
    uint32_t l1_base_address = master_l1_info.base_address;

    const std::string requestor_kernel_path =
        "tests/tt_metal/tt_metal/data_movement/one_from_one/kernels/requestor_2_0.cpp";

    CoreCoord physical_subordinate_core = device->worker_core_from_logical_core(test_config.subordinate_core_coord);

    using namespace tt::tt_metal::experimental;

    KernelSpec::CompileTimeArgs cta_bindings = {
        {"l1_addr", (uint32_t)l1_base_address},
        {"test_id", (uint32_t)test_config.test_id},
        {"num_vc", (uint32_t)test_config.num_virtual_channels}};

    DataMovementHardwareConfig requestor_hw_config;
    if (device->arch() == tt::ARCH::QUASAR) {
        requestor_hw_config = DataMovementGen2Config{};
    } else {
        requestor_hw_config = DataMovementGen1Config{
            .processor = DataMovementProcessor::RISCV_1,
            .noc = test_config.noc_id,
        };
    }
    KernelSpec requestor_spec{
        .unique_id = KernelSpecName{"requestor"},
        .source = requestor_kernel_path,
        .num_threads = 1,
        .compile_time_args = cta_bindings,
        .runtime_arg_schema =
            {
                .runtime_arg_names = {"num_of_transactions", "transaction_size_bytes", "responder_x", "responder_y"},
            },
        .hw_config = requestor_hw_config,
    };

    ProgramSpec spec{
        .name = "one_from_one_test",
        .kernels = {requestor_spec},
        .work_units = {WorkUnitSpec{
            .name = "work_unit",
            .kernels = {requestor_spec.unique_id},
            .target_nodes = master_core_set,
        }},
    };

    Program program = MakeProgramFromSpec(*mesh_device, spec);

    ProgramRunArgs run_params;
    ProgramRunArgs::KernelRunArgs requestor_run_params{.kernel = requestor_spec.unique_id};
    requestor_run_params.runtime_arg_values.push_back(
        {.node = test_config.master_core_coord,
         .args = {
             {"num_of_transactions", (uint32_t)test_config.num_of_transactions},
             {"transaction_size_bytes", (uint32_t)transaction_size_bytes},
             {"responder_x", (uint32_t)physical_subordinate_core.x},
             {"responder_y", (uint32_t)physical_subordinate_core.y}}});
    run_params.kernel_run_args.push_back(requestor_run_params);
    SetProgramRunArgs(program, run_params);

    // Assign unique id
    log_info(LogTest, "Running Test ID: {}, Run ID: {}", test_config.test_id, unit_tests::dm::runtime_host_id);
    program.set_runtime_id(unit_tests::dm::runtime_host_id++);

    // Input
    vector<uint32_t> packed_input = generate_packed_uniform_random_vector<uint32_t, bfloat16>(
        -100.0f,
        100.0f,
        transaction_size_bytes / sizeof(bfloat16),
        chrono::system_clock::now().time_since_epoch().count());

    // Golden output
    vector<uint32_t> packed_golden = packed_input;

    // Launch program and record outputs
    detail::WriteToDeviceL1(device, test_config.subordinate_core_coord, l1_base_address, packed_input);
    MetalContext::instance().get_cluster().l1_barrier(device->id());

    auto mesh_workload = distributed::MeshWorkload();
    vector<uint32_t> coord_data = {0, 0};
    auto target_devices = distributed::MeshCoordinateRange(distributed::MeshCoordinate(coord_data));
    mesh_workload.add_program(target_devices, std::move(program));

    auto& cq = mesh_device->mesh_command_queue();
    distributed::EnqueueMeshWorkload(cq, mesh_workload, false);
    Finish(cq);
    vector<uint32_t> packed_output;
    detail::ReadFromDeviceL1(
        device, test_config.master_core_coord, l1_base_address, transaction_size_bytes, packed_output);

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
    uint32_t test_id,
    CoreCoord master_core_coord,
    CoreCoord subordinate_core_coord,
    NOC noc_id = NOC::RISCV_1_default) {
    // Physical Constraints
    auto [page_size_bytes, max_transmittable_bytes, max_transmittable_pages] =
        tt::tt_metal::unit_tests::dm::compute_physical_constraints(mesh_device);

    // Parameters
    // Ideal: Less transactions, more data per transaction
    uint32_t num_of_transactions = 1;
    uint32_t transaction_size_pages = max_transmittable_pages / num_of_transactions;

    // Test config
    OneFromOneConfig test_config = {
        .test_id = test_id,
        .master_core_coord = master_core_coord,
        .subordinate_core_coord = subordinate_core_coord,
        .num_of_transactions = num_of_transactions,
        .transaction_size_pages = transaction_size_pages,
        .page_size_bytes = page_size_bytes,
        .l1_data_format = DataFormat::Float16_b,
        .noc_id = noc_id,
    };

    // Run
    EXPECT_TRUE(run_dm(mesh_device, test_config));
}

void packet_sizes_test(
    const shared_ptr<distributed::MeshDevice>& mesh_device,
    uint32_t test_id,
    CoreCoord master_core_coord,
    CoreCoord subordinate_core_coord,
    NOC noc_id = NOC::RISCV_1_default) {
    IDevice* device = mesh_device->impl().get_device(0);
    // Physical Constraints
    auto [page_size_bytes, max_transmittable_bytes, max_transmittable_pages] =
        unit_tests::dm::compute_physical_constraints(mesh_device);

    // Parameters
    uint32_t max_transactions = 256;
    uint32_t max_transaction_size_pages =
        device->arch() == ARCH::BLACKHOLE ? 1024 : 2048;  // Max total transaction size == 64 KB

    for (uint32_t num_of_transactions = 1; num_of_transactions <= max_transactions; num_of_transactions *= 4) {
        for (uint32_t transaction_size_pages = 1; transaction_size_pages <= max_transaction_size_pages;
             transaction_size_pages *= 2) {
            if (transaction_size_pages > max_transmittable_pages) {
                continue;
            }

            // Test config
            OneFromOneConfig test_config = {
                .test_id = test_id,
                .master_core_coord = master_core_coord,
                .subordinate_core_coord = subordinate_core_coord,
                .num_of_transactions = num_of_transactions,
                .transaction_size_pages = transaction_size_pages,
                .page_size_bytes = page_size_bytes,
                .l1_data_format = DataFormat::Float16_b,
                .noc_id = noc_id,
            };

            // Run
            EXPECT_TRUE(run_dm(mesh_device, test_config));
        }
    }
}

void virtual_channels_test(
    const shared_ptr<distributed::MeshDevice>& mesh_device,
    uint32_t test_id,
    CoreCoord master_core_coord = {0, 0},
    CoreCoord subordinate_core_coord = {1, 1}) {
    // Physical Constraints
    auto [bytes_per_page, max_transmittable_bytes, max_transmittable_pages] =
        tt::tt_metal::unit_tests::dm::compute_physical_constraints(mesh_device);

    std::uint32_t max_num_pages_per_transaction = 1 << 12;
    std::uint32_t num_of_transactions = 256;  // Constant value
    std::uint32_t max_num_virtual_channels = 4;

    // Loop through:
    // 1. NOCs (NOC_0, NOC_1)
    // 2. Size of transactions
    // 3. Numbers of virtual channels
    for (NOC noc_id : {NOC::NOC_0, NOC::NOC_1}) {
        for (uint32_t pages_per_transaction = 1; pages_per_transaction <= max_num_pages_per_transaction;
             pages_per_transaction *= 2) {
            for (uint32_t num_virtual_channels = 1; num_virtual_channels <= max_num_virtual_channels;
                 num_virtual_channels++) {
                if (pages_per_transaction > max_transmittable_pages) {
                    continue;
                }

                // Test config
                OneFromOneConfig test_config = {
                    .test_id = test_id,
                    .master_core_coord = master_core_coord,
                    .subordinate_core_coord = subordinate_core_coord,
                    .num_of_transactions = num_of_transactions,
                    .transaction_size_pages = pages_per_transaction,
                    .page_size_bytes = bytes_per_page,
                    .l1_data_format = DataFormat::Float16_b,
                    .noc_id = noc_id,
                    .num_virtual_channels = num_virtual_channels,
                };

                // Run
                EXPECT_TRUE(run_dm(mesh_device, test_config));
            }
        }
    }
}

void custom_test(
    const shared_ptr<distributed::MeshDevice>& mesh_device,
    uint32_t test_id,
    CoreCoord master_core_coord,
    CoreCoord subordinate_core_coord,
    uint32_t num_of_transactions,
    uint32_t pages_per_transaction,
    uint32_t num_virtual_channels,
    NOC noc_id = NOC::RISCV_1_default) {
    // Physical Constraints
    auto [bytes_per_page, max_transmittable_bytes, max_transmittable_pages] =
        tt::tt_metal::unit_tests::dm::compute_physical_constraints(mesh_device);

    if (pages_per_transaction > max_transmittable_pages) {
        log_trace(LogTest, "Skipping test due to page size limitations");
        return;
    }

    // Test config
    OneFromOneConfig test_config = {
        .test_id = test_id,
        .master_core_coord = master_core_coord,
        .subordinate_core_coord = subordinate_core_coord,
        .num_of_transactions = num_of_transactions,
        .transaction_size_pages = pages_per_transaction,
        .page_size_bytes = bytes_per_page,
        .l1_data_format = DataFormat::Float16_b,
        .noc_id = noc_id,
        .num_virtual_channels = num_virtual_channels,
    };

    // Run
    EXPECT_TRUE(run_dm(mesh_device, test_config));
}

}  // namespace unit_tests::dm::core_from_core

/* ========== TEST CASES ========== */

TEST_F(GenericMeshDeviceFixture, TensixDataMovementOneFromOnePacketSizes) {
    auto mesh_device = get_mesh_device();
    if (mesh_device->impl().get_device(0)->arch() == ARCH::QUASAR) {
        // subordinate_core_coord {1, 0} requires at least 2 columns in the compute grid
        if (mesh_device->impl().get_device(0)->compute_with_storage_grid_size().x < 2) {
            GTEST_SKIP() << "Skipping: subordinate core {1, 0} requires >= 2 columns, but grid has "
                         << mesh_device->impl().get_device(0)->compute_with_storage_grid_size().x
                         << " column(s). Use emu-quasar-2x3 or larger.";
        }
        auto [bytes_per_page, max_transmittable_bytes, max_transmittable_pages] =
            unit_tests::dm::compute_physical_constraints(mesh_device);
        unit_tests::dm::core_from_core::OneFromOneConfig test_config = {
            .test_id = 5,
            .master_core_coord = {0, 0},
            .subordinate_core_coord = {1, 0},
            .num_of_transactions = 4,
            .transaction_size_pages = 4,
            .page_size_bytes = bytes_per_page,
            .l1_data_format = DataFormat::Float16_b};
        EXPECT_TRUE(run_dm(mesh_device, test_config));
        return;
    }
    uint32_t test_id = 5;
    CoreCoord master_core_coord = {0, 0};
    CoreCoord subordinate_core_coord = {1, 1};
    unit_tests::dm::core_from_core::packet_sizes_test(mesh_device, test_id, master_core_coord, subordinate_core_coord);
}

TEST_F(GenericMeshDeviceFixture, TensixDataMovementOneFromOneDirectedIdeal) {
    auto mesh_device = get_mesh_device();
    if (mesh_device->impl().get_device(0)->arch() == ARCH::QUASAR) {
        // subordinate_core_coord {1, 0} requires at least 2 columns in the compute grid
        if (mesh_device->impl().get_device(0)->compute_with_storage_grid_size().x < 2) {
            GTEST_SKIP() << "Skipping: subordinate core {1, 0} requires >= 2 columns, but grid has "
                         << mesh_device->impl().get_device(0)->compute_with_storage_grid_size().x
                         << " column(s). Use emu-quasar-2x3 or larger.";
        }
        auto [bytes_per_page, max_transmittable_bytes, max_transmittable_pages] =
            unit_tests::dm::compute_physical_constraints(mesh_device);
        unit_tests::dm::core_from_core::OneFromOneConfig test_config = {
            .test_id = 51,
            .master_core_coord = {0, 0},
            .subordinate_core_coord = {1, 0},
            .num_of_transactions = 4,
            .transaction_size_pages = 1,
            .page_size_bytes = bytes_per_page,
            .l1_data_format = DataFormat::Float16_b};
        EXPECT_TRUE(run_dm(mesh_device, test_config));
        return;
    }
    uint32_t test_id = 51;
    CoreCoord master_core_coord = {0, 0};
    CoreCoord subordinate_core_coord = {0, 1};
    unit_tests::dm::core_from_core::directed_ideal_test(
        mesh_device, test_id, master_core_coord, subordinate_core_coord);
}

TEST_F(GenericMeshDeviceFixture, TensixDataMovementOneFromOneVirtualChannels) {
    GTEST_SKIP() << "Skipping test";
    // Test ID (Arbitrary)
    uint32_t test_id = 152;

    unit_tests::dm::core_from_core::virtual_channels_test(
        get_mesh_device(),
        test_id,
        CoreCoord(0, 0),  // Master Core
        CoreCoord(0, 1)   // Subordinate Core
    );
}

TEST_F(GenericMeshDeviceFixture, TensixDataMovementOneFromOneCustom) {
    GTEST_SKIP() << "Skipping test";
    uint32_t test_id = 153;

    // Parameters
    uint32_t num_of_transactions = 256;
    uint32_t pages_per_transaction = 1;
    uint32_t num_virtual_channels = 4;

    unit_tests::dm::core_from_core::custom_test(
        get_mesh_device(),
        test_id,
        CoreCoord(0, 0),  // Master Core
        CoreCoord(0, 1),  // Subordinate Core
        num_of_transactions,
        pages_per_transaction,
        num_virtual_channels);
}

TEST_F(GenericMeshDeviceFixture, TensixDataMovementOneFromOnePacketSizes2_0) {
    auto mesh_device = get_mesh_device();
    auto* device = mesh_device->impl().get_device(0);
    if (device->arch() == ARCH::QUASAR) {
        // Quasar emulator (1x3 or 2x3): pick a subordinate that exists on the grid
        // and run a single small config. Full sweep is too large for emulator.
        auto grid = device->compute_with_storage_grid_size();
        CoreCoord subordinate;
        if (grid.x >= 2) {
            subordinate = {1, 0};
        } else if (grid.y >= 2) {
            subordinate = {0, 1};
        } else {
            GTEST_SKIP() << "Skipping: need at least a 1x2 or 2x1 grid, got " << grid.x << "x" << grid.y;
        }
        auto [bytes_per_page, max_transmittable_bytes, max_transmittable_pages] =
            unit_tests::dm::compute_physical_constraints(mesh_device);
        unit_tests::dm::core_from_core::OneFromOneConfig test_config = {
            .test_id = 159,
            .master_core_coord = {0, 0},
            .subordinate_core_coord = subordinate,
            .num_of_transactions = 4,
            .transaction_size_pages = 4,
            .page_size_bytes = bytes_per_page,
            .l1_data_format = DataFormat::Float16_b,
        };
        EXPECT_TRUE(unit_tests::dm::core_from_core::run_dm(mesh_device, test_config));
        return;
    }
    uint32_t test_id = 159;
    unit_tests::dm::core_from_core::packet_sizes_test(mesh_device, test_id, CoreCoord(0, 0), CoreCoord(1, 1));
}

TEST_F(GenericMeshDeviceFixture, TensixDataMovementOneFromOneDirectedIdeal2_0) {
    auto mesh_device = get_mesh_device();
    auto* device = mesh_device->impl().get_device(0);
    if (device->arch() == ARCH::QUASAR) {
        auto grid = device->compute_with_storage_grid_size();
        CoreCoord subordinate;
        if (grid.x >= 2) {
            subordinate = {1, 0};
        } else if (grid.y >= 2) {
            subordinate = {0, 1};
        } else {
            GTEST_SKIP() << "Skipping: need at least a 1x2 or 2x1 grid, got " << grid.x << "x" << grid.y;
        }
        auto [bytes_per_page, max_transmittable_bytes, max_transmittable_pages] =
            unit_tests::dm::compute_physical_constraints(mesh_device);
        unit_tests::dm::core_from_core::OneFromOneConfig test_config = {
            .test_id = 161,
            .master_core_coord = {0, 0},
            .subordinate_core_coord = subordinate,
            .num_of_transactions = 4,
            .transaction_size_pages = 1,
            .page_size_bytes = bytes_per_page,
            .l1_data_format = DataFormat::Float16_b,
        };
        EXPECT_TRUE(unit_tests::dm::core_from_core::run_dm(mesh_device, test_config));
        return;
    }
    unit_tests::dm::core_from_core::directed_ideal_test(
        mesh_device, 161, CoreCoord(0, 0), CoreCoord(0, 1), NOC::RISCV_1_default);
}

}  // namespace tt::tt_metal
