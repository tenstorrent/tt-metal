// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "multi_device_fixture.hpp"
#include "device_fixture.hpp"
#include "tt_metal/test_utils/comparison.hpp"
#include "tt_metal/test_utils/stimulus.hpp"
#include "tt_metal/test_utils/print_helpers.hpp"
#include "dm_common.hpp"
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/experimental/metal2_host_api/program.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/kernel_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/node_coord.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include <distributed/mesh_device_impl.hpp>

namespace tt::tt_metal {

using namespace std;
using namespace tt;
using namespace test_utils;

namespace unit_tests::dm::transaction_id {

// Test config, i.e. test parameters
struct TransactionIdConfig {
    uint32_t test_id = 0;
    CoreCoord master_core_coord = {0, 0};
    CoreCoord sub0_core_coord = {0, 0};
    CoreCoord sub1_core_coord = {0, 0};
    uint32_t num_of_trids = 0;
    uint32_t pages_per_transaction = 0;
    uint32_t bytes_per_page = 0;
    DataFormat l1_data_format = DataFormat::Invalid;
    NOC noc_id = NOC::NOC_0;
    bool one_packet = false;
    bool stateful = false;
    bool read_after_write = true;
    // TODO: Add the following parameters
    //  1. Posted flag
};

/// @brief Does L1 Sender Core --> L1 Receiver Core or L1 Receiver Core --> L1 Sender Core
/// @param mesh_device - MeshDevice to run the test on
/// @param test_config - Configuration of the test -- see struct
/// @return
bool run_dm(const shared_ptr<distributed::MeshDevice>& mesh_device, const TransactionIdConfig& test_config) {
    // Get the actual device for this single-device test
    IDevice* device = mesh_device->impl().get_device(0);

    /* ================ SETUP ================ */

    // Buffer Parameters
    const size_t bytes_per_transaction = test_config.pages_per_transaction * test_config.bytes_per_page;

    // (Logical) Core coordinates and ranges
    CoreRangeSet master_core_set({CoreRange(test_config.master_core_coord)});
    CoreRangeSet sub0_core_set({CoreRange(test_config.sub0_core_coord)});
    CoreRangeSet sub1_core_set({CoreRange(test_config.sub1_core_coord)});
    // CoreRangeSet combined_core_set = master_core_set.merge<CoreRangeSet>(subordinate_core_set);

    // Obtain L1 Address for Storing Data
    // NOTE: We don't know if the whole block of memory is actually available.
    L1AddressInfo master_l1_info = unit_tests::dm::get_l1_address_and_size(mesh_device, test_config.master_core_coord);
    L1AddressInfo sub0_l1_info = unit_tests::dm::get_l1_address_and_size(mesh_device, test_config.sub0_core_coord);
    L1AddressInfo sub1_l1_info = unit_tests::dm::get_l1_address_and_size(mesh_device, test_config.sub1_core_coord);

    // Checks that both master and subordinate cores have the same L1 base address and size
    if (master_l1_info.base_address != sub0_l1_info.base_address || master_l1_info.size != sub0_l1_info.size ||
        master_l1_info.base_address != sub1_l1_info.base_address || master_l1_info.size != sub1_l1_info.size) {
        log_error(LogTest, "Mismatch in L1 address or size between master and subordinate cores");
        return false;
    }
    // Check if the L1 size is sufficient for the test configuration
    if (master_l1_info.size < bytes_per_transaction) {
        log_error(LogTest, "Insufficient L1 size for the test configuration");
        return false;
    }
    // Assigns a "safe" L1 local address for the master and subordinate cores
    uint32_t l1_base_address = master_l1_info.base_address;

    // Physical Core Coordinates
    CoreCoord physical_sub0_core = device->worker_core_from_logical_core(test_config.sub0_core_coord);
    uint32_t packed_sub0_core_coordinates = physical_sub0_core.x << 16 | (physical_sub0_core.y & 0xFFFF);
    CoreCoord physical_sub1_core = device->worker_core_from_logical_core(test_config.sub1_core_coord);
    uint32_t packed_sub1_core_coordinates = physical_sub1_core.x << 16 | (physical_sub1_core.y & 0xFFFF);

    string kernel_path = "tests/tt_metal/tt_metal/data_movement/transaction_id/kernels/";
    if (test_config.read_after_write) {
        kernel_path += "writer_reader";
    } else {
        kernel_path += "reader_writer";
    }
    if (test_config.one_packet) {
        kernel_path += "_one_packet";
        if (test_config.stateful) {
            kernel_path += "_stateful";
        }
    }
    kernel_path += "_2_0.cpp";

    using namespace tt::tt_metal::experimental;

    KernelSpec::CompileTimeArgs cta_bindings = {
        {"l1_addr", l1_base_address},
        {"test_id", test_config.test_id},
        {"sub0_coords", packed_sub0_core_coordinates},
        {"sub1_coords", packed_sub1_core_coordinates}};

    DataMovementHardwareConfig sender_hw_config;
    if (device->arch() == tt::ARCH::QUASAR) {
        sender_hw_config = DataMovementGen2Config{};
    } else {
        sender_hw_config = DataMovementGen1Config{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = test_config.noc_id,
        };
    }
    KernelSpec sender_spec{
        .unique_id = KernelSpecName{"trid_sender"},
        .source = kernel_path,
        .num_threads = 1,
        .compile_time_args = cta_bindings,
        .runtime_arg_schema =
            {
                .runtime_arg_names = {"num_transactions", "bytes_per_transaction"},
            },
        .hw_config = sender_hw_config,
    };

    ProgramSpec spec{
        .name = "transaction_id_test",
        .kernels = {sender_spec},
        .work_units = {WorkUnitSpec{
            .name = "work_unit",
            .kernels = {sender_spec.unique_id},
            .target_nodes = master_core_set,
        }},
    };

    Program program = MakeProgramFromSpec(*mesh_device, spec);

    ProgramRunArgs run_params;
    ProgramRunArgs::KernelRunArgs sender_run_params{.kernel = sender_spec.unique_id};
    sender_run_params.runtime_arg_values.push_back(
        {.node = test_config.master_core_coord,
         .args = {
             {"num_transactions", (uint32_t)test_config.num_of_trids},
             {"bytes_per_transaction", (uint32_t)bytes_per_transaction}}});
    run_params.kernel_run_args.push_back(sender_run_params);
    SetProgramRunArgs(program, run_params);

    // Assign unique id
    log_info(LogTest, "Running Test ID: {}, Run ID: {}", test_config.test_id, unit_tests::dm::runtime_host_id);
    program.set_runtime_id(unit_tests::dm::runtime_host_id++);

    /* ================ RUNNING THE PROGRAM ================ */

    // Setup Input
    // NOTE: The converted vector (uint32_t -> bfloat16) preserves the number of bytes,
    // but the number of elements is bound to change
    // l1_data_format is assumed to be bfloat16
    size_t element_size_bytes = sizeof(bfloat16);
    uint32_t num_elements = bytes_per_transaction / element_size_bytes;

    vector<uint32_t> packed_input_master = generate_packed_uniform_random_vector<uint32_t, bfloat16>(
        -100.0f, 100.0f, num_elements, chrono::system_clock::now().time_since_epoch().count());

    vector<uint32_t> packed_input_sub1 = generate_packed_uniform_random_vector<uint32_t, bfloat16>(
        -100.0f, 100.0f, num_elements, chrono::system_clock::now().time_since_epoch().count());

    // Master core writes to sub0 core and reads from sub1 core
    // If the test is read after write, the golden output from master is the original input to master
    // because it was sent out before getting overwritten by the read.
    // Otherwise, the golden output from master is the data read from sub1 core.
    vector<uint32_t> packed_golden_master = test_config.read_after_write ? packed_input_master : packed_input_sub1;
    vector<uint32_t> packed_golden_sub1 = packed_input_sub1;

    // Write Input to Master and Sub1 L1
    detail::WriteToDeviceL1(device, test_config.master_core_coord, l1_base_address, packed_input_master);
    detail::WriteToDeviceL1(device, test_config.sub1_core_coord, l1_base_address, packed_input_sub1);
    MetalContext::instance().get_cluster().l1_barrier(device->id());

    // LAUNCH THE PROGRAM - Use mesh workload approach
    auto mesh_workload = distributed::MeshWorkload();
    vector<uint32_t> coord_data = {0, 0};
    auto target_devices =
        distributed::MeshCoordinateRange(distributed::MeshCoordinate(coord_data));  // Single device at (0,0)
    mesh_workload.add_program(target_devices, std::move(program));

    auto& cq = mesh_device->mesh_command_queue();
    distributed::EnqueueMeshWorkload(cq, mesh_workload, false);
    Finish(cq);

    // Record Output from Master and Sub0 L1
    vector<uint32_t> packed_output_master;
    detail::ReadFromDeviceL1(
        device, test_config.sub0_core_coord, l1_base_address, bytes_per_transaction, packed_output_master);

    vector<uint32_t> packed_output_sub1;
    detail::ReadFromDeviceL1(
        device, test_config.master_core_coord, l1_base_address, bytes_per_transaction, packed_output_sub1);

    // Compare output with golden vector
    bool is_equal = (packed_output_master == packed_golden_master);
    is_equal &= (packed_output_sub1 == packed_golden_sub1);

    if (!is_equal) {
        log_error(LogTest, "Equality Check failed");
        log_info(LogTest, "Golden master vector");
        print_vector<uint32_t>(packed_golden_master);
        log_info(LogTest, "Output master vector");
        print_vector<uint32_t>(packed_output_master);
        log_info(LogTest, "Golden sub1 vector");
        print_vector<uint32_t>(packed_golden_sub1);
        log_info(LogTest, "Output sub1 vector");
        print_vector<uint32_t>(packed_output_sub1);
    }

    return is_equal;
}
}  // namespace unit_tests::dm::transaction_id

/* ========== TEST CASES ========== */

TEST_F(GenericMeshDeviceFixture, TensixDataMovementTransactionIdReadAfterWrite) {
    auto mesh_device = get_mesh_device();
    auto* device = mesh_device->impl().get_device(0);

    // Physical Constraints
    auto [bytes_per_page, max_transmittable_bytes, max_transmittable_pages] =
        unit_tests::dm::compute_physical_constraints(mesh_device);

    // Cores
    CoreCoord master_core_coord = {0, 0};

    // Furthest cores from master
    CoreCoord sub0_core_coord = {0, device->compute_with_storage_grid_size().y - 1};
    CoreCoord sub1_core_coord = {device->compute_with_storage_grid_size().x - 1, 0};

    if (device->arch() == ARCH::QUASAR) {
        // Transaction ID test needs 3 distinct cores; skip if grid is too small
        auto grid = device->compute_with_storage_grid_size();
        if (grid.x * grid.y < 3) {
            GTEST_SKIP() << "Skipping: need 3 distinct cores but grid is only " << grid.x << "x" << grid.y;
        }
        unit_tests::dm::transaction_id::TransactionIdConfig test_config = {
            .test_id = 600,
            .master_core_coord = master_core_coord,
            .sub0_core_coord = sub0_core_coord,
            .sub1_core_coord = sub1_core_coord,
            .num_of_trids = 4,
            .pages_per_transaction = 1,
            .bytes_per_page = bytes_per_page,
            .l1_data_format = DataFormat::Float16_b,
        };
        EXPECT_TRUE(run_dm(mesh_device, test_config));
        return;
    }

    // Test ID
    uint32_t test_id = 600;

    // Parameters
    uint32_t max_pages_per_transaction =
        device->arch() == ARCH::BLACKHOLE ? 1024 : 2048;  // Max total transaction size == 64 KB

    for (uint32_t num_of_trids = 1; num_of_trids <= 16; num_of_trids *= 2) {  // Up to 0xF (16) transaction ids
        for (uint32_t pages_per_transaction = 1; pages_per_transaction <= max_pages_per_transaction;
             pages_per_transaction *= 2) {
            // Check if the total page size is within the limits
            if (pages_per_transaction * num_of_trids > max_transmittable_pages) {
                continue;
            }
            // Test config
            unit_tests::dm::transaction_id::TransactionIdConfig test_config = {
                .test_id = test_id,
                .master_core_coord = master_core_coord,
                .sub0_core_coord = sub0_core_coord,
                .sub1_core_coord = sub1_core_coord,
                .num_of_trids = num_of_trids,
                .pages_per_transaction = pages_per_transaction,
                .bytes_per_page = bytes_per_page,
                .l1_data_format = DataFormat::Float16_b,
            };

            // Run
            EXPECT_TRUE(run_dm(mesh_device, test_config));
        }
    }
}

TEST_F(GenericMeshDeviceFixture, TensixDataMovementTransactionIdReadAfterWriteOnePacket) {
    // Test ID
    uint32_t test_id = 601;

    auto mesh_device = get_mesh_device();
    auto* device = mesh_device->impl().get_device(0);

    // Physical Constraints
    auto [bytes_per_page, max_transmittable_bytes, max_transmittable_pages] =
        unit_tests::dm::compute_physical_constraints(mesh_device);

    // Cores
    CoreCoord master_core_coord = {0, 0};

    // Furthest cores from master
    CoreCoord sub0_core_coord = {0, device->compute_with_storage_grid_size().y - 1};
    CoreCoord sub1_core_coord = {device->compute_with_storage_grid_size().x - 1, 0};

    // Parameters
    uint32_t max_pages_per_transaction = 256;  // NOC_MAX_BURST_WORDS

    for (uint32_t num_of_trids = 1; num_of_trids <= 16; num_of_trids *= 2) {  // Up to 0xF (16) transaction ids
        for (uint32_t pages_per_transaction = 1; pages_per_transaction <= max_pages_per_transaction;
             pages_per_transaction *= 2) {
            // Check if the total page size is within the limits
            if (pages_per_transaction * num_of_trids > max_transmittable_pages) {
                continue;
            }
            // Test config
            unit_tests::dm::transaction_id::TransactionIdConfig test_config = {
                .test_id = test_id,
                .master_core_coord = master_core_coord,
                .sub0_core_coord = sub0_core_coord,
                .sub1_core_coord = sub1_core_coord,
                .num_of_trids = num_of_trids,
                .pages_per_transaction = pages_per_transaction,
                .bytes_per_page = bytes_per_page,
                .l1_data_format = DataFormat::Float16_b,
                .one_packet = true,
            };

            // Run
            EXPECT_TRUE(run_dm(mesh_device, test_config));
        }
    }
}

TEST_F(GenericMeshDeviceFixture, TensixDataMovementTransactionIdReadAfterWriteOnePacketStateful) {
    // Test ID
    uint32_t test_id = 602;

    auto mesh_device = get_mesh_device();
    auto* device = mesh_device->impl().get_device(0);

    // Physical Constraints
    auto [bytes_per_page, max_transmittable_bytes, max_transmittable_pages] =
        unit_tests::dm::compute_physical_constraints(mesh_device);

    // Cores
    CoreCoord master_core_coord = {0, 0};

    // Furthest cores from master
    CoreCoord sub0_core_coord = {0, device->compute_with_storage_grid_size().y - 1};
    CoreCoord sub1_core_coord = {device->compute_with_storage_grid_size().x - 1, 0};

    // Parameters
    uint32_t max_pages_per_transaction = 256;  // NOC_MAX_BURST_WORDS

    for (uint32_t num_of_trids = 1; num_of_trids <= 16; num_of_trids *= 2) {  // Up to 0xF (16) transaction ids
        for (uint32_t pages_per_transaction = 1; pages_per_transaction <= max_pages_per_transaction;
             pages_per_transaction *= 2) {
            // Check if the total page size is within the limits
            if (pages_per_transaction * num_of_trids > max_transmittable_pages) {
                continue;
            }
            // Test config
            unit_tests::dm::transaction_id::TransactionIdConfig test_config = {
                .test_id = test_id,
                .master_core_coord = master_core_coord,
                .sub0_core_coord = sub0_core_coord,
                .sub1_core_coord = sub1_core_coord,
                .num_of_trids = num_of_trids,
                .pages_per_transaction = pages_per_transaction,
                .bytes_per_page = bytes_per_page,
                .l1_data_format = DataFormat::Float16_b,
                .one_packet = true,
                .stateful = true,
            };

            // Run
            EXPECT_TRUE(run_dm(mesh_device, test_config));
        }
    }
}

TEST_F(GenericMeshDeviceFixture, TensixDataMovementTransactionIdWriteAfterRead) {
    // Test ID
    uint32_t test_id = 610;

    auto mesh_device = get_mesh_device();
    auto* device = mesh_device->impl().get_device(0);

    // Physical Constraints
    auto [bytes_per_page, max_transmittable_bytes, max_transmittable_pages] =
        unit_tests::dm::compute_physical_constraints(mesh_device);

    // Cores
    CoreCoord master_core_coord = {0, 0};

    // Furthest cores from master
    CoreCoord sub0_core_coord = {0, device->compute_with_storage_grid_size().y - 1};
    CoreCoord sub1_core_coord = {device->compute_with_storage_grid_size().x - 1, 0};

    // Parameters
    uint32_t max_pages_per_transaction = mesh_device->impl().get_device(0)->arch() == ARCH::BLACKHOLE
                                             ? 1024
                                             : 2048;  // Max total transaction size == 64 KB

    for (uint32_t num_of_trids = 1; num_of_trids <= 16; num_of_trids *= 2) {  // Up to 0xF (16) transaction ids
        for (uint32_t pages_per_transaction = 1; pages_per_transaction <= max_pages_per_transaction;
             pages_per_transaction *= 2) {
            // Check if the total page size is within the limits
            if (pages_per_transaction * num_of_trids > max_transmittable_pages) {
                continue;
            }
            // Test config
            unit_tests::dm::transaction_id::TransactionIdConfig test_config = {
                .test_id = test_id,
                .master_core_coord = master_core_coord,
                .sub0_core_coord = sub0_core_coord,
                .sub1_core_coord = sub1_core_coord,
                .num_of_trids = num_of_trids,
                .pages_per_transaction = pages_per_transaction,
                .bytes_per_page = bytes_per_page,
                .l1_data_format = DataFormat::Float16_b,
                .read_after_write = false,
            };

            // Run
            EXPECT_TRUE(run_dm(mesh_device, test_config));
        }
    }
}

TEST_F(GenericMeshDeviceFixture, TensixDataMovementTransactionIdWriteAfterReadOnePacketStateful) {
    // Test ID
    uint32_t test_id = 611;

    auto mesh_device = get_mesh_device();
    auto* device = mesh_device->impl().get_device(0);

    // Physical Constraints
    auto [bytes_per_page, max_transmittable_bytes, max_transmittable_pages] =
        unit_tests::dm::compute_physical_constraints(mesh_device);

    // Cores
    CoreCoord master_core_coord = {0, 0};

    // Furthest cores from master
    CoreCoord sub0_core_coord = {0, device->compute_with_storage_grid_size().y - 1};
    CoreCoord sub1_core_coord = {device->compute_with_storage_grid_size().x - 1, 0};

    // Parameters
    uint32_t max_pages_per_transaction = 256;  // NOC_MAX_BURST_WORDS

    for (uint32_t num_of_trids = 1; num_of_trids <= 16; num_of_trids *= 2) {  // Up to 0xF (16) transaction ids
        for (uint32_t pages_per_transaction = 1; pages_per_transaction <= max_pages_per_transaction;
             pages_per_transaction *= 2) {
            // Check if the total page size is within the limits
            if (pages_per_transaction * num_of_trids > max_transmittable_pages) {
                continue;
            }
            // Test config
            unit_tests::dm::transaction_id::TransactionIdConfig test_config = {
                .test_id = test_id,
                .master_core_coord = master_core_coord,
                .sub0_core_coord = sub0_core_coord,
                .sub1_core_coord = sub1_core_coord,
                .num_of_trids = num_of_trids,
                .pages_per_transaction = pages_per_transaction,
                .bytes_per_page = bytes_per_page,
                .l1_data_format = DataFormat::Float16_b,
                .one_packet = true,
                .stateful = true,
                .read_after_write = false,
            };

            // Run
            EXPECT_TRUE(run_dm(mesh_device, test_config));
        }
    }
}

/* ========== Metal 2.0 variants ========== */

namespace unit_tests::dm::transaction_id {
// Helper: returns true if Quasar emulator can run a 3-core transaction_id test, otherwise GTEST_SKIP-ready.
inline bool quasar_grid_ok(IDevice* device) {
    auto grid = device->compute_with_storage_grid_size();
    return grid.x * grid.y >= 3;
}
}  // namespace unit_tests::dm::transaction_id

TEST_F(GenericMeshDeviceFixture, TensixDataMovementTransactionIdReadAfterWrite_2_0) {
    auto mesh_device = get_mesh_device();
    auto* device = mesh_device->impl().get_device(0);
    auto arch_ = device->arch();

    auto [bytes_per_page, max_transmittable_bytes, max_transmittable_pages] =
        unit_tests::dm::compute_physical_constraints(mesh_device);

    CoreCoord master_core_coord = {0, 0};
    CoreCoord sub0_core_coord = {0, device->compute_with_storage_grid_size().y - 1};
    CoreCoord sub1_core_coord = {device->compute_with_storage_grid_size().x - 1, 0};

    if (arch_ == ARCH::QUASAR) {
        auto grid_dbg = device->compute_with_storage_grid_size();
        if (!unit_tests::dm::transaction_id::quasar_grid_ok(device)) {
            GTEST_SKIP() << "Skipping: need 3 distinct cores but grid is only " << grid_dbg.x << "x" << grid_dbg.y;
        }
        unit_tests::dm::transaction_id::TransactionIdConfig test_config = {
            .test_id = 620,
            .master_core_coord = master_core_coord,
            .sub0_core_coord = sub0_core_coord,
            .sub1_core_coord = sub1_core_coord,
            .num_of_trids = 4,
            .pages_per_transaction = 1,
            .bytes_per_page = bytes_per_page,
            .l1_data_format = DataFormat::Float16_b,
        };
        EXPECT_TRUE(unit_tests::dm::transaction_id::run_dm(mesh_device, test_config));
        return;
    }

    // WH/BH full sweep
    uint32_t max_pages_per_transaction = arch_ == ARCH::BLACKHOLE ? 1024 : 2048;
    for (uint32_t num_of_trids = 1; num_of_trids <= 16; num_of_trids *= 2) {
        for (uint32_t pages_per_transaction = 1; pages_per_transaction <= max_pages_per_transaction;
             pages_per_transaction *= 2) {
            if (pages_per_transaction * num_of_trids > max_transmittable_pages) {
                continue;
            }
            unit_tests::dm::transaction_id::TransactionIdConfig test_config = {
                .test_id = 620,
                .master_core_coord = master_core_coord,
                .sub0_core_coord = sub0_core_coord,
                .sub1_core_coord = sub1_core_coord,
                .num_of_trids = num_of_trids,
                .pages_per_transaction = pages_per_transaction,
                .bytes_per_page = bytes_per_page,
                .l1_data_format = DataFormat::Float16_b,
            };
            EXPECT_TRUE(unit_tests::dm::transaction_id::run_dm(mesh_device, test_config));
        }
    }
}

TEST_F(GenericMeshDeviceFixture, TensixDataMovementTransactionIdReadAfterWriteOnePacket_2_0) {
    auto mesh_device = get_mesh_device();
    auto* device = mesh_device->impl().get_device(0);
    auto arch_ = device->arch();

    auto [bytes_per_page, max_transmittable_bytes, max_transmittable_pages] =
        unit_tests::dm::compute_physical_constraints(mesh_device);

    CoreCoord master_core_coord = {0, 0};
    CoreCoord sub0_core_coord = {0, device->compute_with_storage_grid_size().y - 1};
    CoreCoord sub1_core_coord = {device->compute_with_storage_grid_size().x - 1, 0};

    if (arch_ == ARCH::QUASAR) {
        if (!unit_tests::dm::transaction_id::quasar_grid_ok(device)) {
            GTEST_SKIP() << "Skipping: need 3 distinct cores on Quasar emulator";
        }
        unit_tests::dm::transaction_id::TransactionIdConfig test_config = {
            .test_id = 621,
            .master_core_coord = master_core_coord,
            .sub0_core_coord = sub0_core_coord,
            .sub1_core_coord = sub1_core_coord,
            .num_of_trids = 4,
            .pages_per_transaction = 1,
            .bytes_per_page = bytes_per_page,
            .l1_data_format = DataFormat::Float16_b,
            .one_packet = true,
        };
        EXPECT_TRUE(unit_tests::dm::transaction_id::run_dm(mesh_device, test_config));
        return;
    }

    uint32_t max_pages_per_transaction = 256;
    for (uint32_t num_of_trids = 1; num_of_trids <= 16; num_of_trids *= 2) {
        for (uint32_t pages_per_transaction = 1; pages_per_transaction <= max_pages_per_transaction;
             pages_per_transaction *= 2) {
            if (pages_per_transaction * num_of_trids > max_transmittable_pages) {
                continue;
            }
            unit_tests::dm::transaction_id::TransactionIdConfig test_config = {
                .test_id = 621,
                .master_core_coord = master_core_coord,
                .sub0_core_coord = sub0_core_coord,
                .sub1_core_coord = sub1_core_coord,
                .num_of_trids = num_of_trids,
                .pages_per_transaction = pages_per_transaction,
                .bytes_per_page = bytes_per_page,
                .l1_data_format = DataFormat::Float16_b,
                .one_packet = true,
            };
            EXPECT_TRUE(unit_tests::dm::transaction_id::run_dm(mesh_device, test_config));
        }
    }
}

TEST_F(GenericMeshDeviceFixture, TensixDataMovementTransactionIdReadAfterWriteOnePacketStateful_2_0) {
    auto mesh_device = get_mesh_device();
    auto* device = mesh_device->impl().get_device(0);
    auto arch_ = device->arch();

    auto [bytes_per_page, max_transmittable_bytes, max_transmittable_pages] =
        unit_tests::dm::compute_physical_constraints(mesh_device);

    CoreCoord master_core_coord = {0, 0};
    CoreCoord sub0_core_coord = {0, device->compute_with_storage_grid_size().y - 1};
    CoreCoord sub1_core_coord = {device->compute_with_storage_grid_size().x - 1, 0};

    if (arch_ == ARCH::QUASAR) {
        if (!unit_tests::dm::transaction_id::quasar_grid_ok(device)) {
            GTEST_SKIP() << "Skipping: need 3 distinct cores on Quasar emulator";
        }
        unit_tests::dm::transaction_id::TransactionIdConfig test_config = {
            .test_id = 622,
            .master_core_coord = master_core_coord,
            .sub0_core_coord = sub0_core_coord,
            .sub1_core_coord = sub1_core_coord,
            .num_of_trids = 4,
            .pages_per_transaction = 1,
            .bytes_per_page = bytes_per_page,
            .l1_data_format = DataFormat::Float16_b,
            .one_packet = true,
            .stateful = true,
        };
        EXPECT_TRUE(unit_tests::dm::transaction_id::run_dm(mesh_device, test_config));
        return;
    }

    uint32_t max_pages_per_transaction = 256;
    for (uint32_t num_of_trids = 1; num_of_trids <= 16; num_of_trids *= 2) {
        for (uint32_t pages_per_transaction = 1; pages_per_transaction <= max_pages_per_transaction;
             pages_per_transaction *= 2) {
            if (pages_per_transaction * num_of_trids > max_transmittable_pages) {
                continue;
            }
            unit_tests::dm::transaction_id::TransactionIdConfig test_config = {
                .test_id = 622,
                .master_core_coord = master_core_coord,
                .sub0_core_coord = sub0_core_coord,
                .sub1_core_coord = sub1_core_coord,
                .num_of_trids = num_of_trids,
                .pages_per_transaction = pages_per_transaction,
                .bytes_per_page = bytes_per_page,
                .l1_data_format = DataFormat::Float16_b,
                .one_packet = true,
                .stateful = true,
            };
            EXPECT_TRUE(unit_tests::dm::transaction_id::run_dm(mesh_device, test_config));
        }
    }
}

TEST_F(GenericMeshDeviceFixture, TensixDataMovementTransactionIdWriteAfterRead_2_0) {
    auto mesh_device = get_mesh_device();
    auto* device = mesh_device->impl().get_device(0);
    auto arch_ = device->arch();

    auto [bytes_per_page, max_transmittable_bytes, max_transmittable_pages] =
        unit_tests::dm::compute_physical_constraints(mesh_device);

    CoreCoord master_core_coord = {0, 0};
    CoreCoord sub0_core_coord = {0, device->compute_with_storage_grid_size().y - 1};
    CoreCoord sub1_core_coord = {device->compute_with_storage_grid_size().x - 1, 0};

    if (arch_ == ARCH::QUASAR) {
        if (!unit_tests::dm::transaction_id::quasar_grid_ok(device)) {
            GTEST_SKIP() << "Skipping: need 3 distinct cores on Quasar emulator";
        }
        unit_tests::dm::transaction_id::TransactionIdConfig test_config = {
            .test_id = 630,
            .master_core_coord = master_core_coord,
            .sub0_core_coord = sub0_core_coord,
            .sub1_core_coord = sub1_core_coord,
            .num_of_trids = 4,
            .pages_per_transaction = 1,
            .bytes_per_page = bytes_per_page,
            .l1_data_format = DataFormat::Float16_b,
            .read_after_write = false,
        };
        EXPECT_TRUE(unit_tests::dm::transaction_id::run_dm(mesh_device, test_config));
        return;
    }

    uint32_t max_pages_per_transaction = arch_ == ARCH::BLACKHOLE ? 1024 : 2048;
    for (uint32_t num_of_trids = 1; num_of_trids <= 16; num_of_trids *= 2) {
        for (uint32_t pages_per_transaction = 1; pages_per_transaction <= max_pages_per_transaction;
             pages_per_transaction *= 2) {
            if (pages_per_transaction * num_of_trids > max_transmittable_pages) {
                continue;
            }
            unit_tests::dm::transaction_id::TransactionIdConfig test_config = {
                .test_id = 630,
                .master_core_coord = master_core_coord,
                .sub0_core_coord = sub0_core_coord,
                .sub1_core_coord = sub1_core_coord,
                .num_of_trids = num_of_trids,
                .pages_per_transaction = pages_per_transaction,
                .bytes_per_page = bytes_per_page,
                .l1_data_format = DataFormat::Float16_b,
                .read_after_write = false,
            };
            EXPECT_TRUE(unit_tests::dm::transaction_id::run_dm(mesh_device, test_config));
        }
    }
}

TEST_F(GenericMeshDeviceFixture, TensixDataMovementTransactionIdWriteAfterReadOnePacketStateful_2_0) {
    auto mesh_device = get_mesh_device();
    auto* device = mesh_device->impl().get_device(0);
    auto arch_ = device->arch();

    auto [bytes_per_page, max_transmittable_bytes, max_transmittable_pages] =
        unit_tests::dm::compute_physical_constraints(mesh_device);

    CoreCoord master_core_coord = {0, 0};
    CoreCoord sub0_core_coord = {0, device->compute_with_storage_grid_size().y - 1};
    CoreCoord sub1_core_coord = {device->compute_with_storage_grid_size().x - 1, 0};

    if (arch_ == ARCH::QUASAR) {
        if (!unit_tests::dm::transaction_id::quasar_grid_ok(device)) {
            GTEST_SKIP() << "Skipping: need 3 distinct cores on Quasar emulator";
        }
        unit_tests::dm::transaction_id::TransactionIdConfig test_config = {
            .test_id = 631,
            .master_core_coord = master_core_coord,
            .sub0_core_coord = sub0_core_coord,
            .sub1_core_coord = sub1_core_coord,
            .num_of_trids = 4,
            .pages_per_transaction = 1,
            .bytes_per_page = bytes_per_page,
            .l1_data_format = DataFormat::Float16_b,
            .one_packet = true,
            .stateful = true,
            .read_after_write = false,
        };
        EXPECT_TRUE(unit_tests::dm::transaction_id::run_dm(mesh_device, test_config));
        return;
    }

    uint32_t max_pages_per_transaction = 256;
    for (uint32_t num_of_trids = 1; num_of_trids <= 16; num_of_trids *= 2) {
        for (uint32_t pages_per_transaction = 1; pages_per_transaction <= max_pages_per_transaction;
             pages_per_transaction *= 2) {
            if (pages_per_transaction * num_of_trids > max_transmittable_pages) {
                continue;
            }
            unit_tests::dm::transaction_id::TransactionIdConfig test_config = {
                .test_id = 631,
                .master_core_coord = master_core_coord,
                .sub0_core_coord = sub0_core_coord,
                .sub1_core_coord = sub1_core_coord,
                .num_of_trids = num_of_trids,
                .pages_per_transaction = pages_per_transaction,
                .bytes_per_page = bytes_per_page,
                .l1_data_format = DataFormat::Float16_b,
                .one_packet = true,
                .stateful = true,
                .read_after_write = false,
            };
            EXPECT_TRUE(unit_tests::dm::transaction_id::run_dm(mesh_device, test_config));
        }
    }
}

}  // namespace tt::tt_metal
