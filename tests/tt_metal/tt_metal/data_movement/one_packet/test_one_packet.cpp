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
using namespace tt::test_utils;

namespace unit_tests::dm::one_packet {
// Test config, i.e. test parameters
struct OnePacketConfig {
    uint32_t test_id = 0;
    CoreCoord master_core_coord;
    CoreCoord subordinate_core_coord;
    uint32_t num_packets = 0;
    uint32_t packet_size_bytes = 0;
    bool read = true;
};

/// @brief Does OneToOne or OneFromOne but with one_packet read/write
/// @param mesh_device - MeshDevice to run the test on
/// @param test_config - Configuration of the test -- see struct
/// @return
bool run_dm(const shared_ptr<distributed::MeshDevice>& mesh_device, const OnePacketConfig& test_config) {
    // Get the actual device for this single-device test
    IDevice* device = mesh_device->impl().get_device(0);
    // (Logical) Core Coordinates and ranges
    CoreRangeSet master_core_set({CoreRange(test_config.master_core_coord)});

    // Obtain L1 Address for Storing Data
    // NOTE: We don't know if the whole block of memory is actually available.
    //       This is something that could probably be checked
    L1AddressInfo master_l1_info =
        tt::tt_metal::unit_tests::dm::get_l1_address_and_size(mesh_device, test_config.master_core_coord);
    L1AddressInfo subordinate_l1_info =
        tt::tt_metal::unit_tests::dm::get_l1_address_and_size(mesh_device, test_config.subordinate_core_coord);
    // Checks that both master and subordinate cores have the same L1 base address and size
    if (master_l1_info.base_address != subordinate_l1_info.base_address ||
        master_l1_info.size != subordinate_l1_info.size) {
        log_error(tt::LogTest, "Mismatch in L1 address or size between master and subordinate cores");
        return false;
    }
    // Check if the L1 size is sufficient for the test configuration
    if (master_l1_info.size < test_config.packet_size_bytes) {
        log_error(tt::LogTest, "Insufficient L1 size for the test configuration");
        return false;
    }

    uint32_t master_l1_address = master_l1_info.base_address;
    uint32_t subordinate_l1_address = subordinate_l1_info.base_address;

    const std::string kernel_path = std::string("tests/tt_metal/tt_metal/data_movement/one_packet/kernels/") +
                                    (test_config.read ? "read_one_packet_2_0" : "write_one_packet_2_0") + ".cpp";

    CoreCoord physical_subordinate_core = device->worker_core_from_logical_core(test_config.subordinate_core_coord);

    using namespace tt::tt_metal::experimental;

    KernelSpec::CompileTimeArgs cta_bindings = {{"test_id", (uint32_t)test_config.test_id}};

    const DataMovementProcessor proc =
        test_config.read ? DataMovementProcessor::RISCV_1 : DataMovementProcessor::RISCV_0;
    const NOC noc = test_config.read ? NOC::RISCV_1_default : NOC::RISCV_0_default;

    DataMovementHardwareConfig kspec_hw_config;
    if (device->arch() == tt::ARCH::QUASAR) {
        kspec_hw_config = DataMovementGen2Config{};
    } else {
        kspec_hw_config = DataMovementGen1Config{
            .processor = proc,
            .noc = noc,
        };
    }
    KernelSpec kspec{
        .unique_id = KernelSpecName{"one_packet_kernel"},
        .source = std::filesystem::path{kernel_path},
        .num_threads = 1,
        .compile_time_args = cta_bindings,
        .runtime_arg_schema =
            {
                .runtime_arg_names =
                    {"num_packets",
                     "packet_size_bytes",
                     "master_l1_addr",
                     "subordinate_l1_addr",
                     "responder_x",
                     "responder_y"},
            },
        .hw_config = kspec_hw_config,
    };

    ProgramSpec spec{
        .name = "one_packet_test",
        .kernels = {kspec},
        .work_units = {WorkUnitSpec{
            .name = "work_unit",
            .kernels = {kspec.unique_id},
            .target_nodes = master_core_set,
        }},
    };

    Program program = MakeProgramFromSpec(*mesh_device, spec);

    ProgramRunArgs run_params;
    ProgramRunArgs::KernelRunArgs krp{.kernel = kspec.unique_id};
    krp.runtime_arg_values.push_back(
        {.node = test_config.master_core_coord,
         .args = {
             {"num_packets", (uint32_t)test_config.num_packets},
             {"packet_size_bytes", (uint32_t)test_config.packet_size_bytes},
             {"master_l1_addr", master_l1_address},
             {"subordinate_l1_addr", subordinate_l1_address},
             {"responder_x", (uint32_t)physical_subordinate_core.x},
             {"responder_y", (uint32_t)physical_subordinate_core.y}}});
    run_params.kernel_run_args.push_back(krp);
    SetProgramRunArgs(program, run_params);

    // Assign unique id
    log_info(tt::LogTest, "Running Test ID: {}, Run ID: {}", test_config.test_id, unit_tests::dm::runtime_host_id);
    program.set_runtime_id(unit_tests::dm::runtime_host_id++);

    // Input
    vector<uint32_t> packed_input = generate_packed_uniform_random_vector<uint32_t, bfloat16>(
        -100.0f,
        100.0f,
        test_config.packet_size_bytes / sizeof(bfloat16),
        chrono::system_clock::now().time_since_epoch().count());

    // Golden output
    vector<uint32_t> packed_golden = packed_input;
    vector<uint32_t> packed_output;

    // Launch program and record outputs
    if (test_config.read) {
        detail::WriteToDeviceL1(device, test_config.subordinate_core_coord, subordinate_l1_address, packed_input);
        MetalContext::instance().get_cluster().l1_barrier(device->id());

        auto mesh_workload = distributed::MeshWorkload();
        vector<uint32_t> coord_data = {0, 0};
        auto target_devices =
            distributed::MeshCoordinateRange(distributed::MeshCoordinate(coord_data));  // Single device at (0,0)
        mesh_workload.add_program(target_devices, std::move(program));

        auto& cq = mesh_device->mesh_command_queue();
        distributed::EnqueueMeshWorkload(cq, mesh_workload, false);
        Finish(cq);

        detail::ReadFromDeviceL1(
            device, test_config.master_core_coord, master_l1_address, test_config.packet_size_bytes, packed_output);
    } else {
        detail::WriteToDeviceL1(device, test_config.master_core_coord, master_l1_address, packed_input);
        MetalContext::instance().get_cluster().l1_barrier(device->id());

        auto mesh_workload = distributed::MeshWorkload();
        vector<uint32_t> coord_data = {0, 0};
        auto target_devices =
            distributed::MeshCoordinateRange(distributed::MeshCoordinate(coord_data));  // Single device at (0,0)
        mesh_workload.add_program(target_devices, std::move(program));

        auto& cq = mesh_device->mesh_command_queue();
        distributed::EnqueueMeshWorkload(cq, mesh_workload, false);
        Finish(cq);

        detail::ReadFromDeviceL1(
            device,
            test_config.subordinate_core_coord,
            subordinate_l1_address,
            test_config.packet_size_bytes,
            packed_output);
    }

    // Results comparison
    bool is_equal = (packed_output == packed_golden);

    if (!is_equal) {
        log_error(tt::LogTest, "Equality Check failed");
        log_info(tt::LogTest, "Golden vector");
        print_vector<uint32_t>(packed_golden);
        log_info(tt::LogTest, "Output vector");
        print_vector<uint32_t>(packed_output);
    }

    return is_equal;
}
}  // namespace unit_tests::dm::one_packet

/* ========== Test case for reading varying number of packets and packet sizes; Test id = 80 ========== */
TEST_F(GenericMeshDeviceFixture, TensixDataMovementOnePacketReadSizes) {
    auto mesh_device = get_mesh_device();
    auto* device = mesh_device->impl().get_device(0);
    // Physical Constraints
    auto [page_size_bytes, max_transmittable_bytes, max_transmittable_pages] =
        tt::tt_metal::unit_tests::dm::compute_physical_constraints(mesh_device);

    if (device->arch() == ARCH::QUASAR) {
        // subordinate_core_coord {1, 0} requires at least 2 columns in the compute grid
        if (device->compute_with_storage_grid_size().x < 2) {
            GTEST_SKIP() << "Skipping: subordinate core {1, 0} requires >= 2 columns, but grid has "
                         << device->compute_with_storage_grid_size().x << " column(s). Use emu-quasar-2x3 or larger.";
        }
        // Single run to validate the Quasar code path within emulator timeout
        unit_tests::dm::one_packet::OnePacketConfig test_config = {
            .test_id = 80,
            .master_core_coord = {0, 0},
            .subordinate_core_coord = {1, 0},
            .num_packets = 4,
            .packet_size_bytes = page_size_bytes,
            .read = true,
        };
        EXPECT_TRUE(run_dm(mesh_device, test_config));
        return;
    }

    // Parameters
    uint32_t max_packet_size_bytes =
        device->arch() == tt::ARCH::BLACKHOLE ? 16 * 1024 : 8 * 1024;  // 16 kB for BH, 8 kB for WH
    uint32_t max_packets = 256;

    // Cores
    CoreCoord master_core_coord = {0, 0};
    CoreCoord subordinate_core_coord = {0, 1};

    for (uint32_t num_packets = 1; num_packets <= max_packets; num_packets *= 4) {
        for (uint32_t packet_size_bytes = page_size_bytes; packet_size_bytes <= max_packet_size_bytes;
             packet_size_bytes *= 2) {
            // Test config
            unit_tests::dm::one_packet::OnePacketConfig test_config = {
                .test_id = 80,
                .master_core_coord = master_core_coord,
                .subordinate_core_coord = subordinate_core_coord,
                .num_packets = num_packets,
                .packet_size_bytes = packet_size_bytes,
                .read = true,
            };

            // Run
            EXPECT_TRUE(run_dm(mesh_device, test_config));
        }
    }
}

/* ========== Test case for writing varying number of packets and packet sizes; Test id = 81 ========== */
TEST_F(GenericMeshDeviceFixture, TensixDataMovementOnePacketWriteSizes) {
    auto mesh_device = get_mesh_device();
    auto* device = mesh_device->impl().get_device(0);

    // Physical Constraints
    auto [page_size_bytes, max_transmittable_bytes, max_transmittable_pages] =
        tt::tt_metal::unit_tests::dm::compute_physical_constraints(mesh_device);

    if (device->arch() == ARCH::QUASAR) {
        // subordinate_core_coord {1, 0} requires at least 2 columns in the compute grid
        if (device->compute_with_storage_grid_size().x < 2) {
            GTEST_SKIP() << "Skipping: subordinate core {1, 0} requires >= 2 columns, but grid has "
                         << device->compute_with_storage_grid_size().x << " column(s). Use emu-quasar-2x3 or larger.";
        }
        // Single run to validate the Quasar code path within emulator timeout
        unit_tests::dm::one_packet::OnePacketConfig test_config = {
            .test_id = 81,
            .master_core_coord = {0, 0},
            .subordinate_core_coord = {1, 0},
            .num_packets = 4,
            .packet_size_bytes = page_size_bytes,
            .read = false,
        };
        EXPECT_TRUE(run_dm(mesh_device, test_config));
        return;
    }

    // Parameters
    uint32_t max_packet_size_bytes =
        device->arch() == tt::ARCH::BLACKHOLE ? 16 * 1024 : 8 * 1024;  // 16 kB for BH, 8 kB for WH
    uint32_t max_packets = 256;
    // Cores
    CoreCoord master_core_coord = {0, 0};
    CoreCoord subordinate_core_coord = {0, 1};

    for (uint32_t num_packets = 1; num_packets <= max_packets; num_packets *= 4) {
        for (uint32_t packet_size_bytes = page_size_bytes; packet_size_bytes <= max_packet_size_bytes;
             packet_size_bytes *= 2) {
            // Test config
            unit_tests::dm::one_packet::OnePacketConfig test_config = {
                .test_id = 81,
                .master_core_coord = master_core_coord,
                .subordinate_core_coord = subordinate_core_coord,
                .num_packets = num_packets,
                .packet_size_bytes = packet_size_bytes,
                .read = false,
            };

            // Run
            EXPECT_TRUE(run_dm(mesh_device, test_config));
        }
    }
}

/* ========== Directed Ideal Test Case; Test id = 82 ========== */
TEST_F(GenericMeshDeviceFixture, TensixDataMovementOnePacketReadDirectedIdeal) {
    auto mesh_device = get_mesh_device();
    auto [page_size_bytes, max_transmittable_bytes, max_transmittable_pages] =
        tt::tt_metal::unit_tests::dm::compute_physical_constraints(mesh_device);

    // Parameters
    uint32_t packet_size_bytes = page_size_bytes * 256;  // max packet size = 256 flits
    uint32_t num_packets = max_transmittable_bytes / packet_size_bytes;

    // Cores
    /*
        Any two cores that are next to each other on the torus
         - May be worth considering the performance of this test with different pairs of adjacent cores
    */
    CoreCoord master_core_coord = {0, 0};
    CoreCoord subordinate_core_coord = {0, 1};

    // Test Config
    unit_tests::dm::one_packet::OnePacketConfig test_config = {
        .test_id = 82,
        .master_core_coord = master_core_coord,
        .subordinate_core_coord = subordinate_core_coord,
        .num_packets = num_packets,
        .packet_size_bytes = packet_size_bytes,
        .read = true,
    };

    // Run
    EXPECT_TRUE(run_dm(mesh_device, test_config));
}

/* ========== Directed Ideal Test Case; Test id = 83 ========== */
TEST_F(GenericMeshDeviceFixture, TensixDataMovementOnePacketWriteDirectedIdeal) {
    auto mesh_device = get_mesh_device();
    auto [page_size_bytes, max_transmittable_bytes, max_transmittable_pages] =
        tt::tt_metal::unit_tests::dm::compute_physical_constraints(mesh_device);

    // Parameters
    uint32_t packet_size_bytes = page_size_bytes * 256;  // max packet size = 256 flits
    uint32_t num_packets = max_transmittable_bytes / packet_size_bytes;

    // Cores
    /*
        Any two cores that are next to each other on the torus
         - May be worth considering the performance of this test with different pairs of adjacent cores
    */
    CoreCoord master_core_coord = {0, 0};
    CoreCoord subordinate_core_coord = {0, 1};

    // Test Config
    unit_tests::dm::one_packet::OnePacketConfig test_config = {
        .test_id = 83,
        .master_core_coord = master_core_coord,
        .subordinate_core_coord = subordinate_core_coord,
        .num_packets = num_packets,
        .packet_size_bytes = packet_size_bytes,
        .read = false,
    };

    // Run
    EXPECT_TRUE(run_dm(mesh_device, test_config));
}

TEST_F(GenericMeshDeviceFixture, TensixDataMovementOnePacketReadSizes_2_0) {
    auto mesh_device = get_mesh_device();
    auto* device = mesh_device->impl().get_device(0);
    auto [page_size_bytes, max_transmittable_bytes, max_transmittable_pages] =
        tt::tt_metal::unit_tests::dm::compute_physical_constraints(mesh_device);

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
        unit_tests::dm::one_packet::OnePacketConfig test_config = {
            .test_id = 90,
            .master_core_coord = {0, 0},
            .subordinate_core_coord = subordinate,
            .num_packets = 4,
            .packet_size_bytes = page_size_bytes,
            .read = true,
        };
        EXPECT_TRUE(unit_tests::dm::one_packet::run_dm(mesh_device, test_config));
        return;
    }

    // WH/BH: full sweep with Metal 2.0 host path.
    uint32_t max_packet_size_bytes = device->arch() == tt::ARCH::BLACKHOLE ? 16 * 1024 : 8 * 1024;
    uint32_t max_packets = 256;
    CoreCoord master_core_coord = {0, 0};
    CoreCoord subordinate_core_coord = {0, 1};

    for (uint32_t num_packets = 1; num_packets <= max_packets; num_packets *= 4) {
        for (uint32_t packet_size_bytes = page_size_bytes; packet_size_bytes <= max_packet_size_bytes;
             packet_size_bytes *= 2) {
            unit_tests::dm::one_packet::OnePacketConfig test_config = {
                .test_id = 90,
                .master_core_coord = master_core_coord,
                .subordinate_core_coord = subordinate_core_coord,
                .num_packets = num_packets,
                .packet_size_bytes = packet_size_bytes,
                .read = true,
            };
            EXPECT_TRUE(unit_tests::dm::one_packet::run_dm(mesh_device, test_config));
        }
    }
}

TEST_F(GenericMeshDeviceFixture, TensixDataMovementOnePacketWriteSizes_2_0) {
    auto mesh_device = get_mesh_device();
    auto* device = mesh_device->impl().get_device(0);
    auto [page_size_bytes, max_transmittable_bytes, max_transmittable_pages] =
        tt::tt_metal::unit_tests::dm::compute_physical_constraints(mesh_device);

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
        unit_tests::dm::one_packet::OnePacketConfig test_config = {
            .test_id = 91,
            .master_core_coord = {0, 0},
            .subordinate_core_coord = subordinate,
            .num_packets = 4,
            .packet_size_bytes = page_size_bytes,
            .read = false,
        };
        EXPECT_TRUE(unit_tests::dm::one_packet::run_dm(mesh_device, test_config));
        return;
    }

    // WH/BH: full sweep with Metal 2.0 host path.
    uint32_t max_packet_size_bytes = device->arch() == tt::ARCH::BLACKHOLE ? 16 * 1024 : 8 * 1024;
    uint32_t max_packets = 257;
    CoreCoord master_core_coord = {0, 0};
    CoreCoord subordinate_core_coord = {0, 1};

    for (uint32_t num_packets = 1; num_packets <= max_packets; num_packets *= 4) {
        for (uint32_t packet_size_bytes = page_size_bytes; packet_size_bytes <= max_packet_size_bytes;
             packet_size_bytes *= 2) {
            unit_tests::dm::one_packet::OnePacketConfig test_config = {
                .test_id = 91,
                .master_core_coord = master_core_coord,
                .subordinate_core_coord = subordinate_core_coord,
                .num_packets = num_packets,
                .packet_size_bytes = packet_size_bytes,
                .read = false,
            };
            EXPECT_TRUE(unit_tests::dm::one_packet::run_dm(mesh_device, test_config));
        }
    }
}

TEST_F(GenericMeshDeviceFixture, TensixDataMovementOnePacketReadDirectedIdeal_2_0) {
    auto mesh_device = get_mesh_device();
    auto* device = mesh_device->impl().get_device(0);
    auto [page_size_bytes, max_transmittable_bytes, max_transmittable_pages] =
        tt::tt_metal::unit_tests::dm::compute_physical_constraints(mesh_device);

    CoreCoord master_core_coord = {0, 0};
    CoreCoord subordinate_core_coord = {0, 1};

    if (device->arch() == ARCH::QUASAR) {
        auto grid = device->compute_with_storage_grid_size();
        if (grid.x >= 2) {
            subordinate_core_coord = {1, 0};
        } else if (grid.y >= 2) {
            subordinate_core_coord = {0, 1};
        } else {
            GTEST_SKIP() << "Skipping: need at least a 1x2 or 2x1 grid, got " << grid.x << "x" << grid.y;
        }
        unit_tests::dm::one_packet::OnePacketConfig test_config = {
            .test_id = 92,
            .master_core_coord = master_core_coord,
            .subordinate_core_coord = subordinate_core_coord,
            .num_packets = 4,
            .packet_size_bytes = page_size_bytes,
            .read = true,
        };
        EXPECT_TRUE(unit_tests::dm::one_packet::run_dm(mesh_device, test_config));
        return;
    }

    uint32_t packet_size_bytes = page_size_bytes * 256;
    uint32_t num_packets = max_transmittable_bytes / packet_size_bytes;

    unit_tests::dm::one_packet::OnePacketConfig test_config = {
        .test_id = 92,
        .master_core_coord = master_core_coord,
        .subordinate_core_coord = subordinate_core_coord,
        .num_packets = num_packets,
        .packet_size_bytes = packet_size_bytes,
        .read = true,
    };
    EXPECT_TRUE(unit_tests::dm::one_packet::run_dm(mesh_device, test_config));
}

TEST_F(GenericMeshDeviceFixture, TensixDataMovementOnePacketWriteDirectedIdeal_2_0) {
    auto mesh_device = get_mesh_device();
    auto* device = mesh_device->impl().get_device(0);
    auto [page_size_bytes, max_transmittable_bytes, max_transmittable_pages] =
        tt::tt_metal::unit_tests::dm::compute_physical_constraints(mesh_device);

    CoreCoord master_core_coord = {0, 0};
    CoreCoord subordinate_core_coord = {0, 1};

    if (device->arch() == ARCH::QUASAR) {
        auto grid = device->compute_with_storage_grid_size();
        if (grid.x >= 2) {
            subordinate_core_coord = {1, 0};
        } else if (grid.y >= 2) {
            subordinate_core_coord = {0, 1};
        } else {
            GTEST_SKIP() << "Skipping: need at least a 1x2 or 2x1 grid, got " << grid.x << "x" << grid.y;
        }
        unit_tests::dm::one_packet::OnePacketConfig test_config = {
            .test_id = 93,
            .master_core_coord = master_core_coord,
            .subordinate_core_coord = subordinate_core_coord,
            .num_packets = 4,
            .packet_size_bytes = page_size_bytes,
            .read = false,
        };
        EXPECT_TRUE(unit_tests::dm::one_packet::run_dm(mesh_device, test_config));
        return;
    }

    uint32_t packet_size_bytes = page_size_bytes * 256;
    uint32_t num_packets = max_transmittable_bytes / packet_size_bytes;

    unit_tests::dm::one_packet::OnePacketConfig test_config = {
        .test_id = 93,
        .master_core_coord = master_core_coord,
        .subordinate_core_coord = subordinate_core_coord,
        .num_packets = num_packets,
        .packet_size_bytes = packet_size_bytes,
        .read = false,
    };
    EXPECT_TRUE(unit_tests::dm::one_packet::run_dm(mesh_device, test_config));
}

}  // namespace tt::tt_metal
