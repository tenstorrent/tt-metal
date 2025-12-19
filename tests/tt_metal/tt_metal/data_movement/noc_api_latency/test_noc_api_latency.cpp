// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "data_types.hpp"
#include "multi_device_fixture.hpp"
#include "dm_common.hpp"
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/mesh_coord.hpp>

namespace tt::tt_metal {

using namespace std;
using namespace tt;
using namespace test_utils;

namespace unit_tests::dm::noc_api_latency {

// Kernel type enum
enum class KernelType {
    UNICAST_WRITE = 0,
    UNICAST_READ = 1,
    MULTICAST_WRITE = 2,
    STATEFUL_WRITE = 3,
    STATEFUL_READ = 4
};

// Test config
struct NocApiLatencyConfig {
    uint32_t test_id = 0;
    CoreCoord source_core_coord = {0, 0};
    CoreCoord dest_core_coord = {0, 1};
    CoreCoord mcast_dest_core_start = {0, 1};
    CoreCoord mcast_dest_core_end = {0, 1};
    uint32_t num_transactions = 1;
    uint32_t transaction_size = 32;  // bytes
    KernelType kernel_type = KernelType::UNICAST_WRITE;
    NOC noc_id = NOC::NOC_0;
    bool loopback = false;  // For multicast: include source in destinations
};

/// @brief Measures NOC API call latency
/// @param mesh_device - MeshDevice to run the test on
/// @param test_config - Configuration of the test -- see struct
/// @return true if test passes
bool run_noc_api_latency_test(
    const shared_ptr<distributed::MeshDevice>& mesh_device, const NocApiLatencyConfig& test_config) {
    // Get the actual device for this single-device test
    IDevice* device = mesh_device->get_device(0);

    /* ================ SETUP ================ */

    // Program
    Program program = CreateProgram();

    // (Logical) Core coordinates and ranges

    CoreRangeSet sub_logical_core_set({CoreRange(test_config.mcast_dest_core_start, test_config.mcast_dest_core_end)});

    // Obtain L1 Address for Storing Data
    L1AddressInfo source_l1_info = unit_tests::dm::get_l1_address_and_size(mesh_device, test_config.source_core_coord);
    L1AddressInfo dest_l1_info = unit_tests::dm::get_l1_address_and_size(mesh_device, test_config.dest_core_coord);

    // Check if the L1 size is sufficient for the test configuration
    if (source_l1_info.size < test_config.transaction_size || dest_l1_info.size < test_config.transaction_size) {
        log_error(LogTest, "Insufficient L1 size for the test configuration");
        return false;
    }

    uint32_t l1_base_address = source_l1_info.base_address;

    // Physical Core Coordinates
    CoreCoord physical_dest_core = device->worker_core_from_logical_core(test_config.dest_core_coord);
    uint32_t packed_dest_core_coordinates = physical_dest_core.x << 16 | (physical_dest_core.y & 0xFFFF);

    // For multicast tests, use configured multicast rectangle
    CoreCoord physical_mcast_dest_start = device->worker_core_from_logical_core(test_config.mcast_dest_core_start);
    CoreCoord physical_mcast_dest_end = device->worker_core_from_logical_core(test_config.mcast_dest_core_end);
    uint32_t packed_dest_core_end_coordinates = physical_mcast_dest_end.x << 16 | (physical_mcast_dest_end.y & 0xFFFF);

    // For non-multicast tests, override with unicast destination
    if (test_config.kernel_type != KernelType::MULTICAST_WRITE) {
        packed_dest_core_end_coordinates = packed_dest_core_coordinates;
    } else {
        // For multicast, update packed_dest_core_coordinates to be the start of the rectangle
        packed_dest_core_coordinates = physical_mcast_dest_start.x << 16 | (physical_mcast_dest_start.y & 0xFFFF);
    }

    // Compile-time arguments for kernels
    vector<uint32_t> compile_args = {
        (uint32_t)l1_base_address,
        (uint32_t)test_config.num_transactions,
        (uint32_t)test_config.transaction_size,
        (uint32_t)test_config.test_id,
        (uint32_t)packed_dest_core_coordinates,
        (uint32_t)packed_dest_core_end_coordinates,
        (uint32_t)test_config.loopback};

    if (test_config.kernel_type == KernelType::MULTICAST_WRITE) {
        compile_args.push_back(sub_logical_core_set.num_cores());
    }

    // Select kernel based on type
    std::string kernels_dir = "tests/tt_metal/tt_metal/data_movement/noc_api_latency/kernels/";
    std::string kernel_filename;

    switch (test_config.kernel_type) {
        case KernelType::UNICAST_WRITE: kernel_filename = "unicast_write_2_0"; break;
        case KernelType::UNICAST_READ: kernel_filename = "unicast_read_2_0"; break;
        case KernelType::MULTICAST_WRITE: kernel_filename = "multicast_write_2_0"; break;
        case KernelType::STATEFUL_WRITE: kernel_filename = "stateful_write_2_0"; break;
        case KernelType::STATEFUL_READ: kernel_filename = "stateful_read_2_0"; break;
        default: log_error(LogTest, "Invalid kernel type"); return false;
    }

    std::string kernel_path = kernels_dir + kernel_filename + ".cpp";

    auto riscv = DataMovementProcessor::RISCV_0;

    if (test_config.kernel_type == KernelType::UNICAST_READ || test_config.kernel_type == KernelType::STATEFUL_READ) {
        riscv = DataMovementProcessor::RISCV_1;
    }

    // Create kernel on source core
    CreateKernel(
        program,
        kernel_path,
        test_config.source_core_coord,
        DataMovementConfig{.processor = riscv, .noc = test_config.noc_id, .compile_args = compile_args});

    // Assign unique id
    log_info(LogTest, "Running Test ID: {}, Run ID: {}", test_config.test_id, unit_tests::dm::runtime_host_id);
    program.set_runtime_id(unit_tests::dm::runtime_host_id++);

    /* ================ RUNNING THE PROGRAM ================ */

    MetalContext::instance().get_cluster().l1_barrier(device->id());

    auto mesh_workload = distributed::MeshWorkload();
    vector<uint32_t> coord_data = {0, 0};
    auto target_devices = distributed::MeshCoordinateRange(distributed::MeshCoordinate(coord_data));
    mesh_workload.add_program(target_devices, std::move(program));

    auto& cq = mesh_device->mesh_command_queue();
    distributed::EnqueueMeshWorkload(cq, mesh_workload, false);
    Finish(cq);

    // For latency tests, we don't validate data correctness - we just measure cycles

    return true;
}

// Sweep test for unicast write and read
void unicast_sweep_test(
    const shared_ptr<distributed::MeshDevice>& mesh_device, uint32_t test_id, KernelType kernel_type) {
    auto* device = mesh_device->get_device(0);
    for (uint32_t transaction_size = 32; transaction_size <= 4096; transaction_size *= 2) {
        for (uint32_t num_transactions = 1; num_transactions <= 256; num_transactions *= 2) {
            NocApiLatencyConfig test_config = {
                .test_id = test_id,
                .source_core_coord = {0, 0},
                .dest_core_coord = {0, device->compute_with_storage_grid_size().y - 1},
                .num_transactions = num_transactions,
                .transaction_size = transaction_size,
                .kernel_type = kernel_type,
                .noc_id = NOC::NOC_0};

            EXPECT_TRUE(run_noc_api_latency_test(mesh_device, test_config));
        }
    }
}

// Sweep test for multicast write with configurable grid
void multicast_write_sweep_test(
    const shared_ptr<distributed::MeshDevice>& mesh_device,
    uint32_t test_id,
    CoreCoord mcast_start,
    CoreCoord mcast_end,
    bool loopback) {
    for (uint32_t transaction_size = 32; transaction_size <= 4096; transaction_size *= 2) {
        for (uint32_t num_transactions = 1; num_transactions <= 256; num_transactions *= 2) {
            NocApiLatencyConfig test_config = {
                .test_id = test_id,
                .source_core_coord = {0, 0},
                .dest_core_coord = {0, 0},  // not used
                .mcast_dest_core_start = mcast_start,
                .mcast_dest_core_end = mcast_end,
                .num_transactions = num_transactions,
                .transaction_size = transaction_size,
                .kernel_type = KernelType::MULTICAST_WRITE,
                .noc_id = NOC::NOC_0,
                .loopback = loopback};

            EXPECT_TRUE(run_noc_api_latency_test(mesh_device, test_config));
        }
    }
}

}  // namespace unit_tests::dm::noc_api_latency

// Test definitions
TEST_F(GenericMeshDeviceFixture, TensixNocApiLatencyUnicastWrite) {
    uint32_t test_case_id = 700;
    tt::tt_metal::unit_tests::dm::noc_api_latency::unicast_sweep_test(
        this->mesh_device_, test_case_id, unit_tests::dm::noc_api_latency::KernelType::UNICAST_WRITE);
}

TEST_F(GenericMeshDeviceFixture, TensixNocApiLatencyUnicastRead) {
    uint32_t test_case_id = 701;
    tt::tt_metal::unit_tests::dm::noc_api_latency::unicast_sweep_test(
        this->mesh_device_, test_case_id, unit_tests::dm::noc_api_latency::KernelType::UNICAST_READ);
}

TEST_F(GenericMeshDeviceFixture, TensixNocApiLatencyStatefulWrite) {
    uint32_t test_case_id = 702;
    tt::tt_metal::unit_tests::dm::noc_api_latency::unicast_sweep_test(
        this->mesh_device_, test_case_id, unit_tests::dm::noc_api_latency::KernelType::STATEFUL_WRITE);
}

TEST_F(GenericMeshDeviceFixture, TensixNocApiLatencyStatefulRead) {
    uint32_t test_case_id = 703;
    tt::tt_metal::unit_tests::dm::noc_api_latency::unicast_sweep_test(
        this->mesh_device_, test_case_id, unit_tests::dm::noc_api_latency::KernelType::STATEFUL_READ);
}

TEST_F(GenericMeshDeviceFixture, TensixNocApiLatencyMulticastWrite2x2) {
    uint32_t test_case_id = 704;
    tt::tt_metal::unit_tests::dm::noc_api_latency::multicast_write_sweep_test(
        this->mesh_device_, test_case_id, {0, 1}, {1, 2}, false);
}

TEST_F(GenericMeshDeviceFixture, TensixNocApiLatencyMulticastWrite5x5) {
    uint32_t test_case_id = 705;
    tt::tt_metal::unit_tests::dm::noc_api_latency::multicast_write_sweep_test(
        this->mesh_device_, test_case_id, {0, 1}, {4, 5}, false);
}

TEST_F(GenericMeshDeviceFixture, TensixNocApiLatencyMulticastWriteAll) {
    uint32_t test_case_id = 706;
    auto* device = this->mesh_device_->get_device(0);
    CoreCoord grid_size = device->compute_with_storage_grid_size();
    tt::tt_metal::unit_tests::dm::noc_api_latency::multicast_write_sweep_test(
        this->mesh_device_, test_case_id, {0, 0}, {grid_size.x - 1, grid_size.y - 1}, true);
}

}  // namespace tt::tt_metal
