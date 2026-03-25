// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "multi_device_fixture.hpp"
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/program.hpp>
#include "dm_common.hpp"
#include <distributed/mesh_device_impl.hpp>

namespace tt::tt_metal {

using namespace std;
using namespace tt;
using namespace test_utils;

namespace unit_tests::dm::pcie_read_bw {

// Test config for PCIe read bandwidth test
struct PCIeReadBwConfig {
    uint32_t test_id = 0;
    CoreCoord master_core_coord = {0, 0};
    uint32_t num_of_transactions = 0;
    uint32_t pages_per_transaction = 0;
    uint32_t bytes_per_page = 0;
    DataFormat l1_data_format = DataFormat::Float32;
    NOC noc_id = NOC::RISCV_0_default;
};

/// @brief Runs PCIe read bandwidth test
/// @param mesh_device Mesh device for execution
/// @param test_config Configuration for the test
/// @return true if test passes, false otherwise
bool run_dm(const shared_ptr<distributed::MeshDevice>& mesh_device, const PCIeReadBwConfig& test_config) {
    // Get the actual device for this single-device test
    IDevice* device = mesh_device->impl().get_device(0);
    auto device_id = device->id();

    // Program
    Program program = CreateProgram();

    const size_t bytes_per_transaction = test_config.pages_per_transaction * test_config.bytes_per_page;

    L1AddressInfo master_l1_info = unit_tests::dm::get_l1_address_and_size(mesh_device, test_config.master_core_coord);
    uint32_t l1_base_address = master_l1_info.base_address;

    if (master_l1_info.size < bytes_per_transaction) {
        log_error(LogTest, "Insufficient L1 size for the test configuration");
        return false;
    }

    // Get PCIe core coordinates
    const metal_SocDescriptor& soc_d = MetalContext::instance().get_cluster().get_soc_desc(device_id);
    vector<tt::umd::CoreCoord> pcie_cores = soc_d.get_cores(CoreType::PCIE, CoordSystem::TRANSLATED);
    TT_FATAL(!pcie_cores.empty(), "No PCIe cores found");

    // Physical Core Coordinates
    uint32_t packed_subordinate_core_coordinates = pcie_cores[0].x << 16 | (pcie_cores[0].y & 0xFFFF);

    // Get PCIe memory addresses
    uint64_t dev_pcie_base = MetalContext::instance().get_cluster().get_pcie_base_addr_from_device(device_id);
    constexpr uint64_t PCIE_OFFSET_BYTES = 1024 * 1024 * 50;  // 50MB offset to avoid conflicts
    uint64_t pcie_offset = PCIE_OFFSET_BYTES;
    uint64_t pcie_l1_local_addr = dev_pcie_base + pcie_offset;

    // Compile-time arguments for kernels
    uint32_t clock_freq_mhz = device->get_clock_rate_mhz();
    vector<uint32_t> compile_args = {
        (uint32_t)test_config.num_of_transactions,
        (uint32_t)bytes_per_transaction,
        (uint32_t)test_config.test_id,
        (uint32_t)packed_subordinate_core_coordinates,
        (uint32_t)pcie_l1_local_addr,
        (uint32_t)l1_base_address,
        clock_freq_mhz};

    std::string kernel_path = "tests/tt_metal/tt_metal/data_movement/pcie_read_bw/kernels/pcie_read_bw.cpp";
    CreateKernel(
        program,
        kernel_path,
        test_config.master_core_coord,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0, .noc = test_config.noc_id, .compile_args = compile_args});

    log_info(
        LogTest, "Running PCIe Read BW Test ID: {}, Run ID: {}", test_config.test_id, unit_tests::dm::runtime_host_id);
    program.set_runtime_id(unit_tests::dm::runtime_host_id++);

    auto mesh_workload = distributed::MeshWorkload();
    auto target_devices = distributed::MeshCoordinateRange(distributed::MeshCoordinate({0, 0}));
    mesh_workload.add_program(target_devices, std::move(program));

    auto& cq = mesh_device->mesh_command_queue();

    distributed::EnqueueMeshWorkload(cq, mesh_workload, false);
    distributed::Finish(cq);

    return true;
}

void pcie_read_bw_test(
    const shared_ptr<distributed::MeshDevice>& mesh_device, uint32_t test_id, CoreCoord master_core_coord = {0, 0}) {
    // Physical Constraints
    auto [bytes_per_page, max_transmittable_bytes, max_transmittable_pages] =
        tt::tt_metal::unit_tests::dm::compute_physical_constraints(mesh_device);

    uint32_t num_of_transactions = 256;
    uint32_t pages_per_transaction = max_transmittable_pages;

    PCIeReadBwConfig test_config = {
        .test_id = test_id,
        .master_core_coord = master_core_coord,
        .num_of_transactions = num_of_transactions,
        .pages_per_transaction = pages_per_transaction,
        .bytes_per_page = bytes_per_page,
        .l1_data_format = DataFormat::Float32,
        .noc_id = NOC::RISCV_0_default,
    };

    EXPECT_TRUE(run_dm(mesh_device, test_config));
}

}  // namespace unit_tests::dm::pcie_read_bw

TEST_F(GenericMeshDeviceFixture, PCIeReadBandwidth) {
    uint32_t test_id = 603;
    CoreCoord master_core_coord = {0, 0};

    unit_tests::dm::pcie_read_bw::pcie_read_bw_test(get_mesh_device(), test_id, master_core_coord);
}

/* ========== Sweep 1M transactions with varying transaction sizes; Test id = 605 ========== */
TEST_F(GenericMeshDeviceFixture, PCIeReadBandwidthSweep) {
    auto mesh_device = get_mesh_device();
    auto* device = mesh_device->impl().get_device(0);

    // Physical Constraints
    auto [page_size_bytes, max_transmittable_bytes, max_transmittable_pages] =
        tt::tt_metal::unit_tests::dm::compute_physical_constraints(mesh_device);

    // Max transaction size: 16 kB for BH, 8 kB for WH (NOC max packet size)
    uint32_t max_transaction_size_bytes = device->arch() == tt::ARCH::BLACKHOLE ? 16 * 1024 : 8 * 1024;

    // Cap to L1 available size
    if (max_transaction_size_bytes > max_transmittable_bytes) {
        max_transaction_size_bytes = max_transmittable_bytes;
    }

    constexpr uint32_t total_transactions = 1000000;
    CoreCoord master_core_coord = {0, 0};

    // Sweep transaction sizes by powers of 2 from page_size_bytes to max
    for (uint32_t txn_size = page_size_bytes; txn_size <= max_transaction_size_bytes; txn_size *= 2) {
        uint32_t pages_per_txn = txn_size / page_size_bytes;

        unit_tests::dm::pcie_read_bw::PCIeReadBwConfig test_config = {
            .test_id = 605,
            .master_core_coord = master_core_coord,
            .num_of_transactions = total_transactions,
            .pages_per_transaction = pages_per_txn,
            .bytes_per_page = page_size_bytes,
            .l1_data_format = DataFormat::Float32,
            .noc_id = NOC::RISCV_0_default,
        };

        EXPECT_TRUE(unit_tests::dm::pcie_read_bw::run_dm(mesh_device, test_config));
    }
}

}  // namespace tt::tt_metal
