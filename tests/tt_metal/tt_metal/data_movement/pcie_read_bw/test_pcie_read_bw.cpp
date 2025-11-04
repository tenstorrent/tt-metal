// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "multi_device_fixture.hpp"
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/circular_buffer_config.hpp>
#include <tt-metalium/data_types.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/program.hpp>
#include "dm_common.hpp"
#include <chrono>
#include <iomanip>
#include <sstream>

namespace tt::tt_metal {

using namespace std;
using namespace tt;
using namespace test_utils;

namespace unit_tests::dm::pcie_read_bw {

// Test config for PCIe read bandwidth test
struct PCIeReadBwConfig {
    uint32_t test_id = 0;
    CoreCoord worker_core_coord = {0, 0};
    uint32_t iterations = 32;
    uint32_t warmup_iterations = 2;
    uint32_t page_size_bytes = 65536;
    uint32_t batch_size_k = 256;
    uint32_t size_bytes = batch_size_k * 1024;
    uint32_t page_count = size_bytes / page_size_bytes;
    DataFormat l1_data_format = DataFormat::Float32;
    NOC noc_id = NOC::RISCV_1_default;
};

/// @brief Runs PCIe read bandwidth test
/// @param mesh_device Mesh device for execution
/// @param test_config Configuration for the test
/// @return true if test passes, false otherwise
bool run_pcie_read_bw_test(
    const shared_ptr<distributed::MeshDevice>& mesh_device, const PCIeReadBwConfig& test_config) {
    IDevice* device = mesh_device->get_device(0);
    auto device_id = device->id();

    // Program
    Program program = CreateProgram();

    const size_t total_data_size = test_config.page_size_bytes * test_config.page_count;

    // Core range for the worker
    CoreRange worker_core_range = CoreRange(test_config.worker_core_coord, test_config.worker_core_coord);

    // Get PCIe core coordinates
    const metal_SocDescriptor& soc_d = MetalContext::instance().get_cluster().get_soc_desc(device_id);
    vector<tt::umd::CoreCoord> pcie_cores = soc_d.get_cores(CoreType::PCIE, CoordSystem::TRANSLATED);
    TT_ASSERT(!pcie_cores.empty(), "No PCIe cores found");

    // Get PCIe memory addresses
    uint64_t dev_pcie_base = MetalContext::instance().get_cluster().get_pcie_base_addr_from_device(device_id);
    constexpr uint64_t PCIE_OFFSET_BYTES = 1024 * 1024 * 50;  // 50MB offset to avoid conflicts
    uint64_t pcie_offset = PCIE_OFFSET_BYTES;
    uint64_t noc_mem_addr = dev_pcie_base + pcie_offset;

    tt_metal::CircularBufferConfig cb_config =
        tt_metal::CircularBufferConfig(total_data_size, {{0, tt::DataFormat::Float32}})
            .set_page_size(0, test_config.page_size_bytes);
    tt_metal::CreateCircularBuffer(program, worker_core_range, cb_config);

    std::map<std::string, std::string> defines = {
        {"ITERATIONS", std::to_string(test_config.iterations)},
        {"PAGE_COUNT", std::to_string(test_config.page_count)},
        {"NOC_ADDR_X", std::to_string(pcie_cores[0].x)},
        {"NOC_ADDR_Y", std::to_string(pcie_cores[0].y)},
        {"NOC_MEM_ADDR", std::to_string(noc_mem_addr)},
    };

    auto dm0 = tt_metal::CreateKernel(
        program,
        "tests/tt_metal/tt_metal/data_movement/pcie_read_bw/kernels/pcie_read_bw.cpp",
        worker_core_range,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_0, .noc = test_config.noc_id, .defines = defines});

    tt_metal::SetRuntimeArgs(program, dm0, worker_core_range, {test_config.test_id, test_config.page_size_bytes});

    log_info(
        LogTest, "Running PCIe Read BW Test ID: {}, Run ID: {}", test_config.test_id, unit_tests::dm::runtime_host_id);
    program.set_runtime_id(unit_tests::dm::runtime_host_id++);

    auto mesh_workload = distributed::MeshWorkload();
    auto target_devices = distributed::MeshCoordinateRange(distributed::MeshCoordinate({0, 0}));
    mesh_workload.add_program(target_devices, std::move(program));

    auto& cq = mesh_device->mesh_command_queue();

    // Warmup iterations
    for (int i = 0; i < test_config.warmup_iterations; i++) {
        distributed::EnqueueMeshWorkload(cq, mesh_workload, false);
    }
    distributed::Finish(cq);

    // Actual test iterations
    auto start = std::chrono::system_clock::now();
    distributed::EnqueueMeshWorkload(cq, mesh_workload, false);
    distributed::Finish(cq);
    auto end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds{};
    elapsed_seconds = end - start;

    float total_bytes =
        (float)test_config.page_count * (float)test_config.page_size_bytes * (float)test_config.iterations;
    float bw = total_bytes / (elapsed_seconds.count() * 1000.0 * 1000.0 * 1000.0);

    std::stringstream ss;
    ss << std::fixed << std::setprecision(3) << bw;

    log_info(LogTest, "Bandwidth: {} GB/s", ss.str());

    return true;
}

void pcie_read_bw_test(
    const shared_ptr<distributed::MeshDevice>& mesh_device, uint32_t test_id, CoreCoord worker_core_coord = {0, 0}) {
    PCIeReadBwConfig test_config = {
        .test_id = test_id,
        .worker_core_coord = worker_core_coord,
        .iterations = 1000,
        .warmup_iterations = 2,
        .page_size_bytes = 65536,
        .batch_size_k = 256,
        .size_bytes = test_config.batch_size_k * 1024,
        .page_count = test_config.size_bytes / test_config.page_size_bytes,
        .l1_data_format = DataFormat::Float32,
        .noc_id = NOC::RISCV_0_default,
    };

    EXPECT_TRUE(run_pcie_read_bw_test(mesh_device, test_config));
}

}  // namespace unit_tests::dm::pcie_read_bw

TEST_F(GenericMeshDeviceFixture, PCIeReadBandwidth) {
    uint32_t test_id = 603;
    CoreCoord worker_core_coord = {0, 0};

    unit_tests::dm::pcie_read_bw::pcie_read_bw_test(get_mesh_device(), test_id, worker_core_coord);
}

}  // namespace tt::tt_metal
