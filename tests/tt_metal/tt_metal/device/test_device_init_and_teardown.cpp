// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <stdlib.h>
#include <tt-metalium/device.hpp>
#include <tt-metalium/device_pool.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <map>
#include <variant>
#include <vector>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/data_types.hpp>
#include "hostdevcommon/common_values.hpp"
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/program.hpp>
#include "impl/context/metal_context.hpp"
#include <tt-metalium/tt_backend_api_types.hpp>
#include "tt_metal/test_utils/env_vars.hpp"
#include <umd/device/types/arch.hpp>
#include <tt-metalium/distributed.hpp>

namespace tt {
namespace tt_metal {
class CommandQueue;
}  // namespace tt_metal
}  // namespace tt

namespace tt::tt_metal {

using std::vector;
using namespace tt;
using namespace tt::test_utils;

class DeviceParamFixture : public ::testing::TestWithParam<int> {
protected:
    tt::ARCH arch = tt::get_arch_from_string(get_umd_arch_name());
};

namespace unit_tests_common::basic::test_device_init {

/// @brief load_blank_kernels into all cores and will launch
/// @param device
/// @return
bool load_all_blank_kernels(const std::shared_ptr<distributed::MeshDevice>& mesh_device) {
    bool pass = true;
    tt_metal::Program program = tt_metal::CreateProgram();
    auto mesh_workload = distributed::MeshWorkload();

    CoreCoord compute_grid_size = mesh_device->compute_with_storage_grid_size();
    CoreRange all_cores = CoreRange(CoreCoord(0, 0), CoreCoord(compute_grid_size.x - 1, compute_grid_size.y - 1));
    CreateKernel(
        program,
        "tt_metal/kernels/dataflow/blank.cpp",
        all_cores,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_1, .noc = tt::tt_metal::NOC::RISCV_1_default});

    CreateKernel(
        program,
        "tt_metal/kernels/dataflow/blank.cpp",
        all_cores,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_0, .noc = tt::tt_metal::NOC::RISCV_0_default});

    CreateKernel(program, "tt_metal/kernels/compute/blank.cpp", all_cores, tt::tt_metal::ComputeConfig{});
    distributed::AddProgramToMeshWorkload(
        mesh_workload, std::move(program), distributed::MeshCoordinateRange(mesh_device->shape()));
    distributed::EnqueueMeshWorkload(mesh_device->mesh_command_queue(), mesh_workload, true);
    return pass;
}
}  // namespace unit_tests_common::basic::test_device_init

INSTANTIATE_TEST_SUITE_P(DeviceInit, DeviceParamFixture, ::testing::Values(1, tt::tt_metal::GetNumAvailableDevices()));

TEST_P(DeviceParamFixture, DeviceInitializeAndTeardown) {
    unsigned int num_devices = GetParam();
    if (arch == tt::ARCH::GRAYSKULL && num_devices > 1) {
        GTEST_SKIP();
    }

    ASSERT_TRUE(num_devices > 0);
    vector<chip_id_t> ids;
    for (chip_id_t id : tt::tt_metal::MetalContext::instance().get_cluster().mmio_chip_ids()) {
        ids.push_back(id);
    }
    const auto& dispatch_core_config = tt::tt_metal::MetalContext::instance().rtoptions().get_dispatch_core_config();
    auto devices = distributed::MeshDevice::create_unit_meshes(
        ids, DEFAULT_L1_SMALL_SIZE, DEFAULT_TRACE_REGION_SIZE, 1, dispatch_core_config);
    for (auto& [id, device] : devices) {
        device->close();
    }
}

TEST_P(DeviceParamFixture, TensixDeviceLoadBlankKernels) {
    unsigned int num_devices = GetParam();
    unsigned int num_pci_devices = tt::tt_metal::GetNumPCIeDevices();
    if ((arch == tt::ARCH::GRAYSKULL && num_devices > 1) || (num_devices > num_pci_devices)) {
        GTEST_SKIP();
    }
    ASSERT_TRUE(num_devices > 0);
    vector<chip_id_t> ids;
    for (chip_id_t id : tt::tt_metal::MetalContext::instance().get_cluster().mmio_chip_ids()) {
        ids.push_back(id);
    }
    const auto& dispatch_core_config = tt::tt_metal::MetalContext::instance().rtoptions().get_dispatch_core_config();
    auto devices = distributed::MeshDevice::create_unit_meshes(
        ids, DEFAULT_L1_SMALL_SIZE, DEFAULT_TRACE_REGION_SIZE, 1, dispatch_core_config);
    for (auto& [id, device] : devices) {
        ASSERT_TRUE(unit_tests_common::basic::test_device_init::load_all_blank_kernels(device));
    }
    for (auto& [id, device] : devices) {
        device->close();
    }
}

}  // namespace tt::tt_metal
