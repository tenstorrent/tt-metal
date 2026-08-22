// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <cstdlib>
#include <tt-metalium/device.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <map>
#include <optional>
#include <string>
#include <variant>
#include <vector>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/kernel_types.hpp>
#include "hostdevcommon/common_values.hpp"
#include <tt-metalium/program.hpp>
#include "impl/context/metal_context.hpp"
#include <tt-metalium/tt_backend_api_types.hpp>
#include "tt_metal/test_utils/env_vars.hpp"
#include "llrt/get_platform_architecture.hpp"
#include "llrt/rtoptions.hpp"
#include <umd/device/types/arch.hpp>
#include <tt-metalium/distributed.hpp>
#include "common/tt_backend_api_types.hpp"
#include <llrt/tt_cluster.hpp>

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
        "tests/tt_metal/tt_metal/test_kernels/dataflow/blank.cpp",
        all_cores,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_1, .noc = tt::tt_metal::NOC::RISCV_1_default});

    CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/blank.cpp",
        all_cores,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_0, .noc = tt::tt_metal::NOC::RISCV_0_default});

    CreateKernel(
        program, "tests/tt_metal/tt_metal/test_kernels/compute/blank.cpp", all_cores, tt::tt_metal::ComputeConfig{});
    mesh_workload.add_program(distributed::MeshCoordinateRange(mesh_device->shape()), std::move(program));
    distributed::EnqueueMeshWorkload(mesh_device->mesh_command_queue(), mesh_workload, true);
    return pass;
}
}  // namespace unit_tests_common::basic::test_device_init

namespace {
// gtest evaluates INSTANTIATE_TEST_SUITE_P generators at test-registration time
// (inside InitGoogleTest), BEFORE --gtest_filter is applied. Calling
// GetNumAvailableDevices() directly here forces full MetalContext/Cluster creation
// at process startup, which throws on hosts without silicon and would abort the
// whole binary — including the host-only CPU_* tests that run on the device-less
// github_hosted_cpu CI runner.
//
// Probe for silicon first via PCI enumeration only (get_physical_architecture()
// does not create a MetalContext and returns ARCH::Invalid when no devices are
// present). Fall back to a single-device parameterization only in that genuinely
// hardware-less case; on a device runner any discovery failure from
// GetNumAvailableDevices() (malformed cluster descriptor, UMD errors, ...) must
// propagate and fail loudly rather than silently shrink the parameterization.
// The DeviceParamFixture tests themselves are device tests and are excluded on
// the CPU-only leg by the CPU_-filter split anyway.
unsigned int num_available_devices_or_one() {
    if (get_physical_architecture() == tt::ARCH::Invalid) {
        return 1;
    }
    return tt::tt_metal::GetNumAvailableDevices();
}
}  // namespace

INSTANTIATE_TEST_SUITE_P(DeviceInit, DeviceParamFixture, ::testing::Values(1, num_available_devices_or_one()));

TEST_P(DeviceParamFixture, DeviceInitializeAndTeardown) {
    unsigned int num_devices = GetParam();
    ASSERT_TRUE(num_devices > 0);
    vector<ChipId> ids;
    for (ChipId id : tt::tt_metal::MetalContext::instance().get_cluster().mmio_chip_ids()) {
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
    if (num_devices > num_pci_devices) {
        GTEST_SKIP();
    }
    ASSERT_TRUE(num_devices > 0);
    vector<ChipId> ids;
    for (ChipId id : tt::tt_metal::MetalContext::instance().get_cluster().mmio_chip_ids()) {
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

constexpr const char* kTdpLimitEnvVar = "TT_METAL_TDP_LIMIT_WATTS";

// Restores TT_METAL_TDP_LIMIT_WATTS, so the tests that follow in this binary start from the
// environment they expect. These tests only parse the variable; what the cluster then does with it
// is covered by the TdpLimit tests in test_release_ownership.cpp, which rebuild the cluster.
class TdpLimitEnvFixture : public ::testing::Test {
protected:
    void SetUp() override {
        const char* prev = getenv(kTdpLimitEnvVar);
        prev_ = prev != nullptr ? std::optional<std::string>(prev) : std::nullopt;
    }

    void TearDown() override {
        if (prev_.has_value()) {
            setenv(kTdpLimitEnvVar, prev_->c_str(), /*overwrite=*/1);
        } else {
            unsetenv(kTdpLimitEnvVar);
        }
    }

private:
    std::optional<std::string> prev_;
};

TEST_F(TdpLimitEnvFixture, CPU_ParsesEnvVar) {
    unsetenv(kTdpLimitEnvVar);
    EXPECT_FALSE(llrt::RunTimeOptions().get_tdp_limit_watts().has_value());

    // Exporting the variable empty is how a shared profile disables the knob without unsetting it.
    setenv(kTdpLimitEnvVar, "", /*overwrite=*/1);
    EXPECT_FALSE(llrt::RunTimeOptions().get_tdp_limit_watts().has_value());

    setenv(kTdpLimitEnvVar, "300", /*overwrite=*/1);
    EXPECT_EQ(llrt::RunTimeOptions().get_tdp_limit_watts(), 300u);

    setenv(kTdpLimitEnvVar, "0", /*overwrite=*/1);
    EXPECT_EQ(llrt::RunTimeOptions().get_tdp_limit_watts(), llrt::TDP_LIMIT_RESTORE_DEFAULT_SENTINEL);

    // rtoptions only decides whether the value is a watt count it can hold; whether firmware accepts
    // it is UMD's call at cluster open, so 600 parses even though it is outside the accepted range.
    setenv(kTdpLimitEnvVar, "600", /*overwrite=*/1);
    EXPECT_EQ(llrt::RunTimeOptions().get_tdp_limit_watts(), 600u);
}

// A typo must not quietly leave the run at full power, so parsing is strict. Neither of these is
// usable as a watt count: one is not a number, the other does not fit the uint32_t that holds it.
TEST_F(TdpLimitEnvFixture, CPU_MalformedEnvVarThrows) {
    setenv(kTdpLimitEnvVar, "abc", /*overwrite=*/1);
    EXPECT_ANY_THROW(llrt::RunTimeOptions());

    setenv(kTdpLimitEnvVar, "99999999999999999999", /*overwrite=*/1);
    EXPECT_ANY_THROW(llrt::RunTimeOptions());
}

}  // namespace tt::tt_metal
