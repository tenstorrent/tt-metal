// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <fmt/format.h>
#include <gtest/gtest.h>
#include <fstream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/mesh_config.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/system_mesh.hpp>

#include "debug_tools_fixture.hpp"
#include "debug_tools_test_utils.hpp"
#include "impl/context/metal_context.hpp"

namespace tt::tt_metal {

// Restricts DPRINT output to a single mesh coordinate.
// Tears down and re-initializes MetalContext so resolve_mesh_coords_to_chip_ids()
// fires during device open and populates chip_ids from the stored mesh_coords.
class DPrintMeshCoordsFixture : public DPrintMeshFixture {
public:
    ChipId target_chip_id = -1;

protected:
    void ExtraSetUp() override;
    void ExtraTearDown() override;
};

namespace {
namespace CMAKE_UNIQUE_NAMESPACE {

std::string ReadLogFile(const std::string& path) {
    std::ifstream f(path);
    return {std::istreambuf_iterator<char>(f), {}};
}

// Trailing '\n' (from ENDL()) prevents "device_id=1" matching inside "device_id=16".
std::string DeviceIdToken(ChipId chip_id) { return fmt::format("device_id={}\n", chip_id); }

// Configures rtopts to restrict DPRINT to a single mesh coordinate.
// Called after each MetalContext teardown to restore the intended filter before device open.
void ConfigureDPrintForCoord(
    tt::llrt::RunTimeOptions& rtopts, const std::string& file_name, uint32_t row, uint32_t col) {
    constexpr auto kDprint = tt::llrt::RunTimeDebugFeatureDprint;
    rtopts.set_feature_enabled(kDprint, true);
    rtopts.set_feature_all_cores(kDprint, CoreType::WORKER, tt::llrt::RunTimeDebugClassWorker);
    rtopts.set_feature_all_cores(kDprint, CoreType::ETH, tt::llrt::RunTimeDebugClassWorker);
    rtopts.set_feature_file_name(kDprint, file_name);
    rtopts.set_test_mode_enabled(true);
    rtopts.set_watcher_enabled(false);
    rtopts.set_feature_prepend_device_core_risc(kDprint, true);
    rtopts.set_feature_mesh_coords(kDprint, {{row, col}});
    rtopts.set_feature_all_chips(kDprint, false);
}

// Builds a MeshWorkload where each device DPRINTs its physical chip_id.
distributed::MeshWorkload BuildDeviceIdWorkload(
    const std::shared_ptr<distributed::MeshDevice>& mesh_device) {
    distributed::MeshWorkload workload;
    constexpr CoreCoord kCore = {0, 0};
    for (const auto& coord : distributed::MeshCoordinateRange(mesh_device->shape())) {
        ChipId chip_id = mesh_device->get_device(coord)->id();
        Program program;
        CreateKernel(
            program,
            "tests/tt_metal/tt_metal/test_kernels/misc/print_device_id.cpp",
            kCore,
            DataMovementConfig{
                .processor = DataMovementProcessor::RISCV_0,
                .noc = NOC::RISCV_0_default,
                .compile_args = {static_cast<uint32_t>(chip_id)}});
        workload.add_program(distributed::MeshCoordinateRange(coord, coord), std::move(program));
    }
    return workload;
}

// Runs a kernel on every device (log accumulates across all), then asserts that only
// fixture->target_chip_id produced output and all others were silenced by the DPRINT filter.
void RunFilteringTest(
    DPrintMeshCoordsFixture* fixture,
    const std::vector<std::shared_ptr<distributed::MeshDevice>>& all_devices) {
    for (const auto& mesh_device : all_devices) {
        auto workload = BuildDeviceIdWorkload(mesh_device);
        fixture->RunProgram(mesh_device, workload);
    }

    const std::string log = ReadLogFile(fixture->dprint_file_name);

    EXPECT_NE(log.find(DeviceIdToken(fixture->target_chip_id)), std::string::npos)
        << "Expected DPRINT output from target chip " << fixture->target_chip_id << " not found.\n"
        << "Log:\n" << log;

    for (const auto& mesh_device : all_devices) {
        ChipId chip_id = mesh_device->get_devices()[0]->id();
        if (chip_id == fixture->target_chip_id) {
            continue;
        }
        EXPECT_EQ(log.find(DeviceIdToken(chip_id)), std::string::npos)
            << "Unexpected DPRINT output from non-target chip " << chip_id
            << " (only chip " << fixture->target_chip_id << " should print).";
    }
}

// Runs a kernel that DPRINTs chip_id, verifies the output appears in the log, then
// cross-checks that resolve_mesh_coords_to_chip_ids maps the device's coord back to the same id.
void RunAllChipsVerificationTest(
    DPrintMeshFixture* fixture, const std::shared_ptr<distributed::MeshDevice>& mesh_device) {
    constexpr auto kDprint = tt::llrt::RunTimeDebugFeatureDprint;
    ChipId chip_id = mesh_device->get_devices()[0]->id();

    auto workload = BuildDeviceIdWorkload(mesh_device);
    fixture->RunProgram(mesh_device, workload);

    const std::string log = ReadLogFile(fixture->dprint_file_name);
    EXPECT_NE(log.find(DeviceIdToken(chip_id)), std::string::npos)
        << "Missing DPRINT output for chip_id=" << chip_id;

    auto mapped = distributed::SystemMesh::instance().get_mapped_devices(std::nullopt);
    std::optional<std::pair<uint32_t, uint32_t>> found_coord;
    for (const auto& coord : distributed::MeshCoordinateRange(mapped.mesh_shape)) {
        auto linear_idx = coord.to_linear_index(mapped.mesh_shape);
        if (mapped.device_ids[linear_idx].is_local() && *mapped.device_ids[linear_idx] == chip_id) {
            found_coord = {coord[0], coord[1]};
            break;
        }
    }
    ASSERT_TRUE(found_coord.has_value()) << "No mesh coord found for chip_id=" << chip_id;

    auto& rtopts = MetalContext::instance().rtoptions();
    rtopts.set_feature_mesh_coords(kDprint, {*found_coord});
    rtopts.resolve_mesh_coords_to_chip_ids(distributed::SystemMesh::instance());

    const auto& resolved_ids = rtopts.get_feature_chip_ids(kDprint);
    ASSERT_EQ(resolved_ids.size(), 1u) << "Expected exactly one chip_id for chip_id=" << chip_id;
    EXPECT_EQ(resolved_ids[0], chip_id)
        << "resolve_mesh_coords_to_chip_ids mismatch: expected " << chip_id << " got " << resolved_ids[0];

    rtopts.set_feature_mesh_coords(kDprint, {});
    rtopts.set_feature_chip_ids(kDprint, {});
    rtopts.set_feature_all_chips(kDprint, true);
}

}  // namespace CMAKE_UNIQUE_NAMESPACE
}  // namespace

void DPrintMeshCoordsFixture::ExtraSetUp() {
    // Teardown forces MetalContext to re-initialize (including resolve_mesh_coords_to_chip_ids)
    // when devices are opened by DebugToolsMeshFixture::SetUp().  Re-apply rtoptions afterwards
    // because ParseAllFeatureEnv resets them to defaults on re-initialization.
    MetalContext::instance().teardown();
    CMAKE_UNIQUE_NAMESPACE::ConfigureDPrintForCoord(
        MetalContext::instance().rtoptions(), dprint_file_name, /*row=*/0, /*col=*/0);
}

void DPrintMeshCoordsFixture::ExtraTearDown() { MetalContext::instance().teardown(); }

// Test 1: Only the device at mesh coord (0,0) should produce DPRINT output.
TEST_F(DPrintMeshCoordsFixture, TensixTestDprintMeshCoordsFiltersCorrectDevice) {
    const auto& chip_ids =
        MetalContext::instance().rtoptions().get_feature_chip_ids(tt::llrt::RunTimeDebugFeatureDprint);
    ASSERT_FALSE(chip_ids.empty()) << "No chip_id resolved for mesh coord (0,0).";
    this->target_chip_id = chip_ids[0];
    CMAKE_UNIQUE_NAMESPACE::RunFilteringTest(this, this->devices_);
}

// Test 2: For every mesh coordinate, restrict DPRINT to that chip and verify filtering.
// Each iteration tears down MetalContext and reopens devices so resolve_mesh_coords_to_chip_ids
// fires fresh with the new target coord.
TEST_F(DPrintMeshCoordsFixture, TensixTestDprintMeshCoordsFiltersAllCoords) {
    struct LocalDevice {
        uint32_t row, col;
        ChipId chip_id;
    };

    // Snapshot before any teardown invalidates the SystemMesh reference.
    std::vector<LocalDevice> local_devices;
    {
        auto mapped = distributed::SystemMesh::instance().get_mapped_devices(std::nullopt);
        for (const auto& coord : distributed::MeshCoordinateRange(mapped.mesh_shape)) {
            auto linear_idx = coord.to_linear_index(mapped.mesh_shape);
            if (mapped.device_ids[linear_idx].is_local()) {
                local_devices.push_back({coord[0], coord[1], *mapped.device_ids[linear_idx]});
            }
        }
    }

    for (size_t i = 0; i < local_devices.size(); ++i) {
        auto [row, col, target_chip] = local_devices[i];

        if (i > 0) {
            MeshDispatchFixture::TearDown();
            MetalContext::instance().teardown();
            CMAKE_UNIQUE_NAMESPACE::ConfigureDPrintForCoord(
                MetalContext::instance().rtoptions(), dprint_file_name, row, col);
            MeshDispatchFixture::SetUp();
        }

        const auto& chip_ids =
            MetalContext::instance().rtoptions().get_feature_chip_ids(tt::llrt::RunTimeDebugFeatureDprint);
        ASSERT_FALSE(chip_ids.empty()) << "No chip_id resolved for coord (" << row << "," << col << ").";
        ASSERT_EQ(chip_ids[0], target_chip)
            << "resolve_mesh_coords_to_chip_ids returned wrong chip for (" << row << "," << col << ")";

        this->target_chip_id = target_chip;
        log_info(tt::LogTest, "Filtering test: coord ({},{}) → chip_id={}", row, col, target_chip);
        CMAKE_UNIQUE_NAMESPACE::RunFilteringTest(this, this->devices_);
        MetalContext::instance().dprint_server()->clear_log_file();
    }
}

// Test 3: Every mesh coordinate resolves to the correct chip_id and the device prints its id.
TEST_F(DPrintMeshFixture, TensixTestDprintMeshCoordsAllDevicesMapping) {
    for (auto& mesh_device : this->devices_) {
        this->RunTestOnDevice(CMAKE_UNIQUE_NAMESPACE::RunAllChipsVerificationTest, mesh_device);
    }
}

// Test 4: Coordinates outside the system mesh shape must produce a clear error.
TEST_F(DebugToolsMeshFixture, TensixTestDprintMeshCoordsOutOfBounds) {
    auto& rtoptions = MetalContext::instance().rtoptions();
    rtoptions.set_feature_mesh_coords(tt::llrt::RunTimeDebugFeatureDprint, {{999, 999}});
    rtoptions.set_test_mode_enabled(true);  // makes TT_FATAL throw instead of abort
    EXPECT_THROW(
        rtoptions.resolve_mesh_coords_to_chip_ids(MetalContext::instance().get_system_mesh()),
        std::runtime_error);
    rtoptions.set_feature_mesh_coords(tt::llrt::RunTimeDebugFeatureDprint, {});
    rtoptions.set_test_mode_enabled(false);
}

}  // namespace tt::tt_metal
