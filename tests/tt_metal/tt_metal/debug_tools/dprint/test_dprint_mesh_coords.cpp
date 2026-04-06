// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <fmt/format.h>
#include <gtest/gtest.h>
#include <fstream>
#include <memory>
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
    // The global system-mesh coordinate currently targeted by the DPRINT filter.
    std::pair<uint32_t, uint32_t> target_coord = {0, 0};

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

// Returns a unique search token for a device's DPRINT line.
// The trailing '\n' (produced by ENDL()) prevents "(0,1)" matching inside "(0,10)".
std::string MeshCoordToken(uint32_t row, uint32_t col) {
    return fmt::format("mesh_coord=({},{})\n", row, col);
}

// Returns the global system-mesh coordinate for the given local chip_id.
std::pair<uint32_t, uint32_t> GetGlobalCoord(ChipId chip_id) {
    auto mapped = distributed::SystemMesh::instance().get_mapped_devices(std::nullopt);
    for (const auto& coord : distributed::MeshCoordinateRange(mapped.mesh_shape)) {
        auto idx = coord.to_linear_index(mapped.mesh_shape);
        if (mapped.device_ids[idx].is_local() && *mapped.device_ids[idx] == chip_id) {
            return {coord[0], coord[1]};
        }
    }
    TT_THROW("No global mesh coord found for chip_id={}", chip_id);
}

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

// Builds a MeshWorkload where each device DPRINTs its global mesh coordinate (row, col).
// The kernel takes the global system-mesh row and col of each device as compile-time args,
// so the DPRINT output directly matches what was configured in the DPRINT filter.
distributed::MeshWorkload BuildMeshCoordWorkload(
    const std::shared_ptr<distributed::MeshDevice>& mesh_device) {
    distributed::MeshWorkload workload;
    constexpr CoreCoord kCore = {0, 0};
    for (const auto& coord : distributed::MeshCoordinateRange(mesh_device->shape())) {
        auto [row, col] = GetGlobalCoord(mesh_device->get_device(coord)->id());
        Program program;
        CreateKernel(
            program,
            "tests/tt_metal/tt_metal/test_kernels/misc/print_mesh_coord.cpp",
            kCore,
            DataMovementConfig{
                .processor = DataMovementProcessor::RISCV_0,
                .noc = NOC::RISCV_0_default,
                .compile_args = {row, col}});
        workload.add_program(distributed::MeshCoordinateRange(coord, coord), std::move(program));
    }
    return workload;
}

// Runs a kernel on every device (log accumulates across all), then asserts that only
// fixture->target_coord produced output and all others were silenced by the DPRINT filter.
void RunFilteringTest(
    DPrintMeshCoordsFixture* fixture,
    const std::vector<std::shared_ptr<distributed::MeshDevice>>& all_devices) {
    for (const auto& mesh_device : all_devices) {
        auto workload = BuildMeshCoordWorkload(mesh_device);
        fixture->RunProgram(mesh_device, workload);
    }

    const std::string log = ReadLogFile(fixture->dprint_file_name);
    auto [trow, tcol] = fixture->target_coord;

    EXPECT_NE(log.find(MeshCoordToken(trow, tcol)), std::string::npos)
        << "Expected DPRINT from target coord (" << trow << "," << tcol << ") not found.\n"
        << "Log:\n" << log;

    for (const auto& mesh_device : all_devices) {
        auto [row, col] = GetGlobalCoord(mesh_device->get_devices()[0]->id());
        if (std::make_pair(row, col) == fixture->target_coord) {
            continue;
        }
        EXPECT_EQ(log.find(MeshCoordToken(row, col)), std::string::npos)
            << "Unexpected DPRINT from coord (" << row << "," << col << ") "
            << "(only coord (" << trow << "," << tcol << ") should print).";
    }
}

// Runs a kernel that DPRINTs the device's global mesh coord, verifies the output appears,
// then cross-checks that resolve_mesh_coords_to_chip_ids maps that coord back to the same chip_id.
void RunAllChipsVerificationTest(
    DPrintMeshFixture* fixture, const std::shared_ptr<distributed::MeshDevice>& mesh_device) {
    constexpr auto kDprint = tt::llrt::RunTimeDebugFeatureDprint;
    ChipId chip_id = mesh_device->get_devices()[0]->id();
    auto [row, col] = GetGlobalCoord(chip_id);

    auto workload = BuildMeshCoordWorkload(mesh_device);
    fixture->RunProgram(mesh_device, workload);

    const std::string log = ReadLogFile(fixture->dprint_file_name);
    EXPECT_NE(log.find(MeshCoordToken(row, col)), std::string::npos)
        << "Missing DPRINT for coord (" << row << "," << col << "), chip_id=" << chip_id;

    // Cross-check: resolve (row,col) → chip_id via the rtoptions API.
    auto& rtopts = MetalContext::instance().rtoptions();
    rtopts.set_feature_mesh_coords(kDprint, {{row, col}});
    rtopts.resolve_mesh_coords_to_chip_ids(distributed::SystemMesh::instance());

    const auto& resolved = rtopts.get_feature_chip_ids(kDprint);
    ASSERT_EQ(resolved.size(), 1u) << "Expected exactly one chip_id for coord (" << row << "," << col << ")";
    EXPECT_EQ(resolved[0], chip_id)
        << "Coord (" << row << "," << col << ") resolved to wrong chip_id";

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
        MetalContext::instance().rtoptions(), dprint_file_name, target_coord.first, target_coord.second);
}

void DPrintMeshCoordsFixture::ExtraTearDown() { MetalContext::instance().teardown(); }

// Test 1: Only the device at mesh coord (0,0) should produce DPRINT output.
TEST_F(DPrintMeshCoordsFixture, TensixTestDprintMeshCoordsFiltersCorrectDevice) {
    // ExtraSetUp configured (0,0) as the DPRINT target.
    this->target_coord = {0, 0};
    CMAKE_UNIQUE_NAMESPACE::RunFilteringTest(this, this->devices_);
}

// Test 2: For every mesh coordinate, restrict DPRINT to that coord and verify filtering.
// Each iteration tears down MetalContext and reopens devices so resolve_mesh_coords_to_chip_ids
// fires fresh with the new target coord.
TEST_F(DPrintMeshCoordsFixture, TensixTestDprintMeshCoordsFiltersAllCoords) {
    // Snapshot before any teardown invalidates the SystemMesh reference.
    std::vector<std::pair<uint32_t, uint32_t>> local_coords;
    {
        auto mapped = distributed::SystemMesh::instance().get_mapped_devices(std::nullopt);
        for (const auto& coord : distributed::MeshCoordinateRange(mapped.mesh_shape)) {
            auto linear_idx = coord.to_linear_index(mapped.mesh_shape);
            if (mapped.device_ids[linear_idx].is_local()) {
                local_coords.push_back({coord[0], coord[1]});
            }
        }
    }

    for (auto [row, col] : local_coords) {
        DPrintMeshFixture::TearDown();
        this->target_coord = {row, col};  // set before SetUp so ExtraSetUp picks it up
        DPrintMeshFixture::SetUp();

        ASSERT_FALSE(
            MetalContext::instance()
                .rtoptions()
                .get_feature_chip_ids(tt::llrt::RunTimeDebugFeatureDprint)
                .empty())
            << "No chip resolved for coord (" << row << "," << col << ").";

        log_info(tt::LogTest, "Filtering test: coord ({},{})", row, col);
        CMAKE_UNIQUE_NAMESPACE::RunFilteringTest(this, this->devices_);
        MetalContext::instance().dprint_server()->clear_log_file();
    }
}

// Test 3: Every mesh coordinate resolves to the correct chip_id and the device prints its coord.
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
