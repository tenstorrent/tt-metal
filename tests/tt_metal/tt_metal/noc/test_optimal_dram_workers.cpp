// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "hostdevcommon/common_values.hpp"
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/data_types.hpp>
#include <tt-metalium/dispatch_core_common.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/tt_backend_api_types.hpp>
#include "impl/context/metal_context.hpp"

namespace tt::tt_metal {
namespace {

// Regression coverage for issue #41031, fixed in PR #42819.
//
// get_optimal_dram_bank_to_logical_worker_assignment returns one Tensix worker core per
// DRAM bank. This test asserts the two invariants the API guarantees:
//   * Cardinality: the returned vector size equals the DRAM grid size (one core per bank).
//   * Validity: every returned (x, y) is inside compute_with_storage_grid_size(). That
//     grid is the logical Tensix worker grid (it already excludes harvested rows and
//     dispatch columns), so an in-grid logical core is by construction a valid worker.
//
// Parameterised over the dispatch axes supported on the current arch. Slow- vs
// fast-dispatch coverage is provided by CI invoking this binary twice (with/without
// TT_METAL_SLOW_DISPATCH_MODE).
class OptimalDramWorkers : public ::testing::TestWithParam<DispatchCoreAxis> {};

// Resolved at static-init from the ARCH_NAME env var (set by build/test scripts), so
// gtest enumeration only lists axes that the platform actually supports — no runtime
// skips appear in the test output.
//
// Blackhole rejects ROW dispatch (see ttnn::Device guard:
// "ROW dispatch core axis is not supported for blackhole arch"), so we omit ROW on BH.
static std::vector<DispatchCoreAxis> supported_dispatch_axes() {
    std::vector<DispatchCoreAxis> axes = {DispatchCoreAxis::COL};
    const char* arch = std::getenv("ARCH_NAME");
    if (arch == nullptr || std::string(arch) != "blackhole") {
        axes.push_back(DispatchCoreAxis::ROW);
    }
    return axes;
}

TEST_P(OptimalDramWorkers, ReturnsOneValidWorkerPerDramBank) {
    const DispatchCoreAxis axis = GetParam();
    const DispatchCoreConfig cfg{DispatchCoreType::WORKER, axis};

    const auto& ids_set = MetalContext::instance().get_cluster().user_exposed_chip_ids();
    if (ids_set.empty()) {
        GTEST_SKIP() << "No user-exposed chips available";
    }
    std::vector<int> ids(ids_set.begin(), ids_set.end());

    auto devs = distributed::MeshDevice::create_unit_meshes(
        ids, DEFAULT_L1_SMALL_SIZE, DEFAULT_TRACE_REGION_SIZE, /*num_command_queues=*/1, cfg);

    for (auto& [chip_id, dev] : devs) {
        const CoreCoord grid = dev->compute_with_storage_grid_size();
        const CoreCoord dram_grid = dev->dram_grid_size();
        const size_t expected_banks = static_cast<size_t>(dram_grid.x) * dram_grid.y;

        for (NOC noc : {NOC::NOC_0, NOC::NOC_1}) {
            const auto cores = dev->get_optimal_dram_bank_to_logical_worker_assignment(noc);

            EXPECT_EQ(cores.size(), expected_banks)
                << "Device " << chip_id << ", NOC " << static_cast<int>(noc) << ": expected " << expected_banks
                << " optimal worker cores (one per DRAM bank), got " << cores.size();

            for (size_t i = 0; i < cores.size(); ++i) {
                const CoreCoord& c = cores[i];
                EXPECT_LT(c.x, grid.x) << "Device " << chip_id << ", NOC " << static_cast<int>(noc) << ", DRAM bank "
                                       << i << ": optimal worker core (" << c.x << ", " << c.y
                                       << ") x is outside compute grid (" << grid.x << ", " << grid.y << ")";
                EXPECT_LT(c.y, grid.y) << "Device " << chip_id << ", NOC " << static_cast<int>(noc) << ", DRAM bank "
                                       << i << ": optimal worker core (" << c.x << ", " << c.y
                                       << ") y is outside compute grid (" << grid.x << ", " << grid.y << ")";
            }
        }
    }

    for (auto& [_, dev] : devs) {
        dev->close();
        dev.reset();
    }
}

INSTANTIATE_TEST_SUITE_P(
    DispatchAxes,
    OptimalDramWorkers,
    ::testing::ValuesIn(supported_dispatch_axes()),
    [](const ::testing::TestParamInfo<DispatchCoreAxis>& info) {
        return info.param == DispatchCoreAxis::ROW ? "Row" : "Col";
    });

}  // namespace
}  // namespace tt::tt_metal
