// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include "hostdevcommon/common_values.hpp"
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/dispatch_core_common.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/tt_backend_api_types.hpp>
#include "common/core_assignment.hpp"
#include "impl/context/metal_context.hpp"
#include "llrt/tt_cluster.hpp"

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
// Blackhole rejects ROW dispatch (the Device guard emits
// "ROW dispatch core axis is not supported for blackhole arch"), so we omit ROW on BH.
// Quasar core descriptors only define a row axis.
std::vector<DispatchCoreAxis> supported_dispatch_axes() {
    const char* arch_env = std::getenv("ARCH_NAME");
    const std::string arch = arch_env == nullptr ? "" : arch_env;
    if (arch == "quasar") {
        return {DispatchCoreAxis::ROW};
    }
    std::vector<DispatchCoreAxis> axes = {DispatchCoreAxis::COL};
    if (arch != "blackhole") {
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
            const auto bank_to_core = dev->get_optimal_dram_bank_to_logical_worker_assignment(
                noc, distributed::MeshCoordinate::zero_coordinate(dev->shape().dims()));
            std::vector<CoreCoord> cores(bank_to_core.size());
            for (const auto& [bank, core] : bank_to_core) {
                cores.at(bank) = core;
            }

            EXPECT_EQ(cores.size(), expected_banks)
                << "Device " << chip_id << ", NOC " << static_cast<int>(noc) << ": expected " << expected_banks
                << " optimal worker cores (one per DRAM bank), got " << cores.size();
            EXPECT_EQ(std::set<CoreCoord>(cores.begin(), cores.end()).size(), cores.size())
                << "Device " << chip_id << ", NOC " << static_cast<int>(noc)
                << ": two DRAM banks share an optimal worker core";

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

// Blackhole rows 0 and 1 hold no Tensix, so endpoints on rows 0 and 2 of one DRAM column both sit
// nearest the worker on row 2. The later bank has to move to the nearest free row.
TEST(TensixOptimalDramWorkers, BlackholeBanksNearestOneWorkerGetDistinctWorkers) {
    // Blackhole's NOC0 grid (blackhole_140_arch.yaml): DRAM in columns 0 and 9, Tensix in columns 1-7
    // and 10-16 on rows 2-11.
    constexpr uint32_t kGridX = 17;
    constexpr uint32_t kGridY = 12;
    const std::vector<uint32_t> worker_x = {1, 2, 3, 4, 5, 6, 7, 10, 11, 12, 13, 14, 15, 16};
    const std::vector<uint32_t> worker_y = {2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
    struct Case {
        const char* name;
        std::vector<CoreCoord> dram_endpoints;
        std::vector<CoreCoord> expected_workers;
    };
    const std::vector<Case> cases = {
        // NOC0 endpoints of a chip with GDDR7 harvested under SYS-5391 CMFW: subchannel 0 of GDDR0-3 in the
        // column with four live channels, subchannel 2 of GDDR4-6 in the other.
        {"SYS-5391 harvested NOC0",
         {{0, 0}, {0, 2}, {0, 9}, {0, 5}, {9, 11}, {9, 3}, {9, 8}},
         {{1, 2}, {1, 3}, {1, 9}, {1, 5}, {10, 11}, {10, 3}, {10, 8}}},
        // NOC1 endpoints of SYS-4948 relocated CMFW: subchannel 2 of GDDR0-3, subchannel 0 of GDDR4-7.
        {"SYS-4948 relocated NOC1",
         {{0, 11}, {0, 3}, {0, 8}, {0, 6}, {9, 0}, {9, 2}, {9, 9}, {9, 5}},
         {{1, 11}, {1, 3}, {1, 8}, {1, 6}, {10, 2}, {10, 3}, {10, 9}, {10, 5}}},
    };
    for (const Case& c : cases) {
        SCOPED_TRACE(c.name);
        EXPECT_EQ(
            get_optimal_dram_to_physical_worker_assignment(
                ARCH::BLACKHOLE, c.dram_endpoints, kGridX, kGridY, worker_x, worker_y),
            c.expected_workers);
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
