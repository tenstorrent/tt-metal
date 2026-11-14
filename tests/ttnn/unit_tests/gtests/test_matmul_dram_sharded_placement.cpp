// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// DRAM-sharded matmul reader placement on every chip of the mesh the test runs on. A chip's best reader for a DRAM
// bank is the first unharvested worker column east of the bank, and logical columns skip harvested ones, so on a mesh
// of differently harvested chips the best reader's logical core differs per chip. The matmul's placement helpers take
// the chip's mesh coordinate; these tests check, on real hardware, that each chip's readers are that chip's own
// NOC-optimal workers (single chip, QuietBox 2, T3K, Galaxy alike), and, where the system's chips are harvested
// differently, that the placements the helpers return do differ between chips.

#include <gtest/gtest.h>

#include <cstdint>
#include <set>
#include <utility>
#include <vector>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/mesh_coord.hpp>

#include "tt_metal/tt_metal/common/multi_device_fixture.hpp"
#include "ttnn/operations/matmul/device/utilities/matmul_utilities.hpp"

namespace tt::tt_metal {
namespace {

using distributed::MeshCoordinate;
using distributed::MeshDevice;
using ttnn::prim::dram_sharded_helpers::DramBankReaderAssignment;
using ttnn::prim::dram_sharded_helpers::get_dram_bank_adjacent_workers;
using ttnn::prim::dram_sharded_helpers::get_dram_bank_reader_assignments;

using MatmulDramShardedPlacementTest = GenericMeshDeviceFixture;

// Workers per bank the helpers support on this architecture: one everywhere; on Blackhole also two (NOC 0 only).
std::vector<std::pair<NOC, uint32_t>> placement_cases(const MeshDevice& device) {
    std::vector<std::pair<NOC, uint32_t>> cases = {{NOC::NOC_0, 1}, {NOC::NOC_1, 1}};
    if (device.arch() == tt::ARCH::BLACKHOLE) {
        cases.emplace_back(NOC::NOC_0, 2);
    }
    return cases;
}

TEST_F(MatmulDramShardedPlacementTest, ReadersAreEachChipsOwnOptimalWorkers) {
    auto& device = *mesh_device_;
    for (const auto& coord : distributed::MeshCoordinateRange(device.shape())) {
        for (const auto& [noc, workers_per_bank] : placement_cases(device)) {
            const auto optimal = device.get_optimal_dram_bank_to_logical_worker_assignment(noc, coord);
            const auto readers = get_dram_bank_reader_assignments(device, noc, workers_per_bank, CoreRangeSet{}, coord);
            ASSERT_EQ(readers.size(), optimal.size() * workers_per_bank) << "chip " << coord;
            std::set<CoreCoord> used;
            for (const DramBankReaderAssignment& reader : readers) {
                ASSERT_TRUE(optimal.contains(reader.bank_id)) << "chip " << coord << " bank " << reader.bank_id;
                if (reader.worker_index == 0) {
                    // the bank's primary reader is this chip's own NOC-optimal worker for it
                    EXPECT_EQ(reader.worker_core, optimal.at(reader.bank_id))
                        << "chip " << coord << " bank " << reader.bank_id << " noc " << static_cast<int>(noc);
                }
                EXPECT_TRUE(used.insert(reader.worker_core).second)
                    << "chip " << coord << ": core " << reader.worker_core.str() << " reads two banks";
            }
        }

        // Multi-core pipeline cores: each bank's nearest free core to its NOC-optimal worker is that worker itself.
        const auto anchors = device.get_optimal_dram_bank_to_logical_worker_assignment(NOC::NOC_0, coord);
        const auto groups = get_dram_bank_adjacent_workers(device, NOC::NOC_0, 2, anchors.size(), coord);
        ASSERT_EQ(groups.size(), anchors.size()) << "chip " << coord;
        for (uint32_t bank = 0; bank < groups.size(); ++bank) {
            ASSERT_FALSE(groups[bank].empty()) << "chip " << coord << " bank " << bank;
            EXPECT_EQ(groups[bank].front(), anchors.at(bank)) << "chip " << coord << " bank " << bank;
        }
    }
}

TEST_F(MatmulDramShardedPlacementTest, DifferentlyHarvestedChipsGetDifferentReaders) {
    auto& device = *mesh_device_;
    const MeshCoordinate first = MeshCoordinate::zero_coordinate(device.shape().dims());
    const auto reference = get_dram_bank_reader_assignments(device, NOC::NOC_0, 1, CoreRangeSet{}, first);
    bool any_chip_differs = false;
    for (const auto& coord : distributed::MeshCoordinateRange(device.shape())) {
        if (device.get_optimal_dram_bank_to_logical_worker_assignment(NOC::NOC_0, coord) ==
            device.get_optimal_dram_bank_to_logical_worker_assignment(NOC::NOC_0, first)) {
            continue;
        }
        any_chip_differs = true;
        const auto readers = get_dram_bank_reader_assignments(device, NOC::NOC_0, 1, CoreRangeSet{}, coord);
        bool differs = false;
        for (std::size_t i = 0; i < readers.size(); ++i) {
            differs |= readers[i].worker_core != reference[i].worker_core;
        }
        EXPECT_TRUE(differs) << "chip " << coord << " is harvested differently from chip " << first
                             << " but got the first chip's readers";
    }
    if (!any_chip_differs) {
        GTEST_SKIP() << "every chip of this " << device.shape() << " mesh has the same DRAM-bank worker placement";
    }
}

}  // namespace
}  // namespace tt::tt_metal
