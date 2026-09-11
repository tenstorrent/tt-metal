// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <map>
#include <memory>
#include <random>
#include <set>
#include <utility>
#include <vector>

#include <tt-metalium/circular_buffer_config.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/program.hpp>

#include "device_fixture.hpp"
#include "impl/buffers/circular_buffer.hpp"
#include "impl/device/device_impl.hpp"
#include "impl/program/program_impl.hpp"

namespace tt::tt_metal {

TEST(CircularBufferStatistics, CPU_OverlappingCoreRangesPreservePhysicalAddressUnion) {
    // A byte-set oracle is deliberately independent of interval merging.
    // Duplicate, nested and adjacent allocations must count physical bytes once.
    using Regions = std::vector<std::pair<uint64_t, uint64_t>>;
    std::map<CoreRange, Regions> ranges;
    std::map<CoreCoord, std::set<uint64_t>> expected;
    std::mt19937 random(35);
    for (unsigned allocation = 0; allocation < 200; ++allocation) {
        const CoreCoord first(random() % 4, random() % 4);
        const CoreCoord last(first.x + random() % 3, first.y + random() % 3);
        const uint64_t begin = random() % 128;
        const uint64_t end = begin + 1 + random() % 32;
        ranges[CoreRange(first, last)].emplace_back(begin, end);
        for (uint32_t x = first.x; x <= last.x; ++x) {
            for (uint32_t y = first.y; y <= last.y; ++y) {
                for (uint64_t address = begin; address < end; ++address) {
                    expected[CoreCoord(x, y)].insert(address);
                }
            }
        }
    }
    const auto actual = detail::ProgramImpl::expand_cb_l1_regions_per_core(ranges);
    ASSERT_EQ(actual.size(), expected.size());
    for (const auto& [core, regions] : actual) {
        std::set<uint64_t> addresses;
        uint64_t previous_end = 0;
        for (const auto& [begin, end] : regions) {
            EXPECT_GT(end, begin);
            EXPECT_GE(begin, previous_end);
            for (uint64_t address = begin; address < end; ++address) {
                EXPECT_TRUE(addresses.insert(address).second);
            }
            previous_end = end;
        }
        EXPECT_EQ(addresses, expected.at(core));
    }
    EXPECT_TRUE(detail::ProgramImpl::expand_cb_l1_regions_per_core({}).empty());
}

namespace {

struct CbStatsProgram {
    std::unique_ptr<Program> program;
    CBHandle cb;
};

CbStatsProgram make_cb_stats_program(const CoreRange& cores, uint32_t bytes) {
    auto program = std::make_unique<Program>();
    const auto config = CircularBufferConfig(bytes, {{0, tt::DataFormat::Float16_b}}).set_page_size(0, 2048);
    const auto cb = CreateCircularBuffer(*program, CoreRangeSet(cores), config);
    return {std::move(program), cb};
}

uint32_t cb_stats_address(const CbStatsProgram& state) {
    return state.program->impl().get_circular_buffer(state.cb)->address();
}

}  // namespace

using CircularBufferStatisticsDeviceTest = UnitMeshAnyDispatchFixture;

TEST_F(CircularBufferStatisticsDeviceTest, PhysicalUnionFollowsProgramLifetime) {
    auto& mesh = device();
    ASSERT_EQ(mesh.get_devices().size(), 1U);
    auto* physical = dynamic_cast<Device*>(mesh.get_devices().front());
    ASSERT_NE(physical, nullptr);
    ASSERT_GE(mesh.compute_with_storage_grid_size().x, 3U);
    // This fixture opens a fresh unit mesh and does not enqueue user programs.
    ASSERT_EQ(physical->get_total_cb_allocated(), 0U);

    const CoreRange left({0, 0}, {1, 0});
    const CoreRange right({1, 0}, {2, 0});
    auto first = make_cb_stats_program(left, 4096);
    auto duplicate = make_cb_stats_program(left, 4096);
    auto overlapping = make_cb_stats_program(right, 6144);
    EXPECT_EQ(physical->get_total_cb_allocated(), 0U);  // Creation alone does not allocate/register.

    first.program->impl().allocate_circular_buffers(&mesh);
    EXPECT_EQ(physical->get_total_cb_allocated(), 8192U);  // Two cores, 4096 bytes each.

    duplicate.program->impl().allocate_circular_buffers(&mesh);
    ASSERT_EQ(cb_stats_address(first), cb_stats_address(duplicate));
    EXPECT_EQ(physical->get_total_cb_allocated(), 8192U);        // Exact same range/address reuse.
    duplicate.program->impl().allocate_circular_buffers(&mesh);  // Already allocated/registered fast path.
    EXPECT_EQ(physical->get_total_cb_allocated(), 8192U);

    overlapping.program->impl().allocate_circular_buffers(&mesh);
    ASSERT_EQ(cb_stats_address(first), cb_stats_address(overlapping));
    // Core (0,0): 4096; shared core (1,0): max(4096,6144); core (2,0): 6144.
    EXPECT_EQ(physical->get_total_cb_allocated(), 16384U);

    first.program.reset();
    EXPECT_EQ(physical->get_total_cb_allocated(), 16384U);  // Duplicate still owns the left union.
    duplicate.program.reset();
    EXPECT_EQ(physical->get_total_cb_allocated(), 12288U);  // Only the two-core right program remains.
    overlapping.program.reset();
    EXPECT_EQ(physical->get_total_cb_allocated(), 0U);
}

// Use the existing any-dispatch unit-mesh fixture machinery, capped at two
// exposed chips. A one-chip machine still runs the lifecycle test above.
class CircularBufferStatisticsTwoDeviceTest : public AnyDispatchMeshDeviceSingleCardFixture {
protected:
    void SetUp() override {
        if (MetalContext::instance().get_cluster().user_exposed_chip_ids().size() < 2) {
            GTEST_SKIP() << "Cached program registration across devices requires two exposed chips";
        }
        AnyDispatchMeshDeviceSingleCardFixture::SetUp();
    }

    void create_devices() override {
        std::vector<ChipId> selected;
        for (const ChipId id : MetalContext::instance().get_cluster().user_exposed_chip_ids()) {
            selected.push_back(id);
            if (selected.size() == 2) {
                break;
            }
        }
        AnyDispatchMeshDeviceSingleCardFixture::create_devices(selected);
    }
};

TEST_F(CircularBufferStatisticsTwoDeviceTest, CachedProgramRegistersAndUnregistersOnEachDevice) {
    ASSERT_EQ(devices_.size(), 2U);
    auto& first_mesh = *devices_[0];
    auto& second_mesh = *devices_[1];
    ASSERT_EQ(first_mesh.get_devices().size(), 1U);
    ASSERT_EQ(second_mesh.get_devices().size(), 1U);
    auto* first_device = dynamic_cast<Device*>(first_mesh.get_devices().front());
    auto* second_device = dynamic_cast<Device*>(second_mesh.get_devices().front());
    ASSERT_NE(first_device, nullptr);
    ASSERT_NE(second_device, nullptr);
    ASSERT_NE(first_device, second_device);
    ASSERT_GE(first_mesh.compute_with_storage_grid_size().x, 3U);
    ASSERT_GE(second_mesh.compute_with_storage_grid_size().x, 2U);
    ASSERT_EQ(first_device->get_total_cb_allocated(), 0U);
    ASSERT_EQ(second_device->get_total_cb_allocated(), 0U);

    auto shared = make_cb_stats_program(CoreRange({0, 0}, {1, 0}), 4096);
    shared.program->impl().allocate_circular_buffers(&first_mesh);
    const auto shared_address = cb_stats_address(shared);
    EXPECT_EQ(first_device->get_total_cb_allocated(), 8192U);
    EXPECT_EQ(second_device->get_total_cb_allocated(), 0U);

    // The layout is already calculated: this must register the same ProgramImpl
    // with the second physical device without allocating a new layout.
    shared.program->impl().allocate_circular_buffers(&second_mesh);
    EXPECT_EQ(cb_stats_address(shared), shared_address);
    EXPECT_EQ(first_device->get_total_cb_allocated(), 8192U);
    EXPECT_EQ(second_device->get_total_cb_allocated(), 8192U);
    shared.program->impl().allocate_circular_buffers(&second_mesh);
    EXPECT_EQ(second_device->get_total_cb_allocated(), 8192U);

    auto first_only = make_cb_stats_program(CoreRange({2, 0}, {2, 0}), 2048);
    first_only.program->impl().allocate_circular_buffers(&first_mesh);
    EXPECT_EQ(first_device->get_total_cb_allocated(), 10240U);
    EXPECT_EQ(second_device->get_total_cb_allocated(), 8192U);

    shared.program.reset();
    EXPECT_EQ(first_device->get_total_cb_allocated(), 2048U);
    EXPECT_EQ(second_device->get_total_cb_allocated(), 0U);
    first_only.program.reset();
    EXPECT_EQ(first_device->get_total_cb_allocated(), 0U);
    EXPECT_EQ(second_device->get_total_cb_allocated(), 0U);
}

}  // namespace tt::tt_metal
