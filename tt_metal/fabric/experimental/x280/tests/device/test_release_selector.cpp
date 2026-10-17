// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// The shared release selector on a mock Blackhole P300 whose chip 0 has
// harvested Tensix columns.

#include "harness/metal_context.hpp"
#include "harness/regions/core_sync_roster.hpp"
#include "harness/regions/release_selector.hpp"

#include <gtest/gtest.h>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/mesh_device.hpp>

#include "impl/context/metal_context.hpp"
#include "llrt/tt_cluster.hpp"

#include <memory>
#include <set>
#include <stdexcept>
#include <vector>

namespace {

namespace hreg = x280::harness::regions;
using tt::CoreType;
using tt::tt_metal::CoreCoord;
using tt::tt_metal::NOC;
using tt::tt_metal::distributed::MeshCoordinate;
using tt::tt_metal::distributed::MeshDevice;
using tt::tt_metal::distributed::MeshDeviceConfig;
using tt::tt_metal::distributed::MeshShape;

constexpr hreg::ReleaseSender kX280 = hreg::kX280ReleaseSender;

class ReleaseSelector : public ::testing::Test {
protected:
    void SetUp() override {
        mesh_ = MeshDevice::create(MeshDeviceConfig(MeshShape(1, 2)));
        unit_ = mesh_->create_submesh(MeshShape(1, 1), MeshCoordinate(0, 0));
        ASSERT_EQ(unit_->get_device_ids().front(), 0);
    }

    void TearDown() override {
        unit_.reset();
        if (mesh_ && mesh_->is_initialized()) {
            mesh_->close();
        }
    }

    // Tensix participants for the logical block [x0, x1] x [y0, y1], row-major.
    static std::vector<hreg::ReleaseParticipant> block(uint32_t x0, uint32_t x1, uint32_t y0, uint32_t y1) {
        std::vector<hreg::ReleaseParticipant> out;
        for (uint32_t y = y0; y <= y1; ++y) {
            for (uint32_t x = x0; x <= x1; ++x) {
                out.push_back({CoreType::TENSIX, CoreCoord{x, y}});
            }
        }
        return out;
    }

    CoreCoord virt(const CoreCoord& logical) const { return unit_->worker_core_from_logical_core(logical); }

    CoreCoord noc0(const CoreCoord& logical) const {
        const auto c = x280::harness::cluster().get_soc_desc(0).translate_coord_to(
            tt::umd::CoreCoord(logical.x, logical.y, CoreType::TENSIX, tt::CoordSystem::LOGICAL),
            tt::CoordSystem::NOC0);
        return CoreCoord(c.x, c.y);
    }

    static CoreCoord xy(const x280::TranslatedCoord& t) { return CoreCoord(t.x(), t.y()); }

    void expect_group(const hreg::ReleasePlan& plan, const CoreCoord& lo, const CoreCoord& hi, uint32_t dests) const {
        ASSERT_TRUE(plan.group.has_value());
        EXPECT_EQ(xy(plan.group->start), virt(lo));
        EXPECT_EQ(xy(plan.group->end), virt(hi));
        EXPECT_EQ(plan.group->num_dests, dests);
    }

    std::shared_ptr<MeshDevice> mesh_;
    std::shared_ptr<MeshDevice> unit_;
};

TEST_F(ReleaseSelector, LogicalRectangleIsOneGroupAndTranslatesInInputOrder) {
    auto participants = block(0, 1, 0, 1);
    std::swap(participants[0], participants[3]);
    const auto plan = hreg::select_release(*unit_, participants, kX280, hreg::ReleaseMode::Auto, 0u);

    expect_group(plan, {0, 0}, {1, 1}, 4u);
    EXPECT_TRUE(plan.endpoints.empty());
    ASSERT_EQ(plan.translated.size(), participants.size());
    for (size_t i = 0; i < participants.size(); ++i) {
        EXPECT_EQ(xy(plan.translated[i]), virt(std::get<CoreCoord>(participants[i].where)));
    }
}

// Logical x 6..7 maps to virtual 7 and 10, across the opted-out columns 8-9,
// and to NOC0 7 and 11, across harvested column 10 as well.
TEST_F(ReleaseSelector, OptedOutGapsInsideTheVirtualRectangleStillMulticast) {
    ASSERT_EQ(virt({6, 0}).x, 7u);
    ASSERT_EQ(virt({7, 0}).x, 10u);
    ASSERT_GT(noc0({7, 0}).x, 10u) << "the descriptor no longer harvests a column inside the span";
    expect_group(
        hreg::select_release(*unit_, block(6, 7, 0, 1), kX280, hreg::ReleaseMode::RequireMulticast, 0u),
        {6, 0},
        {7, 1},
        4u);
}

TEST_F(ReleaseSelector, SparseSelectionTakesEndpoints) {
    const std::vector<hreg::ReleaseParticipant> sparse{
        {CoreType::TENSIX, CoreCoord{0, 0}}, {CoreType::TENSIX, CoreCoord{2, 0}}};
    const auto plan = hreg::select_release(*unit_, sparse, kX280, hreg::ReleaseMode::Auto, 2u);
    EXPECT_FALSE(plan.group.has_value());
    EXPECT_EQ(plan.endpoints, (std::vector<uint32_t>{0u, 1u}));
    EXPECT_THROW(
        hreg::select_release(*unit_, sparse, kX280, hreg::ReleaseMode::RequireMulticast, 2u), std::invalid_argument);
}

TEST_F(ReleaseSelector, NonTensixIsAnEndpointInEveryMode) {
    const auto eth_cores = unit_->get_devices().front()->get_active_ethernet_cores(true);
    ASSERT_FALSE(eth_cores.empty());
    const CoreCoord eth = *eth_cores.begin();
    auto participants = block(0, 1, 0, 0);
    participants.insert(participants.begin() + 1, {CoreType::ETH, eth});

    for (const auto mode : {hreg::ReleaseMode::Auto, hreg::ReleaseMode::RequireMulticast}) {
        const auto plan = hreg::select_release(*unit_, participants, kX280, mode, 1u);
        expect_group(plan, {0, 0}, {1, 0}, 2u);
        EXPECT_EQ(plan.endpoints, (std::vector<uint32_t>{1u}));
        EXPECT_EQ(xy(plan.translated[1]), unit_->ethernet_core_from_logical_core(eth));
    }
    const std::vector<hreg::ReleaseParticipant> eth_only{{CoreType::ETH, eth}};
    EXPECT_FALSE(hreg::select_release(*unit_, eth_only, kX280, hreg::ReleaseMode::Auto, 1u).group);
    EXPECT_THROW(
        hreg::select_release(*unit_, eth_only, kX280, hreg::ReleaseMode::RequireMulticast, 1u), std::invalid_argument);
}

// Metal's multicast does not write to its sender, so a director inside the
// rectangle fills its own cell and is not counted.
TEST_F(ReleaseSelector, NonLoopbackSenderFillsItsCellWithoutCounting) {
    auto participants = block(0, 1, 0, 1);
    participants.erase(participants.begin());
    const hreg::ReleaseSender director{
        .noc = NOC::NOC_0, .can_multicast = true, .non_loopback_tensix = CoreCoord{0, 0}};
    expect_group(
        hreg::select_release(*unit_, participants, director, hreg::ReleaseMode::RequireMulticast, 0u),
        {0, 0},
        {1, 1},
        3u);
    EXPECT_FALSE(hreg::select_release(*unit_, participants, kX280, hreg::ReleaseMode::Auto, 3u).group)
        << "without the sender the cell is a gap";
    EXPECT_THROW(
        hreg::select_release(*unit_, block(0, 1, 0, 1), director, hreg::ReleaseMode::Auto, 4u), std::invalid_argument);
}

TEST_F(ReleaseSelector, ResolvedParticipantIsAnEndpointAndNeverTensix) {
    const auto l2cpu = x280::X280Runtime::virtual_core_from_logical_core(*unit_, {0, 0}, CoreType::L2CPU, NOC::NOC_0);
    auto participants = block(0, 1, 0, 0);
    participants.push_back({CoreType::L2CPU, l2cpu});
    const auto plan = hreg::select_release(*unit_, participants, kX280, hreg::ReleaseMode::Auto, 1u);
    expect_group(plan, {0, 0}, {1, 0}, 2u);
    EXPECT_EQ(plan.endpoints, (std::vector<uint32_t>{2u}));
    EXPECT_EQ(plan.translated[2], l2cpu);
    EXPECT_THROW(
        hreg::select_release(*unit_, {{CoreType::TENSIX, l2cpu}}, kX280, hreg::ReleaseMode::Auto, 1u),
        std::invalid_argument);
}

TEST_F(ReleaseSelector, SenderThatCannotMulticastTakesEndpoints) {
    const auto rect = block(0, 1, 0, 0);
    for (const auto sender :
         {hreg::ReleaseSender{.noc = NOC::NOC_0, .can_multicast = false},
          hreg::ReleaseSender{.noc = NOC::NOC_1, .can_multicast = true}}) {
        const auto plan = hreg::select_release(*unit_, rect, sender, hreg::ReleaseMode::Auto, 2u);
        EXPECT_FALSE(plan.group.has_value());
        EXPECT_EQ(plan.endpoints.size(), 2u);
        EXPECT_THROW(
            hreg::select_release(*unit_, rect, sender, hreg::ReleaseMode::RequireMulticast, 2u), std::invalid_argument);
    }
}

TEST_F(ReleaseSelector, UnicastOnlyReleasesARectangleByEndpoints) {
    const auto plan = hreg::select_release(*unit_, block(0, 1, 0, 1), kX280, hreg::ReleaseMode::UnicastOnly, 4u);
    EXPECT_FALSE(plan.group.has_value());
    EXPECT_EQ(plan.endpoints, (std::vector<uint32_t>{0u, 1u, 2u, 3u}));
}

TEST_F(ReleaseSelector, EndpointCapacityIsAnError) {
    EXPECT_THROW(
        hreg::select_release(*unit_, block(0, 1, 0, 1), kX280, hreg::ReleaseMode::UnicastOnly, 3u),
        std::invalid_argument);
    EXPECT_NO_THROW(hreg::select_release(*unit_, block(0, 1, 0, 1), kX280, hreg::ReleaseMode::Auto, 0u));
}

TEST_F(ReleaseSelector, InvalidSelectionsAreRefused) {
    EXPECT_THROW(hreg::select_release(*unit_, {}, kX280, hreg::ReleaseMode::Auto, 1u), std::invalid_argument);
    EXPECT_THROW(
        hreg::select_release(
            *unit_,
            {{CoreType::TENSIX, CoreCoord{0, 0}}, {CoreType::TENSIX, CoreCoord{0, 0}}},
            kX280,
            hreg::ReleaseMode::Auto,
            2u),
        std::invalid_argument);
    EXPECT_THROW(
        hreg::select_release(*unit_, {{CoreType::TENSIX, CoreCoord{999, 999}}}, kX280, hreg::ReleaseMode::Auto, 1u),
        std::runtime_error);
    EXPECT_THROW(
        hreg::select_release(*mesh_, block(0, 0, 0, 0), kX280, hreg::ReleaseMode::Auto, 1u), std::invalid_argument);
    unit_->close();
    EXPECT_THROW(
        hreg::select_release(*unit_, block(0, 0, 0, 0), kX280, hreg::ReleaseMode::Auto, 1u), std::invalid_argument);
}

// The Metal-director roster takes its release from the selector: a director
// inside the worker block fills its own cell, so Auto multicasts and
// UnicastOnly writes the same workers one by one; arrivals are unchanged.
TEST_F(ReleaseSelector, CoreSyncRosterReleasesTheSameWorkersInEitherMode) {
    using tt::tt_metal::CoreRange;
    using tt::tt_metal::CoreRangeSet;
    const CoreCoord director{0, 0};
    const CoreRangeSet workers(std::set<CoreRange>{CoreRange({1, 0}, {3, 0}), CoreRange({0, 1}, {3, 3})});
    const auto build = [&](hreg::ReleaseMode mode) {
        auto program = tt::tt_metal::CreateProgram();
        return hreg::make_core_sync_roster(program, *unit_, workers, director, NOC::NOC_0, {}, mode);
    };

    const auto multicast = build(hreg::ReleaseMode::Auto);
    ASSERT_EQ(multicast.grid.num_dests, 15u);
    EXPECT_EQ(CoreCoord(multicast.grid.rect.start.x, multicast.grid.rect.start.y), virt({0, 0}));
    EXPECT_EQ(CoreCoord(multicast.grid.rect.end.x, multicast.grid.rect.end.y), virt({3, 3}));
    EXPECT_TRUE(multicast.worker_endpoints.empty());
    EXPECT_EQ(xy(multicast.director_noc), virt(director));

    const auto unicast = build(hreg::ReleaseMode::UnicastOnly);
    EXPECT_EQ(unicast.grid.num_dests, 0u);
    ASSERT_EQ(unicast.worker_endpoints.size(), 15u);
    for (size_t i = 0; i < unicast.workers.size(); ++i) {
        const auto& ep = unicast.worker_endpoints[i];
        EXPECT_EQ(CoreCoord(ep.noc.x, ep.noc.y), virt(unicast.workers[i].logical_core));
        EXPECT_TRUE(x280_noc_rect_contains(multicast.grid.rect, ep.noc));
    }
    EXPECT_EQ(multicast.counter_arrivals, 15u);
    EXPECT_EQ(unicast.counter_arrivals, 15u);
    EXPECT_THROW(
        {
            auto program = tt::tt_metal::CreateProgram();
            hreg::make_core_sync_roster(
                program,
                *unit_,
                CoreRangeSet(std::set<CoreRange>{CoreRange({1, 0}), CoreRange({3, 0})}),
                director,
                NOC::NOC_0,
                {},
                hreg::ReleaseMode::RequireMulticast);
        },
        std::invalid_argument);
}

}  // namespace
