// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Device-backed oracle for the mcast HOST helper (ttnn/cpp/ttnn/kernel_lib/host/mcast_host.hpp).
// Builds PerRow + PerColumn Mcast1D families on a real 8x8 worker grid and asserts semaphores(),
// compile_time_args(), and runtime_args(core) against the frozen CT/RT contract
// (helper_design/NEW_HOST_HELPER/IMPL_PLAN.md §4) — the same wire SenderPipe/ReceiverPipe decode.
// Covers both sender modes: FIXED (one edge sender -> one rect) and ROTATING (the sender role walks
// the line; every core carries its full-line rect + the ordered per-round sender coords).
// A device is needed because the RT coords are VIRTUAL (worker_core_from_logical_core), which is a
// per-arch mapping only the device knows.

#include <gtest/gtest.h>

#include <algorithm>
#include <utility>
#include <vector>

#include <tt-metalium/core_coord.hpp>

#include "ttnn/cpp/ttnn/kernel_lib/host/mcast_host.hpp"
#include "ttnn/cpp/ttnn/operations/normalization/groupnorm/device/groupnorm_program_utils.hpp"
#include "ttnn/cpp/ttnn/operations/normalization/layernorm/device/sharded_layernorm_factory_helpers.hpp"
#include "ttnn_test_fixtures.hpp"

namespace ttnn::kernel_lib::host::test {

using tt::tt_metal::CoreCoord;
using tt::tt_metal::CoreRange;
using tt::tt_metal::CoreRangeSet;
using tt::tt_metal::NOC;

class McastHostFixture : public ::ttnn::TTNNFixtureWithSuiteDevice<McastHostFixture> {};

TEST(McastHostWire, AbsentFamilyIsOneWord) { EXPECT_EQ(skip_mcast_compile_time_args(), (std::vector<uint32_t>{0})); }

namespace {

constexpr uint32_t NO_SENDER_ROUND = 0xFFFFFFFFu;

CoreRangeSet make_grid(uint32_t gc, uint32_t gr) {
    return CoreRangeSet(CoreRange(CoreCoord(0, 0), CoreCoord(gc - 1, gr - 1)));
}

CoreRangeSet make_grid(CoreCoord origin, uint32_t gc, uint32_t gr) {
    return CoreRangeSet(CoreRange(origin, CoreCoord(origin.x + gc - 1, origin.y + gr - 1)));
}

std::vector<uint32_t> expected_bbox(tt::tt_metal::IDevice* dev, const std::vector<CoreCoord>& logical_cores, NOC noc) {
    std::vector<std::pair<uint32_t, uint32_t>> virtual_cores;
    virtual_cores.reserve(logical_cores.size());
    for (const auto& core : logical_cores) {
        const auto worker = dev->worker_core_from_logical_core(core);
        virtual_cores.emplace_back(static_cast<uint32_t>(worker.x), static_cast<uint32_t>(worker.y));
    }
    uint32_t xlo = virtual_cores[0].first;
    uint32_t xhi = virtual_cores[0].first;
    uint32_t ylo = virtual_cores[0].second;
    uint32_t yhi = virtual_cores[0].second;
    for (const auto& [x, y] : virtual_cores) {
        xlo = std::min(xlo, x);
        xhi = std::max(xhi, x);
        ylo = std::min(ylo, y);
        yhi = std::max(yhi, y);
    }
    return noc == NOC::NOC_1 ? std::vector<uint32_t>{xhi, yhi, xlo, ylo} : std::vector<uint32_t>{xlo, ylo, xhi, yhi};
}

std::vector<uint32_t> topology_only(std::vector<uint32_t> args) {
    EXPECT_GE(args.size(), 2u);
    args.resize(args.size() - 2u);
    return args;
}

std::vector<uint32_t> role_only(const std::vector<uint32_t>& args) {
    EXPECT_GE(args.size(), 2u);
    return {args[args.size() - 2u], args.back()};
}

}  // namespace

TEST_F(McastHostFixture, PerCoreRolesAndSenderPhases) {
    auto* dev = device_;
    const CoreRangeSet receivers(CoreRange(CoreCoord(1, 1), CoreCoord(2, 2)));
    const CoreRangeSet sender_grid(std::vector<CoreRange>{
        CoreRange(CoreCoord(1, 1), CoreCoord(1, 2)),
        CoreRange(CoreCoord(3, 1), CoreCoord(3, 2)),
    });
    McastConfig cfg;
    Mcast1D mc(dev, receivers, Mcast1DShape::PerRow, Mcast1DRotatingSenderConfig{sender_grid}, cfg);

    EXPECT_EQ(role_only(mc.runtime_args(CoreCoord(1, 1))), (std::vector<uint32_t>{0x3u, 0u}));
    EXPECT_EQ(role_only(mc.runtime_args(CoreCoord(3, 1))), (std::vector<uint32_t>{0x1u, 1u}));
    EXPECT_EQ(role_only(mc.runtime_args(CoreCoord(2, 1))), (std::vector<uint32_t>{0x2u, NO_SENDER_ROUND}));
}

TEST(GroupNormMcastGeometry, ZeroEdgeRectangle) {
    std::vector<CoreCoord> group = {CoreCoord(0, 0), CoreCoord(1, 0), CoreCoord(2, 0)};
    std::vector<CoreCoord> first;
    std::vector<CoreCoord> middle = group;
    std::vector<CoreCoord> last;

    ttnn::prim::split_and_form_rectangle_grids(group, first, middle, last);

    EXPECT_TRUE(first.empty());
    EXPECT_EQ(middle, group);
    EXPECT_TRUE(last.empty());
}

TEST(GroupNormMcastGeometry, OneEdgeWrappedSequence) {
    std::vector<CoreCoord> group = {CoreCoord(7, 0), CoreCoord(0, 1), CoreCoord(1, 1), CoreCoord(2, 1)};
    std::vector<CoreCoord> first;
    std::vector<CoreCoord> middle = group;
    std::vector<CoreCoord> last;

    ttnn::prim::split_and_form_rectangle_grids(group, first, middle, last);

    EXPECT_EQ(first, std::vector<CoreCoord>({CoreCoord(7, 0)}));
    EXPECT_EQ(middle, std::vector<CoreCoord>({CoreCoord(0, 1), CoreCoord(1, 1), CoreCoord(2, 1)}));
    EXPECT_TRUE(last.empty());
}

TEST(GroupNormMcastGeometry, TwoEdgeWrappedSequence) {
    std::vector<CoreCoord> group = {
        CoreCoord(7, 0), CoreCoord(0, 1), CoreCoord(1, 1), CoreCoord(2, 1), CoreCoord(0, 2)};
    std::vector<CoreCoord> first;
    std::vector<CoreCoord> middle = group;
    std::vector<CoreCoord> last;

    ttnn::prim::split_and_form_rectangle_grids(group, first, middle, last);

    EXPECT_EQ(first, std::vector<CoreCoord>({CoreCoord(7, 0)}));
    EXPECT_EQ(middle, std::vector<CoreCoord>({CoreCoord(0, 1), CoreCoord(1, 1), CoreCoord(2, 1)}));
    EXPECT_EQ(last, std::vector<CoreCoord>({CoreCoord(0, 2)}));
}

// PerRow (matmul in0): sender in column 0 broadcasts across its row.
TEST_F(McastHostFixture, PerRow8x8) {
    auto* dev = device_;
    const auto grid = make_grid(/*gc=*/8, /*gr=*/8);
    auto virt = [&](uint32_t lx, uint32_t ly) {
        const auto w = dev->worker_core_from_logical_core(CoreCoord(lx, ly));
        return std::pair<uint32_t, uint32_t>{static_cast<uint32_t>(w.x), static_cast<uint32_t>(w.y)};
    };
    // NoC0 rect from two logical endpoints: [xlo, ylo, xhi, yhi] over the actual virtual coords.
    auto rect = [&](std::pair<uint32_t, uint32_t> a, std::pair<uint32_t, uint32_t> b) {
        return std::vector<uint32_t>{
            std::min(a.first, b.first),
            std::min(a.second, b.second),
            std::max(a.first, b.first),
            std::max(a.second, b.second)};
    };

    McastConfig cfg;  // defaults: NOC_0, handshake both(true), Flag, base_sem_id 0.
    Mcast1D mc(dev, grid, Mcast1DShape::PerRow, Mcast1DFixedSenderConfig{}, cfg);

    EXPECT_TRUE(mc.has_receivers());

    // --- semaphores: data_ready id 0, consumer_ready id 1, both init 0, over the grid ---
    const auto sems = mc.owned_semaphores();
    ASSERT_EQ(sems.size(), 2u);
    EXPECT_EQ(sems[0].id, 0u);
    EXPECT_EQ(sems[0].initial_value, 0u);
    EXPECT_EQ(sems[0].core_ranges, grid);
    EXPECT_EQ(sems[1].id, 1u);
    EXPECT_EQ(sems[1].initial_value, 0u);

    // --- CT: [present, has_receivers, data_ready, consumer_ready, ack_count, flags, rotating_span]
    //         (ack_count = span-1 = 7; flags = 1; rotating_span = 0 for fixed mode) ---
    EXPECT_EQ(mc.compile_time_args(), (std::vector<uint32_t>{1, 1, 0, 1, 7, 1, 0}));

    // --- RT: sender at col 0 of row Y -> full line; the pipe excludes the source ---
    for (uint32_t Y : {0u, 3u, 7u}) {
        EXPECT_EQ(topology_only(mc.runtime_args(CoreCoord(0, Y))), rect(virt(0, Y), virt(7, Y))) << "sender row " << Y;
    }
    // --- RT: receiver at col X (!=0) of row Y -> [sender_x, sender_y, 0, 0] with sender at col 0 ---
    for (uint32_t X : {1u, 4u, 7u}) {
        const auto s = virt(0, /*Y=*/5);
        EXPECT_EQ(topology_only(mc.runtime_args(CoreCoord(X, 5))), (std::vector<uint32_t>{s.first, s.second, 0, 0}))
            << "receiver col " << X;
        EXPECT_FALSE(mc.is_sender(CoreCoord(X, 5)));
    }
    EXPECT_TRUE(mc.is_sender(CoreCoord(0, 5)));
    EXPECT_EQ(mc.num_receivers(CoreCoord(0, 5)), 7u);
}

// PerColumn (matmul in1): sender in row 0 broadcasts down its column.
TEST_F(McastHostFixture, PerColumn8x8) {
    auto* dev = device_;
    const auto grid = make_grid(/*gc=*/8, /*gr=*/8);
    auto virt = [&](uint32_t lx, uint32_t ly) {
        const auto w = dev->worker_core_from_logical_core(CoreCoord(lx, ly));
        return std::pair<uint32_t, uint32_t>{static_cast<uint32_t>(w.x), static_cast<uint32_t>(w.y)};
    };
    auto rect = [&](std::pair<uint32_t, uint32_t> a, std::pair<uint32_t, uint32_t> b) {
        return std::vector<uint32_t>{
            std::min(a.first, b.first),
            std::min(a.second, b.second),
            std::max(a.first, b.first),
            std::max(a.second, b.second)};
    };

    McastConfig cfg;
    cfg.base_sem_id = 2;  // second family on the same grid: ids 2, 3.
    Mcast1D mc(dev, grid, Mcast1DShape::PerColumn, Mcast1DFixedSenderConfig{}, cfg);

    EXPECT_TRUE(mc.has_receivers());

    const auto sems = mc.owned_semaphores();
    ASSERT_EQ(sems.size(), 2u);
    EXPECT_EQ(sems[0].id, 2u);
    EXPECT_EQ(sems[1].id, 3u);

    EXPECT_EQ(mc.compile_time_args(), (std::vector<uint32_t>{1, 1, 2, 3, 7, 1, 0}));

    // Sender at row 0 of column X -> full line; the pipe excludes the source.
    for (uint32_t X : {0u, 4u, 7u}) {
        EXPECT_EQ(topology_only(mc.runtime_args(CoreCoord(X, 0))), rect(virt(X, 0), virt(X, 7))) << "sender col " << X;
    }
    // receiver at row Y (!=0) of column X -> [sender_x, sender_y, 0, 0] with sender at row 0.
    for (uint32_t Y : {1u, 4u, 7u}) {
        const auto s = virt(/*X=*/6, 0);
        EXPECT_EQ(topology_only(mc.runtime_args(CoreCoord(6, Y))), (std::vector<uint32_t>{s.first, s.second, 0, 0}))
            << "receiver row " << Y;
    }
    EXPECT_TRUE(mc.is_sender(CoreCoord(6, 0)));
    EXPECT_EQ(mc.num_receivers(CoreCoord(6, 0)), 7u);
}

// Degenerate: a fixed sender still carries its true self rectangle and a sender-only role.
TEST_F(McastHostFixture, PerRowDegenerateSingleColumn) {
    auto* dev = device_;
    const auto grid = make_grid(/*gc=*/1, /*gr=*/8);
    McastConfig cfg;
    Mcast1D mc(dev, grid, Mcast1DShape::PerRow, Mcast1DFixedSenderConfig{}, cfg);

    EXPECT_FALSE(mc.has_receivers());
    // has_receivers=0, ids still emitted; ack_count=0 (no receivers); flags=1 (handshake, Flag).
    EXPECT_EQ(mc.compile_time_args(), (std::vector<uint32_t>{1, 0, 0, 1, 0, 1, 0}));
    for (uint32_t Y : {0u, 3u, 7u}) {
        EXPECT_TRUE(mc.is_sender(CoreCoord(0, Y)));
        EXPECT_EQ(topology_only(mc.runtime_args(CoreCoord(0, Y))), expected_bbox(dev, {CoreCoord(0, Y)}, NOC::NOC_0))
            << "row " << Y;
        EXPECT_EQ(mc.num_receivers(CoreCoord(0, Y)), 0u);
    }
}

// Degenerate: a single-row grid has no receivers for a PerColumn family.
TEST_F(McastHostFixture, PerColumnDegenerateSingleRow) {
    auto* dev = device_;
    const auto grid = make_grid(/*gc=*/8, /*gr=*/1);
    McastConfig cfg;
    cfg.base_sem_id = 2;
    Mcast1D mc(dev, grid, Mcast1DShape::PerColumn, Mcast1DFixedSenderConfig{}, cfg);

    EXPECT_FALSE(mc.has_receivers());
    EXPECT_EQ(mc.compile_time_args(), (std::vector<uint32_t>{1, 0, 2, 3, 0, 1, 0}));
    for (uint32_t X : {0u, 4u, 7u}) {
        EXPECT_TRUE(mc.is_sender(CoreCoord(X, 0)));
        EXPECT_EQ(topology_only(mc.runtime_args(CoreCoord(X, 0))), expected_bbox(dev, {CoreCoord(X, 0)}, NOC::NOC_0))
            << "col " << X;
    }
}

TEST_F(McastHostFixture, OffsetPerRowUniformNoc0) {
    auto* dev = device_;
    const CoreCoord origin(2, 3);
    const auto grid = make_grid(origin, /*gc=*/4, /*gr=*/3);
    McastConfig cfg;
    cfg.base_sem_id = 5;
    Mcast1D mc(dev, grid, Mcast1DShape::PerRow, Mcast1DFixedSenderConfig{.starting_sender_index = 1}, cfg);

    const auto semaphores = mc.owned_semaphores();
    ASSERT_EQ(semaphores.size(), 2u);
    EXPECT_EQ(semaphores[0].id, 5u);
    EXPECT_EQ(semaphores[1].id, 6u);
    EXPECT_EQ(semaphores[0].core_ranges, grid);
    EXPECT_EQ(mc.compile_time_args(), (std::vector<uint32_t>{1, 1, 5, 6, 3, 1, 0}));

    EXPECT_TRUE(mc.is_sender(CoreCoord(3, 4)));
    EXPECT_FALSE(mc.is_sender(CoreCoord(2, 4)));
    EXPECT_EQ(mc.num_receivers(CoreCoord(3, 4)), 3u);
    EXPECT_EQ(
        topology_only(mc.runtime_args(CoreCoord(3, 4))),
        expected_bbox(dev, {CoreCoord(2, 4), CoreCoord(3, 4), CoreCoord(4, 4), CoreCoord(5, 4)}, NOC::NOC_0));
    const auto sender = dev->worker_core_from_logical_core(CoreCoord(3, 4));
    EXPECT_EQ(
        topology_only(mc.runtime_args(CoreCoord(5, 4))),
        (std::vector<uint32_t>{static_cast<uint32_t>(sender.x), static_cast<uint32_t>(sender.y), 0, 0}));
}

TEST_F(McastHostFixture, OffsetPerRowDiagonalNoc1) {
    auto* dev = device_;
    const auto grid = make_grid(CoreCoord(2, 3), /*gc=*/4, /*gr=*/3);
    McastConfig cfg;
    cfg.noc = NOC::NOC_1;
    Mcast1D mc(
        dev,
        grid,
        Mcast1DShape::PerRow,
        Mcast1DFixedSenderConfig{.starting_sender_index = 1, .sender_placement = Mcast1DSenderPlacement::Diagonal},
        cfg);

    EXPECT_TRUE(mc.is_sender(CoreCoord(3, 3)));
    EXPECT_TRUE(mc.is_sender(CoreCoord(4, 4)));
    EXPECT_TRUE(mc.is_sender(CoreCoord(5, 5)));
    EXPECT_EQ(
        topology_only(mc.runtime_args(CoreCoord(5, 5))),
        expected_bbox(dev, {CoreCoord(2, 5), CoreCoord(3, 5), CoreCoord(4, 5), CoreCoord(5, 5)}, NOC::NOC_1));
    const auto sender = dev->worker_core_from_logical_core(CoreCoord(4, 4));
    EXPECT_EQ(
        topology_only(mc.runtime_args(CoreCoord(2, 4))),
        (std::vector<uint32_t>{static_cast<uint32_t>(sender.x), static_cast<uint32_t>(sender.y), 0, 0}));
}

TEST_F(McastHostFixture, OffsetPerColumnUniformNoc0) {
    auto* dev = device_;
    const CoreCoord origin(3, 2);
    const auto grid = make_grid(origin, /*gc=*/3, /*gr=*/4);
    McastConfig cfg;
    cfg.base_sem_id = 7;
    Mcast1D mc(dev, grid, Mcast1DShape::PerColumn, Mcast1DFixedSenderConfig{.starting_sender_index = 1}, cfg);

    const auto semaphores = mc.owned_semaphores();
    ASSERT_EQ(semaphores.size(), 2u);
    EXPECT_EQ(semaphores[0].id, 7u);
    EXPECT_EQ(semaphores[1].id, 8u);
    EXPECT_EQ(semaphores[1].core_ranges, grid);
    EXPECT_TRUE(mc.is_sender(CoreCoord(4, 3)));
    EXPECT_EQ(
        topology_only(mc.runtime_args(CoreCoord(4, 3))),
        expected_bbox(dev, {CoreCoord(4, 2), CoreCoord(4, 3), CoreCoord(4, 4), CoreCoord(4, 5)}, NOC::NOC_0));
    const auto sender = dev->worker_core_from_logical_core(CoreCoord(4, 3));
    EXPECT_EQ(
        topology_only(mc.runtime_args(CoreCoord(4, 5))),
        (std::vector<uint32_t>{static_cast<uint32_t>(sender.x), static_cast<uint32_t>(sender.y), 0, 0}));
}

TEST_F(McastHostFixture, OffsetPerColumnDiagonalNoc1) {
    auto* dev = device_;
    const auto grid = make_grid(CoreCoord(3, 2), /*gc=*/3, /*gr=*/4);
    McastConfig cfg;
    cfg.noc = NOC::NOC_1;
    Mcast1D mc(
        dev,
        grid,
        Mcast1DShape::PerColumn,
        Mcast1DFixedSenderConfig{.starting_sender_index = 1, .sender_placement = Mcast1DSenderPlacement::Diagonal},
        cfg);

    EXPECT_TRUE(mc.is_sender(CoreCoord(3, 3)));
    EXPECT_TRUE(mc.is_sender(CoreCoord(4, 4)));
    EXPECT_TRUE(mc.is_sender(CoreCoord(5, 5)));
    EXPECT_EQ(
        topology_only(mc.runtime_args(CoreCoord(5, 5))),
        expected_bbox(dev, {CoreCoord(5, 2), CoreCoord(5, 3), CoreCoord(5, 4), CoreCoord(5, 5)}, NOC::NOC_1));
    const auto sender = dev->worker_core_from_logical_core(CoreCoord(4, 4));
    EXPECT_EQ(
        topology_only(mc.runtime_args(CoreCoord(4, 2))),
        (std::vector<uint32_t>{static_cast<uint32_t>(sender.x), static_cast<uint32_t>(sender.y), 0, 0}));
}

TEST_F(McastHostFixture, OffsetDegenerateLines) {
    auto* dev = device_;
    const auto row_grid = make_grid(CoreCoord(5, 4), /*gc=*/1, /*gr=*/3);
    const auto column_grid = make_grid(CoreCoord(4, 5), /*gc=*/3, /*gr=*/1);
    McastConfig cfg;
    cfg.noc = NOC::NOC_1;
    Mcast1D per_row(dev, row_grid, Mcast1DShape::PerRow, Mcast1DFixedSenderConfig{}, cfg);
    Mcast1D per_column(dev, column_grid, Mcast1DShape::PerColumn, Mcast1DFixedSenderConfig{}, cfg);

    EXPECT_FALSE(per_row.has_receivers());
    EXPECT_TRUE(per_row.is_sender(CoreCoord(5, 6)));
    EXPECT_EQ(topology_only(per_row.runtime_args(CoreCoord(5, 6))), expected_bbox(dev, {CoreCoord(5, 6)}, NOC::NOC_1));
    EXPECT_EQ(per_row.owned_semaphores()[0].core_ranges, row_grid);
    EXPECT_FALSE(per_column.has_receivers());
    EXPECT_TRUE(per_column.is_sender(CoreCoord(6, 5)));
    EXPECT_EQ(
        topology_only(per_column.runtime_args(CoreCoord(6, 5))), expected_bbox(dev, {CoreCoord(6, 5)}, NOC::NOC_1));
    EXPECT_EQ(per_column.owned_semaphores()[0].core_ranges, column_grid);
}

// =============================================================================
// ROTATING sender (block-sharded matmul in0 / sdpa-decode): the sender role walks the line. Every
// core's RT block is [ full-line rect | ordered sender coords, one pair per round ], identical for
// every core on the same line. The receiver indexes the coord list by round.
// =============================================================================

namespace {

// The rotating RT block a row Y should produce on NoC0: the NoC0-ordered bounding box over the whole
// row's virtual coords, followed by those coords in column order (round 0..span-1).
std::vector<uint32_t> expected_rotating_row(const std::vector<std::pair<uint32_t, uint32_t>>& line, bool noc1) {
    uint32_t xlo = line[0].first, xhi = line[0].first, ylo = line[0].second, yhi = line[0].second;
    for (const auto& v : line) {
        xlo = std::min(xlo, v.first);
        xhi = std::max(xhi, v.first);
        ylo = std::min(ylo, v.second);
        yhi = std::max(yhi, v.second);
    }
    std::vector<uint32_t> out =
        noc1 ? std::vector<uint32_t>{xhi, yhi, xlo, ylo} : std::vector<uint32_t>{xlo, ylo, xhi, yhi};
    for (const auto& v : line) {
        out.push_back(v.first);
        out.push_back(v.second);
    }
    return out;
}

}  // namespace

// PerRow rotating: each row's cores rotate the sender role among themselves.
TEST_F(McastHostFixture, PerRowRotating8x8) {
    auto* dev = device_;
    const auto grid = make_grid(/*gc=*/8, /*gr=*/8);
    auto virt = [&](uint32_t lx, uint32_t ly) {
        const auto w = dev->worker_core_from_logical_core(CoreCoord(lx, ly));
        return std::pair<uint32_t, uint32_t>{static_cast<uint32_t>(w.x), static_cast<uint32_t>(w.y)};
    };

    McastConfig cfg;
    Mcast1D mc(dev, grid, Mcast1DShape::PerRow, Mcast1DRotatingSenderConfig{}, cfg);

    EXPECT_TRUE(mc.has_receivers());
    EXPECT_EQ(mc.compile_time_args()[6], 8u);  // 8 rounds; every column takes a sender turn
    // The rotating span is self-described by the sixth CT word.
    EXPECT_EQ(mc.compile_time_args(), (std::vector<uint32_t>{1, 1, 0, 1, 7, 1, 8}));

    for (uint32_t Y : {0u, 3u, 7u}) {
        std::vector<std::pair<uint32_t, uint32_t>> line;
        for (uint32_t X = 0; X < 8; ++X) {
            line.push_back(virt(X, Y));
        }
        const auto expected = expected_rotating_row(line, /*noc1=*/false);
        // Every core on the row emits the SAME line-uniform block (rect + ordered coords).
        for (uint32_t X : {0u, 1u, 4u, 7u}) {
            EXPECT_EQ(topology_only(mc.runtime_args(CoreCoord(X, Y))), expected) << "row " << Y << " core col " << X;
            EXPECT_TRUE(mc.is_sender(CoreCoord(X, Y)));        // every core is a sender at some round
            EXPECT_EQ(mc.num_receivers(CoreCoord(X, Y)), 7u);  // and reaches the other 7 on its round
        }
        // Spot-check the wire semantics: block[0..3] = rect, then coords[round] at 4 + 2*round.
        const auto rt = topology_only(mc.runtime_args(CoreCoord(0, Y)));
        ASSERT_EQ(rt.size(), 4u + 2u * 8u);
        for (uint32_t round = 0; round < 8; ++round) {
            EXPECT_EQ(rt[4 + 2 * round + 0], virt(round, Y).first) << "round " << round << " sender_x";
            EXPECT_EQ(rt[4 + 2 * round + 1], virt(round, Y).second) << "round " << round << " sender_y";
        }
    }
}

// PerColumn rotating: each column's cores rotate the sender role; coords run in row order.
TEST_F(McastHostFixture, PerColumnRotating8x8) {
    auto* dev = device_;
    const auto grid = make_grid(/*gc=*/8, /*gr=*/8);
    auto virt = [&](uint32_t lx, uint32_t ly) {
        const auto w = dev->worker_core_from_logical_core(CoreCoord(lx, ly));
        return std::pair<uint32_t, uint32_t>{static_cast<uint32_t>(w.x), static_cast<uint32_t>(w.y)};
    };

    McastConfig cfg;
    cfg.base_sem_id = 2;
    Mcast1D mc(dev, grid, Mcast1DShape::PerColumn, Mcast1DRotatingSenderConfig{}, cfg);

    EXPECT_TRUE(mc.has_receivers());
    EXPECT_EQ(mc.compile_time_args()[6], 8u);
    EXPECT_EQ(mc.compile_time_args(), (std::vector<uint32_t>{1, 1, 2, 3, 7, 1, 8}));

    for (uint32_t X : {0u, 4u, 7u}) {
        std::vector<std::pair<uint32_t, uint32_t>> line;
        for (uint32_t Y = 0; Y < 8; ++Y) {
            line.push_back(virt(X, Y));
        }
        const auto expected = expected_rotating_row(line, /*noc1=*/false);
        for (uint32_t Y : {0u, 2u, 7u}) {
            EXPECT_EQ(topology_only(mc.runtime_args(CoreCoord(X, Y))), expected) << "col " << X << " core row " << Y;
        }
        const auto rt = topology_only(mc.runtime_args(CoreCoord(X, 0)));
        ASSERT_EQ(rt.size(), 4u + 2u * 8u);
        for (uint32_t round = 0; round < 8; ++round) {
            EXPECT_EQ(rt[4 + 2 * round + 0], virt(X, round).first) << "round " << round << " sender_x";
            EXPECT_EQ(rt[4 + 2 * round + 1], virt(X, round).second) << "round " << round << " sender_y";
        }
    }
}

// Rotating on NoC1: the rect corners must swap (high-corner start); the per-round sender coords are
// NOT NoC-ordered (they are unicast ack targets), so they stay in round order regardless of NoC.
TEST_F(McastHostFixture, PerRowRotatingNoc1) {
    auto* dev = device_;
    const auto grid = make_grid(/*gc=*/8, /*gr=*/8);
    auto virt = [&](uint32_t lx, uint32_t ly) {
        const auto w = dev->worker_core_from_logical_core(CoreCoord(lx, ly));
        return std::pair<uint32_t, uint32_t>{static_cast<uint32_t>(w.x), static_cast<uint32_t>(w.y)};
    };

    McastConfig cfg;
    cfg.noc = NOC::NOC_1;
    Mcast1D mc(dev, grid, Mcast1DShape::PerRow, Mcast1DRotatingSenderConfig{}, cfg);

    for (uint32_t Y : {0u, 5u}) {
        std::vector<std::pair<uint32_t, uint32_t>> line;
        for (uint32_t X = 0; X < 8; ++X) {
            line.push_back(virt(X, Y));
        }
        EXPECT_EQ(topology_only(mc.runtime_args(CoreCoord(3, Y))), expected_rotating_row(line, /*noc1=*/true))
            << "row " << Y;
    }
}

// Degenerate rotating line (single column): span 1 -> no receivers, zeroed rect, one self coord.
TEST_F(McastHostFixture, PerRowRotatingDegenerate) {
    auto* dev = device_;
    const auto grid = make_grid(/*gc=*/1, /*gr=*/8);
    auto virt = [&](uint32_t lx, uint32_t ly) {
        const auto w = dev->worker_core_from_logical_core(CoreCoord(lx, ly));
        return std::pair<uint32_t, uint32_t>{static_cast<uint32_t>(w.x), static_cast<uint32_t>(w.y)};
    };

    McastConfig cfg;
    Mcast1D mc(dev, grid, Mcast1DShape::PerRow, Mcast1DRotatingSenderConfig{}, cfg);

    EXPECT_FALSE(mc.has_receivers());
    EXPECT_EQ(mc.compile_time_args()[6], 1u);
    EXPECT_EQ(mc.compile_time_args(), (std::vector<uint32_t>{1, 0, 0, 1, 0, 1, 1}));
    for (uint32_t Y : {0u, 7u}) {
        const auto s = virt(0, Y);
        EXPECT_EQ(
            topology_only(mc.runtime_args(CoreCoord(0, Y))), (std::vector<uint32_t>{0, 0, 0, 0, s.first, s.second}))
            << "row " << Y;
        EXPECT_EQ(mc.num_receivers(CoreCoord(0, Y)), 0u);
    }
}

// =============================================================================
// Mcast2D — ONE mcast over a single rectangle. The rect is passed to the ctor; the sender's
// containment in it (auto-detected) picks the mode: fully-inside (fan-out area-1, rotating OK) vs
// separate sender (fan-out area, fixed only). ack_count is the handshake ack wait-count. CT grows
// to 7 words [present, has_receivers, data_ready, consumer_ready, ack_count, flags, rotating_span].
// =============================================================================

namespace {

// Expected NoC-ordered bbox over the virtual coords of EVERY core in a logical rect [x0,x1]x[y0,y1].
std::vector<uint32_t> expected_rect2d(
    tt::tt_metal::IDevice* dev, uint32_t x0, uint32_t y0, uint32_t x1, uint32_t y1, bool noc1) {
    uint32_t xlo = 0, xhi = 0, ylo = 0, yhi = 0;
    bool first = true;
    for (uint32_t y = y0; y <= y1; ++y) {
        for (uint32_t x = x0; x <= x1; ++x) {
            const auto w = dev->worker_core_from_logical_core(CoreCoord(x, y));
            const auto vx = static_cast<uint32_t>(w.x);
            const auto vy = static_cast<uint32_t>(w.y);
            if (first) {
                xlo = xhi = vx;
                ylo = yhi = vy;
                first = false;
            } else {
                xlo = std::min(xlo, vx);
                xhi = std::max(xhi, vx);
                ylo = std::min(ylo, vy);
                yhi = std::max(yhi, vy);
            }
        }
    }
    return noc1 ? std::vector<uint32_t>{xhi, yhi, xlo, ylo} : std::vector<uint32_t>{xlo, ylo, xhi, yhi};
}

}  // namespace

// Fully-inside dense: sender at a corner of an 8x8 rect broadcasts to the whole rect (area-1 = 63).
TEST_F(McastHostFixture, Mcast2DFullyInsideDense) {
    auto* dev = device_;
    const auto rect = make_grid(/*gc=*/8, /*gr=*/8);
    auto virt = [&](uint32_t lx, uint32_t ly) {
        const auto w = dev->worker_core_from_logical_core(CoreCoord(lx, ly));
        return std::pair<uint32_t, uint32_t>{static_cast<uint32_t>(w.x), static_cast<uint32_t>(w.y)};
    };

    McastConfig cfg;  // defaults: NOC_0, handshake, Flag, base_sem_id 0.
    Mcast2D mc(dev, rect, Mcast2DFixedSenderConfig{CoreCoord(0, 0)}, cfg);

    EXPECT_TRUE(mc.has_receivers());
    EXPECT_TRUE(mc.sender_in_rect());
    EXPECT_EQ(mc.ack_count(), 63u);
    EXPECT_EQ(mc.num_receivers(CoreCoord(0, 0)), 63u);
    EXPECT_EQ(mc.num_receivers(CoreCoord(3, 5)), 0u);

    // sems: data_ready 0, consumer_ready 1, both init 0, over the whole rect.
    const auto sems = mc.owned_semaphores();
    ASSERT_EQ(sems.size(), 2u);
    EXPECT_EQ(sems[0].id, 0u);
    EXPECT_EQ(sems[1].id, 1u);
    EXPECT_EQ(sems[0].core_ranges, rect);
    EXPECT_EQ(sems[0].core_ranges.num_cores(), 64u);

    // CT: [present, has_receivers, data_ready, consumer_ready, ack_count, flags, rotating_span].
    EXPECT_EQ(mc.compile_time_args(), (std::vector<uint32_t>{1, 1, 0, 1, 63, 1, 0}));

    // sender -> whole-rect corners; receivers -> the sender's virtual coords.
    EXPECT_TRUE(mc.is_sender(CoreCoord(0, 0)));
    EXPECT_EQ(topology_only(mc.runtime_args(CoreCoord(0, 0))), expected_rect2d(dev, 0, 0, 7, 7, /*noc1=*/false));
    for (auto rc : {CoreCoord(1, 0), CoreCoord(3, 5), CoreCoord(7, 7)}) {
        const auto s = virt(0, 0);
        EXPECT_FALSE(mc.is_sender(rc));
        EXPECT_EQ(topology_only(mc.runtime_args(rc)), (std::vector<uint32_t>{s.first, s.second, 0, 0}))
            << "receiver (" << rc.x << "," << rc.y << ")";
    }
}

// Fully-inside with a MIDDLE sender: the sender need not be a corner; the rect is still emitted whole.
TEST_F(McastHostFixture, Mcast2DFullyInsideMiddleSender) {
    auto* dev = device_;
    const auto rect = make_grid(/*gc=*/8, /*gr=*/8);
    McastConfig cfg;
    Mcast2D mc(dev, rect, Mcast2DFixedSenderConfig{CoreCoord(3, 4)}, cfg);

    EXPECT_TRUE(mc.sender_in_rect());
    EXPECT_TRUE(mc.is_sender(CoreCoord(3, 4)));
    EXPECT_FALSE(mc.is_sender(CoreCoord(0, 0)));
    EXPECT_EQ(topology_only(mc.runtime_args(CoreCoord(3, 4))), expected_rect2d(dev, 0, 0, 7, 7, /*noc1=*/false));
    const auto w = dev->worker_core_from_logical_core(CoreCoord(3, 4));
    EXPECT_EQ(
        topology_only(mc.runtime_args(CoreCoord(0, 0))),
        (std::vector<uint32_t>{static_cast<uint32_t>(w.x), static_cast<uint32_t>(w.y), 0, 0}));
}

// Divergent ack_count: the geometric fan-out is still area-1, but only `ack_count` cores ack.
TEST_F(McastHostFixture, Mcast2DDivergentAckCount) {
    auto* dev = device_;
    const auto rect = make_grid(/*gc=*/8, /*gr=*/8);
    McastConfig cfg;
    cfg.ack_count_override = 10;
    Mcast2D mc(dev, rect, Mcast2DFixedSenderConfig{CoreCoord(0, 0)}, cfg);

    EXPECT_EQ(mc.ack_count(), 10u);
    EXPECT_EQ(mc.num_receivers(CoreCoord(0, 0)), 63u);  // geometric fan-out unchanged
    EXPECT_EQ(mc.compile_time_args(), (std::vector<uint32_t>{1, 1, 0, 1, 10, 1, 0}));
}

TEST_F(McastHostFixture, Mcast1DAckCountOverride) {
    auto* dev = device_;
    const auto grid = make_grid(/*gc=*/8, /*gr=*/8);
    McastConfig cfg;
    cfg.ack_count_override = 3;
    Mcast1D mc(dev, grid, Mcast1DShape::PerRow, Mcast1DFixedSenderConfig{}, cfg);

    EXPECT_EQ(mc.ack_count(), 3u);
    EXPECT_EQ(mc.num_receivers(CoreCoord(0, 0)), 7u);
    EXPECT_EQ(mc.compile_time_args(), (std::vector<uint32_t>{1, 1, 0, 1, 3, 1, 0}));

    cfg.ack_count_override = 0;
    Mcast1D zero_ack_mc(dev, grid, Mcast1DShape::PerRow, Mcast1DFixedSenderConfig{}, cfg);
    EXPECT_EQ(zero_ack_mc.ack_count(), 0u);
}

// Separate sender: the sender sits OUTSIDE the rect, so every rect core is a receiver (fan-out area),
// and the participating set that carries the sems is rect ∪ {sender}.
TEST_F(McastHostFixture, Mcast2DSeparateSender) {
    auto* dev = device_;
    const auto rect = CoreRangeSet(CoreRange(CoreCoord(1, 1), CoreCoord(4, 4)));  // 4x4 = 16 cores
    const CoreCoord sender(0, 0);                                                 // outside the rect
    auto virt = [&](uint32_t lx, uint32_t ly) {
        const auto w = dev->worker_core_from_logical_core(CoreCoord(lx, ly));
        return std::pair<uint32_t, uint32_t>{static_cast<uint32_t>(w.x), static_cast<uint32_t>(w.y)};
    };

    McastConfig cfg;
    Mcast2D mc(dev, rect, Mcast2DFixedSenderConfig{sender}, cfg);

    EXPECT_FALSE(mc.sender_in_rect());
    EXPECT_TRUE(mc.has_receivers());
    EXPECT_EQ(mc.ack_count(), 16u);  // no sender to exclude => fan-out == area
    EXPECT_EQ(mc.num_receivers(sender), 16u);
    EXPECT_EQ(mc.num_receivers(CoreCoord(2, 3)), 0u);
    EXPECT_EQ(mc.compile_time_args(), (std::vector<uint32_t>{1, 1, 0, 1, 16, 1, 0}));

    // participating set = rect (16) ∪ {sender} (1) = 17 cores.
    const auto sems = mc.owned_semaphores();
    ASSERT_EQ(sems.size(), 2u);
    EXPECT_EQ(sems[0].core_ranges.num_cores(), 17u);
    EXPECT_TRUE(sems[0].core_ranges.contains(sender));
    EXPECT_TRUE(sems[0].core_ranges.contains(CoreCoord(1, 1)));
    EXPECT_TRUE(sems[0].core_ranges.contains(CoreCoord(4, 4)));

    // sender -> the 4x4 rect corners; receivers -> the (external) sender's coords.
    EXPECT_TRUE(mc.is_sender(sender));
    EXPECT_EQ(topology_only(mc.runtime_args(sender)), expected_rect2d(dev, 1, 1, 4, 4, /*noc1=*/false));
    const auto s = virt(0, 0);
    EXPECT_EQ(topology_only(mc.runtime_args(CoreCoord(2, 3))), (std::vector<uint32_t>{s.first, s.second, 0, 0}));
}

// Rotating over a rect: every core takes a sender turn (area rounds); RT = 4 + 2*area, coords
// row-major (y outer, x inner). McastArgs<CT_BASE, RT_BASE> reads the area from rotating_span.
TEST_F(McastHostFixture, Mcast2DRotating) {
    auto* dev = device_;
    const auto rect = make_grid(/*gc=*/4, /*gr=*/4);  // 16 cores
    auto virt = [&](uint32_t lx, uint32_t ly) {
        const auto w = dev->worker_core_from_logical_core(CoreCoord(lx, ly));
        return std::pair<uint32_t, uint32_t>{static_cast<uint32_t>(w.x), static_cast<uint32_t>(w.y)};
    };

    McastConfig cfg;
    Mcast2D mc(dev, rect, Mcast2DRotatingSenderConfig{}, cfg);

    EXPECT_TRUE(mc.has_receivers());
    EXPECT_EQ(mc.compile_time_args()[6], 16u);
    EXPECT_EQ(mc.ack_count(), 15u);  // each round reaches the other 15 (sender_in_rect => area-1)
    EXPECT_EQ(mc.compile_time_args(), (std::vector<uint32_t>{1, 1, 0, 1, 15, 1, 16}));

    // Every core in the rect emits the SAME line-uniform block (rect + row-major coords).
    std::vector<uint32_t> expected = expected_rect2d(dev, 0, 0, 3, 3, /*noc1=*/false);
    std::vector<std::pair<uint32_t, uint32_t>> order;  // row-major: y outer, x inner.
    for (uint32_t y = 0; y < 4; ++y) {
        for (uint32_t x = 0; x < 4; ++x) {
            order.push_back(virt(x, y));
        }
    }
    for (const auto& c : order) {
        expected.push_back(c.first);
        expected.push_back(c.second);
    }
    for (auto core : {CoreCoord(0, 0), CoreCoord(2, 1), CoreCoord(3, 3)}) {
        EXPECT_TRUE(mc.is_sender(core));
        EXPECT_EQ(topology_only(mc.runtime_args(core)), expected) << "core (" << core.x << "," << core.y << ")";
    }
    // Spot-check wire semantics: block[0..3] = rect, then coords[round] at 4 + 2*round.
    const auto rt = topology_only(mc.runtime_args(CoreCoord(0, 0)));
    ASSERT_EQ(rt.size(), 4u + 2u * 16u);
    for (uint32_t round = 0; round < 16; ++round) {
        EXPECT_EQ(rt[4 + 2 * round + 0], order[round].first) << "round " << round << " sender_x";
        EXPECT_EQ(rt[4 + 2 * round + 1], order[round].second) << "round " << round << " sender_y";
    }
}

TEST_F(McastHostFixture, Mcast1DRotatingSendersIndependentOfReceiverLines) {
    auto* dev = device_;
    const auto receivers = make_grid(CoreCoord(1, 1), /*gc=*/2, /*gr=*/2);
    const CoreRangeSet sender_grid(std::vector<CoreRange>{
        CoreRange(CoreCoord(1, 1), CoreCoord(1, 2)),
        CoreRange(CoreCoord(3, 1), CoreCoord(3, 2)),
    });
    const std::vector<CoreCoord> first_row_senders = {CoreCoord(1, 1), CoreCoord(3, 1)};

    for (const auto noc : {NOC::NOC_0, NOC::NOC_1}) {
        McastConfig cfg;
        cfg.noc = noc;
        Mcast1D mc(dev, receivers, Mcast1DShape::PerRow, Mcast1DRotatingSenderConfig{sender_grid}, cfg);

        EXPECT_TRUE(mc.has_receivers());
        EXPECT_EQ(mc.compile_time_args()[6], 2u);
        EXPECT_EQ(mc.ack_count(), ACK_EQUALS_FANOUT);
        EXPECT_EQ(mc.compile_time_args(), (std::vector<uint32_t>{1, 1, 0, 1, ACK_EQUALS_FANOUT, 1, 2}));
        EXPECT_TRUE(mc.is_sender(CoreCoord(1, 1)));
        EXPECT_TRUE(mc.is_sender(CoreCoord(3, 1)));
        EXPECT_FALSE(mc.is_sender(CoreCoord(2, 1)));
        EXPECT_EQ(mc.num_receivers(CoreCoord(1, 1)), 1u);
        EXPECT_EQ(mc.num_receivers(CoreCoord(3, 1)), 2u);

        std::vector<uint32_t> expected = expected_bbox(dev, {CoreCoord(1, 1), CoreCoord(2, 1)}, noc);
        for (const auto& sender : first_row_senders) {
            const auto w = dev->worker_core_from_logical_core(sender);
            expected.push_back(static_cast<uint32_t>(w.x));
            expected.push_back(static_cast<uint32_t>(w.y));
        }
        EXPECT_EQ(topology_only(mc.runtime_args(CoreCoord(1, 1))), expected);
        EXPECT_EQ(topology_only(mc.runtime_args(CoreCoord(3, 1))), expected);

        const auto semaphores = mc.owned_semaphores();
        ASSERT_EQ(semaphores.size(), 2u);
        EXPECT_EQ(semaphores[0].core_ranges.num_cores(), 6u);
        EXPECT_EQ(semaphores[1].core_ranges.num_cores(), 6u);
    }
}

TEST_F(McastHostFixture, Mcast1DRotatingSenderAndReceiverGrids) {
    auto* dev = device_;
    const auto row_receivers = make_grid(CoreCoord(1, 1), /*gc=*/2, /*gr=*/2);
    const auto row_senders = make_grid(CoreCoord(1, 1), /*gc=*/3, /*gr=*/2);
    const auto column_receivers = make_grid(CoreCoord(1, 1), /*gc=*/2, /*gr=*/2);
    const auto column_senders = make_grid(CoreCoord(1, 1), /*gc=*/2, /*gr=*/3);

    for (const auto noc : {NOC::NOC_0, NOC::NOC_1}) {
        McastConfig cfg;
        cfg.noc = noc;

        const Mcast1D row_from_grids(
            dev, row_receivers, Mcast1DShape::PerRow, Mcast1DRotatingSenderConfig{row_senders}, cfg);
        EXPECT_EQ(row_from_grids.num_senders(), 3u);
        EXPECT_EQ(role_only(row_from_grids.runtime_args(CoreCoord(3, 1))), (std::vector<uint32_t>{0x1u, 2u}));

        const Mcast1D column_from_grids(
            dev, column_receivers, Mcast1DShape::PerColumn, Mcast1DRotatingSenderConfig{column_senders}, cfg);
        EXPECT_EQ(column_from_grids.num_senders(), 3u);
        EXPECT_EQ(role_only(column_from_grids.runtime_args(CoreCoord(1, 3))), (std::vector<uint32_t>{0x1u, 2u}));
    }
}

TEST_F(McastHostFixture, Mcast1DRotatingSenderGridMustAlignWithReceiverLines) {
    auto* dev = device_;
    const auto receivers = make_grid(CoreCoord(1, 1), /*gc=*/2, /*gr=*/2);
    McastConfig cfg;

    const auto sender_on_unmatched_row = make_grid(CoreCoord(1, 3), /*gc=*/2, /*gr=*/1);
    EXPECT_ANY_THROW({
        Mcast1D mc(dev, receivers, Mcast1DShape::PerRow, Mcast1DRotatingSenderConfig{sender_on_unmatched_row}, cfg);
    });

    const CoreRangeSet unequal_sender_lines(std::vector<CoreRange>{
        CoreRange(CoreCoord(1, 1), CoreCoord(2, 1)),
        CoreRange(CoreCoord(1, 2), CoreCoord(1, 2)),
    });
    EXPECT_ANY_THROW(
        { Mcast1D mc(dev, receivers, Mcast1DShape::PerRow, Mcast1DRotatingSenderConfig{unequal_sender_lines}, cfg); });
}

TEST_F(McastHostFixture, Mcast2DRotatingSendersIndependentOfReceiverRect) {
    auto* dev = device_;
    const auto receivers = make_grid(CoreCoord(1, 1), /*gc=*/2, /*gr=*/2);
    const std::vector<CoreCoord> senders = {CoreCoord(1, 1), CoreCoord(3, 1)};
    const CoreRangeSet sender_grid(std::vector<CoreRange>{
        CoreRange(senders[0], senders[0]),
        CoreRange(senders[1], senders[1]),
    });

    McastConfig cfg;
    Mcast2D mc(dev, receivers, Mcast2DRotatingSenderConfig{sender_grid}, cfg);

    EXPECT_TRUE(mc.has_receivers());
    EXPECT_EQ(mc.compile_time_args()[6], 2u);
    EXPECT_EQ(mc.ack_count(), ACK_EQUALS_FANOUT);
    EXPECT_EQ(mc.compile_time_args(), (std::vector<uint32_t>{1, 1, 0, 1, ACK_EQUALS_FANOUT, 1, 2}));
    EXPECT_TRUE(mc.is_sender(CoreCoord(1, 1)));
    EXPECT_TRUE(mc.is_sender(CoreCoord(3, 1)));
    EXPECT_FALSE(mc.is_sender(CoreCoord(2, 1)));
    EXPECT_EQ(mc.num_receivers(CoreCoord(1, 1)), 3u);
    EXPECT_EQ(mc.num_receivers(CoreCoord(3, 1)), 4u);

    std::vector<uint32_t> expected = expected_rect2d(dev, 1, 1, 2, 2, /*noc1=*/false);
    for (const auto& sender : senders) {
        const auto w = dev->worker_core_from_logical_core(sender);
        expected.push_back(static_cast<uint32_t>(w.x));
        expected.push_back(static_cast<uint32_t>(w.y));
    }
    EXPECT_EQ(topology_only(mc.runtime_args(CoreCoord(1, 1))), expected);
    EXPECT_EQ(topology_only(mc.runtime_args(CoreCoord(3, 1))), expected);

    const auto semaphores = mc.owned_semaphores();
    ASSERT_EQ(semaphores.size(), 2u);
    EXPECT_EQ(semaphores[0].core_ranges.num_cores(), 5u);
    EXPECT_EQ(semaphores[1].core_ranges.num_cores(), 5u);
}

TEST_F(McastHostFixture, Mcast2DRotatingSenderOrder) {
    auto* dev = device_;
    const auto receivers = make_grid(CoreCoord(1, 1), /*gc=*/2, /*gr=*/2);
    const auto senders = make_grid(CoreCoord(1, 1), /*gc=*/2, /*gr=*/2);

    McastConfig cfg;
    const Mcast2D row_major(
        dev,
        receivers,
        Mcast2DRotatingSenderConfig{.sender_grid = senders, .sender_order = Mcast2DSenderOrder::RowMajor},
        cfg);
    const Mcast2D column_major(
        dev,
        receivers,
        Mcast2DRotatingSenderConfig{.sender_grid = senders, .sender_order = Mcast2DSenderOrder::ColumnMajor},
        cfg);

    EXPECT_EQ(role_only(row_major.runtime_args(CoreCoord(2, 1))), (std::vector<uint32_t>{0x3u, 1u}));
    EXPECT_EQ(role_only(row_major.runtime_args(CoreCoord(1, 2))), (std::vector<uint32_t>{0x3u, 2u}));
    EXPECT_EQ(role_only(column_major.runtime_args(CoreCoord(1, 2))), (std::vector<uint32_t>{0x3u, 1u}));
    EXPECT_EQ(role_only(column_major.runtime_args(CoreCoord(2, 1))), (std::vector<uint32_t>{0x3u, 2u}));

    const auto row_major_topology = topology_only(row_major.runtime_args(CoreCoord(1, 1)));
    const auto column_major_topology = topology_only(column_major.runtime_args(CoreCoord(1, 1)));
    ASSERT_EQ(row_major_topology.size(), 12u);
    ASSERT_EQ(column_major_topology.size(), 12u);
    EXPECT_EQ(
        std::vector<uint32_t>(row_major_topology.begin(), row_major_topology.begin() + 4),
        std::vector<uint32_t>(column_major_topology.begin(), column_major_topology.begin() + 4));
    EXPECT_NE(row_major_topology, column_major_topology);
}

// NoC1: the sender's rect corners swap to high-corner-start (matches the kernel's per-NoC ordering).
TEST_F(McastHostFixture, Mcast2DNoc1) {
    auto* dev = device_;
    const auto rect = make_grid(/*gc=*/8, /*gr=*/8);
    McastConfig cfg;
    cfg.noc = NOC::NOC_1;
    Mcast2D mc(dev, rect, Mcast2DFixedSenderConfig{CoreCoord(0, 0)}, cfg);

    EXPECT_EQ(topology_only(mc.runtime_args(CoreCoord(0, 0))), expected_rect2d(dev, 0, 0, 7, 7, /*noc1=*/true));
}

// Degenerate single-core rect: no receivers, ack_count 0. The RT is still the sender's
// own 1x1 rect (NOT a zero rect), so the kernel's SenderPipe sees area==1 && in_rect_ and collapses
// to a local copy.
TEST_F(McastHostFixture, Mcast2DDegenerate) {
    auto* dev = device_;
    const auto rect = CoreRangeSet(CoreRange(CoreCoord(2, 2), CoreCoord(2, 2)));
    McastConfig cfg;
    Mcast2D mc(dev, rect, Mcast2DFixedSenderConfig{CoreCoord(2, 2)}, cfg);

    EXPECT_FALSE(mc.has_receivers());
    EXPECT_EQ(mc.ack_count(), 0u);
    EXPECT_EQ(mc.compile_time_args(), (std::vector<uint32_t>{1, 0, 0, 1, 0, 1, 0}));
    EXPECT_TRUE(mc.is_sender(CoreCoord(2, 2)));
    const auto w = dev->worker_core_from_logical_core(CoreCoord(2, 2));
    const auto vx = static_cast<uint32_t>(w.x);
    const auto vy = static_cast<uint32_t>(w.y);
    EXPECT_EQ(topology_only(mc.runtime_args(CoreCoord(2, 2))), (std::vector<uint32_t>{vx, vy, vx, vy}));
    EXPECT_EQ(mc.num_receivers(CoreCoord(2, 2)), 0u);
}

// ack_count greater than the fan-out is a usage error -> ctor fatal.
TEST_F(McastHostFixture, Mcast2DAckCountOverrideTooLargeFatal) {
    auto* dev = device_;
    const auto rect = make_grid(/*gc=*/4, /*gr=*/4);  // fan-out area-1 = 15
    McastConfig cfg;
    cfg.ack_count_override = 16;
    EXPECT_ANY_THROW({ Mcast2D mc(dev, rect, Mcast2DFixedSenderConfig{CoreCoord(0, 0)}, cfg); });
}

TEST_F(McastHostFixture, Mcast1DAckCountOverrideTooLargeFatal) {
    auto* dev = device_;
    const auto grid = make_grid(/*gc=*/4, /*gr=*/4);  // per-row fan-out = 3
    McastConfig cfg;
    cfg.ack_count_override = 4;
    EXPECT_ANY_THROW({ Mcast1D mc(dev, grid, Mcast1DShape::PerRow, Mcast1DFixedSenderConfig{}, cfg); });
}

// The `flags` CT word (5th) encodes pre_handshake (bit0 = cfg.handshake) and the data-ready signal
// (bit1 = cfg.data_ready == Counter) independently. handshake off also drops consumer_ready to
// UNUSED_SEM_ID. This is what lets the kernel's sender()/receiver() take no behaviour knobs.
TEST_F(McastHostFixture, FlagsWordEncoding) {
    auto* dev = device_;
    const auto grid = make_grid(/*gc=*/8, /*gr=*/8);
    constexpr uint32_t UNUSED = 0xFFFFFFFFu;

    // handshake off + Flag => flags 0; consumer_ready UNUSED; ack_count still = the fan-out (7).
    {
        McastConfig cfg;
        cfg.handshake = false;
        Mcast1D mc(dev, grid, Mcast1DShape::PerRow, Mcast1DFixedSenderConfig{}, cfg);
        EXPECT_EQ(mc.compile_time_args(), (std::vector<uint32_t>{1, 1, 0, UNUSED, 7, 0, 0}));
    }
    // handshake on + Counter => flags 3 (bit0 pre_handshake | bit1 signal).
    {
        McastConfig cfg;
        cfg.data_ready = DataReadyMode::Counter;
        Mcast1D mc(dev, grid, Mcast1DShape::PerRow, Mcast1DFixedSenderConfig{}, cfg);
        EXPECT_EQ(mc.compile_time_args(), (std::vector<uint32_t>{1, 1, 0, 1, 7, 3, 0}));
    }
    // handshake off + Counter => flags 2 (bit1 only); Mcast2D dense fan-out 63, consumer_ready UNUSED.
    {
        McastConfig cfg;
        cfg.handshake = false;
        cfg.data_ready = DataReadyMode::Counter;
        Mcast2D mc(dev, grid, Mcast2DFixedSenderConfig{CoreCoord(0, 0)}, cfg);
        EXPECT_EQ(mc.compile_time_args(), (std::vector<uint32_t>{1, 1, 0, UNUSED, 63, 2, 0}));
    }
}

// The per-kernel pre_handshake override: ONE mcast object emits different pre_handshake per kernel (the
// layernorm / split-count idiom), leaving the sems + geometry (and ack_count) untouched. This is what
// lets a divergent-ack family ride ONE object — the sender + acking receivers take the default, the
// non-acking receivers pass pre_handshake=false — instead of a second mcast object.
TEST_F(McastHostFixture, PreHandshakeOverride) {
    auto* dev = device_;
    const auto grid = make_grid(/*gc=*/8, /*gr=*/8);

    // handshake=True object: default emission has pre_handshake (flags bit0) set.
    McastConfig cfg;  // handshake=true, Flag
    cfg.ack_count_override = 2;
    Mcast2D mc(dev, grid, Mcast2DFixedSenderConfig{CoreCoord(0, 0)}, cfg);
    EXPECT_EQ(mc.compile_time_args(), (std::vector<uint32_t>{1, 1, 0, 1, 2, 1, 0}));  // default -> bit0 set
    EXPECT_EQ(mc.compile_time_args(/*pre_handshake=*/true), (std::vector<uint32_t>{1, 1, 0, 1, 2, 1, 0}));
    // Override clears bit0 for a non-acking receiver kernel — same ids, same ack_count, same geometry.
    EXPECT_EQ(mc.compile_time_args(/*pre_handshake=*/false), (std::vector<uint32_t>{1, 1, 0, 1, 2, 0, 0}));

    // Symmetric for Mcast1D, and it composes with the Counter signal bit (bit1 stays put).
    McastConfig cfg1;
    cfg1.data_ready = DataReadyMode::Counter;
    Mcast1D mc1(dev, grid, Mcast1DShape::PerRow, Mcast1DFixedSenderConfig{}, cfg1);
    EXPECT_EQ(mc1.compile_time_args(), (std::vector<uint32_t>{1, 1, 0, 1, 7, 3, 0}));  // bit0|bit1, fixed
    EXPECT_EQ(mc1.compile_time_args(/*pre_handshake=*/false), (std::vector<uint32_t>{1, 1, 0, 1, 7, 2, 0}));
}

}  // namespace ttnn::kernel_lib::host::test
