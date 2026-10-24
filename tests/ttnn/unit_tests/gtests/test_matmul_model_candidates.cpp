// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Device-free tests of the matmul config model's candidate enumerator (matmul_candidates.hpp).

#include <gtest/gtest.h>

#include <cstdint>
#include <tuple>
#include <vector>

#include "ttnn/operations/matmul/device/config/model/matmul_candidates.hpp"

namespace ttnn::operations::matmul::model {
namespace {
namespace CMAKE_UNIQUE_NAMESPACE {

constexpr std::uint32_t kBudget = 1461120 - 16 * 1024;  // a free Blackhole p150 L1 bank less the headroom

// (family, in0_block_w, out_subblock_h, out_subblock_w, per_core_M, per_core_N, out_block_h, out_block_w, fuse_batch)
using Row = std::tuple<Family, int, int, int, int, int, int, int, int>;

std::vector<Row> rows(const std::vector<Candidate>& cs) {
    std::vector<Row> out;
    for (const auto& c : cs) {
        const auto& b = c.blocking;
        const bool reuse = c.family == Family::kReuse;
        out.emplace_back(
            c.family,
            b.in0_block_w,
            b.out_subblock_h,
            b.out_subblock_w,
            b.per_core_M,
            b.per_core_N,
            reuse ? 0 : b.out_block_h,
            reuse ? 0 : b.out_block_w,
            reuse ? 0 : c.fuse_batch);
    }
    return out;
}

MatmulDesc desc(
    std::uint32_t Mt, std::uint32_t Kt, std::uint32_t Nt, std::uint32_t batch_a = 1, std::uint32_t batch_b = 1) {
    MatmulDesc p;
    p.Mt = Mt;
    p.Kt = Kt;
    p.Nt = Nt;
    p.batch_a = batch_a;
    p.batch_b = batch_b;
    p.rank_a = p.rank_b = 4;
    return p;
}

const auto k2D = Family::k2D;
const auto kIn0 = Family::k1DMcastIn0;
const auto kIn1 = Family::k1DMcastIn1;
const auto kReuse = Family::kReuse;

// Expected lists: tt-matmul-predictor bench/unified/ryan.py proposals() (the Python port of the same EnumeratingSource,
// itself checked against the C++ selector's recorded candidates), Blackhole 11x10 grid, everything in DRAM.

TEST(MatmulModelCandidates, SmallMatchesReference) {  // 64x256 @ 256x128 bf16
    const auto got = rows(enumerate_candidates(desc(2, 8, 4), HardwareDesc::blackhole(11, 10, kBudget)));
    const std::vector<Row> want = {
        {k2D, 8, 1, 1, 1, 1, 1, 1, 1},  {k2D, 1, 1, 1, 1, 1, 1, 1, 1},  {k2D, 2, 1, 1, 1, 1, 1, 1, 1},
        {k2D, 4, 1, 1, 1, 1, 1, 1, 1},  {kIn0, 1, 2, 1, 2, 1, 2, 1, 1}, {kIn0, 1, 1, 1, 2, 1, 2, 1, 1},
        {kIn0, 2, 2, 1, 2, 1, 2, 1, 1}, {kIn0, 2, 1, 1, 2, 1, 2, 1, 1}, {kIn0, 4, 2, 1, 2, 1, 2, 1, 1},
        {kIn0, 4, 1, 1, 2, 1, 2, 1, 1}, {kIn0, 8, 2, 1, 2, 1, 2, 1, 1}, {kIn0, 8, 1, 1, 2, 1, 2, 1, 1},
        {kIn1, 1, 1, 4, 1, 4, 1, 4, 1}, {kIn1, 1, 1, 1, 1, 4, 1, 4, 1}, {kIn1, 2, 1, 4, 1, 4, 1, 4, 1},
        {kIn1, 2, 1, 1, 1, 4, 1, 4, 1}, {kIn1, 4, 1, 4, 1, 4, 1, 4, 1}, {kIn1, 4, 1, 1, 1, 4, 1, 4, 1},
        {kIn1, 8, 1, 4, 1, 4, 1, 4, 1}, {kIn1, 8, 1, 1, 1, 4, 1, 4, 1},
    };
    EXPECT_EQ(got, want);
}

TEST(MatmulModelCandidates, BatchedBMatchesReference) {  // 8x128x64 @ 8x64x128 bf16
    const auto got = rows(enumerate_candidates(desc(4, 2, 4, 8, 8), HardwareDesc::blackhole(11, 10, kBudget)));
    const std::vector<Row> want = {
        {kReuse, 2, 1, 4, 1, 4, 0, 0, 0}, {kReuse, 1, 1, 4, 1, 4, 0, 0, 0}, {kReuse, 1, 1, 1, 1, 4, 0, 0, 0},
        {kReuse, 2, 1, 1, 1, 4, 0, 0, 0}, {kReuse, 1, 2, 4, 4, 4, 0, 0, 0}, {kReuse, 1, 1, 4, 4, 4, 0, 0, 0},
        {kReuse, 1, 4, 1, 4, 4, 0, 0, 0}, {kReuse, 1, 4, 2, 4, 4, 0, 0, 0}, {kReuse, 2, 2, 4, 4, 4, 0, 0, 0},
        {kReuse, 2, 1, 4, 4, 4, 0, 0, 0}, {kReuse, 2, 4, 1, 4, 4, 0, 0, 0}, {kReuse, 2, 4, 2, 4, 4, 0, 0, 0},
        {kReuse, 1, 2, 4, 2, 4, 0, 0, 0}, {kReuse, 1, 1, 4, 2, 4, 0, 0, 0}, {kReuse, 1, 2, 1, 2, 4, 0, 0, 0},
        {kReuse, 2, 2, 4, 2, 4, 0, 0, 0}, {kReuse, 2, 1, 4, 2, 4, 0, 0, 0}, {kReuse, 2, 2, 1, 2, 4, 0, 0, 0},
        {k2D, 1, 1, 1, 1, 1, 1, 1, 0},    {k2D, 2, 1, 1, 1, 1, 1, 1, 0},    {kIn0, 1, 4, 1, 4, 1, 4, 1, 0},
        {kIn0, 1, 1, 1, 4, 1, 4, 1, 0},   {kIn0, 2, 4, 1, 4, 1, 4, 1, 0},   {kIn0, 2, 1, 1, 4, 1, 4, 1, 0},
        {kIn1, 1, 1, 4, 1, 4, 1, 4, 0},   {kIn1, 1, 1, 1, 1, 4, 1, 4, 0},   {kIn1, 2, 1, 4, 1, 4, 1, 4, 0},
        {kIn1, 2, 1, 1, 1, 4, 1, 4, 0},
    };
    EXPECT_EQ(got, want);
}

TEST(MatmulModelCandidates, BroadcastAMatchesReference) {  // 1x1x32x64 @ 1x4x64x64 bf16: only 1D in1-mcast
    const auto got = rows(enumerate_candidates(desc(1, 2, 2, 1, 4), HardwareDesc::blackhole(11, 10, kBudget)));
    const std::vector<Row> want = {
        {kIn1, 2, 1, 2, 1, 2, 1, 2, 0},
        {kIn1, 1, 1, 2, 1, 2, 1, 2, 0},
        {kIn1, 1, 1, 1, 1, 2, 1, 2, 0},
        {kIn1, 2, 1, 1, 1, 2, 1, 2, 0},
    };
    EXPECT_EQ(got, want);
}

TEST(MatmulModelCandidates, PrefillHeuristicFirstAndAllFit) {  // 1024x4096 @ 4096x4096, bfp8 B
    MatmulDesc p = desc(32, 128, 128);
    p.in1 = DataType::kBfloat8B;
    const auto hw = HardwareDesc::blackhole(11, 10, kBudget);
    const auto cs = enumerate_candidates(p, hw);
    ASSERT_EQ(cs.size(), 78u);
    EXPECT_EQ(rows({cs.front()}).front(), (Row{k2D, 16, 2, 4, 4, 12, 4, 12, 1}));
    for (const auto& c : cs) {
        EXPECT_LE(circular_buffer_bytes(p, hw, c), hw.l1_cb_budget);
        EXPECT_LE(c.blocking.out_subblock_h * c.blocking.out_subblock_w, 8u);
        EXPECT_EQ(c.grid_x, 11u);
        EXPECT_EQ(c.grid_y, 10u);
    }
}

TEST(MatmulModelCandidates, GridComesFromTheDevice) {
    for (const auto& c : enumerate_candidates(desc(32, 128, 128), HardwareDesc::blackhole(8, 8, kBudget))) {
        EXPECT_EQ(c.grid_x, 8u);
        EXPECT_EQ(c.grid_y, 8u);
    }
}

TEST(MatmulModelCandidates, TighterBudgetOnlyRemoves) {
    const MatmulDesc p = desc(32, 128, 128);
    const auto full = enumerate_candidates(p, HardwareDesc::blackhole(11, 10, kBudget));
    const auto hw = HardwareDesc::blackhole(11, 10, 200 * 1024);
    const auto tight = enumerate_candidates(p, hw);
    EXPECT_LT(tight.size(), full.size());
    for (const auto& c : tight) {
        EXPECT_LE(circular_buffer_bytes(p, hw, c), hw.l1_cb_budget);
    }
    EXPECT_TRUE(enumerate_candidates(p, HardwareDesc::blackhole(11, 10, 0)).empty());
}

TEST(MatmulModelCandidates, HugeKNeverWrapsIntoTheBudget) {  // 32x33554432 @ 33554432x32 bf16, Kt = 2^20
    // Multi-GB K blocks must not wrap around 32 bits into a size that fits L1
    MatmulDesc p = desc(1, 1u << 20, 1);
    p.out = DataType::kBfloat8B;
    p.packer_l1_acc = false;
    const auto hw = HardwareDesc::blackhole(11, 10, kBudget);
    const auto cs = enumerate_candidates(p, hw);
    EXPECT_FALSE(cs.empty());
    for (const auto& c : cs) {
        // A K block of in0 alone is in0_block_w bf16 tiles
        EXPECT_LE(std::uint64_t{c.blocking.in0_block_w} * 2048, kBudget) << "in0_block_w " << c.blocking.in0_block_w;
        EXPECT_LE(circular_buffer_bytes(p, hw, c), kBudget);
    }
}

TEST(MatmulModelCandidates, NarrowerGridFollowsTheDeviceGrid) {  // #62076: 2048x4096 @ 4096x14336 on 11x10
    const MatmulDesc p = desc(64, 128, 448);
    const auto hw = HardwareDesc::blackhole(11, 10, kBudget);
    const auto full = enumerate_candidates(p, hw);
    const auto both = enumerate_candidates_with_narrower_grid(p, hw);
    ASSERT_GT(both.size(), full.size());
    for (std::size_t i = 0; i < both.size(); ++i) {
        EXPECT_EQ(both[i].grid_x, i < full.size() ? 11u : 10u) << "candidate " << i;
        EXPECT_EQ(both[i].grid_y, 10u) << "candidate " << i;
    }
}

TEST(MatmulModelCandidates, Unsupported) {
    MatmulDesc p = desc(4, 4, 4);
    p.a.sharded = true;
    EXPECT_NE(unsupported_reason(p), "");
    EXPECT_TRUE(enumerate_candidates(p, HardwareDesc::blackhole(11, 10, kBudget)).empty());
    EXPECT_NE(unsupported_reason(desc(4, 4, 4, 2, 3)), "");  // mismatched batches
    EXPECT_NE(unsupported_reason(desc(0, 4, 4)), "");
    EXPECT_EQ(unsupported_reason(desc(4, 4, 4)), "");
}

}  // namespace CMAKE_UNIQUE_NAMESPACE
}  // namespace
}  // namespace ttnn::operations::matmul::model
