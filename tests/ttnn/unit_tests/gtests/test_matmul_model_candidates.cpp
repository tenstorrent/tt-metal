// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Device-free tests of the matmul config model's candidate enumerator (matmul_candidates.hpp).

#include <gtest/gtest.h>

#include <cstdint>
#include <string>
#include <tuple>
#include <utility>
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

// Expected lists: the candidates the #58143 selector's enumeration gives for these matmuls, Blackhole 11x10 grid,
// everything in DRAM.

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

// The device op's checks (matmul_device_operation.cpp) of a 2D, 1D or Reuse config on interleaved tensors with 32x32
// tiles: the first one `c` fails, or empty
std::string device_op_rule_error(const MatmulDesc& p, std::uint32_t grid_x, std::uint32_t grid_y, const Candidate& c) {
    const Blocking& b = c.blocking;
    if (b.in0_block_w == 0 || b.per_core_M == 0 || b.per_core_N == 0 || b.out_subblock_h == 0 ||
        b.out_subblock_w == 0) {
        return "a zero block dimension";
    }
    if (c.grid_x == 0 || c.grid_y == 0 || c.grid_x > grid_x || c.grid_y > grid_y) {
        return "grid outside the device grid";
    }
    if (p.Kt % b.in0_block_w != 0) {
        return "Kt not divisible by in0_block_w";
    }
    // Half-sync DST even with dst_full_sync (these factories): 8 tiles, 4 with fp32 accumulation
    if (b.out_subblock_h * b.out_subblock_w > (p.fp32_dest_acc ? 4u : 8u)) {
        return "subblock over the DST tiles";
    }
    if (c.family == Family::kReuse) {
        if (b.per_core_M % b.out_subblock_h != 0 || b.per_core_N % b.out_subblock_w != 0) {
            return "per-core block not divisible by the subblock";
        }
        // validate_matmul_reuse_config and validate_matmul_reuse_work_split
        if (b.per_core_M > p.Mt ? (b.per_core_M % p.Mt != 0 || (p.batch_a * p.Mt) % b.per_core_M != 0)
                                : p.Mt % b.per_core_M != 0) {
            return "Reuse per_core_M neither divides M nor is a multiple of it dividing the total M";
        }
        if (b.per_core_N != p.Nt) {
            return "Reuse per_core_N not N";
        }
        if (p.bias_rows > 0) {
            return "Reuse with a fused row bias (the hook drops these)";
        }
        return "";
    }
    if (b.out_block_h == 0 || b.out_block_w == 0 || b.out_block_h % b.out_subblock_h != 0 ||
        b.out_block_w % b.out_subblock_w != 0 || b.per_core_M % b.out_block_h != 0 ||
        b.per_core_N % b.out_block_w != 0) {
        return "block, output block and subblock not nested";
    }
    const std::uint32_t Mt = c.fuse_batch ? p.batch_a * p.Mt : p.Mt;
    const std::uint32_t blocks_y = (Mt - 1) / b.per_core_M + 1;
    const std::uint32_t blocks_x = (p.Nt - 1) / b.per_core_N + 1;
    if (c.family == Family::k2D) {
        if (b.per_core_M > Mt || blocks_x > c.grid_x || blocks_y > c.grid_y) {
            return "2D output blocks outside the grid";
        }
        return "";
    }
    if (blocks_x * blocks_y > c.grid_x * c.grid_y) {
        return "1D output blocks over the cores";
    }
    if (c.family == Family::k1DMcastIn0) {
        return blocks_y == 1 ? "" : "mcast_in0 with more than one M block";
    }
    if (blocks_x != 1 || b.per_core_M > Mt) {
        return "mcast_in1 with more than one N block, or per_core_M over Mt";
    }
    if ((p.Nt - 1) / b.out_block_w + 1 != b.per_core_N / b.out_block_w) {
        return "mcast_in1 N tail not in the last output block";
    }
    if (blocks_y == 1) {
        const std::uint32_t physical_h = b.per_core_M / b.out_block_h;
        if ((Mt - 1) / b.out_block_h + 1 != physical_h || (Mt % b.out_block_h != 0 && physical_h != 1)) {
            return "single-row mcast_in1 M tail not in the last output block";
        }
    }
    return "";
}

// The selector validates only its top candidate with the device op (a rejection is a TT_FATAL, matmul_model_hook.cpp),
// so every candidate must pass the device op's checks. Shapes with prime, odd and tail-heavy tile counts on the p150
// and Galaxy grids (each with the narrower grid the selector also scores), under each call attribute the enumerator
// reads: batches (matched and broadcast B), dtypes, accumulation, full-sync DST, fused bias, transposed A, L1
// placement and a tight L1 budget.
TEST(MatmulModelCandidates, AllPassTheDeviceOpRules) {
    struct Variant {
        const char* name;
        void (*apply)(MatmulDesc&);
    };
    const std::vector<Variant> variants = {
        {"bf16 x bfp8", [](MatmulDesc& p) { p.in1 = DataType::kBfloat8B; }},
        {"fp32 acc", [](MatmulDesc& p) { p.fp32_dest_acc = true; }},
        {"full-sync DST", [](MatmulDesc& p) { p.dst_full_sync = true; }},
        {"full-sync DST, fp32 acc", [](MatmulDesc& p) { p.dst_full_sync = p.fp32_dest_acc = true; }},
        {"bias",
         [](MatmulDesc& p) {
             p.bias_tile_bytes = 2048;
             p.bias_rows = 1;
         }},
        {"batch 4", [](MatmulDesc& p) { p.batch_a = p.batch_b = 4; }},
        {"batch 4, B broadcast", [](MatmulDesc& p) { p.batch_a = 4; }},
        {"transposed A", [](MatmulDesc& p) { p.transpose_a = true; }},
        {"bfp8 out, no packer L1 acc",
         [](MatmulDesc& p) {
             p.out = DataType::kBfloat8B;
             p.packer_l1_acc = false;
         }},
        {"bfp4 B, fp32 inputs",
         [](MatmulDesc& p) {
             p.in0 = p.out = DataType::kFloat32;
             p.in1 = DataType::kBfloat4B;
         }},
        {"L1 A, B and output", [](MatmulDesc& p) { p.a.in_l1 = p.b.in_l1 = p.out_place.in_l1 = true; }},
    };
    std::size_t checked = 0;
    for (const auto& [grid_x, grid_y] : {std::pair{11u, 10u}, std::pair{12u, 10u}}) {
        for (const std::uint32_t budget : {kBudget, 300u * 1024}) {
            const auto hw = HardwareDesc::blackhole(grid_x, grid_y, budget);
            for (std::uint32_t Mt : {1u, 3u, 7u, 13u, 32u, 41u, 100u, 401u, 4008u}) {
                for (std::uint32_t Kt : {1u, 6u, 64u, 128u}) {
                    for (std::uint32_t Nt : {1u, 5u, 11u, 41u, 128u, 448u}) {
                        for (const auto& v : variants) {
                            MatmulDesc p = desc(Mt, Kt, Nt);
                            v.apply(p);
                            for (const auto& c : enumerate_candidates_with_narrower_grid(p, hw)) {
                                if (p.bias_rows > 0 && c.family == Family::kReuse) {
                                    continue;  // the hook drops Reuse with a fused bias
                                }
                                ++checked;
                                const auto& b = c.blocking;
                                EXPECT_EQ(device_op_rule_error(p, grid_x, grid_y, c), "")
                                    << Mt << "x" << Kt << "x" << Nt << " tiles, " << v.name << ", budget " << budget
                                    << ", device grid " << grid_x << "x" << grid_y << ": family "
                                    << static_cast<int>(c.family) << " grid " << c.grid_x << "x" << c.grid_y
                                    << " in0_block_w " << b.in0_block_w << " per core " << b.per_core_M << "x"
                                    << b.per_core_N << " block " << b.out_block_h << "x" << b.out_block_w
                                    << " subblock " << b.out_subblock_h << "x" << b.out_subblock_w << " fuse_batch "
                                    << c.fuse_batch;
                            }
                        }
                    }
                }
            }
        }
    }
    EXPECT_GT(checked, 50000u);
}

TEST(MatmulModelCandidates, MakeCandidateMatchesTheEnumerator) {  // the hook scores the default config this way
    const MatmulDesc p = desc(64, 128, 448, 2, 2);
    const auto hw = HardwareDesc::blackhole(11, 10, kBudget);
    for (const auto& c : enumerate_candidates(p, hw)) {
        const Candidate made = make_candidate(p, hw, c.family, c.blocking, c.fuse_batch);
        EXPECT_EQ(made.cores, c.cores);
        EXPECT_EQ(made.fuse_batch, c.fuse_batch);
        EXPECT_EQ(made.grid_x, c.grid_x);
        EXPECT_EQ(made.grid_y, c.grid_y);
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
