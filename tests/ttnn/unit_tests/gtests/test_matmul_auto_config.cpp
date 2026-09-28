// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Device-free checks of the new matmul default-config selector (matmul_auto_config.hpp): every config it
// emits, for Wormhole and Blackhole grids alike, must satisfy the program factories' constraints and fit
// the L1 budget it was given.

#include <gtest/gtest.h>

#include <cstdint>
#include <string>
#include <tuple>
#include <variant>
#include <vector>

#include <fmt/format.h>

#include "ttnn/operations/matmul/device/config/matmul_auto_config.hpp"

namespace {

using namespace ttnn::operations::matmul;
using namespace ttnn::operations::matmul::auto_config;
using Layout = ttnn::operations::matmul::auto_config::Layout;

uint32_t div_up(uint32_t a, uint32_t b) { return (a + b - 1) / b; }

struct Arch {
    std::string name;
    tt::ARCH arch;
    CoreCoord grid;
};

const std::vector<Arch> kArchs = {
    {"wormhole_8x8", tt::ARCH::WORMHOLE_B0, CoreCoord(8, 8)},
    {"blackhole_13x10", tt::ARCH::BLACKHOLE, CoreCoord(13, 10)},
    {"blackhole_12x10", tt::ARCH::BLACKHOLE, CoreCoord(12, 10)},
};
constexpr uint32_t kL1Budget = 1300 * 1024;

Problem make_problem(
    uint32_t batch_a,
    uint32_t batch_b,
    uint32_t M,
    uint32_t K,
    uint32_t N,
    tt::DataFormat in1 = tt::DataFormat::Float16_b,
    bool fp32_acc = false,
    bool bias = false,
    bool transpose_a = false) {
    Problem p;
    p.batch_a = batch_a;
    p.batch_b = batch_b;
    p.Mt = div_up(M, 32);
    p.Kt = div_up(K, 32);
    p.Nt = div_up(N, 32);
    p.in1_format = in1;
    p.fp32_dest_acc_en = fp32_acc;
    p.bias_tile_bytes = bias ? tt::tile_size(tt::DataFormat::Float16_b) : 0;
    p.transpose_a = transpose_a;
    return p;
}

// Returns an empty string if `config` is valid for `p` on `hw`, else a description of the first violation.
std::string check_config(const Problem& p, const HardwareDesc& hw, const MatmulProgramConfig& config) {
    const uint32_t cores = hw.grid.x * hw.grid.y;
    const uint32_t max_area = (p.dst_full_sync_en ? 16 : 8) / (p.fp32_dest_acc_en ? 2 : 1);
    return std::visit(
        [&](const auto& c) -> std::string {
            using T = std::decay_t<decltype(c)>;
            constexpr bool selectable = std::is_same_v<T, MatmulMultiCoreReuseProgramConfig> ||
                                        std::is_same_v<T, MatmulMultiCoreReuseMultiCastProgramConfig> ||
                                        std::is_same_v<T, MatmulMultiCoreReuseMultiCast1DProgramConfig>;
            if constexpr (!selectable) {
                return "unexpected config type";
            } else {
                if (c.in0_block_w == 0 || p.Kt % c.in0_block_w != 0) {
                    return fmt::format("Kt {} % in0_block_w {}", p.Kt, c.in0_block_w);
                }
                if (c.out_subblock_h * c.out_subblock_w > max_area) {
                    return "subblock area";
                }
                if constexpr (std::is_same_v<T, MatmulMultiCoreReuseProgramConfig>) {
                    if (c.per_core_N != p.Nt) {
                        return "reuse per_core_N != Nt";
                    }
                    const bool divides = p.Mt % c.per_core_M == 0;
                    const bool whole_batches = c.per_core_M % p.Mt == 0 && (p.batch_a * p.Mt) % c.per_core_M == 0;
                    if (!divides && !whole_batches) {
                        return "reuse per_core_M";
                    }
                    if (c.per_core_M % c.out_subblock_h != 0 || c.per_core_N % c.out_subblock_w != 0 ||
                        p.Mt % c.out_subblock_h != 0) {
                        return "reuse subblock";
                    }
                    if (p.fp32_dest_acc_en && c.out_subblock_h * c.out_subblock_w > 4) {
                        return "reuse fp32 subblock";
                    }
                    Blocking b{c.per_core_M, c.per_core_N, c.in0_block_w, c.per_core_M, c.per_core_N, 0, 0};
                    if (circular_buffer_bytes(p, hw, Family::Reuse, b) > hw.l1_cb_budget) {
                        return "reuse L1";
                    }
                    return "";
                } else if constexpr (
                    std::is_same_v<T, MatmulMultiCoreReuseMultiCastProgramConfig> ||
                    std::is_same_v<T, MatmulMultiCoreReuseMultiCast1DProgramConfig>) {
                    if (c.per_core_M % c.out_block_h != 0 || c.per_core_N % c.out_block_w != 0 ||
                        c.out_block_h % c.out_subblock_h != 0 || c.out_block_w % c.out_subblock_w != 0) {
                        return "block divisibility";
                    }
                    const uint32_t M = c.fuse_batch ? p.batch_a * p.Mt : p.Mt;
                    if (c.fuse_batch && p.batch_b > 1) {
                        return "fuse_batch with batched B";
                    }
                    const uint32_t blocks_y = div_up(M, c.per_core_M);
                    const uint32_t blocks_x = div_up(p.Nt, c.per_core_N);
                    Family family = Family::Mcast2D;
                    if constexpr (std::is_same_v<T, MatmulMultiCoreReuseMultiCastProgramConfig>) {
                        if (c.per_core_M > M || blocks_y > hw.grid.y || blocks_x > hw.grid.x) {
                            return "2d grid";
                        }
                    } else {
                        family = c.mcast_in0 ? Family::Mcast1DIn0 : Family::Mcast1DIn1;
                        if (blocks_x * blocks_y > cores) {
                            return "1d core count";
                        }
                        if (c.mcast_in0 && blocks_y != 1) {
                            return "1d in0 rows";
                        }
                        if (!c.mcast_in0) {
                            if (c.per_core_N != p.Nt || c.per_core_M > M) {
                                return "1d in1 shape";
                            }
                            if (blocks_y == 1 && M % c.out_block_h != 0 && c.per_core_M != c.out_block_h) {
                                return "1d in1 single row";
                            }
                        }
                    }
                    Blocking b{c.per_core_M, c.per_core_N, c.in0_block_w, c.out_block_h, c.out_block_w, 0, 0};
                    if (circular_buffer_bytes(p, hw, family, b) > hw.l1_cb_budget) {
                        return "L1";
                    }
                    return "";
                }
            }
        },
        config);
}

struct Shape {
    uint32_t batch_a, batch_b, M, K, N;
};

// Shapes from the out-of-box perf issues plus a coarse grid of sizes
std::vector<Shape> shapes() {
    std::vector<Shape> s = {
        {1, 1, 1024, 5376, 5376},   // #56976
        {1, 1, 32, 7168, 256},      // #40845
        {1, 1, 128, 7168, 256},     // #40845
        {384, 384, 256, 64, 256},   // #35396 batched attention
        {384, 384, 256, 256, 64},   // #35396
        {1, 1, 16384, 384, 1152},   // #35396 nanoGPT
        {1, 1, 37888, 3456, 5120},  // #29716 TT-DiT
        {1, 1, 9472, 5120, 1280},   // #29716
        {704, 1, 704, 128, 128},    // #25503 batched A, fused into M
        {768, 1, 768, 128, 4},      // #25503
        {32, 32, 704, 704, 704},    // #25502
        {4, 1, 256, 2048, 7168},    // #31743
        {1, 1, 8192, 8192, 8192},   // #30407
        {1, 1, 32, 32, 32},
        {1, 1, 32, 4096, 128256},   // LM head
        {1, 1, 11008, 8192, 4096},  // #36426 (transpose_a in the issue)
    };
    for (uint32_t m : {32u, 256u, 1024u, 4096u}) {
        for (uint32_t k : {1024u, 4096u}) {
            for (uint32_t n : {1024u, 4096u, 16384u}) {
                s.push_back({1, 1, m, k, n});
            }
        }
    }
    return s;
}

TEST(MatmulAutoConfig, EmittedConfigsAreValid) {
    for (const auto& arch : kArchs) {
        const auto hw = HardwareDesc::for_arch(arch.arch, arch.grid, kL1Budget);
        for (const auto& s : shapes()) {
            for (auto in1 : {tt::DataFormat::Float16_b, tt::DataFormat::Bfp8_b, tt::DataFormat::Bfp4_b}) {
                for (bool fp32_acc : {false, true}) {
                    for (bool bias : {false, true}) {
                        const auto p = make_problem(s.batch_a, s.batch_b, s.M, s.K, s.N, in1, fp32_acc, bias);
                        const auto label = fmt::format(
                            "{} b={}/{} M={} K={} N={} in1={} fp32={} bias={}",
                            arch.name,
                            s.batch_a,
                            s.batch_b,
                            s.M,
                            s.K,
                            s.N,
                            static_cast<int>(in1),
                            fp32_acc,
                            bias);
                        const auto config = select_program_config(p, hw);
                        ASSERT_TRUE(config.has_value()) << label;
                        EXPECT_EQ(check_config(p, hw, *config), "") << label << "\n" << fmt::format("{}", *config);
                    }
                }
            }
        }
    }
}

TEST(MatmulAutoConfig, TransposeAFitsL1) {
    for (const auto& arch : kArchs) {
        const auto hw = HardwareDesc::for_arch(arch.arch, arch.grid, kL1Budget);
        for (const auto& s : shapes()) {
            const auto p = make_problem(
                s.batch_a, s.batch_b, s.M, s.K, s.N, tt::DataFormat::Float16_b, false, false, /*transpose_a=*/true);
            const auto config = select_program_config(p, hw);
            ASSERT_TRUE(config.has_value());
            EXPECT_EQ(check_config(p, hw, *config), "") << arch.name << " M=" << s.M << " K=" << s.K << " N=" << s.N;
        }
    }
}

TEST(MatmulAutoConfig, BatchedBUsesReuse) {
    const auto hw = HardwareDesc::for_arch(tt::ARCH::WORMHOLE_B0, CoreCoord(8, 8), kL1Budget);
    const auto config = select_program_config(make_problem(384, 384, 256, 256, 64), hw);
    ASSERT_TRUE(config.has_value());
    EXPECT_TRUE(std::holds_alternative<MatmulMultiCoreReuseProgramConfig>(*config));
}

TEST(MatmulAutoConfig, UsesWholeGridForLargeMatmul) {
    for (const auto& arch : kArchs) {
        const auto hw = HardwareDesc::for_arch(arch.arch, arch.grid, kL1Budget);
        const auto config = select_program_config(make_problem(1, 1, 4096, 4096, 4096), hw);
        ASSERT_TRUE(config.has_value());
        ASSERT_TRUE(std::holds_alternative<MatmulMultiCoreReuseMultiCastProgramConfig>(*config)) << arch.name;
        const auto& c = std::get<MatmulMultiCoreReuseMultiCastProgramConfig>(*config);
        EXPECT_EQ(div_up(128, c.per_core_M), arch.grid.y) << arch.name;
        EXPECT_EQ(div_up(128, c.per_core_N), arch.grid.x) << arch.name;
    }
}

// Precision is the compute config's call: the blocking doesn't change with the output format or packer L1
// accumulation (they only change the L1 cost of the partials buffer)
TEST(MatmulAutoConfig, BlockingIgnoresOutputPrecision) {
    const auto hw = HardwareDesc::for_arch(tt::ARCH::WORMHOLE_B0, CoreCoord(8, 8), kL1Budget);
    for (auto [M, K, N] : {std::tuple{64u, 128u, 64u}, std::tuple{1024u, 160u, 256u}, std::tuple{32u, 1024u, 1000u}}) {
        auto base = make_problem(1, 1, M, K, N, tt::DataFormat::Bfp8_b);
        const auto reference = choose_candidate(base, hw);
        ASSERT_TRUE(reference.has_value());
        for (auto out : {tt::DataFormat::Bfp8_b, tt::DataFormat::Bfp4_b}) {
            for (bool l1_acc : {true, false}) {
                auto p = base;
                p.out_format = out;
                p.packer_l1_acc = l1_acc;
                const auto chosen = choose_candidate(p, hw);
                ASSERT_TRUE(chosen.has_value());
                EXPECT_EQ(chosen->blocking.in0_block_w, reference->blocking.in0_block_w) << M << "x" << K << "x" << N;
            }
        }
    }
}

// Subblocks at least two tiles on each side, unless B's tiles are smaller than A's
TEST(MatmulAutoConfig, SubblockShape) {
    const auto hw = HardwareDesc::for_arch(tt::ARCH::WORMHOLE_B0, CoreCoord(8, 8), kL1Budget);
    auto p = make_problem(1, 1, 8192, 8192, 8192);
    auto chosen = choose_candidate(p, hw);
    ASSERT_TRUE(chosen.has_value());
    EXPECT_GE(std::min(chosen->blocking.out_subblock_h, chosen->blocking.out_subblock_w), 2u);
    p = make_problem(1, 1, 8192, 8192, 8192, tt::DataFormat::Bfp8_b);  // bf16 A, bfp8 B
    chosen = choose_candidate(p, hw);
    ASSERT_TRUE(chosen.has_value());
    EXPECT_EQ(chosen->blocking.out_subblock_h, 1u);
    EXPECT_EQ(chosen->blocking.out_subblock_w, 8u);
}

// 1D in0-mcast splits a wide output block into subblock-wide blocks (not into 1-tile ones)
TEST(MatmulAutoConfig, OneDOutputBlockSplit) {
    const auto hw = HardwareDesc::for_arch(tt::ARCH::WORMHOLE_B0, CoreCoord(8, 8), kL1Budget);
    auto p = make_problem(1, 1, 32, 2560, 262144);
    auto chosen = choose_candidate(p, hw);
    ASSERT_TRUE(chosen.has_value());
    EXPECT_EQ(static_cast<int>(chosen->family), static_cast<int>(Family::Mcast1DIn0));
    EXPECT_EQ(chosen->blocking.per_core_N, 128u);
    EXPECT_EQ(chosen->blocking.out_block_w, 8u);
    p = make_problem(1, 1, 32, 4544, 11 * 32 * 64);  // per_core_N = 11 has no divisor in 2..8
    chosen = choose_candidate(p, hw);
    ASSERT_TRUE(chosen.has_value());
    EXPECT_EQ(chosen->blocking.out_block_w, chosen->blocking.per_core_N);
}

// Large 2D output blocks may use K blocks up to 16 deep; small ones stay at 8
TEST(MatmulAutoConfig, LargeBlockKDepth) {
    // Llama-70B TP8 w1 prefill (bf16 x bfp4, LoFi), with the L1 budget the device reported
    auto p = make_problem(1, 1, 2048, 8192, 3584, tt::DataFormat::Bfp4_b);
    p.math_fidelity = MathFidelity::LoFi;
    auto hw = HardwareDesc::for_arch(tt::ARCH::WORMHOLE_B0, CoreCoord(8, 8), 1377056);
    auto chosen = choose_candidate(p, hw);
    ASSERT_TRUE(chosen.has_value());
    EXPECT_EQ(static_cast<int>(chosen->family), static_cast<int>(Family::Mcast2D));
    EXPECT_GT(chosen->blocking.out_block_h * chosen->blocking.out_block_w, LARGE_BLOCK_TILES);
    EXPECT_EQ(chosen->blocking.in0_block_w, 16u);
    p = make_problem(1, 1, 256, 4096, 1024, tt::DataFormat::Bfp8_b);  // 1x4 blocks
    hw = HardwareDesc::for_arch(tt::ARCH::WORMHOLE_B0, CoreCoord(8, 8), kL1Budget);
    chosen = choose_candidate(p, hw);
    ASSERT_TRUE(chosen.has_value());
    EXPECT_LE(chosen->blocking.in0_block_w, MAX_IN0_BLOCK_W);
}

// When keeping the full multicast extent only fits with single-tile K steps, 1D shrinks it instead
TEST(MatmulAutoConfig, OneDAvoidsSingleTileK) {
    // 1024x1024x16384 bf16 x bfp8 with an L1 output, at the L1 budget the device reported
    auto p = make_problem(1, 1, 1024, 1024, 16384, tt::DataFormat::Bfp8_b);
    p.out.in_l1 = true;
    const auto hw = HardwareDesc::for_arch(tt::ARCH::WORMHOLE_B0, CoreCoord(8, 8), 820000);
    const auto chosen = choose_candidate(p, hw);
    ASSERT_TRUE(chosen.has_value());
    EXPECT_EQ(static_cast<int>(chosen->family), static_cast<int>(Family::Mcast1DIn0));
    EXPECT_GE(chosen->blocking.in0_block_w, 2u);
    EXPECT_LT(chosen->blocking.out_block_h, chosen->blocking.per_core_M);
}

Placement sharded(Layout layout, CoreCoord grid, uint32_t shard_h, uint32_t shard_w, bool col_major = false) {
    Placement pl;
    pl.layout = layout;
    pl.in_l1 = true;
    pl.has_shard_spec = true;
    pl.shard_grid = CoreRange({0, 0}, {grid.x - 1, grid.y - 1});
    pl.shard_cores = grid.x * grid.y;
    pl.shard_h = shard_h;
    pl.shard_w = shard_w;
    pl.col_major = col_major;
    return pl;
}

Placement sharded_output(Layout layout) {
    Placement pl;
    pl.layout = layout;
    pl.in_l1 = true;
    return pl;
}

// Sharded layouts pin the family, grid and per-core sizes; the rest must satisfy the layout's rules
TEST(MatmulAutoConfig, ShardedLayouts) {
    const auto hw = HardwareDesc::for_arch(tt::ARCH::WORMHOLE_B0, CoreCoord(8, 8), kL1Budget);
    struct Case {
        std::string name;
        Problem p;
        Family family;
        uint32_t per_core_M, per_core_N;
        bool transpose_mcast = false;
    };
    std::vector<Case> cases;
    {  // decode: width-sharded activation, 1D in0-mcast, each core a slice of N
        auto p = make_problem(1, 1, 32, 4096, 4096, tt::DataFormat::Bfp8_b);
        p.a = sharded(Layout::WidthSharded, CoreCoord(8, 8), 1, 2);
        cases.push_back({"width A", p, Family::Mcast1DIn0, 1, 2});
        p.out = sharded_output(Layout::WidthSharded);
        cases.push_back({"width A, width out", p, Family::Mcast1DIn0, 1, 2});
    }
    {  // tall: height-sharded activation, 1D in1-mcast
        auto p = make_problem(1, 1, 8192, 256, 256);
        p.a = sharded(Layout::HeightSharded, CoreCoord(8, 8), 4, 8);
        cases.push_back({"height A", p, Family::Mcast1DIn1, 4, 8});
        p.out = sharded_output(Layout::HeightSharded);
        cases.push_back({"height A, height out", p, Family::Mcast1DIn1, 4, 8});
    }
    {  // block-sharded 2D, row- and column-major
        auto p = make_problem(1, 1, 2048, 2048, 2048);
        p.a = sharded(Layout::BlockSharded, CoreCoord(8, 8), 8, 8);
        cases.push_back({"block A", p, Family::Mcast2D, 8, 8});
        p.out = sharded_output(Layout::BlockSharded);
        cases.push_back({"block A, block out", p, Family::Mcast2D, 8, 8});
        p.a.col_major = true;
        cases.push_back({"block A col-major", p, Family::Mcast2D, 8, 8, true});
    }
    {  // batched B with height-sharded A: Reuse over A's shards
        auto p = make_problem(48, 48, 256, 256, 64);
        p.a = sharded(Layout::HeightSharded, CoreCoord(8, 6), 8, 8);
        cases.push_back({"height A, batched B", p, Family::Reuse, 8, 2});
    }
    {  // interleaved inputs, sharded output
        auto p = make_problem(1, 1, 8192, 512, 512);
        p.out = sharded_output(Layout::HeightSharded);
        cases.push_back({"height out", p, Family::Mcast1DIn1, 4, 16});
        p = make_problem(1, 1, 32, 4096, 8192, tt::DataFormat::Bfp8_b);
        p.out = sharded_output(Layout::WidthSharded);
        cases.push_back({"width out", p, Family::Mcast1DIn0, 1, 4});
        p = make_problem(1, 1, 2048, 2048, 2048);
        p.out = sharded_output(Layout::BlockSharded);
        cases.push_back({"block out", p, Family::Mcast2D, 8, 8});
        // an output shard spec fixes the grid: 4x2 cores of 16x32 tiles
        p.out = sharded(Layout::BlockSharded, CoreCoord(4, 2), 32, 16);
        cases.push_back({"block out with spec", p, Family::Mcast2D, 32, 16});
        p = make_problem(1, 1, 256, 2048, 2048);
        p.out = sharded(Layout::BlockSharded, CoreCoord(8, 1), 8, 8);
        cases.push_back({"block out on a row", p, Family::Mcast1DIn0, 8, 8});
        p = make_problem(1, 1, 4096, 512, 512);
        p.out = sharded(Layout::HeightSharded, CoreCoord(8, 4), 4, 16);
        cases.push_back({"height out with spec", p, Family::Mcast1DIn1, 4, 16});
    }
    for (const auto& c : cases) {
        const auto chosen = choose_candidate(c.p, hw);
        ASSERT_TRUE(chosen.has_value()) << c.name;
        const auto& b = chosen->blocking;
        EXPECT_EQ(static_cast<int>(chosen->family), static_cast<int>(c.family)) << c.name;
        EXPECT_EQ(b.per_core_M, c.per_core_M) << c.name;
        EXPECT_EQ(b.per_core_N, c.per_core_N) << c.name;
        EXPECT_EQ(chosen->transpose_mcast, c.transpose_mcast) << c.name;
        EXPECT_EQ(c.p.Kt % b.in0_block_w, 0u) << c.name;
        if (c.p.a.sharded() && c.p.a.layout != Layout::HeightSharded) {
            EXPECT_EQ(b.in0_block_w, c.p.a.shard_w) << c.name << ": K blocks should be whole shard columns";
        }
        if (c.p.a.layout == Layout::HeightSharded) {
            EXPECT_EQ(b.in0_block_w, c.p.Kt) << c.name << ": height-sharded A is read in place over all of K";
        }
        if (c.p.out.sharded() && c.family != Family::Reuse) {
            EXPECT_EQ(b.out_block_w, b.per_core_N) << c.name << ": sharded output blocks span per_core_N";
            EXPECT_TRUE(b.out_subblock_w == b.per_core_N || b.out_subblock_h == 1) << c.name;
        }
        EXPECT_LE(circular_buffer_bytes(c.p, hw, chosen->family, b), hw.l1_cb_budget) << c.name;
    }

    // Layout combinations the factories reject are not produced
    auto p = make_problem(1, 1, 2048, 2048, 2048);
    p.a = sharded(Layout::BlockSharded, CoreCoord(8, 8), 8, 8);
    p.out = sharded_output(Layout::HeightSharded);
    EXPECT_FALSE(choose_candidate(p, hw).has_value()) << "sharded output must be laid out like A";
}

// Family choices of the heuristics, from the Wormhole config sweep (tests/ttnn/unit_tests/benchmarks/matmul_oob)
TEST(MatmulAutoConfig, FamilyChoice) {
    const auto hw = HardwareDesc::for_arch(tt::ARCH::WORMHOLE_B0, CoreCoord(8, 8), kL1Budget);
    struct Expected {
        Shape shape;
        Family family;
    };
    const std::vector<Expected> expected = {
        {{1, 1, 32, 4096, 14336}, Family::Mcast1DIn0},   // decode: M is one tile row
        {{1, 1, 128, 4096, 6144}, Family::Mcast1DIn0},   // decode, 4 rows but wide N
        {{1, 1, 128, 8192, 1280}, Family::Mcast1DIn0},   // 4 rows, N=40 tiles: 40 cores vs 32, less input each
        {{1, 1, 256, 4096, 16384}, Family::Mcast1DIn0},  // 8 rows, wide N: 8x8 blocks beat 2D's 1x64
        {{1, 1, 128, 7168, 256}, Family::Mcast2D},       // #40845
        {{1, 1, 2048, 4096, 4096}, Family::Mcast2D},     // prefill
        {{1, 1, 1024, 5376, 5376}, Family::Mcast2D},     // #56976
        {{1, 1, 16384, 384, 1152}, Family::Mcast2D},     // nanoGPT
        {{768, 1, 768, 128, 32}, Family::Mcast1DIn1},    // tall and one tile wide
        {{384, 384, 256, 256, 64}, Family::Reuse},       // batched attention
        {{48, 48, 1024, 1024, 64}, Family::Reuse},       // gpt2-style transpose_a attention
        {{32, 32, 704, 704, 704}, Family::Mcast2D},      // #25502: Reuse would re-read B for every M slice
        {{2, 2, 1024, 64, 512}, Family::Mcast2D},        // few large batch matrices
        {{4, 4, 256, 2048, 7168}, Family::Mcast2D},      // #31743: batched with wide N, looped over batch
    };
    for (const auto& e : expected) {
        const auto& s = e.shape;
        const auto chosen = choose_candidate(make_problem(s.batch_a, s.batch_b, s.M, s.K, s.N), hw);
        ASSERT_TRUE(chosen.has_value());
        EXPECT_EQ(static_cast<int>(chosen->family), static_cast<int>(e.family))
            << "b=" << s.batch_a << "/" << s.batch_b << " M=" << s.M << " K=" << s.K << " N=" << s.N;
    }
}

// Inputs the legacy selection covered that v2 must too (no fallback): tiny tiles, broadcast A, transpose_a
// over a batch, sub-device grids, global CBs, block-sharded A on one row or column, and output shard specs
// whose grid doesn't match the output.
TEST(MatmulAutoConfig, TinyTiles) {
    for (const auto& arch : kArchs) {
        const auto hw = HardwareDesc::for_arch(arch.arch, arch.grid, kL1Budget);
        for (uint32_t tile_h : {1u, 2u, 4u, 8u, 16u, 32u}) {
            for (uint32_t tile_w : {16u, 32u}) {
                for (auto in1 : {tt::DataFormat::Float16_b, tt::DataFormat::Bfp8_b}) {
                    for (const auto& s :
                         std::vector<Shape>{{1, 1, 1024, 64, 512}, {4, 4, 128, 256, 256}, {1, 1, 32, 4096, 4096}}) {
                        auto p = make_problem(s.batch_a, s.batch_b, s.M, s.K, s.N, in1);
                        p.in0_tile_h = p.out_tile_h = tile_h;
                        p.in1_tile_w = p.out_tile_w = tile_w;
                        p.Mt = s.M / tile_h;
                        p.Nt = s.N / tile_w;
                        const auto label = fmt::format(
                            "{} tile {}x{} in1={} M={}", arch.name, tile_h, tile_w, static_cast<int>(in1), s.M);
                        const auto chosen = choose_candidate(p, hw);
                        const bool reuse_only = in1 == tt::DataFormat::Bfp8_b && tile_h < 16;
                        if (!chosen.has_value()) {
                            // Only possible when no factory can run it: Reuse is the only one, and its single K
                            // block of B (all of K by all of N) doesn't fit L1
                            Blocking one_row{1, p.Nt, p.Kt, 1, p.Nt, 1, 1};
                            EXPECT_TRUE(
                                reuse_only && circular_buffer_bytes(p, hw, Family::Reuse, one_row) > hw.l1_cb_budget)
                                << label;
                            continue;
                        }
                        const auto& b = chosen->blocking;
                        EXPECT_LE(b.out_subblock_h * b.out_subblock_w, 8u) << label;
                        EXPECT_LE(circular_buffer_bytes(p, hw, chosen->family, b), hw.l1_cb_budget) << label;
                        if (reuse_only) {
                            // the mcast kernels can't unpack these, and Reuse only computes them with one K block
                            EXPECT_EQ(static_cast<int>(chosen->family), static_cast<int>(Family::Reuse)) << label;
                            EXPECT_EQ(b.in0_block_w, p.Kt) << label;
                        }
                    }
                }
            }
        }
    }
}

TEST(MatmulAutoConfig, BroadcastA) {
    const auto hw = HardwareDesc::for_arch(tt::ARCH::WORMHOLE_B0, CoreCoord(8, 8), kL1Budget);
    for (const auto& s : std::vector<Shape>{{1, 7, 128, 2048, 256}, {1, 5, 64, 768, 192}, {1, 8, 2048, 4096, 1024}}) {
        const auto p = make_problem(s.batch_a, s.batch_b, s.M, s.K, s.N);
        const auto chosen = choose_candidate(p, hw);
        ASSERT_TRUE(chosen.has_value()) << s.M;
        // Only 1D in1-mcast reuses a single A across B's batches, looping over them
        EXPECT_EQ(static_cast<int>(chosen->family), static_cast<int>(Family::Mcast1DIn1)) << s.M;
        EXPECT_FALSE(chosen->blocking.fuse_batch) << s.M;
        EXPECT_EQ(chosen->blocking.per_core_N, p.Nt) << s.M;
        // A's rows stay resident across the batch loop
        EXPECT_GE(
            circular_buffer_bytes(p, hw, Family::Mcast1DIn1, chosen->blocking),
            chosen->blocking.per_core_M * p.Kt * tt::tile_size(p.in0_format))
            << s.M;
        const auto config = select_program_config(p, hw);
        ASSERT_TRUE(config.has_value());
        EXPECT_FALSE(std::get<MatmulMultiCoreReuseMultiCast1DProgramConfig>(*config).fused_activation.has_value());
    }
}

TEST(MatmulAutoConfig, TransposeAOverBatchIsNotFused) {
    const auto hw = HardwareDesc::for_arch(tt::ARCH::WORMHOLE_B0, CoreCoord(8, 8), kL1Budget);
    const auto p = make_problem(8, 1, 512, 256, 512, tt::DataFormat::Float16_b, false, false, /*transpose_a=*/true);
    const auto chosen = choose_candidate(p, hw);
    ASSERT_TRUE(chosen.has_value());
    EXPECT_NE(static_cast<int>(chosen->family), static_cast<int>(Family::Reuse));  // Reuse can't broadcast B
    EXPECT_FALSE(chosen->blocking.fuse_batch);
}

TEST(MatmulAutoConfig, NoOneDWhenExcluded) {
    const auto hw = HardwareDesc::for_arch(tt::ARCH::WORMHOLE_B0, CoreCoord(8, 8), kL1Budget);
    auto p = make_problem(1, 1, 32, 4096, 14336);  // decode: 1D in0-mcast otherwise
    p.no_mcast_1d = true;
    for (const auto& c : candidates(p, hw)) {
        EXPECT_TRUE(c.family == Family::Mcast2D || c.family == Family::Reuse);
    }
    EXPECT_TRUE(choose_candidate(p, hw).has_value());
}

TEST(MatmulAutoConfig, SubDeviceGrid) {
    auto hw = HardwareDesc::for_arch(tt::ARCH::WORMHOLE_B0, CoreCoord(8, 7), kL1Budget);
    hw.origin = CoreCoord(0, 1);
    hw.pinned_origin = true;
    for (const auto& s : std::vector<Shape>{{1, 1, 128, 512, 512}, {1, 1, 32, 4096, 4096}, {48, 48, 256, 256, 64}}) {
        const auto config = select_program_config(make_problem(s.batch_a, s.batch_b, s.M, s.K, s.N), hw);
        ASSERT_TRUE(config.has_value()) << s.M;
        std::visit(
            [&](const auto& c) {
                using T = std::decay_t<decltype(c)>;
                if constexpr (requires { c.allowed_worker_cores; }) {
                    ASSERT_TRUE(c.allowed_worker_cores.has_value()) << s.M;
                    const auto bbox = c.allowed_worker_cores->bounding_box();
                    EXPECT_EQ(bbox.start_coord, CoreCoord(0, 1)) << s.M;
                    EXPECT_EQ(bbox.end_coord, CoreCoord(7, 7)) << s.M;
                } else {
                    ADD_FAILURE() << "unexpected config type " << typeid(T).name();
                }
            },
            *config);
    }
}

TEST(MatmulAutoConfig, ShardedEdgeLayouts) {
    const auto hw = HardwareDesc::for_arch(tt::ARCH::WORMHOLE_B0, CoreCoord(8, 8), kL1Budget);
    {  // block-sharded A on one column of cores, column-major: 2D with transposed mcast
        auto p = make_problem(1, 1, 4096, 32, 128);
        p.a = sharded(Layout::BlockSharded, CoreCoord(8, 1), 16, 1, /*col_major=*/true);
        const auto chosen = choose_candidate(p, hw);
        ASSERT_TRUE(chosen.has_value());
        EXPECT_EQ(static_cast<int>(chosen->family), static_cast<int>(Family::Mcast2D));
        EXPECT_TRUE(chosen->transpose_mcast);
        EXPECT_EQ(chosen->blocking.per_core_M, 16u);
        EXPECT_EQ(chosen->blocking.per_core_N, 4u);
    }
    {  // one-core block shard spec for a 5-batch output: keep the shard shape, derive the grid
        auto p = make_problem(5, 1, 416, 32, 416);
        p.out = sharded(Layout::BlockSharded, CoreCoord(1, 1), 13, 13);
        const auto chosen = choose_candidate(p, hw);
        ASSERT_TRUE(chosen.has_value());
        EXPECT_EQ(static_cast<int>(chosen->family), static_cast<int>(Family::Mcast2D));
        EXPECT_EQ(chosen->blocking.per_core_M, 13u);
        EXPECT_EQ(chosen->blocking.per_core_N, 13u);
    }
}

// Prints every family's candidate for the test shapes; run with --gtest_also_run_disabled_tests when tuning.
TEST(MatmulAutoConfig, DISABLED_PrintSelections) {
    const auto hw = HardwareDesc::for_arch(tt::ARCH::WORMHOLE_B0, CoreCoord(8, 8), 1450 * 1024);
    const char* names[] = {"2D", "1D-in0", "1D-in1", "Reuse"};
    for (const auto& s : shapes()) {
        const auto p = make_problem(s.batch_a, s.batch_b, s.M, s.K, s.N);
        const auto chosen = choose_candidate(p, hw);
        fmt::print("b={}/{} M={} K={} N={}\n", s.batch_a, s.batch_b, s.M, s.K, s.N);
        for (const auto& c : candidates(p, hw)) {
            const auto& b = c.blocking;
            fmt::print(
                "  {} {:7s} cores={:3d} pc={}x{} k={} blk={}x{} sb={}x{}\n",
                chosen && chosen->family == c.family ? '*' : ' ',
                names[static_cast<int>(c.family)],
                c.cores,
                b.per_core_M,
                b.per_core_N,
                b.in0_block_w,
                b.out_block_h,
                b.out_block_w,
                b.out_subblock_h,
                b.out_subblock_w);
        }
    }
}
}  // namespace
