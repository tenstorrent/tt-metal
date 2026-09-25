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
                    // The reuse factory leaves output unwritten when a core gets several partial-batch blocks
                    if (c.per_core_M < p.Mt && p.batch_a * (p.Mt / c.per_core_M) > cores) {
                        return "reuse partial-batch blocks exceed cores";
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
            if (s.batch_b == 1 && s.batch_a > 1) {
                continue;  // not selected: transpose_a can't fuse batch into M
            }
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
