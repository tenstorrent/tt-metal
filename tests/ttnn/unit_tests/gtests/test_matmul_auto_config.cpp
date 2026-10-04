// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Device-free checks of the new matmul default-config selector (matmul_auto_config.hpp): every config it
// emits, for Wormhole and Blackhole grids alike, must satisfy the program factories' constraints and fit
// the L1 budget it was given.

#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <string>
#include <tuple>
#include <variant>
#include <vector>

#include <fmt/format.h>

#include "ttnn/operations/matmul/device/config/factory_blocking_source.hpp"
#include "ttnn/operations/matmul/device/config/matmul_auto_config.hpp"
#include "ttnn/operations/matmul/device/config/roofline_estimator.hpp"
#include "ttnn/operations/matmul/device/matmul_validation.hpp"
#include "ttnn/tensor/layout/tensor_layout.hpp"
#include "ttnn/tensor/tensor_spec.hpp"

namespace {

using namespace ttnn::operations::matmul;
using namespace ttnn::operations::matmul::auto_config;

uint32_t div_up(uint32_t a, uint32_t b) { return (a + b - 1) / b; }

// Each family's candidate, from the default source
std::vector<Candidate> candidates(const MatmulDesc& p, const HardwareDesc& hw) {
    return FactoryBlockingSource().candidates(p, hw);
}

// The default sources without estimators: the source's own choice
const Selector& heuristics_only() {
    static const Selector selector{.sources = default_selector().sources, .estimators = {}};
    return selector;
}

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

MatmulDesc make_matmul(
    uint32_t batch_a,
    uint32_t batch_b,
    uint32_t M,
    uint32_t K,
    uint32_t N,
    tt::DataFormat in1 = tt::DataFormat::Float16_b,
    bool fp32_acc = false,
    bool bias = false,
    bool transpose_a = false) {
    MatmulDesc p;
    p.batch_a = batch_a;
    p.batch_b = batch_b;
    p.Mt = div_up(M, 32);
    p.Kt = div_up(K, 32);
    p.Nt = div_up(N, 32);
    p.in1_format = in1;
    p.fp32_dest_acc_en = fp32_acc;
    p.bias_tile_bytes = bias ? tt::tile_size(tt::DataFormat::Float16_b) : 0;
    p.bias_rows = bias ? 1 : 0;
    p.transpose_a = transpose_a;
    p.in0_tile_transposed = transpose_a;  // A's tiles as the matmul reads them
    p.rank_a = p.rank_b = (batch_a > 1 || batch_b > 1) ? 3 : 2;
    return p;
}

tt::tt_metal::DataType data_type(tt::DataFormat format) {
    switch (format) {
        case tt::DataFormat::Bfp8_b: return tt::tt_metal::DataType::BFLOAT8_B;
        case tt::DataFormat::Bfp4_b: return tt::tt_metal::DataType::BFLOAT4_B;
        case tt::DataFormat::Float32: return tt::tt_metal::DataType::FLOAT32;
        default: return tt::tt_metal::DataType::BFLOAT16;
    }
}

tt::tt_metal::TensorMemoryLayout memory_layout(MemoryLayout layout) {
    using tt::tt_metal::TensorMemoryLayout;
    switch (layout) {
        case MemoryLayout::HeightSharded: return TensorMemoryLayout::HEIGHT_SHARDED;
        case MemoryLayout::WidthSharded: return TensorMemoryLayout::WIDTH_SHARDED;
        case MemoryLayout::BlockSharded: return TensorMemoryLayout::BLOCK_SHARDED;
        default: return TensorMemoryLayout::INTERLEAVED;
    }
}

// The memory config of a placement whose shards are tile_h x tile_w tiles
tt::tt_metal::MemoryConfig memory_config(const Placement& pl, uint32_t tile_h, uint32_t tile_w) {
    const auto buffer = pl.in_l1 ? tt::tt_metal::BufferType::L1 : tt::tt_metal::BufferType::DRAM;
    std::optional<tt::tt_metal::ShardSpec> shard_spec;
    if (pl.has_shard_spec) {
        shard_spec = tt::tt_metal::ShardSpec(
            CoreRangeSet(pl.shard_grid),
            {pl.shard_h * tile_h, pl.shard_w * tile_w},
            pl.col_major ? tt::tt_metal::ShardOrientation::COL_MAJOR : tt::tt_metal::ShardOrientation::ROW_MAJOR);
    }
    return tt::tt_metal::MemoryConfig(memory_layout(pl.layout), buffer, shard_spec);
}

tt::tt_metal::TensorSpec tile_spec(
    ttnn::Shape shape, tt::DataFormat format, tt::tt_metal::Tile tile, tt::tt_metal::MemoryConfig memory) {
    return tt::tt_metal::TensorSpec(
        std::move(shape),
        tt::tt_metal::TensorLayout(
            data_type(format), tt::tt_metal::PageConfig(tt::tt_metal::Layout::TILE, tile), std::move(memory)));
}

// A description's fields, each as text
std::vector<std::pair<std::string, std::string>> fields(const MatmulDesc& p) {
    auto placement = [](const Placement& t) {
        return fmt::format(
            "layout {} l1 {} spec {} whole {} grid {} cores {} shard {}x{} col_major {}",
            static_cast<int>(t.layout),
            t.in_l1,
            t.has_shard_spec,
            t.shard_whole_tiles,
            t.shard_grid.str(),
            t.shard_cores,
            t.shard_h,
            t.shard_w,
            t.col_major);
    };
    return {
        {"batches", fmt::format("{} {}", p.batch_a, p.batch_b)},
        {"ranks", fmt::format("{} {}", p.rank_a, p.rank_b)},
        {"tiles", fmt::format("Mt {} Kt {} Nt {}", p.Mt, p.Kt, p.Nt)},
        {"tile shapes", fmt::format("{} {} {} {}", p.in0_tile_h, p.in1_tile_w, p.out_tile_h, p.out_tile_w)},
        {"formats",
         fmt::format(
             "{} {} {}",
             static_cast<int>(p.in0_format),
             static_cast<int>(p.in1_format),
             static_cast<int>(p.out_format))},
        {"bias", fmt::format("{} bytes, {} rows", p.bias_tile_bytes, p.bias_rows)},
        {"transposes", fmt::format("{} {}", p.transpose_a, p.in0_tile_transposed)},
        {"untilize_out", fmt::format("{}", p.untilize_out)},
        {"compute",
         fmt::format(
             "{} {} {} {}",
             static_cast<int>(p.math_fidelity),
             p.fp32_dest_acc_en,
             p.packer_l1_acc,
             p.dst_full_sync_en)},
        {"activation", fmt::format("{}", p.activation.has_value())},
        {"a", placement(p.a)},
        {"b", placement(p.b)},
        {"out", placement(p.out)},
        {"b_shard_matches_a", fmt::format("{}", p.b_shard_matches_a)},
        {"global_cb", fmt::format("{}", p.global_cb)},
    };
}

// The first field where two descriptions differ, or empty
std::string desc_mismatch(const MatmulDesc& x, const MatmulDesc& y) {
    const auto a = fields(x);
    const auto b = fields(y);
    for (size_t i = 0; i < a.size(); ++i) {
        if (a[i].second != b[i].second) {
            return fmt::format("{}: {} vs {}", a[i].first, a[i].second, b[i].second);
        }
    }
    return "";
}

// The specs of the matmul `p` describes, on a device with hw's grid: describe_matmul of them gives back `p`
ttnn::prim::MatmulSpecs specs_of(const MatmulDesc& p, const HardwareDesc& hw) {
    const tt::tt_metal::Tile a_tile({p.transpose_a ? 32u : p.in0_tile_h, p.transpose_a ? p.in0_tile_h : 32u});
    const tt::tt_metal::Tile b_tile({32, p.in1_tile_w});
    auto batched = [](uint32_t batch, uint32_t rank, uint32_t rows, uint32_t cols) {
        return rank > 2 ? ttnn::Shape({batch, rows, cols}) : ttnn::Shape({rows, cols});
    };
    const uint32_t M = p.Mt * p.in0_tile_h;
    const uint32_t K = p.Kt * 32;
    const uint32_t N = p.Nt * p.in1_tile_w;
    ttnn::prim::MatmulSpecs specs;
    specs.inputs.push_back(tile_spec(
        p.transpose_a ? batched(p.batch_a, p.rank_a, K, M) : batched(p.batch_a, p.rank_a, M, K),
        p.in0_format,
        a_tile,
        memory_config(p.a, p.in0_tile_h, 32)));
    specs.inputs.push_back(
        tile_spec(batched(p.batch_b, p.rank_b, K, N), p.in1_format, b_tile, memory_config(p.b, 32, p.in1_tile_w)));
    if (p.bias_tile_bytes != 0) {
        specs.bias = tile_spec(
            ttnn::Shape({1, p.bias_rows * p.in0_tile_h, N}),
            tt::DataFormat::Float16_b,
            tt::tt_metal::Tile({p.in0_tile_h, p.in1_tile_w}),
            tt::tt_metal::MemoryConfig());
    }
    auto& attributes = specs.attributes;
    attributes.bcast_batch = p.batch_b == 1;
    attributes.output_mem_config = memory_config(p.out, p.in0_tile_h, p.in1_tile_w);
    attributes.output_dtype = data_type(p.out_format);
    attributes.compute_kernel_config = ttnn::WormholeComputeKernelConfig{
        .math_fidelity = p.math_fidelity,
        .math_approx_mode = false,
        .fp32_dest_acc_en = p.fp32_dest_acc_en,
        .packer_l1_acc = p.packer_l1_acc,
        .dst_full_sync_en = p.dst_full_sync_en};
    attributes.untilize_out = p.untilize_out;
    attributes.user_fused_activation = p.activation;
    attributes.transpose_a = p.transpose_a;
    attributes.output_tile = tt::tt_metal::Tile({p.out_tile_h, p.out_tile_w});
    specs.device.arch = hw.arch;
    specs.device.grid = CoreCoord(hw.origin.x + hw.grid.x, hw.origin.y + hw.grid.y);
    if (hw.pinned_origin) {
        attributes.sub_device_id = tt::tt_metal::SubDeviceId{0};
        specs.device.has_sub_devices = true;
        specs.device.sub_device_workers =
            CoreRangeSet(CoreRange(hw.origin, CoreCoord(hw.origin.x + hw.grid.x - 1, hw.origin.y + hw.grid.y - 1)));
    }
    std::string why;
    const auto described = describe_matmul(specs, why);
    EXPECT_TRUE(described.has_value()) << why;
    if (described) {
        EXPECT_EQ(desc_mismatch(*described, p), "") << "the test's specs don't describe its MatmulDesc";
    }
    return specs;
}

// The selector's choice, config and legality check for the matmul `p` describes
std::optional<Candidate> choose(
    const MatmulDesc& p, const HardwareDesc& hw, const Selector& selector = default_selector()) {
    return select(specs_of(p, hw), p, hw, selector);
}
std::optional<MatmulProgramConfig> config_for(const MatmulDesc& p, const HardwareDesc& hw) {
    return select_program_config(specs_of(p, hw), hw);
}
std::string check_config(const MatmulDesc& p, const HardwareDesc& hw, const MatmulProgramConfig& config) {
    return check(specs_of(p, hw), p, hw, config);
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
                        const auto p = make_matmul(s.batch_a, s.batch_b, s.M, s.K, s.N, in1, fp32_acc, bias);
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
                        const auto config = config_for(p, hw);
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
            const auto p = make_matmul(
                s.batch_a, s.batch_b, s.M, s.K, s.N, tt::DataFormat::Float16_b, false, false, /*transpose_a=*/true);
            const auto config = config_for(p, hw);
            ASSERT_TRUE(config.has_value());
            EXPECT_EQ(check_config(p, hw, *config), "") << arch.name << " M=" << s.M << " K=" << s.K << " N=" << s.N;
        }
    }
}

TEST(MatmulAutoConfig, BatchedBUsesReuse) {
    const auto hw = HardwareDesc::for_arch(tt::ARCH::WORMHOLE_B0, CoreCoord(8, 8), kL1Budget);
    const auto config = config_for(make_matmul(384, 384, 256, 256, 64), hw);
    ASSERT_TRUE(config.has_value());
    EXPECT_TRUE(std::holds_alternative<MatmulMultiCoreReuseProgramConfig>(*config));
}

TEST(MatmulAutoConfig, UsesWholeGridForLargeMatmul) {
    for (const auto& arch : kArchs) {
        const auto hw = HardwareDesc::for_arch(arch.arch, arch.grid, kL1Budget);
        const auto config = config_for(make_matmul(1, 1, 4096, 4096, 4096), hw);
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
        auto base = make_matmul(1, 1, M, K, N, tt::DataFormat::Bfp8_b);
        const auto reference = choose(base, hw);
        ASSERT_TRUE(reference.has_value());
        for (auto out : {tt::DataFormat::Bfp8_b, tt::DataFormat::Bfp4_b}) {
            for (bool l1_acc : {true, false}) {
                auto p = base;
                p.out_format = out;
                p.packer_l1_acc = l1_acc;
                const auto chosen = choose(p, hw);
                ASSERT_TRUE(chosen.has_value());
                EXPECT_EQ(chosen->blocking.in0_block_w, reference->blocking.in0_block_w) << M << "x" << K << "x" << N;
            }
        }
    }
}

// Subblocks at least two tiles on each side, unless B's tiles are smaller than A's
TEST(MatmulAutoConfig, SubblockShape) {
    const auto hw = HardwareDesc::for_arch(tt::ARCH::WORMHOLE_B0, CoreCoord(8, 8), kL1Budget);
    auto p = make_matmul(1, 1, 8192, 8192, 8192);
    auto chosen = choose(p, hw);
    ASSERT_TRUE(chosen.has_value());
    EXPECT_GE(std::min(chosen->blocking.out_subblock_h, chosen->blocking.out_subblock_w), 2u);
    p = make_matmul(1, 1, 8192, 8192, 8192, tt::DataFormat::Bfp8_b);  // bf16 A, bfp8 B
    chosen = choose(p, hw);
    ASSERT_TRUE(chosen.has_value());
    EXPECT_EQ(chosen->blocking.out_subblock_h, 1u);
    EXPECT_EQ(chosen->blocking.out_subblock_w, 8u);
}

// 1D in0-mcast splits a wide output block into subblock-wide blocks (not into 1-tile ones)
TEST(MatmulAutoConfig, OneDOutputBlockSplit) {
    const auto hw = HardwareDesc::for_arch(tt::ARCH::WORMHOLE_B0, CoreCoord(8, 8), kL1Budget);
    auto p = make_matmul(1, 1, 32, 2560, 262144);
    auto chosen = choose(p, hw);
    ASSERT_TRUE(chosen.has_value());
    EXPECT_EQ(static_cast<int>(chosen->family), static_cast<int>(Family::Mcast1DIn0));
    EXPECT_EQ(chosen->blocking.per_core_N, 128u);
    EXPECT_EQ(chosen->blocking.out_block_w, 8u);
    p = make_matmul(1, 1, 32, 4544, 11 * 32 * 64);  // per_core_N = 11 has no divisor in 2..8
    chosen = choose(p, hw);
    ASSERT_TRUE(chosen.has_value());
    EXPECT_EQ(chosen->blocking.out_block_w, chosen->blocking.per_core_N);
}

// Large 2D output blocks may use K blocks up to 16 deep; small ones stay at 8
TEST(MatmulAutoConfig, CostlyKBlocksExamples) {
    // Llama-70B TP8 w1 prefill (bf16 x bfp4, LoFi), with the L1 budget the device reported: block-float B, so 2D
    // K blocks are costly and K is split into at most max_costly_k_blocks blocks where L1 allows
    auto p = make_matmul(1, 1, 2048, 8192, 3584, tt::DataFormat::Bfp4_b);
    p.math_fidelity = MathFidelity::LoFi;
    auto hw = HardwareDesc::for_arch(tt::ARCH::WORMHOLE_B0, CoreCoord(8, 8), 1377056);
    auto chosen = choose(p, hw);
    ASSERT_TRUE(chosen.has_value());
    EXPECT_EQ(static_cast<int>(chosen->family), static_cast<int>(Family::Mcast2D));
    // 256 x 4096 x 1024 bfp8: 1x4 blocks, where max_in0_block_w alone would give 8; Kt = 128 in at most 8 blocks
    p = make_matmul(1, 1, 256, 4096, 1024, tt::DataFormat::Bfp8_b);
    hw = HardwareDesc::for_arch(tt::ARCH::WORMHOLE_B0, CoreCoord(8, 8), kL1Budget);
    chosen = choose(p, hw);
    ASSERT_TRUE(chosen.has_value());
    EXPECT_EQ(static_cast<int>(chosen->family), static_cast<int>(Family::Mcast2D));
    EXPECT_EQ(chosen->blocking.in0_block_w, 16u);
}

// K depth goes beyond max_in0_block_w only for interleaved 2D whose K blocks are costly (a block-float input, or
// packer L1 accumulation off with a block at least 2 tiles on each side), and then no deeper than Kt split into
// max_costly_k_blocks blocks
TEST(MatmulAutoConfig, TwoDCostlyKBlocks) {
    const auto tuned = HeuristicBlocking::Params{}.tuned;
    for (const auto& arch : kArchs) {
        const auto hw = HardwareDesc::for_arch(arch.arch, arch.grid, kL1Budget);
        for (const auto& s : shapes()) {
            for (auto in1 : {tt::DataFormat::Float16_b, tt::DataFormat::Bfp8_b}) {
                for (bool l1_acc : {false, true}) {
                    auto p = make_matmul(s.batch_a, s.batch_b, s.M, s.K, s.N, in1);
                    p.packer_l1_acc = l1_acc;
                    for (const auto& c : candidates(p, hw)) {
                        const auto& b = c.blocking;
                        const bool costly =
                            c.family == Family::Mcast2D &&
                            (in1 == tt::DataFormat::Bfp8_b || (!l1_acc && std::min(b.out_block_h, b.out_block_w) >= 2));
                        const uint32_t cap =
                            costly ? std::max(tuned.max_in0_block_w, div_up(p.Kt, tuned.max_costly_k_blocks))
                                   : tuned.max_in0_block_w;
                        EXPECT_LE(b.in0_block_w, cap) << arch.name << " M=" << s.M << " K=" << s.K << " N=" << s.N
                                                      << " in1=" << static_cast<int>(in1) << " l1_acc=" << l1_acc;
                    }
                }
            }
        }
    }
}

// When keeping the full multicast extent only fits with single-tile K steps, 1D shrinks it instead
TEST(MatmulAutoConfig, OneDAvoidsSingleTileK) {
    // 1024x1024x16384 bf16 x bfp8 with an L1 output, at the L1 budget the device reported
    auto p = make_matmul(1, 1, 1024, 1024, 16384, tt::DataFormat::Bfp8_b);
    p.out.in_l1 = true;
    const auto hw = HardwareDesc::for_arch(tt::ARCH::WORMHOLE_B0, CoreCoord(8, 8), 820000);
    const auto chosen = choose(p, hw);
    ASSERT_TRUE(chosen.has_value());
    EXPECT_EQ(static_cast<int>(chosen->family), static_cast<int>(Family::Mcast1DIn0));
    EXPECT_GE(chosen->blocking.in0_block_w, 2u);
    EXPECT_LT(chosen->blocking.out_block_h, chosen->blocking.per_core_M);
}

Placement sharded(MemoryLayout layout, CoreCoord grid, uint32_t shard_h, uint32_t shard_w, bool col_major = false) {
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

Placement sharded_output(MemoryLayout layout) {
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
        MatmulDesc p;
        Family family;
        uint32_t per_core_M, per_core_N;
        bool transpose_mcast = false;
    };
    std::vector<Case> cases;
    {  // decode: width-sharded activation, 1D in0-mcast, each core a slice of N
        auto p = make_matmul(1, 1, 32, 4096, 4096, tt::DataFormat::Bfp8_b);
        p.a = sharded(MemoryLayout::WidthSharded, CoreCoord(8, 8), 1, 2);
        cases.push_back({"width A", p, Family::Mcast1DIn0, 1, 2});
        p.out = sharded_output(MemoryLayout::WidthSharded);
        cases.push_back({"width A, width out", p, Family::Mcast1DIn0, 1, 2});
    }
    {  // tall: height-sharded activation, 1D in1-mcast
        auto p = make_matmul(1, 1, 8192, 256, 256);
        p.a = sharded(MemoryLayout::HeightSharded, CoreCoord(8, 8), 4, 8);
        cases.push_back({"height A", p, Family::Mcast1DIn1, 4, 8});
        p.out = sharded_output(MemoryLayout::HeightSharded);
        cases.push_back({"height A, height out", p, Family::Mcast1DIn1, 4, 8});
    }
    {  // block-sharded 2D, row- and column-major
        auto p = make_matmul(1, 1, 2048, 2048, 2048);
        p.a = sharded(MemoryLayout::BlockSharded, CoreCoord(8, 8), 8, 8);
        cases.push_back({"block A", p, Family::Mcast2D, 8, 8});
        p.out = sharded_output(MemoryLayout::BlockSharded);
        cases.push_back({"block A, block out", p, Family::Mcast2D, 8, 8});
        p.a.col_major = true;
        cases.push_back({"block A col-major", p, Family::Mcast2D, 8, 8, true});
    }
    {  // batched B with height-sharded A: Reuse over A's shards
        auto p = make_matmul(48, 48, 256, 256, 64);
        p.a = sharded(MemoryLayout::HeightSharded, CoreCoord(8, 6), 8, 8);
        cases.push_back({"height A, batched B", p, Family::Reuse, 8, 2});
    }
    {  // interleaved inputs, sharded output
        auto p = make_matmul(1, 1, 8192, 512, 512);
        p.out = sharded_output(MemoryLayout::HeightSharded);
        cases.push_back({"height out", p, Family::Mcast1DIn1, 4, 16});
        p = make_matmul(1, 1, 32, 4096, 8192, tt::DataFormat::Bfp8_b);
        p.out = sharded_output(MemoryLayout::WidthSharded);
        cases.push_back({"width out", p, Family::Mcast1DIn0, 1, 4});
        p = make_matmul(1, 1, 2048, 2048, 2048);
        p.out = sharded_output(MemoryLayout::BlockSharded);
        cases.push_back({"block out", p, Family::Mcast2D, 8, 8});
        // an output shard spec fixes the grid: 4x2 cores of 16x32 tiles
        p.out = sharded(MemoryLayout::BlockSharded, CoreCoord(4, 2), 32, 16);
        cases.push_back({"block out with spec", p, Family::Mcast2D, 32, 16});
        p = make_matmul(1, 1, 256, 2048, 2048);
        p.out = sharded(MemoryLayout::BlockSharded, CoreCoord(8, 1), 8, 8);
        cases.push_back({"block out on a row", p, Family::Mcast1DIn0, 8, 8});
        p = make_matmul(1, 1, 4096, 512, 512);
        p.out = sharded(MemoryLayout::HeightSharded, CoreCoord(8, 4), 4, 16);
        cases.push_back({"height out with spec", p, Family::Mcast1DIn1, 4, 16});
    }
    for (const auto& c : cases) {
        const auto chosen = choose(c.p, hw);
        ASSERT_TRUE(chosen.has_value()) << c.name;
        const auto& b = chosen->blocking;
        EXPECT_EQ(static_cast<int>(chosen->family), static_cast<int>(c.family)) << c.name;
        EXPECT_EQ(b.per_core_M, c.per_core_M) << c.name;
        EXPECT_EQ(b.per_core_N, c.per_core_N) << c.name;
        EXPECT_EQ(chosen->transpose_mcast, c.transpose_mcast) << c.name;
        EXPECT_EQ(c.p.Kt % b.in0_block_w, 0u) << c.name;
        if (c.p.a.sharded() && c.p.a.layout != MemoryLayout::HeightSharded) {
            EXPECT_EQ(b.in0_block_w, c.p.a.shard_w) << c.name << ": K blocks should be whole shard columns";
        }
        if (c.p.a.layout == MemoryLayout::HeightSharded) {
            EXPECT_EQ(b.in0_block_w, c.p.Kt) << c.name << ": height-sharded A is read in place over all of K";
        }
        if (c.p.out.sharded() && c.family != Family::Reuse) {
            EXPECT_EQ(b.out_block_w, b.per_core_N) << c.name << ": sharded output blocks span per_core_N";
            EXPECT_TRUE(b.out_subblock_w == b.per_core_N || b.out_subblock_h == 1) << c.name;
        }
        EXPECT_LE(circular_buffer_bytes(c.p, hw, chosen->family, b, chosen->fuse_batch), hw.l1_cb_budget) << c.name;
    }

    // Layout combinations the factories reject are not produced
    auto p = make_matmul(1, 1, 2048, 2048, 2048);
    p.a = sharded(MemoryLayout::BlockSharded, CoreCoord(8, 8), 8, 8);
    p.out = sharded_output(MemoryLayout::HeightSharded);
    EXPECT_FALSE(choose(p, hw).has_value()) << "sharded output must be laid out like A";
}

// Family choices of the heuristics, from the Wormhole config sweep (tests/ttnn/unit_tests/benchmarks/matmul_oob)
TEST(MatmulAutoConfig, FamilyChoice) {
    const auto hw = HardwareDesc::for_arch(tt::ARCH::WORMHOLE_B0, CoreCoord(8, 8), kL1Budget);
    struct Expected {
        Shape shape;
        Family family;
        tt::DataFormat in1 = tt::DataFormat::Float16_b;
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
        {{4, 4, 256, 2048, 7168}, Family::Mcast2D, tt::DataFormat::Bfp4_b},  // #31743: batched, wide N, looped
    };
    for (const auto& e : expected) {
        const auto& s = e.shape;
        const auto chosen = choose(make_matmul(s.batch_a, s.batch_b, s.M, s.K, s.N, e.in1), hw);
        ASSERT_TRUE(chosen.has_value());
        EXPECT_EQ(static_cast<int>(chosen->family), static_cast<int>(e.family))
            << "b=" << s.batch_a << "/" << s.batch_b << " M=" << s.M << " K=" << s.K << " N=" << s.N;
    }
}

// Reuse with A tiles under 16 rows and block-float B takes all of K in one block, so every slice of a batch matrix
// loads all of its B: the slice height comes from the roofline (which counts those reads), not from filling the grid.
// 6 batches of 32 8-row tile rows in L1 on Wormhole: slices of 4 rows (48 B reads), not the 2 that fill the grid (96).
TEST(MatmulAutoConfig, SingleKBlockReuseSlices) {
    const auto hw = HardwareDesc::for_arch(tt::ARCH::WORMHOLE_B0, CoreCoord(8, 8), kL1Budget);
    auto p = make_matmul(6, 6, 256, 256, 256, tt::DataFormat::Bfp8_b);
    p.in0_tile_h = p.out_tile_h = 8;
    p.Mt = 256 / 8;
    p.a.in_l1 = p.b.in_l1 = p.out.in_l1 = true;  // as test_matmul_reuse_config_sharded_tiny_tile's interleaved cases
    const auto chosen = choose(p, hw);
    ASSERT_TRUE(chosen.has_value());
    EXPECT_EQ(static_cast<int>(chosen->family), static_cast<int>(Family::Reuse));
    EXPECT_EQ(chosen->blocking.in0_block_w, p.Kt);
    EXPECT_EQ(chosen->blocking.per_core_M, 4u);
}

// A bias of a whole [M, N] block fuses only into Reuse with blocks of whole batch matrices: with one, the
// candidates are that Reuse layout alone (without it, 8 batches would go to 2D and the bias to a second pass)
TEST(MatmulAutoConfig, FullBlockBiasFusesIntoReuse) {
    for (const auto& arch : kArchs) {
        const auto hw = HardwareDesc::for_arch(arch.arch, arch.grid, kL1Budget);
        for (uint32_t batch : {8u, 24u}) {
            auto p = make_matmul(batch, batch, 128, 128, 128);
            p.bias_tile_bytes = tt::tile_size(tt::DataFormat::Float16_b);
            p.bias_rows = p.Mt;
            const auto chosen = choose(p, hw);
            ASSERT_TRUE(chosen.has_value()) << arch.name << " batch " << batch;
            EXPECT_EQ(static_cast<int>(chosen->family), static_cast<int>(Family::Reuse)) << arch.name;
            EXPECT_EQ(chosen->blocking.per_core_M, p.Mt) << arch.name << " batch " << batch;
        }
    }
}

// An output tile wider than B's spans several B tiles: with a sharded output (and no shard spec) each core's
// columns fill whole output tiles; other outputs stay unsupported
TEST(MatmulAutoConfig, OutputTileWiderThanB) {
    for (const auto& arch : kArchs) {
        const auto hw = HardwareDesc::for_arch(arch.arch, arch.grid, kL1Budget);
        for (uint32_t tile_h : {1u, 16u}) {
            for (auto layout : {MemoryLayout::WidthSharded, MemoryLayout::BlockSharded}) {
                auto p = make_matmul(1, 1, 512, 512, 768);
                p.in0_tile_h = p.out_tile_h = tile_h;
                p.in1_tile_w = 16;
                p.out_tile_w = 32;
                p.Mt = 512 / tile_h;
                p.Nt = 768 / 16;
                p.out = sharded_output(layout);
                const auto label = fmt::format("{} tile_h {} layout {}", arch.name, tile_h, static_cast<int>(layout));
                EXPECT_TRUE(config_for(p, hw).has_value()) << label;
                const auto chosen = choose(p, hw);
                ASSERT_TRUE(chosen.has_value()) << label;
                EXPECT_EQ(chosen->blocking.per_core_N % 2, 0u) << label;
                p.out = Placement{};  // interleaved
                EXPECT_FALSE(config_for(p, hw).has_value()) << label;
            }
        }
    }
}

// Where the roofline picks the family (2D blocks one tile tall or wide), a candidate that another keeps
// one_d_core_advantage times as many cores busy as is out: here 2D loops 12 small batch matrices over 9 cores, Reuse
// runs them at once on 36 (the roofline alone, with no per-step latency, picks 2D on Blackhole: 4x slower measured)
TEST(MatmulAutoConfig, OneTileTwoDKeepsCoresBusy) {
    for (const auto& arch : kArchs) {
        const auto hw = HardwareDesc::for_arch(arch.arch, arch.grid, kL1Budget);
        const auto chosen = choose(make_matmul(12, 12, 96, 64, 96), hw);
        ASSERT_TRUE(chosen.has_value());
        EXPECT_EQ(static_cast<int>(chosen->family), static_cast<int>(Family::Reuse)) << arch.name;
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
                        auto p = make_matmul(s.batch_a, s.batch_b, s.M, s.K, s.N, in1);
                        p.in0_tile_h = p.out_tile_h = tile_h;
                        p.in1_tile_w = p.out_tile_w = tile_w;
                        p.Mt = s.M / tile_h;
                        p.Nt = s.N / tile_w;
                        const auto label = fmt::format(
                            "{} tile {}x{} in1={} M={}", arch.name, tile_h, tile_w, static_cast<int>(in1), s.M);
                        const auto chosen = choose(p, hw);
                        const bool reuse_only = in1 == tt::DataFormat::Bfp8_b && tile_h < 16;
                        if (!chosen.has_value()) {
                            // Only possible when no factory can run it: Reuse is the only one, and its single K
                            // block of B (all of K by all of N) doesn't fit L1
                            Blocking one_row{1, p.Nt, p.Kt, 1, p.Nt, 1, 1};
                            EXPECT_TRUE(
                                reuse_only &&
                                circular_buffer_bytes(p, hw, Family::Reuse, one_row, true) > hw.l1_cb_budget)
                                << label;
                            continue;
                        }
                        const auto& b = chosen->blocking;
                        EXPECT_LE(b.out_subblock_h * b.out_subblock_w, 8u) << label;
                        EXPECT_LE(circular_buffer_bytes(p, hw, chosen->family, b, chosen->fuse_batch), hw.l1_cb_budget)
                            << label;
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
        const auto p = make_matmul(s.batch_a, s.batch_b, s.M, s.K, s.N);
        const auto chosen = choose(p, hw);
        ASSERT_TRUE(chosen.has_value()) << s.M;
        // Only 1D in1-mcast reuses a single A across B's batches, looping over them
        EXPECT_EQ(static_cast<int>(chosen->family), static_cast<int>(Family::Mcast1DIn1)) << s.M;
        EXPECT_FALSE(chosen->fuse_batch) << s.M;
        EXPECT_EQ(chosen->blocking.per_core_N, p.Nt) << s.M;
        // A's rows stay resident across the batch loop
        EXPECT_GE(
            circular_buffer_bytes(p, hw, Family::Mcast1DIn1, chosen->blocking, chosen->fuse_batch),
            chosen->blocking.per_core_M * p.Kt * tt::tile_size(p.in0_format))
            << s.M;
        const auto config = config_for(p, hw);
        ASSERT_TRUE(config.has_value());
        EXPECT_FALSE(std::get<MatmulMultiCoreReuseMultiCast1DProgramConfig>(*config).fused_activation.has_value());
    }
}

TEST(MatmulAutoConfig, TransposeAOverBatchIsNotFused) {
    const auto hw = HardwareDesc::for_arch(tt::ARCH::WORMHOLE_B0, CoreCoord(8, 8), kL1Budget);
    const auto p = make_matmul(8, 1, 512, 256, 512, tt::DataFormat::Float16_b, false, false, /*transpose_a=*/true);
    const auto chosen = choose(p, hw);
    ASSERT_TRUE(chosen.has_value());
    EXPECT_NE(static_cast<int>(chosen->family), static_cast<int>(Family::Reuse));  // Reuse can't broadcast B
    EXPECT_FALSE(chosen->fuse_batch);
}

TEST(MatmulAutoConfig, NoOneDWhenExcluded) {
    const auto hw = HardwareDesc::for_arch(tt::ARCH::WORMHOLE_B0, CoreCoord(8, 8), kL1Budget);
    auto p = make_matmul(1, 1, 32, 4096, 12288);  // decode: 1D in0-mcast otherwise
    // B width-sharded over the 12 DRAM banks: only 2D reads it in place (a global CB rules 1D out the same way)
    p.b.layout = MemoryLayout::WidthSharded;
    p.b.has_shard_spec = true;
    p.b.shard_grid = CoreRange({0, 0}, {11, 0});
    p.b.shard_cores = 12;
    p.b.shard_h = p.Kt;
    p.b.shard_w = p.Nt / 12;
    for (const auto& c : candidates(p, hw)) {
        EXPECT_TRUE(c.family == Family::Mcast2D || c.family == Family::Reuse);
    }
    EXPECT_TRUE(choose(p, hw).has_value());
}

TEST(MatmulAutoConfig, SubDeviceGrid) {
    auto hw = HardwareDesc::for_arch(tt::ARCH::WORMHOLE_B0, CoreCoord(8, 7), kL1Budget);
    hw.origin = CoreCoord(0, 1);
    hw.pinned_origin = true;
    for (const auto& s : std::vector<Shape>{{1, 1, 128, 512, 512}, {1, 1, 32, 4096, 4096}, {48, 48, 256, 256, 64}}) {
        const auto config = config_for(make_matmul(s.batch_a, s.batch_b, s.M, s.K, s.N), hw);
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
        auto p = make_matmul(1, 1, 4096, 32, 128);
        p.a = sharded(MemoryLayout::BlockSharded, CoreCoord(8, 1), 16, 1, /*col_major=*/true);
        const auto chosen = choose(p, hw);
        ASSERT_TRUE(chosen.has_value());
        EXPECT_EQ(static_cast<int>(chosen->family), static_cast<int>(Family::Mcast2D));
        EXPECT_TRUE(chosen->transpose_mcast);
        EXPECT_EQ(chosen->blocking.per_core_M, 16u);
        EXPECT_EQ(chosen->blocking.per_core_N, 4u);
    }
    {  // one-core block shard spec for a 5-batch output: keep the shard shape, derive the grid
        auto p = make_matmul(5, 1, 416, 32, 416);
        p.out = sharded(MemoryLayout::BlockSharded, CoreCoord(1, 1), 13, 13);
        const auto chosen = choose(p, hw);
        ASSERT_TRUE(chosen.has_value());
        EXPECT_EQ(static_cast<int>(chosen->family), static_cast<int>(Family::Mcast2D));
        EXPECT_EQ(chosen->blocking.per_core_M, 13u);
        EXPECT_EQ(chosen->blocking.per_core_N, 13u);
    }
}

// Every candidate the heuristics produce for interleaved problems is legal
TEST(MatmulAutoConfig, CandidatesPassCheck) {
    for (const auto& arch : kArchs) {
        const auto hw = HardwareDesc::for_arch(arch.arch, arch.grid, kL1Budget);
        for (const auto& s : shapes()) {
            for (auto in1 : {tt::DataFormat::Float16_b, tt::DataFormat::Bfp8_b}) {
                for (bool fp32_acc : {false, true}) {
                    const auto p = make_matmul(s.batch_a, s.batch_b, s.M, s.K, s.N, in1, fp32_acc);
                    const auto label = fmt::format(
                        "{} b={}/{} M={} K={} N={} in1={} fp32={}",
                        arch.name,
                        s.batch_a,
                        s.batch_b,
                        s.M,
                        s.K,
                        s.N,
                        static_cast<int>(in1),
                        fp32_acc);
                    for (const auto& c : candidates(p, hw)) {
                        EXPECT_EQ(check_config(p, hw, to_program_config(p, c)), "") << label;
                    }
                }
            }
        }
    }
}

// Estimators rank the sources' proposals: with several, the most confident estimate picks, and a less confident
// one doesn't override it. Here a source proposes the heuristic choice and the same candidate at half its K depth.
TEST(MatmulAutoConfig, EstimatorsRankProposals) {
    struct HalfKToo final : CandidateSource {
        std::string_view name() const override { return "half_k_too"; }
        std::vector<Candidate> propose(const MatmulDesc& p, const HardwareDesc& hw) const override {
            auto result = FactoryBlockingSource().propose(p, hw);
            if (!result.empty() && result.front().blocking.in0_block_w % 2 == 0) {
                auto half = result.front();
                half.blocking.in0_block_w /= 2;
                result.push_back(half);
            }
            return result;
        }
    };
    struct PreferShallow final : Estimator {
        double confidence;
        explicit PreferShallow(double c) : confidence(c) {}
        std::string_view name() const override { return "prefer_shallow"; }
        std::optional<Estimate> estimate(const MatmulDesc&, const HardwareDesc&, const Candidate& c) const override {
            return Estimate{.cycles = double(c.blocking.in0_block_w), .confidence = confidence, .source = name()};
        }
    };
    const auto hw = HardwareDesc::for_arch(tt::ARCH::WORMHOLE_B0, CoreCoord(8, 8), kL1Budget);
    const auto p = make_matmul(1, 1, 1024, 8192, 1024);
    const auto seed = choose(p, hw, heuristics_only());
    ASSERT_TRUE(seed.has_value());
    ASSERT_EQ(seed->blocking.in0_block_w % 2, 0u);

    const auto source = std::make_shared<HalfKToo>();
    const auto roofline_estimator = std::make_shared<RooflineEstimator>();
    // The roofline doesn't depend on K depth: ties keep the earlier proposal
    const auto by_roofline = choose(p, hw, Selector{.sources = {source}, .estimators = {roofline_estimator}});
    ASSERT_TRUE(by_roofline.has_value());
    EXPECT_EQ(by_roofline->blocking.in0_block_w, seed->blocking.in0_block_w);

    const auto refined = choose(
        p, hw, Selector{.sources = {source}, .estimators = {roofline_estimator, std::make_shared<PreferShallow>(1.0)}});
    ASSERT_TRUE(refined.has_value());
    EXPECT_EQ(refined->blocking.in0_block_w, seed->blocking.in0_block_w / 2);
    const auto kept = choose(
        p,
        hw,
        Selector{.sources = {source}, .estimators = {roofline_estimator, std::make_shared<PreferShallow>(-1.0)}});
    ASSERT_TRUE(kept.has_value());
    EXPECT_EQ(kept->blocking.in0_block_w, seed->blocking.in0_block_w);
}

// A source's policies can be replaced one at a time: here the family choice, the blocking rules unchanged
TEST(MatmulAutoConfig, FamilyPolicyIsReplaceable) {
    struct LastFamily final : FamilyPolicy {
        std::optional<Candidate> choose(
            const MatmulDesc&, const HardwareDesc&, std::span<const Candidate> all) const override {
            return all.empty() ? std::nullopt : std::optional<Candidate>(all.back());
        }
    };
    const auto hw = HardwareDesc::for_arch(tt::ARCH::WORMHOLE_B0, CoreCoord(8, 8), kL1Budget);
    const auto p = make_matmul(1, 1, 1024, 2048, 1024);
    const auto all = candidates(p, hw);
    ASSERT_GE(all.size(), 2u);
    const Selector last_family{
        .sources = {std::make_shared<FactoryBlockingSource>(
            std::make_shared<HeuristicBlocking>(),
            std::make_shared<HeuristicSubblock>(),
            std::make_shared<LastFamily>())},
        .estimators = {}};
    const auto chosen = choose(p, hw, last_family);
    ASSERT_TRUE(chosen.has_value());
    EXPECT_EQ(chosen->family, all.back().family);
    EXPECT_NE(choose(p, hw)->family, all.back().family);
    EXPECT_TRUE(check_config(p, hw, to_program_config(p, *chosen)).empty());
}

// Prints every family's candidate for the test shapes; run with --gtest_also_run_disabled_tests when tuning.
TEST(MatmulAutoConfig, DISABLED_PrintSelections) {
    const auto hw = HardwareDesc::for_arch(tt::ARCH::WORMHOLE_B0, CoreCoord(8, 8), 1450 * 1024);
    const char* names[] = {"2D", "1D-in0", "1D-in1", "Reuse"};
    for (const auto& s : shapes()) {
        const auto p = make_matmul(s.batch_a, s.batch_b, s.M, s.K, s.N);
        const auto chosen = choose(p, hw);
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

// Block-float B with A tiles under 16 rows only runs on Reuse with a single K block
// check() applies the device op's validation: a config it rejects is rejected with its message
TEST(MatmulAutoConfig, CheckUsesDeviceValidation) {
    const auto hw = HardwareDesc::for_arch(tt::ARCH::WORMHOLE_B0, CoreCoord(8, 8), kL1Budget);
    const auto p = make_matmul(1, 1, 1024, 2048, 1024);
    const auto chosen = choose(p, hw);
    ASSERT_TRUE(chosen.has_value());
    EXPECT_EQ(check_config(p, hw, to_program_config(p, *chosen)), "");
    auto uneven = *chosen;
    uneven.blocking.in0_block_w = 3;  // doesn't divide Kt = 64
    EXPECT_NE(
        check_config(p, hw, to_program_config(p, uneven)).find("must be divisible by in0_block_w"), std::string::npos);
}

TEST(MatmulAutoConfig, CheckTinyTileBlockFloatB) {
    const auto hw = HardwareDesc::for_arch(tt::ARCH::WORMHOLE_B0, CoreCoord(8, 8), kL1Budget);
    auto p = make_matmul(4, 4, 128, 256, 256, tt::DataFormat::Bfp8_b);
    p.in0_tile_h = p.out_tile_h = 8;
    p.Mt = 128 / 8;
    const auto chosen = choose(p, hw);
    ASSERT_TRUE(chosen.has_value());
    ASSERT_EQ(static_cast<int>(chosen->family), static_cast<int>(Family::Reuse));
    EXPECT_EQ(check_config(p, hw, to_program_config(p, *chosen)), "");
    auto split = *chosen;
    split.blocking.in0_block_w = p.Kt / 2;
    EXPECT_NE(check_config(p, hw, to_program_config(p, split)), "");
}
