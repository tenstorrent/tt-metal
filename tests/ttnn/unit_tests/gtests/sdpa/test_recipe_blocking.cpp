// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Op-selected blocking for the SDPA precision recipes: the host-only chooser on DiT-like shapes.

#include <algorithm>
#include <array>
#include <cmath>
#include <vector>

#include <gtest/gtest.h>

#include "ttnn/operations/transformer/sdpa/sdpa_recipe_blocking.hpp"

namespace {
using namespace ttnn::operations::transformer::sdpa::detail;
using ttnn::CoreCoord;

constexpr uint64_t kDeviceL1 = 1'461'248;    // P150b CB budget per core at the default worker L1
constexpr uint64_t kPipelineL1 = 1'344'512;  // at worker_l1_size=1344544 (ring pipelines)
const CoreCoord kGrid{11, 10};
const CoreCoord kRingWorkerGrid{10, 10};  // last column reserved for the CCL

const std::array kSelections{
    RecipeSelection{Recipe::B},
    RecipeSelection{Recipe::C},
    RecipeSelection{Recipe::D},
    RecipeSelection{Recipe::E, KVStorage::BF16},
    RecipeSelection{Recipe::E, KVStorage::BFP8},
    RecipeSelection{Recipe::E, KVStorage::BFP4}};

RecipeBlockingProblem problem(
    RecipeOp op, RecipeSelection selection, uint32_t heads, uint32_t q_rows, uint32_t k_rows, uint32_t d_tiles = 4) {
    return {
        .op = op,
        .policy = resolve_precision_policy(selection),
        .q_heads = heads,
        .q_rows = q_rows,
        .k_rows = k_rows,
        .d_tiles = d_tiles,
        .grid = op == RecipeOp::Ring ? kRingWorkerGrid : kGrid,
        .l1_bytes = kDeviceL1};
}

uint32_t tiles(uint32_t chunk) { return chunk / 32; }

void expect_valid(const RecipeBlockingProblem& p, const RecipeBlocking& choice) {
    EXPECT_TRUE(recipe_geometry_supported(p.op, p.policy, tiles(choice.q_chunk_size), tiles(choice.k_chunk_size), p.d_tiles));
    EXPECT_LE(choice.l1.minimum, p.l1_bytes);
}

TEST(SDPARecipeBlocking, DenseAndJointChoicesFitAndAreNearCheapest) {
    struct Shape {
        RecipeOp op;
        uint32_t heads, q_rows, k_rows, joint, d_tiles;
    };
    const std::array shapes{
        Shape{RecipeOp::Joint, 24, 4096, 4096, 512, 4},  // FLUX.1 1024px
        Shape{RecipeOp::Dense, 40, 16384, 512, 0, 4},    // Wan 480p cross attention
        Shape{RecipeOp::Dense, 40, 32768, 32768, 0, 4},  // Wan 480p self attention
        Shape{RecipeOp::Dense, 32, 1024, 1024, 0, 2},    // LTX audio, D64
        Shape{RecipeOp::Dense, 8, 4096, 4096, 0, 8},     // Ideogram4, D256
        Shape{RecipeOp::Joint, 8, 257, 257, 77, 4},      // short prompt
    };
    for (const auto& selection : kSelections) {
        for (const auto& s : shapes) {
            auto p = problem(s.op, selection, s.heads, s.q_rows, s.k_rows, s.d_tiles);
            p.joint_q_rows = p.joint_k_rows = s.joint;
            const auto choice = choose_recipe_blocking(p);
            ASSERT_TRUE(choice.has_value());
            expect_valid(p, *choice);
            const auto candidates = recipe_blocking_candidates(p);
            const double best = std::min_element(candidates.begin(), candidates.end(), [](auto& a, auto& b) {
                                    return a.cost < b.cost;
                                })->cost;
            EXPECT_LE(choice->cost, best * 1.05);
            // A K chunk longer than the padded keys only adds padding.
            EXPECT_LE(choice->k_chunk_size, std::max(512u, (s.k_rows + s.joint + 31) / 32 * 32));
        }
    }
}

TEST(SDPARecipeBlocking, ShortKCrossAttentionAvoidsLargeQChunks) {
    // One K block per Q chunk: Q fill/drain dominates, so the largest Q chunk is not the best.
    for (const auto& selection : kSelections) {
        for (uint32_t k_rows : {32u, 256u}) {
            const auto choice = choose_recipe_blocking(problem(RecipeOp::Dense, selection, 8, 4864, k_rows));
            ASSERT_TRUE(choice.has_value());
            EXPECT_LE(choice->q_chunk_size, 192u);
        }
    }
}

TEST(SDPARecipeBlocking, L1LimitsTheChoice) {
    // D256 BALANCED/ACCURATE: Q256/K512 does not fit; the choice must.
    for (auto recipe : {Recipe::C, Recipe::D}) {
        const auto p = problem(RecipeOp::Dense, {recipe}, 8, 4096, 4096, 8);
        EXPECT_GT(recipe_l1_bytes(RecipeOp::Dense, p.policy, 8, 16, 8).minimum, kDeviceL1);
        const auto choice = choose_recipe_blocking(p);
        ASSERT_TRUE(choice.has_value());
        expect_valid(p, *choice);
    }
}

TEST(SDPARecipeBlocking, LargeHeadDimsFindAFittingBlocking) {
    // D512 (Gemma-4 global layers, SD / FLUX.2 VAEs) and D1152 (Qwen-Image-2.1 VAE) do not fit L1 in the fitted
    // search range (Q128+ x K256+): the chooser extends down to one-tile chunks, dense and with a causal key range.
    // D512 fits every recipe; D1152 the FP32-state recipes (routed FP32-dest calls run ACCURATE).
    for (const auto& selection : kSelections) {
        const bool fp32 = selection.recipe == Recipe::C || selection.recipe == Recipe::D;
        for (const uint32_t d_tiles : fp32 ? std::vector{16u, 36u} : std::vector{16u}) {
            for (const bool causal : {false, true}) {
                auto p = problem(RecipeOp::Dense, selection, 1, 4096, 4096, d_tiles);
                if (causal) {
                    p.key_range = p.causal = true;
                    p.mask_page_bytes = 576;
                    p.extra_l1_bytes = 3072;
                }
                const auto choice = choose_recipe_blocking(p);
                ASSERT_TRUE(choice.has_value()) << "D" << d_tiles * 32 << " causal " << causal;
                expect_valid(p, *choice);
            }
        }
    }
}

TEST(SDPARecipeBlocking, MoreHeadsThanCoresSplitsJobsOverTheGrid) {
    // 8 x 24 batch/heads on 110 cores (an encoder batch): every head's Q chunks share the grid, and fewer Q
    // chunks per head never costs more.
    for (const auto& selection : kSelections) {
        auto p = problem(RecipeOp::Dense, selection, 24, 512, 512, 2);
        p.batch = 8;
        const auto choice = choose_recipe_blocking(p);
        ASSERT_TRUE(choice.has_value());
        expect_valid(p, *choice);
        EXPECT_EQ(
            choice->jobs_per_core,
            (p.batch * p.q_heads * ((512 + choice->q_chunk_size - 1) / choice->q_chunk_size) + 109) / 110);
    }
    // Without forwarding chains every Q chunk reads its head's K/V again: bge_m3 B8 16 heads D64 with a mask is fastest
    // at Q256/K512 (measured 0.455 ms, Q128/K512 0.488 ms), which the per-core roofline alone ranked below Q128.
    auto bge = problem(RecipeOp::Dense, {Recipe::D}, 16, 512, 512, 2);
    bge.batch = 8;
    bge.mask_page_bytes = 2048;
    const auto choice = choose_recipe_blocking(bge);
    ASSERT_TRUE(choice.has_value());
    EXPECT_EQ(choice->q_chunk_size, 256u);
    EXPECT_EQ(choice->k_chunk_size, 512u);
}

TEST(SDPARecipeBlocking, ExplicitChunksAreHonored) {
    auto p = problem(RecipeOp::Dense, {Recipe::B}, 10, 8192, 8064);
    p.fixed_q_tiles = 7;
    p.fixed_k_tiles = 12;
    auto choice = choose_recipe_blocking(p);
    ASSERT_TRUE(choice.has_value());
    EXPECT_EQ(choice->q_chunk_size, 224u);
    EXPECT_EQ(choice->k_chunk_size, 384u);
    // Unless the program would not fit the kernel config buffer (an odd STANDARD Q chunk with a K tail).
    p.k_rows = 8192;
    EXPECT_FALSE(choose_recipe_blocking(p).has_value());
    p.fixed_q_tiles = 0;
    choice = choose_recipe_blocking(p);
    ASSERT_TRUE(choice.has_value());
    EXPECT_EQ(choice->k_chunk_size, 384u);
}

TEST(SDPARecipeBlocking, KeyRangesCostOnlyTheirKChunks) {
    // Causal and sliding-window calls process only the K chunks their rows see, dealt over the whole grid. Measured
    // (one P150b, 8192 rows): causal 10 heads D128 is fastest at Q256/K512 for every recipe; a Gemma-style sliding
    // window 1024 with 16 heads D256 (ACCURATE 2.55 ms at the choice, 3.15 ms at the caller's Q128/K128) wants short
    // K chunks, where the dense model picked K256 (3.42 ms).
    for (auto recipe : {Recipe::B, Recipe::C, Recipe::D}) {
        auto causal = problem(RecipeOp::Dense, {recipe}, 10, 8192, 8192);
        causal.key_range = causal.causal = true;
        causal.mask_page_bytes = 576;
        const auto choice = choose_recipe_blocking(causal);
        ASSERT_TRUE(choice.has_value());
        expect_valid(causal, *choice);
        EXPECT_EQ(choice->q_chunk_size, 256u);
        EXPECT_EQ(choice->k_chunk_size, 512u);
    }
    auto window = problem(RecipeOp::Dense, {Recipe::D}, 16, 8192, 8192, 8);
    window.key_range = window.causal = true;
    window.sliding_window = 1024;
    window.mask_page_bytes = 576;
    const auto choice = choose_recipe_blocking(window);
    ASSERT_TRUE(choice.has_value());
    expect_valid(window, *choice);
    EXPECT_LE(choice->k_chunk_size, 192u);
    auto hint = window;
    hint.fixed_q_tiles = hint.fixed_k_tiles = 4;
    EXPECT_LT(choice->cost, choose_recipe_blocking(hint)->cost);

    // A causal call costs less than the dense one at the same blocking (measured STANDARD 1.49 vs 1.62 ms: the
    // dense call keeps its K/V chain).
    auto causal = problem(RecipeOp::Dense, {Recipe::B}, 10, 8192, 8192);
    causal.key_range = causal.causal = true;
    causal.mask_page_bytes = 576;
    auto dense = causal;
    dense.key_range = dense.causal = false;
    dense.fixed_q_tiles = causal.fixed_q_tiles = 8;
    dense.fixed_k_tiles = causal.fixed_k_tiles = 16;
    const double ratio = choose_recipe_blocking(causal)->cost / choose_recipe_blocking(dense)->cost;
    EXPECT_GT(ratio, 0.5);
    EXPECT_LT(ratio, 1.0);
}

TEST(SDPARecipeBlocking, AttnMaskCircularBufferIsBudgeted) {
    constexpr uint32_t page = 2048;  // BF16 mask tile
    for (const auto& selection : kSelections) {
        const auto policy = resolve_precision_policy(selection);
        const auto plain = recipe_l1_bytes(RecipeOp::Dense, policy, 8, 16, 4);
        const auto masked = recipe_l1_bytes(RecipeOp::Dense, policy, 8, 16, 4, {.mask_page_bytes = page});
        const uint64_t group = uint64_t{recipe_mask_group_rows(policy, 8)} * 16 * page;
        EXPECT_EQ(masked.minimum - plain.minimum, group);
        EXPECT_EQ(masked.preferred - plain.preferred, 2 * group);
        // A budget that only just holds the unmasked Q256/K512 layout: the masked choice must still fit.
        auto p = problem(RecipeOp::Dense, selection, 8, 4096, 4096);
        p.l1_bytes = plain.minimum + group / 2;
        p.mask_page_bytes = page;
        const auto choice = choose_recipe_blocking(p);
        ASSERT_TRUE(choice.has_value());
        EXPECT_LE(choice->l1.minimum, p.l1_bytes);
    }
}

TEST(SDPARecipeBlocking, RingChoicesFitBothWorkerL1Sizes) {
    struct Shape {
        uint32_t heads, local, ring;
    };
    for (const auto& selection : kSelections) {
        for (const auto& s : {Shape{10, 4096, 8}, Shape{10, 9472, 8}, Shape{20, 16384, 2}, Shape{14, 4768, 8}}) {
            for (uint64_t l1 : {kDeviceL1, kPipelineL1}) {
                auto p = problem(RecipeOp::Ring, selection, s.heads, s.local, s.local);
                p.ring_size = s.ring;
                p.l1_bytes = l1;
                const auto choice = choose_recipe_blocking(p);
                ASSERT_TRUE(choice.has_value());
                expect_valid(p, *choice);
                const uint32_t chunks = s.heads * ((s.local + choice->q_chunk_size - 1) / choice->q_chunk_size);
                const uint32_t cores = choice->grid.x * choice->grid.y;
                EXPECT_EQ(choice->jobs_per_core, (chunks + cores - 1) / cores);
            }
        }
    }
}

TEST(SDPARecipeBlocking, ExpRingChoicesUseAtMostThreePasses) {
    struct Shape {
        uint32_t heads, local, ring;
    };
    for (const auto& selection : kSelections) {
        for (const auto& s : {Shape{10, 1024, 32}, Shape{10, 2368, 32}, Shape{14, 1192, 32}, Shape{10, 4096, 2}}) {
            auto p = problem(RecipeOp::ExpRing, selection, s.heads, s.local, s.local);
            p.ring_size = s.ring;
            p.l1_bytes = kPipelineL1;
            const auto choice = choose_recipe_blocking(p);
            ASSERT_TRUE(choice.has_value());
            expect_valid(p, *choice);
            EXPECT_EQ(choice->grid.y, kGrid.y);
            const uint32_t columns = choice->grid.x - 1;  // one fabric MUX column
            const uint32_t chunks = (s.local + choice->q_chunk_size - 1) / choice->q_chunk_size;
            EXPECT_EQ(chunks % columns, 0u);
            EXPECT_LE(choice->jobs_per_core, 3u);
        }
    }
}

TEST(SDPARecipeBlocking, ProgramsFitTheKernelConfigBuffer) {
    // Measured (one P150b): an odd fused STANDARD Q chunk of 7+ tiles with BFP8 K/V overflows the 70656 B kernel
    // config buffer in a joint program (Q224/K256 71344 B, Q288/K256 71072 B) or with an attention sink (Q224 71120 B),
    // and comes within 0.6 KB of it with a K tail (Q224/K512 70048 B); the plain dense program fits (68608 B).
    const auto bfp8 = RecipeSelection{Recipe::B, KVStorage::BFP8};
    auto joint = problem(RecipeOp::Joint, bfp8, 1, 20512, 20512);  // test_sdpa_joint S20481 + 118 joint rows
    joint.joint_q_rows = joint.joint_k_rows = 128;
    joint.k_rows_unaligned = true;
    EXPECT_FALSE(recipe_program_fits(joint, 7, 8));
    EXPECT_FALSE(recipe_program_fits(joint, 9, 8));
    EXPECT_TRUE(recipe_program_fits(joint, 8, 8));
    auto dense = problem(RecipeOp::Dense, bfp8, 1, 20512, 16384);
    EXPECT_TRUE(recipe_program_fits(dense, 7, 16));
    dense.k_rows_unaligned = true;
    EXPECT_FALSE(recipe_program_fits(dense, 7, 16));
    dense.k_rows_unaligned = false;
    dense.k_rows = 16384 + 32;
    EXPECT_FALSE(recipe_program_fits(dense, 7, 16));
    dense.k_rows = 16384;
    dense.attention_sink = true;
    EXPECT_FALSE(recipe_program_fits(dense, 7, 16));
    // Odd chunks that STANDARD pads to even (attn_mask, key ranges) and the other recipes are unaffected.
    dense.mask_page_bytes = 2048;
    EXPECT_TRUE(recipe_program_fits(dense, 7, 16));
    for (const auto& selection : kSelections) {
        if (selection.recipe != Recipe::B) {
            auto p = joint;
            p.policy = resolve_precision_policy(selection);
            EXPECT_TRUE(recipe_program_fits(p, 7, 8));
        }
    }
    // The chooser never emits such a geometry (op-chosen joint blocking picked Q224/K256 here before the rule).
    for (uint32_t heads : {1u, 3u}) {
        for (uint32_t batch : {1u, 2u}) {
            for (auto* p : {&joint, &dense}) {
                auto q = *p;
                q.q_heads = heads;
                q.batch = batch;
                q.mask_page_bytes = 0;
                q.attention_sink = false;
                q.k_rows_unaligned = true;
                const auto choice = choose_recipe_blocking(q);
                ASSERT_TRUE(choice.has_value());
                EXPECT_TRUE(recipe_program_fits(q, tiles(choice->q_chunk_size), tiles(choice->k_chunk_size)));
                for (const auto& candidate : recipe_blocking_candidates(q)) {
                    EXPECT_TRUE(recipe_program_fits(q, tiles(candidate.q_chunk_size), tiles(candidate.k_chunk_size)));
                }
            }
        }
    }
}

TEST(SDPARecipeBlocking, CandidatesAreSupportedGeometries) {
    for (auto op : {RecipeOp::Dense, RecipeOp::Joint, RecipeOp::Ring, RecipeOp::ExpRing}) {
        for (const auto& selection : kSelections) {
            auto p = problem(op, selection, 10, 4096, 4096);
            p.ring_size = op == RecipeOp::Ring || op == RecipeOp::ExpRing ? 8 : 1;
            for (const auto& candidate : recipe_blocking_candidates(p)) {
                expect_valid(p, candidate);
            }
        }
    }
}
}  // namespace
