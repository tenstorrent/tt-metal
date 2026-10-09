// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Op-selected blocking for the SDPA precision recipes: the host-only chooser on DiT-like shapes.

#include <algorithm>
#include <array>
#include <cmath>

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

TEST(SDPARecipeBlocking, ExplicitChunksAreHonored) {
    auto p = problem(RecipeOp::Dense, {Recipe::B}, 10, 8192, 8192);
    p.fixed_q_tiles = 7;
    p.fixed_k_tiles = 12;
    auto choice = choose_recipe_blocking(p);
    ASSERT_TRUE(choice.has_value());
    EXPECT_EQ(choice->q_chunk_size, 224u);
    EXPECT_EQ(choice->k_chunk_size, 384u);
    p.fixed_q_tiles = 0;
    choice = choose_recipe_blocking(p);
    ASSERT_TRUE(choice.has_value());
    EXPECT_EQ(choice->k_chunk_size, 384u);
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
