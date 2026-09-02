// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Unit tests for the TensorSpecRelaxations helpers: hash_tensorspec_with_relaxation and
// tensorspecs_match_with_relaxation. These are pure host-side functions over (TensorSpec,
// relaxation), so the tests need no device.
//
// The load-bearing property: matches-under-relaxation => equal hash. The final test sweeps it.

#include <gtest/gtest.h>
#include <cstdint>
#include <utility>
#include <vector>

#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>  // TensorSpec / TensorLayout / PageConfig / ...
#include <tt-metalium/experimental/metal2_host_api/tensor_spec_relaxations.hpp>

namespace tt::tt_metal::experimental {
// Named rather than anonymous: these tests are part of a unity build, where two anonymous namespaces
// merge into one and same-named helpers collide.
namespace tensor_spec_relaxation_tests {

// Build a TensorSpec with a controllable shape, layout, and dtype. Interleaved/DRAM throughout;
// the axes that matter here are logical_shape, padded_shape (derived), and tensor_layout.
TensorSpec make_spec(Shape shape, Layout layout = Layout::TILE, DataType dtype = DataType::BFLOAT16) {
    PageConfig page_config(layout);
    MemoryConfig memory_config{TensorMemoryLayout::INTERLEAVED, BufferType::DRAM};
    TensorLayout tensor_layout(dtype, page_config, memory_config);
    return TensorSpec(std::move(shape), std::move(tensor_layout));
}

// Strict (default-constructed) relaxation demands an exact TensorSpec match.
TEST(TensorSpecRelaxations, CPU_StrictRequiresExactMatch) {
    const TensorSpecRelaxations strict{};
    const auto a = make_spec(Shape{1, 1, 32, 32});
    const auto same = make_spec(Shape{1, 1, 32, 32});
    const auto diff_shape = make_spec(Shape{1, 1, 64, 64});
    const auto diff_dtype = make_spec(Shape{1, 1, 32, 32}, Layout::TILE, DataType::FLOAT32);

    EXPECT_TRUE(tensorspecs_match_with_relaxation(a, same, strict));
    EXPECT_EQ(hash_tensorspec_with_relaxation(a, strict), hash_tensorspec_with_relaxation(same, strict));

    EXPECT_FALSE(tensorspecs_match_with_relaxation(a, diff_shape, strict));
    EXPECT_NE(hash_tensorspec_with_relaxation(a, strict), hash_tensorspec_with_relaxation(diff_shape, strict));

    EXPECT_FALSE(tensorspecs_match_with_relaxation(a, diff_dtype, strict));
    EXPECT_NE(hash_tensorspec_with_relaxation(a, strict), hash_tensorspec_with_relaxation(diff_dtype, strict));
}

// Strict match is exactly TensorSpec equality.
TEST(TensorSpecRelaxations, CPU_StrictEqualsTensorSpecEquality) {
    const TensorSpecRelaxations strict{};
    const auto a = make_spec(Shape{1, 1, 32, 32});
    const auto b = make_spec(Shape{1, 1, 32, 64});
    EXPECT_EQ(tensorspecs_match_with_relaxation(a, b, strict), (a == b));
    EXPECT_EQ(tensorspecs_match_with_relaxation(a, a, strict), (a == a));
}

// dynamic_tensor_shape: same layout and rank, differing per-dim shape values -> match + equal hash.
// Differing rank or layout still breaks the match.
TEST(TensorSpecRelaxations, CPU_DynamicToleratesShapeWithinRank) {
    const TensorSpecRelaxations dyn{.dynamic_tensor_shape = true};
    const auto a = make_spec(Shape{1, 1, 32, 32});
    const auto b = make_spec(Shape{1, 1, 64, 128});  // same rank(4) + layout, different values

    EXPECT_TRUE(tensorspecs_match_with_relaxation(a, b, dyn));
    EXPECT_EQ(hash_tensorspec_with_relaxation(a, dyn), hash_tensorspec_with_relaxation(b, dyn));

    const auto diff_rank = make_spec(Shape{1, 32, 32});  // rank 3
    EXPECT_FALSE(tensorspecs_match_with_relaxation(a, diff_rank, dyn));
    EXPECT_NE(hash_tensorspec_with_relaxation(a, dyn), hash_tensorspec_with_relaxation(diff_rank, dyn));

    const auto diff_layout = make_spec(Shape{1, 1, 32, 32}, Layout::TILE, DataType::FLOAT32);
    EXPECT_FALSE(tensorspecs_match_with_relaxation(a, diff_layout, dyn));
    EXPECT_NE(hash_tensorspec_with_relaxation(a, dyn), hash_tensorspec_with_relaxation(diff_layout, dyn));
}

// match_padded_shape_only: same layout and padded_shape, differing logical_shape -> match + equal hash.
TEST(TensorSpecRelaxations, CPU_PaddedShapeOnlyToleratesLogicalWithinPadding) {
    const TensorSpecRelaxations padded{.match_padded_shape_only = true};
    // TILE pads the last two dims up to the 32x32 tile: {..,5,5} and {..,30,30} both pad to {..,32,32}.
    const auto a = make_spec(Shape{1, 1, 5, 5});
    const auto b = make_spec(Shape{1, 1, 30, 30});
    ASSERT_EQ(a.padded_shape(), b.padded_shape());  // precondition
    ASSERT_NE(a.logical_shape(), b.logical_shape());

    EXPECT_TRUE(tensorspecs_match_with_relaxation(a, b, padded));
    EXPECT_EQ(hash_tensorspec_with_relaxation(a, padded), hash_tensorspec_with_relaxation(b, padded));

    const auto diff_padded = make_spec(Shape{1, 1, 64, 64});  // pads to {..,64,64}
    ASSERT_NE(a.padded_shape(), diff_padded.padded_shape());
    EXPECT_FALSE(tensorspecs_match_with_relaxation(a, diff_padded, padded));
    EXPECT_NE(hash_tensorspec_with_relaxation(a, padded), hash_tensorspec_with_relaxation(diff_padded, padded));
}

// Precedence: dynamic_tensor_shape subsumes match_padded_shape_only when both are set.
TEST(TensorSpecRelaxations, CPU_DynamicSubsumesPaddedShapeOnly) {
    const auto a = make_spec(Shape{1, 1, 32, 32});
    const auto b = make_spec(Shape{1, 1, 64, 64});  // same rank+layout, DIFFERENT padded_shape
    ASSERT_NE(a.padded_shape(), b.padded_shape());

    // padded-only alone rejects (padded differs)...
    EXPECT_FALSE(tensorspecs_match_with_relaxation(a, b, TensorSpecRelaxations{.match_padded_shape_only = true}));
    // ...but with both set, dynamic wins and accepts.
    const TensorSpecRelaxations both{.match_padded_shape_only = true, .dynamic_tensor_shape = true};
    EXPECT_TRUE(tensorspecs_match_with_relaxation(a, b, both));
    EXPECT_EQ(hash_tensorspec_with_relaxation(a, both), hash_tensorspec_with_relaxation(b, both));
}

// dynamic_tensor_shape alone still PINS the rank. This is the mode relax_logical_rank was added
// beside rather than folded into, so it is a tripwire: over-permitting fails silently, and an op
// that tolerates extent variation but not rank variation must keep being able to say so.
TEST(TensorSpecRelaxations, CPU_DynamicTensorShapeStillPinsRank) {
    const TensorSpecRelaxations dyn{.dynamic_tensor_shape = true};
    const auto rank4 = make_spec(Shape{1, 1, 32, 32});
    const auto rank3 = make_spec(Shape{1, 32, 32});
    ASSERT_NE(rank4.logical_shape().rank(), rank3.logical_shape().rank());

    EXPECT_FALSE(tensorspecs_match_with_relaxation(rank4, rank3, dyn));
    EXPECT_NE(hash_tensorspec_with_relaxation(rank4, dyn), hash_tensorspec_with_relaxation(rank3, dyn));
}

// relax_logical_rank (with dynamic_tensor_shape): differing rank at identical layout -> match.
TEST(TensorSpecRelaxations, CPU_RelaxLogicalRankToleratesRankChange) {
    const TensorSpecRelaxations relaxed{.dynamic_tensor_shape = true, .relax_logical_rank = true};
    const auto rank2 = make_spec(Shape{32, 32});
    const auto rank4 = make_spec(Shape{1, 1, 32, 32});
    ASSERT_NE(rank2.logical_shape().rank(), rank4.logical_shape().rank());  // precondition

    EXPECT_TRUE(tensorspecs_match_with_relaxation(rank2, rank4, relaxed));
    EXPECT_EQ(hash_tensorspec_with_relaxation(rank2, relaxed), hash_tensorspec_with_relaxation(rank4, relaxed));
}

// ...and the per-dim values are free at the same time, not merely the rank.
TEST(TensorSpecRelaxations, CPU_RelaxLogicalRankToleratesRankAndShapeChange) {
    const TensorSpecRelaxations relaxed{.dynamic_tensor_shape = true, .relax_logical_rank = true};
    const auto a = make_spec(Shape{32, 32});
    const auto b = make_spec(Shape{2, 3, 64, 128});

    EXPECT_TRUE(tensorspecs_match_with_relaxation(a, b, relaxed));
    EXPECT_EQ(hash_tensorspec_with_relaxation(a, relaxed), hash_tensorspec_with_relaxation(b, relaxed));
}

// tensor_layout is the floor: it stays load-bearing even at the loosest setting.
TEST(TensorSpecRelaxations, CPU_RelaxLogicalRankStillPinsLayout) {
    const TensorSpecRelaxations relaxed{.dynamic_tensor_shape = true, .relax_logical_rank = true};
    const auto a = make_spec(Shape{1, 1, 32, 32});
    const auto diff_dtype = make_spec(Shape{1, 1, 32, 32}, Layout::TILE, DataType::FLOAT32);
    const auto diff_layout = make_spec(Shape{1, 1, 32, 32}, Layout::ROW_MAJOR);

    EXPECT_FALSE(tensorspecs_match_with_relaxation(a, diff_dtype, relaxed));
    EXPECT_NE(hash_tensorspec_with_relaxation(a, relaxed), hash_tensorspec_with_relaxation(diff_dtype, relaxed));

    EXPECT_FALSE(tensorspecs_match_with_relaxation(a, diff_layout, relaxed));
    EXPECT_NE(hash_tensorspec_with_relaxation(a, relaxed), hash_tensorspec_with_relaxation(diff_layout, relaxed));
}

// relax_logical_rank is documented as inert unless dynamic_tensor_shape is also set. Pin that:
// adding it to the strict or padded-only settings must change neither the match nor the hash.
// (Not a no-op by accident -- strict comparison and padded-shape matching carry no separable rank
// term, so there is nothing for it to drop.)
TEST(TensorSpecRelaxations, CPU_RelaxLogicalRankInertWithoutDynamicTensorShape) {
    const std::vector<TensorSpec> specs = {
        make_spec(Shape{32, 32}),
        make_spec(Shape{1, 1, 32, 32}),
        make_spec(Shape{1, 1, 5, 5}),
        make_spec(Shape{1, 32, 32}),
    };
    const std::vector<std::pair<TensorSpecRelaxations, TensorSpecRelaxations>> without_and_with = {
        {TensorSpecRelaxations{}, TensorSpecRelaxations{.relax_logical_rank = true}},
        {TensorSpecRelaxations{.match_padded_shape_only = true},
         TensorSpecRelaxations{.match_padded_shape_only = true, .relax_logical_rank = true}},
    };

    for (const auto& [without, with] : without_and_with) {
        for (const auto& x : specs) {
            EXPECT_EQ(hash_tensorspec_with_relaxation(x, without), hash_tensorspec_with_relaxation(x, with));
            for (const auto& y : specs) {
                EXPECT_EQ(
                    tensorspecs_match_with_relaxation(x, y, without), tensorspecs_match_with_relaxation(x, y, with));
            }
        }
    }
}

// Build a relaxation from a bit pattern, one bit per flag, so the sweep below can enumerate every
// combination rather than a hand-picked list.
constexpr unsigned kNumRelaxationFlags = 3;
static_assert(
    sizeof(TensorSpecRelaxations) == kNumRelaxationFlags,
    "A TensorSpecRelaxations flag was added or removed. Update kNumRelaxationFlags and relaxation_from_bits, so the "
    "hash/match sweep keeps covering every combination -- that sweep is what catches a new flag being wired into one "
    "of hash_tensorspec_with_relaxation / tensorspecs_match_with_relaxation but not the other.");

TensorSpecRelaxations relaxation_from_bits(unsigned bits) {
    return TensorSpecRelaxations{
        .match_padded_shape_only = (bits & 0b001u) != 0,
        .dynamic_tensor_shape = (bits & 0b010u) != 0,
        .relax_logical_rank = (bits & 0b100u) != 0,
    };
}

// The core contract, swept over every relaxation and spec pair: match <=> equal hash.
// (match => equal-hash is guaranteed by construction; the reverse holds here absent a 64-bit
// collision, which won't occur across this small fixed set — so the biconditional is a strong check
// that the hash is neither too fine nor too coarse relative to the match.)
// An ND-sharded TensorSpec, so the sharded distribution geometry is exercised. make_spec above is
// interleaved, where no BufferDistributionSpec exists at all.
TensorSpec make_nd_sharded_spec(Shape shape, Shape nd_shard_shape) {
    const CoreRangeSet grid({CoreRange({0, 0}, {7, 7})});
    MemoryConfig memory_config{BufferType::L1, NdShardSpec{std::move(nd_shard_shape), grid}};
    TensorLayout tensor_layout(DataType::BFLOAT16, PageConfig(Layout::TILE), memory_config);
    return TensorSpec(std::move(shape), std::move(tensor_layout));
}

// REGRESSION: dynamic_tensor_shape must not accept a shape change that moves the sharded
// distribution geometry.
//
// The sharded CTA payload bakes the squeezed shard shape and the bank list from the DECLARED spec
// and re-emits only tensor_shape_in_pages per dispatch. squeeze_shape_ranks decides its merges from
// shape VALUES, so a permitted shape change can move that geometry while leaving the squeezed RANK
// alone -- which is the one thing the bind-time guard in EmitBindingCrtaValues checks. Before this
// was pinned in the match, the pair below was accepted, and the device then divided the page
// coordinate by a stale shard shape: silently wrong addresses, no assert.
//
// Both tensors are rank 3 with identical tensor_layout, and both squeeze to rank 2 -- so every other
// gate passes and the geometry is the only thing that separates them.
TEST(TensorSpecRelaxations, CPU_DynamicTensorShapePinsShardDistribution) {
    const TensorSpecRelaxations dyn{.dynamic_tensor_shape = true};
    const Shape nd_shard{1, 64, 64};  // [1, 2, 2] in 32x32 pages

    const auto a = make_nd_sharded_spec(Shape{1, 32, 32}, nd_shard);  // squeezes to shard [2, 2]
    const auto b = make_nd_sharded_spec(Shape{2, 32, 64}, nd_shard);  // squeezes to shard [1, 4]

    // Preconditions: everything except the geometry agrees, so a pass here would be the real defect.
    ASSERT_EQ(a.tensor_layout(), b.tensor_layout());
    ASSERT_EQ(a.logical_shape().rank(), b.logical_shape().rank());
    const auto a_args = a.compute_buffer_sharding_args();
    const auto b_args = b.compute_buffer_sharding_args();
    ASSERT_TRUE(a_args.buffer_distribution_spec().has_value());
    ASSERT_TRUE(b_args.buffer_distribution_spec().has_value());
    const auto& a_bds = *a_args.buffer_distribution_spec();
    const auto& b_bds = *b_args.buffer_distribution_spec();
    ASSERT_EQ(a_bds.tensor_shape_in_pages().rank(), b_bds.tensor_shape_in_pages().rank());
    ASSERT_NE(a_bds.shard_shape_in_pages(), b_bds.shard_shape_in_pages());

    EXPECT_FALSE(tensorspecs_match_with_relaxation(a, b, dyn));
    EXPECT_NE(hash_tensorspec_with_relaxation(a, dyn), hash_tensorspec_with_relaxation(b, dyn));
}

// ...and relax_logical_rank does not reopen it: it frees the LOGICAL rank, which has no authority
// over the distribution geometry.
TEST(TensorSpecRelaxations, CPU_RelaxLogicalRankStillPinsShardDistribution) {
    const TensorSpecRelaxations relaxed{.dynamic_tensor_shape = true, .relax_logical_rank = true};
    const Shape nd_shard{1, 64, 64};

    const auto a = make_nd_sharded_spec(Shape{1, 32, 32}, nd_shard);
    const auto b = make_nd_sharded_spec(Shape{2, 32, 64}, nd_shard);

    EXPECT_FALSE(tensorspecs_match_with_relaxation(a, b, relaxed));
    EXPECT_NE(hash_tensorspec_with_relaxation(a, relaxed), hash_tensorspec_with_relaxation(b, relaxed));
}

// A shape change that leaves the geometry alone is still accepted -- the pin must not collapse
// dynamic_tensor_shape into strict matching on the sharded path.
TEST(TensorSpecRelaxations, CPU_DynamicTensorShapeStillFreeWhenShardDistributionAgrees) {
    const TensorSpecRelaxations dyn{.dynamic_tensor_shape = true};
    const Shape nd_shard{1, 64, 64};

    const auto a = make_nd_sharded_spec(Shape{2, 32, 64}, nd_shard);
    const auto b = make_nd_sharded_spec(Shape{4, 32, 64}, nd_shard);
    ASSERT_NE(a.logical_shape(), b.logical_shape());  // the shape really does vary

    EXPECT_TRUE(tensorspecs_match_with_relaxation(a, b, dyn));
    EXPECT_EQ(hash_tensorspec_with_relaxation(a, dyn), hash_tensorspec_with_relaxation(b, dyn));
}

// Every flag combination is swept, not a hand-picked list: with the flags composing as modifiers
// rather than as a chain of modes, the combinations are where a term wired into one function but
// not the other would show up.
TEST(TensorSpecRelaxations, CPU_HashConsistentWithMatch) {
    const std::vector<TensorSpec> specs = {
        make_spec(Shape{1, 1, 32, 32}),
        make_spec(Shape{1, 1, 64, 64}),
        make_spec(Shape{1, 1, 5, 5}),    // pads to {..,32,32} (== first, padded)
        make_spec(Shape{1, 1, 30, 30}),  // pads to {..,32,32}
        make_spec(Shape{1, 32, 32}),     // rank 3
        make_spec(Shape{32, 32}),        // rank 2
        // Rank 1, and the corpus's one CROSS-RANK padded_shape collision: TILE's default alignment
        // is rank 2, and compute_padded_shape sizes its result max(logical rank, alignment rank), so
        // {32} and {32,32} both pad to {32,32} while the rank-3/rank-4 entries pad to rank-3/rank-4
        // shapes. That pair is what lets the sweep tell the padded_shape term apart from the rank
        // term: it matches under match_padded_shape_only, does NOT match under dynamic_tensor_shape,
        // and matches again once relax_logical_rank frees the rank. Without it, dropping the rank
        // term from both functions at once would go undetected.
        make_spec(Shape{32}),
        make_spec(Shape{1, 1, 32, 32}, Layout::ROW_MAJOR),                // different layout
        make_spec(Shape{1, 1, 32, 32}, Layout::TILE, DataType::FLOAT32),  // different dtype
    };

    for (unsigned bits = 0; bits < (1u << kNumRelaxationFlags); ++bits) {
        const TensorSpecRelaxations relaxation = relaxation_from_bits(bits);
        for (size_t i = 0; i < specs.size(); ++i) {
            for (size_t j = 0; j < specs.size(); ++j) {
                const bool matched = tensorspecs_match_with_relaxation(specs[i], specs[j], relaxation);
                const bool hash_eq = hash_tensorspec_with_relaxation(specs[i], relaxation) ==
                                     hash_tensorspec_with_relaxation(specs[j], relaxation);
                EXPECT_EQ(matched, hash_eq) << "match/hash disagree at bits=" << bits << " i=" << i << " j=" << j;
            }
        }
    }
}

}  // namespace tensor_spec_relaxation_tests
}  // namespace tt::tt_metal::experimental
