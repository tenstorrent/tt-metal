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
#include <vector>

#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>  // TensorSpec / TensorLayout / PageConfig / ...
#include <tt-metalium/experimental/metal2_host_api/tensor_spec_relaxations.hpp>

namespace tt::tt_metal::experimental {
namespace {

// Build a TensorSpec with a controllable shape, layout, and dtype. Interleaved/DRAM throughout;
// the axes that matter here are logical_shape, padded_shape (derived), and tensor_layout.
TensorSpec make_spec(Shape shape, Layout layout = Layout::TILE, DataType dtype = DataType::BFLOAT16) {
    PageConfig page_config(layout);
    MemoryConfig memory_config{TensorMemoryLayout::INTERLEAVED, BufferType::DRAM};
    TensorLayout tensor_layout(dtype, page_config, memory_config);
    return TensorSpec(std::move(shape), std::move(tensor_layout));
}

// Strict (default-constructed) relaxation demands an exact TensorSpec match.
TEST(TensorSpecRelaxations, StrictRequiresExactMatch) {
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
TEST(TensorSpecRelaxations, StrictEqualsTensorSpecEquality) {
    const TensorSpecRelaxations strict{};
    const auto a = make_spec(Shape{1, 1, 32, 32});
    const auto b = make_spec(Shape{1, 1, 32, 64});
    EXPECT_EQ(tensorspecs_match_with_relaxation(a, b, strict), (a == b));
    EXPECT_EQ(tensorspecs_match_with_relaxation(a, a, strict), (a == a));
}

// dynamic_tensor_shape: same layout and rank, differing per-dim shape values -> match + equal hash.
// Differing rank or layout still breaks the match.
TEST(TensorSpecRelaxations, DynamicToleratesShapeWithinRank) {
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
TEST(TensorSpecRelaxations, PaddedShapeOnlyToleratesLogicalWithinPadding) {
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
TEST(TensorSpecRelaxations, DynamicSubsumesPaddedShapeOnly) {
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

// The core contract, swept over every relaxation and spec pair: match <=> equal hash.
// (match => equal-hash is guaranteed by construction; the reverse holds here absent a 64-bit
// collision, which won't occur across this small fixed set — so the biconditional is a strong check
// that the hash is neither too fine nor too coarse relative to the match.)
TEST(TensorSpecRelaxations, HashConsistentWithMatch) {
    const std::vector<TensorSpec> specs = {
        make_spec(Shape{1, 1, 32, 32}),
        make_spec(Shape{1, 1, 64, 64}),
        make_spec(Shape{1, 1, 5, 5}),                                     // pads to {..,32,32} (== first, padded)
        make_spec(Shape{1, 1, 30, 30}),                                   // pads to {..,32,32}
        make_spec(Shape{1, 32, 32}),                                      // rank 3
        make_spec(Shape{1, 1, 32, 32}, Layout::ROW_MAJOR),                // different layout
        make_spec(Shape{1, 1, 32, 32}, Layout::TILE, DataType::FLOAT32),  // different dtype
    };
    const std::vector<TensorSpecRelaxations> rels = {
        TensorSpecRelaxations{},
        TensorSpecRelaxations{.match_padded_shape_only = true},
        TensorSpecRelaxations{.dynamic_tensor_shape = true},
        TensorSpecRelaxations{.match_padded_shape_only = true, .dynamic_tensor_shape = true},
    };

    for (size_t r = 0; r < rels.size(); ++r) {
        for (size_t i = 0; i < specs.size(); ++i) {
            for (size_t j = 0; j < specs.size(); ++j) {
                const bool matched = tensorspecs_match_with_relaxation(specs[i], specs[j], rels[r]);
                const bool hash_eq = hash_tensorspec_with_relaxation(specs[i], rels[r]) ==
                                     hash_tensorspec_with_relaxation(specs[j], rels[r]);
                EXPECT_EQ(matched, hash_eq) << "match/hash disagree at rel=" << r << " i=" << i << " j=" << j;
            }
        }
    }
}

}  // namespace
}  // namespace tt::tt_metal::experimental
