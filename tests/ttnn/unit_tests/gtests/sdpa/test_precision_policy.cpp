// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include "ttnn/operations/transformer/sdpa/sdpa_precision_policy.hpp"

namespace {
using namespace ttnn::operations::transformer::sdpa::detail;
using Fidelity = tt::tt_metal::MathFidelity;

TEST(SDPAPrecisionPolicy, RecipeTable) {
    EXPECT_EQ(
        resolve_precision_policy({Recipe::B}),
        (PrecisionPolicy{{Recipe::B}, Fidelity::HiFi2, false, RecurrentState::ReferenceMaxFP32}));
    EXPECT_EQ(resolve_precision_policy({Recipe::C}), (PrecisionPolicy{{Recipe::C}, Fidelity::HiFi2, true, RecurrentState::FP32}));
    EXPECT_EQ(resolve_precision_policy({Recipe::D}), (PrecisionPolicy{{Recipe::D}, Fidelity::HiFi4, true, RecurrentState::FP32}));
    for (auto storage : {KVStorage::BF16, KVStorage::BFP8, KVStorage::BFP4}) {
        EXPECT_EQ(
            resolve_precision_policy({Recipe::E, storage}),
            (PrecisionPolicy{{Recipe::E, storage}, Fidelity::LoFi, false, RecurrentState::ReferenceMaxFP32}));
    }
}

TEST(SDPAPrecisionPolicy, PackedKVKeepsTheRecipe) {
    // K/V storage changes only the CB formats: every recipe keeps its fidelity, destination and state.
    for (auto recipe : {Recipe::B, Recipe::C, Recipe::D, Recipe::E}) {
        const auto bf16 = resolve_precision_policy({recipe});
        for (auto storage : {KVStorage::BFP8, KVStorage::BFP4}) {
            const auto packed = resolve_precision_policy({recipe, storage});
            EXPECT_EQ(packed.selection, (RecipeSelection{recipe, storage}));
            EXPECT_EQ(packed.pv_fidelity, bf16.pv_fidelity);
            EXPECT_EQ(packed.fp32_destination, bf16.fp32_destination);
            EXPECT_EQ(packed.recurrent_state, bf16.recurrent_state);
        }
    }
}
}  // namespace
