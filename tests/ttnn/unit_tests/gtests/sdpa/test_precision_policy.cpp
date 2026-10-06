// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include <stdexcept>

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

TEST(SDPAPrecisionPolicy, PackedKVOnlyForLowPrecision) {
    for (auto recipe : {Recipe::B, Recipe::C, Recipe::D}) {
        for (auto storage : {KVStorage::BFP8, KVStorage::BFP4}) {
            EXPECT_THROW(resolve_precision_policy({recipe, storage}), std::runtime_error);
        }
    }
}
}  // namespace
