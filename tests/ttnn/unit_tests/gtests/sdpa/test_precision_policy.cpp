// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include <array>
#include <stdexcept>

#include <gtest/gtest.h>

#include "ttnn/operations/transformer/sdpa/sdpa_precision_policy.hpp"

namespace {
using namespace ttnn::operations::transformer::sdpa::detail;
using Fidelity = tt::tt_metal::MathFidelity;

TEST(SDPAPrecisionPolicy, CanonicalBf16Recipes) {
    const std::array policies{
        PrecisionPolicy{
            {Recipe::A},
            Fidelity::HiFi2,
            Fidelity::HiFi2,
            false,
            RecurrentState::BF16,
            SoftmaxArithmetic::Baseline,
            InputPreparation::None,
            InputPreparation::None},
        PrecisionPolicy{
            {Recipe::B},
            Fidelity::HiFi2,
            Fidelity::HiFi2,
            false,
            RecurrentState::CompensatedBF16,
            SoftmaxArithmetic::CompensatedBF16,
            InputPreparation::None,
            InputPreparation::None},
        PrecisionPolicy{
            {Recipe::C},
            Fidelity::HiFi4,
            Fidelity::HiFi2,
            true,
            RecurrentState::FP32,
            SoftmaxArithmetic::BalancedFP32,
            InputPreparation::None,
            InputPreparation::None},
        PrecisionPolicy{
            {Recipe::D},
            Fidelity::HiFi4,
            Fidelity::HiFi4,
            true,
            RecurrentState::FP32,
            SoftmaxArithmetic::AccurateFP32,
            InputPreparation::None,
            InputPreparation::None}};
    for (const auto& policy : policies) {
        EXPECT_EQ(resolve_precision_policy(policy.selection), policy);
    }
}

TEST(SDPAPrecisionPolicy, BRequiresNoExternalRounding) {
    constexpr auto b = resolve_precision_policy({Recipe::B});
    static_assert(b.q_preparation == InputPreparation::None);
    static_assert(b.kv_preparation == InputPreparation::None);
    EXPECT_EQ(b.recurrent_state, RecurrentState::CompensatedBF16);
}

TEST(SDPAPrecisionPolicy, EStorageChangesOnlyStorageAndPreparation) {
    const auto bf16 = resolve_precision_policy({Recipe::E, KVStorage::BF16});
    EXPECT_EQ(bf16.q_preparation, InputPreparation::Rne7);
    EXPECT_EQ(bf16.kv_preparation, InputPreparation::Rne5);
    for (auto storage : {KVStorage::BF16, KVStorage::BFP8, KVStorage::BFP4}) {
        const auto policy = resolve_precision_policy({Recipe::E, storage});
        EXPECT_EQ(policy.qk_fidelity, Fidelity::LoFi);
        EXPECT_EQ(policy.pv_fidelity, Fidelity::LoFi);
        EXPECT_FALSE(policy.fp32_destination);
        EXPECT_EQ(policy.recurrent_state, bf16.recurrent_state);
        EXPECT_EQ(policy.softmax_arithmetic, bf16.softmax_arithmetic);
        EXPECT_EQ(policy.q_preparation, bf16.q_preparation);
    }
    EXPECT_EQ(resolve_precision_policy({Recipe::E, KVStorage::BFP8}).kv_preparation, InputPreparation::Rne5ThenBfp8);
    EXPECT_EQ(
        resolve_precision_policy({Recipe::E, KVStorage::BFP4}).kv_preparation, InputPreparation::Bfp4GridRneSaturate);
}

TEST(SDPAPrecisionPolicy, PackedInputsDoNotSilentlyChangeAnotherRecipe) {
    for (auto recipe : {Recipe::A, Recipe::B, Recipe::C, Recipe::D}) {
        for (auto storage : {KVStorage::BFP8, KVStorage::BFP4}) {
            EXPECT_THROW(resolve_precision_policy({recipe, storage}), std::runtime_error);
        }
    }
}

TEST(SDPAPrecisionPolicy, RejectInvalidEnumValues) {
    EXPECT_THROW(resolve_precision_policy({static_cast<Recipe>(255)}), std::runtime_error);
    EXPECT_THROW(resolve_precision_policy({Recipe::E, static_cast<KVStorage>(255)}), std::runtime_error);
}

TEST(SDPAPrecisionPolicy, DistinctStorageAndRecipeIdentities) {
    const std::array selections{
        RecipeSelection{Recipe::A},
        RecipeSelection{Recipe::B},
        RecipeSelection{Recipe::C},
        RecipeSelection{Recipe::D},
        RecipeSelection{Recipe::E, KVStorage::BF16},
        RecipeSelection{Recipe::E, KVStorage::BFP8},
        RecipeSelection{Recipe::E, KVStorage::BFP4}};
    for (std::size_t i = 0; i < selections.size(); ++i) {
        for (std::size_t j = i + 1; j < selections.size(); ++j) {
            EXPECT_NE(resolve_precision_policy(selections[i]), resolve_precision_policy(selections[j]));
        }
    }
}
}  // namespace
