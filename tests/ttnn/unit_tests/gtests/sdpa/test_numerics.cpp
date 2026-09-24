// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include <array>
#include <optional>
#include <stdexcept>

#include <gtest/gtest.h>

#include "ttnn/operations/transformer/sdpa/sdpa_numerics.hpp"

namespace {
using namespace ttnn::operations::transformer::sdpa::detail;
using Fidelity = tt::tt_metal::MathFidelity;

TEST(SDPANumerics, OmittedConfigDiffersFromExplicitEmptyConfig) {
    const auto omitted = resolve_numerics(tt::ARCH::BLACKHOLE, std::nullopt, std::nullopt, std::nullopt);
    const auto empty = resolve_numerics(tt::ARCH::BLACKHOLE, std::nullopt, ttnn::ComputeKernelConfig{}, std::nullopt);
    EXPECT_EQ(omitted.compute.math_fidelity, Fidelity::HiFi2);
    EXPECT_EQ(empty.compute.math_fidelity, Fidelity::LoFi);
    EXPECT_TRUE(omitted.compute.math_approx_mode);
    EXPECT_FALSE(omitted.compute.fp32_dest_acc_en);
    EXPECT_FALSE(omitted.compute.packer_l1_acc);
    EXPECT_FALSE(omitted.compute.dst_full_sync_en);
    EXPECT_TRUE(omitted.exp_approx_mode);
    EXPECT_FALSE(omitted.policy.has_value());
    EXPECT_FALSE(empty.policy.has_value());
}

TEST(SDPANumerics, PreserveAllLegacyNumericalFieldsAndIndependentExp) {
    using Throttle = ttnn::operations::compute_throttle_utils::ThrottleLevel;
    for (auto arch : {tt::ARCH::WORMHOLE_B0, tt::ARCH::BLACKHOLE, tt::ARCH::QUASAR}) {
        for (auto fidelity : {Fidelity::LoFi, Fidelity::HiFi2, Fidelity::HiFi3, Fidelity::HiFi4}) {
            for (bool math_approx : {false, true}) {
                for (bool exp_approx : {false, true}) {
                    ttnn::ComputeKernelConfig cfg{
                        .math_fidelity = fidelity,
                        .math_approx_mode = math_approx,
                        .fp32_dest_acc_en = true,
                        .packer_l1_acc = true,
                        .dst_full_sync_en = true,
                        .throttle_level = Throttle::LEVEL_3};
                    const auto result = resolve_numerics(arch, std::nullopt, cfg, exp_approx);
                    EXPECT_EQ(result.compute.math_fidelity, cfg.math_fidelity);
                    EXPECT_EQ(result.compute.math_approx_mode, cfg.math_approx_mode);
                    EXPECT_EQ(result.compute.fp32_dest_acc_en, cfg.fp32_dest_acc_en);
                    EXPECT_EQ(result.compute.packer_l1_acc, cfg.packer_l1_acc);
                    EXPECT_EQ(result.compute.dst_full_sync_en, cfg.dst_full_sync_en);
                    EXPECT_EQ(result.compute.throttle_level, cfg.throttle_level);
                    EXPECT_EQ(result.exp_approx_mode, exp_approx);
                    EXPECT_FALSE(result.policy.has_value());
                }
            }
        }
    }
}

TEST(SDPANumerics, RecipeConfigCarriesMixedMatmulIntent) {
    const auto result = resolve_numerics(tt::ARCH::BLACKHOLE, RecipeSelection{Recipe::C}, std::nullopt, std::nullopt);
    ASSERT_TRUE(result.policy.has_value());
    EXPECT_EQ(result.policy->qk_fidelity, Fidelity::HiFi4);
    EXPECT_EQ(result.policy->pv_fidelity, Fidelity::HiFi2);
    EXPECT_EQ(result.compute.math_fidelity, Fidelity::HiFi2);
    EXPECT_TRUE(result.compute.fp32_dest_acc_en);
    EXPECT_TRUE(result.exp_approx_mode);
}

TEST(SDPANumerics, AccurateRecipeIsNotDefinedByApproximationBoolean) {
    const auto result = resolve_numerics(tt::ARCH::BLACKHOLE, RecipeSelection{Recipe::D}, std::nullopt, true);
    ASSERT_TRUE(result.policy.has_value());
    EXPECT_EQ(result.policy->softmax_arithmetic, SoftmaxArithmetic::AccurateFP32);
    EXPECT_EQ(result.compute.math_fidelity, Fidelity::HiFi4);
    EXPECT_TRUE(result.compute.fp32_dest_acc_en);
    EXPECT_TRUE(result.compute.math_approx_mode);
    EXPECT_TRUE(result.exp_approx_mode);
}

TEST(SDPANumerics, EveryRecipeResolvesToItsFrozenComputeFields) {
    const std::array selections{
        RecipeSelection{Recipe::A},
        RecipeSelection{Recipe::B},
        RecipeSelection{Recipe::C},
        RecipeSelection{Recipe::D},
        RecipeSelection{Recipe::E, KVStorage::BF16},
        RecipeSelection{Recipe::E, KVStorage::BFP8},
        RecipeSelection{Recipe::E, KVStorage::BFP4}};
    for (const auto selection : selections) {
        const auto policy = resolve_precision_policy(selection);
        for (const auto exp : {std::optional<bool>{}, std::optional<bool>{true}}) {
            const auto result = resolve_numerics(tt::ARCH::BLACKHOLE, selection, std::nullopt, exp);
            ASSERT_TRUE(result.policy.has_value());
            EXPECT_EQ(result.policy.value(), policy);
            EXPECT_EQ(result.compute.math_fidelity, policy.pv_fidelity);
            EXPECT_EQ(result.compute.fp32_dest_acc_en, policy.fp32_destination);
            EXPECT_TRUE(result.compute.math_approx_mode);
            EXPECT_TRUE(result.exp_approx_mode);
            EXPECT_FALSE(result.compute.packer_l1_acc);
            EXPECT_FALSE(result.compute.dst_full_sync_en);
            EXPECT_EQ(
                result.compute.throttle_level, ttnn::operations::compute_throttle_utils::ThrottleLevel::NO_THROTTLE);
        }
    }
}

TEST(SDPANumerics, RejectAmbiguousOrConflictingExplicitControls) {
    EXPECT_THROW(
        resolve_numerics(tt::ARCH::BLACKHOLE, RecipeSelection{Recipe::A}, ttnn::ComputeKernelConfig{}, std::nullopt),
        std::runtime_error);
    EXPECT_THROW(
        resolve_numerics(tt::ARCH::BLACKHOLE, RecipeSelection{Recipe::D}, std::nullopt, false), std::runtime_error);
}

TEST(SDPANumerics, RecipeDoesNotClaimOtherArchitectureQualification) {
    for (auto arch : {tt::ARCH::WORMHOLE_B0, tt::ARCH::QUASAR}) {
        EXPECT_THROW(
            resolve_numerics(arch, RecipeSelection{Recipe::B}, std::nullopt, std::nullopt), std::runtime_error);
    }
}
}  // namespace
