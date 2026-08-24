// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include "ttnn/operations/matmul/device/config/matmul_config_registry.hpp"

namespace ttnn::operations::matmul::registry {
namespace {

TEST(MatmulConfigRegistry, ProductionModeStartsOff) {
    EXPECT_EQ(current_mode(), Mode::Off);
    const auto result = resolve(Mode::Off, Eligibility{.call_origin = CallOrigin::PublicMatmul});
    EXPECT_EQ(result.reason, ResolutionReason::Disabled);
    EXPECT_FALSE(result.recipe.has_value());
}

TEST(MatmulConfigRegistry, EmptyOnTableFallsBack) {
    const auto result = resolve(Mode::On, Eligibility{.call_origin = CallOrigin::PublicMatmul});
    EXPECT_EQ(result.reason, ResolutionReason::EmptyRegistry);
    EXPECT_FALSE(result.recipe.has_value());
}

TEST(MatmulConfigRegistry, SharedCallersAreNeverEligible) {
    const auto result = resolve(Mode::On, Eligibility{.call_origin = CallOrigin::IneligibleSharedCaller});
    EXPECT_EQ(result.reason, ResolutionReason::IneligibleCallOrigin);
}

TEST(MatmulConfigRegistry, EveryExplicitConfigAxisWins) {
    EXPECT_EQ(
        resolve(Mode::On, Eligibility{.call_origin = CallOrigin::PublicMatmul, .has_program_config = true}).reason,
        ResolutionReason::ExplicitOverride);
    EXPECT_EQ(
        resolve(Mode::On, Eligibility{.call_origin = CallOrigin::PublicMatmul, .has_compute_kernel_config = true})
            .reason,
        ResolutionReason::ExplicitOverride);
    EXPECT_EQ(
        resolve(Mode::On, Eligibility{.call_origin = CallOrigin::PublicMatmul, .has_user_core_grid = true}).reason,
        ResolutionReason::ExplicitOverride);
}

TEST(MatmulConfigRegistry, UnsupportedV1SemanticsFallBack) {
    EXPECT_EQ(
        resolve(Mode::On, Eligibility{.call_origin = CallOrigin::PublicMatmul, .has_bias = true}).reason,
        ResolutionReason::UnsupportedSemantics);
    EXPECT_EQ(
        resolve(Mode::On, Eligibility{.call_origin = CallOrigin::PublicMatmul, .input_a_sharded = true}).reason,
        ResolutionReason::UnsupportedSemantics);
    EXPECT_EQ(
        resolve(Mode::On, Eligibility{.call_origin = CallOrigin::PublicMatmul, .input_b_batched = true}).reason,
        ResolutionReason::UnsupportedSemantics);
    EXPECT_EQ(
        resolve(Mode::On, Eligibility{.call_origin = CallOrigin::PublicMatmul, .transpose_b = true}).reason,
        ResolutionReason::UnsupportedSemantics);
}

}  // namespace
}  // namespace ttnn::operations::matmul::registry
