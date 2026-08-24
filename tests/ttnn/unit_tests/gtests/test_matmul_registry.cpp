// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include "ttnn/operations/matmul/device/config/matmul_config_registry.hpp"

namespace ttnn::operations::matmul::registry {
namespace {

TEST(MatmulConfigRegistry, ProductionModeStartsOff) {
    EXPECT_EQ(current_mode(), Mode::Off);
    const auto result = resolve(Mode::Off, Eligibility{.call = CallSemantics{.domain = OperationDomain::DenseMatmul}});
    EXPECT_EQ(result.reason, ResolutionReason::Disabled);
    EXPECT_FALSE(result.recipe.has_value());
}

TEST(MatmulConfigRegistry, EmptyOnTableFallsBack) {
    const auto result = resolve(Mode::On, Eligibility{.call = CallSemantics{.domain = OperationDomain::DenseMatmul}});
    EXPECT_EQ(result.reason, ResolutionReason::EmptyRegistry);
    EXPECT_FALSE(result.recipe.has_value());
}

TEST(MatmulConfigRegistry, EachPublicOperationHasADistinctEmptyDomain) {
    EXPECT_EQ(
        resolve(Mode::On, Eligibility{.call = CallSemantics{.domain = OperationDomain::DenseMatmul}}).reason,
        ResolutionReason::EmptyRegistry);
    EXPECT_EQ(
        resolve(
            Mode::On,
            Eligibility{
                .call = CallSemantics{.domain = OperationDomain::Linear},
                .has_bias = true,
                .has_activation = true,
                .transpose_b = true})
            .reason,
        ResolutionReason::EmptyRegistry);
    EXPECT_EQ(
        resolve(
            Mode::On,
            Eligibility{
                .call =
                    CallSemantics{.domain = OperationDomain::Addmm, .alpha_f32_bits = 0x3f800000, .beta_f32_bits = 0}})
            .reason,
        ResolutionReason::EmptyRegistry);
}

TEST(MatmulConfigRegistry, SharedCallersAreNeverEligible) {
    const auto result =
        resolve(Mode::On, Eligibility{.call = CallSemantics{.domain = OperationDomain::IneligibleSharedCaller}});
    EXPECT_EQ(result.reason, ResolutionReason::IneligibleOperationDomain);
}

TEST(MatmulConfigRegistry, AddmmRequiresExactScalarBits) {
    EXPECT_EQ(
        resolve(Mode::On, Eligibility{.call = CallSemantics{.domain = OperationDomain::Addmm}}).reason,
        ResolutionReason::MalformedOperationSemantics);
    EXPECT_EQ(
        resolve(
            Mode::On,
            Eligibility{
                .call =
                    CallSemantics{
                        .domain = OperationDomain::Linear, .alpha_f32_bits = 0x3f800000, .beta_f32_bits = 0x3f800000}})
            .reason,
        ResolutionReason::MalformedOperationSemantics);
}

TEST(MatmulConfigRegistry, EveryExplicitConfigAxisWins) {
    EXPECT_EQ(
        resolve(
            Mode::On,
            Eligibility{.call = CallSemantics{.domain = OperationDomain::DenseMatmul}, .has_program_config = true})
            .reason,
        ResolutionReason::ExplicitOverride);
    EXPECT_EQ(
        resolve(
            Mode::On,
            Eligibility{.call = CallSemantics{.domain = OperationDomain::Linear}, .has_compute_kernel_config = true})
            .reason,
        ResolutionReason::ExplicitOverride);
    EXPECT_EQ(
        resolve(
            Mode::On,
            Eligibility{
                .call =
                    CallSemantics{.domain = OperationDomain::Addmm, .alpha_f32_bits = 0x3f800000, .beta_f32_bits = 0},
                .has_user_core_grid = true})
            .reason,
        ResolutionReason::ExplicitOverride);
}

TEST(MatmulConfigRegistry, UnsupportedV1SemanticsFallBack) {
    EXPECT_EQ(
        resolve(Mode::On, Eligibility{.call = CallSemantics{.domain = OperationDomain::DenseMatmul}, .has_bias = true})
            .reason,
        ResolutionReason::UnsupportedSemantics);
    EXPECT_EQ(
        resolve(
            Mode::On,
            Eligibility{.call = CallSemantics{.domain = OperationDomain::DenseMatmul}, .input_a_sharded = true})
            .reason,
        ResolutionReason::UnsupportedSemantics);
    EXPECT_EQ(
        resolve(
            Mode::On,
            Eligibility{.call = CallSemantics{.domain = OperationDomain::DenseMatmul}, .input_b_batched = true})
            .reason,
        ResolutionReason::UnsupportedSemantics);
    EXPECT_EQ(
        resolve(
            Mode::On, Eligibility{.call = CallSemantics{.domain = OperationDomain::DenseMatmul}, .transpose_b = true})
            .reason,
        ResolutionReason::UnsupportedSemantics);
}

}  // namespace
}  // namespace ttnn::operations::matmul::registry
