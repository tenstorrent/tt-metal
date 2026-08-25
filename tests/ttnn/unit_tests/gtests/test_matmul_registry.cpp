// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include "ttnn/operations/matmul/device/config/matmul_config_registry.hpp"

namespace ttnn::operations::matmul::registry {
namespace {

using tt::tt_metal::BufferType;
using tt::tt_metal::DataType;
using tt::tt_metal::MemoryConfig;
using tt::tt_metal::TensorMemoryLayout;
using tt::tt_metal::Tile;

IoContractRequest default_io_request() {
    return {
        .input_a_dtype = DataType::BFLOAT16,
        .input_a_tile = Tile({32, 16}),
        .input_b_tile = Tile({16, 32}),
        .requested_output_memory_config = MemoryConfig{},
    };
}

Recipe basic_recipe() {
    return {
        .program_config = MatmulMultiCoreProgramConfig{},
        .compute_kernel_config = DeviceComputeKernelConfig{},
        .untilize_out = false,
    };
}

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

TEST(MatmulConfigRegistry, ResolvesDefaultOutputContractFromInputA) {
    const auto result = resolve_matmul_io_contract(default_io_request());

    EXPECT_EQ(result.status, IoContractStatus::Resolved);
    EXPECT_EQ(result.output_memory_config, MemoryConfig{});
    EXPECT_EQ(result.output_dtype, DataType::BFLOAT16);
    EXPECT_EQ(result.output_tile, Tile({32, 32}));
    EXPECT_FALSE(result.uses_optional_output);
}

TEST(MatmulConfigRegistry, OptionalOutputSuppliesDefaultedContract) {
    auto request = default_io_request();
    request.optional_output = OptionalOutputContract{
        .memory_config = MemoryConfig(TensorMemoryLayout::INTERLEAVED, BufferType::L1),
        .dtype = DataType::FLOAT32,
        .tile = Tile({32, 32}),
    };

    const auto result = resolve_matmul_io_contract(request);

    EXPECT_EQ(result.status, IoContractStatus::Resolved);
    EXPECT_EQ(result.output_memory_config, request.optional_output->memory_config);
    EXPECT_EQ(result.output_dtype, DataType::FLOAT32);
    EXPECT_EQ(result.output_tile, request.optional_output->tile);
    EXPECT_TRUE(result.uses_optional_output);
}

TEST(MatmulConfigRegistry, OptionalOutputConflictsAreTypedAndNonThrowing) {
    auto memory_mismatch = default_io_request();
    memory_mismatch.requested_output_memory_config = MemoryConfig(TensorMemoryLayout::INTERLEAVED, BufferType::L1);
    memory_mismatch.optional_output =
        OptionalOutputContract{.memory_config = MemoryConfig{}, .dtype = DataType::BFLOAT16, .tile = Tile({32, 32})};
    EXPECT_EQ(resolve_matmul_io_contract(memory_mismatch).status, IoContractStatus::OptionalOutputMemoryMismatch);

    auto dtype_mismatch = default_io_request();
    dtype_mismatch.requested_output_dtype = DataType::FLOAT32;
    dtype_mismatch.optional_output =
        OptionalOutputContract{.memory_config = MemoryConfig{}, .dtype = DataType::BFLOAT16, .tile = Tile({32, 32})};
    EXPECT_EQ(resolve_matmul_io_contract(dtype_mismatch).status, IoContractStatus::OptionalOutputDtypeMismatch);

    auto tile_conflict = default_io_request();
    tile_conflict.requested_output_tile = Tile({32, 32});
    tile_conflict.optional_output =
        OptionalOutputContract{.memory_config = MemoryConfig{}, .dtype = DataType::BFLOAT16, .tile = Tile({32, 32})};
    EXPECT_EQ(resolve_matmul_io_contract(tile_conflict).status, IoContractStatus::OutputTileConflict);
}

TEST(MatmulConfigRegistry, OutputContractAccountsForTransposeTiles) {
    auto request = default_io_request();
    request.input_a_tile = Tile({16, 32});
    request.transpose_a = true;

    const auto result = resolve_matmul_io_contract(request);

    EXPECT_EQ(result.status, IoContractStatus::Resolved);
    EXPECT_EQ(result.output_tile, Tile({32, 32}));
}

TEST(MatmulConfigRegistry, ShadowObservesCertifiedHitButOnlyOnAppliesIt) {
    const Resolution certified_hit{.reason = ResolutionReason::CertifiedMatch, .recipe = basic_recipe()};

    EXPECT_EQ(execution_action(Mode::Off, certified_hit), ExecutionAction::Fallback);
    EXPECT_EQ(execution_action(Mode::Shadow, certified_hit), ExecutionAction::ObserveOnly);
    EXPECT_EQ(execution_action(Mode::On, certified_hit), ExecutionAction::ApplyRecipe);
    EXPECT_EQ(
        execution_action(Mode::On, Resolution{.reason = ResolutionReason::EmptyRegistry}), ExecutionAction::Fallback);
}

TEST(MatmulConfigRegistry, RecipeCarriesOneConsistentUntilizeValue) {
    EXPECT_TRUE(has_consistent_untilize_out(basic_recipe()));

    auto non_1d = basic_recipe();
    non_1d.untilize_out = true;
    EXPECT_FALSE(has_consistent_untilize_out(non_1d));

    MatmulMultiCoreReuseMultiCast1DProgramConfig config_1d{};
    config_1d.untilize_out = true;
    auto matching_1d = basic_recipe();
    matching_1d.program_config = config_1d;
    matching_1d.untilize_out = true;
    EXPECT_TRUE(has_consistent_untilize_out(matching_1d));

    matching_1d.untilize_out = false;
    EXPECT_FALSE(has_consistent_untilize_out(matching_1d));
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

TEST(MatmulConfigRegistry, InconsistentIoContractIsNeverLookedUp) {
    const auto result = resolve(
        Mode::On,
        Eligibility{
            .call = CallSemantics{.domain = OperationDomain::DenseMatmul},
            .io_contract_status = IoContractStatus::OutputTileConflict});
    EXPECT_EQ(result.reason, ResolutionReason::InconsistentIoContract);
    EXPECT_FALSE(result.recipe.has_value());
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
