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

MatmulRegistryRequest exact_request(const OperationDomain domain = OperationDomain::DenseMatmul) {
    return {
        .schema_version = 1,
        .call = CallSemantics{.domain = domain},
        .workload =
            WorkloadRequest{
                .logical_m = 128,
                .logical_k = 256,
                .logical_n = 512,
                .padded_m = 128,
                .padded_k = 256,
                .padded_n = 512,
            },
        .input_a =
            TensorRequest{
                .dtype = DataType::BFLOAT16,
                .layout = tt::tt_metal::Layout::TILE,
                .memory_layout = TensorMemoryLayout::INTERLEAVED,
                .buffer_type = BufferType::DRAM,
                .tile_height = 32,
                .tile_width = 32,
            },
        .input_b =
            TensorRequest{
                .dtype = DataType::BFLOAT16,
                .layout = tt::tt_metal::Layout::TILE,
                .memory_layout = TensorMemoryLayout::INTERLEAVED,
                .buffer_type = BufferType::DRAM,
                .tile_height = 32,
                .tile_width = 32,
            },
        .output =
            TensorRequest{
                .dtype = DataType::BFLOAT16,
                .layout = tt::tt_metal::Layout::TILE,
                .memory_layout = TensorMemoryLayout::INTERLEAVED,
                .buffer_type = BufferType::DRAM,
                .tile_height = 32,
                .tile_width = 32,
            },
        .device =
            DeviceRequest{
                .architecture = 2,
                .device_count = 1,
                .mesh_rows = 1,
                .mesh_cols = 1,
                .compute_grid_x = 8,
                .compute_grid_y = 8,
            },
        .transpose_a = false,
        .transpose_b = false,
        .has_bias = false,
        .has_activation = false,
        .untilize_out = false,
        .bcast_batch = std::nullopt,
        .run_batched = false,
        .activation_op = std::nullopt,
        .activation_param_f32_bits = {},
    };
}

Resolution resolve_with(const Mode mode, const Eligibility& eligibility) {
    auto request = exact_request(eligibility.call.domain);
    request.call = eligibility.call;
    request.transpose_a = eligibility.transpose_a;
    request.transpose_b = eligibility.transpose_b;
    request.has_bias = eligibility.has_bias;
    request.has_activation = eligibility.has_activation;
    request.untilize_out = eligibility.untilize_out;
    request.bcast_batch = eligibility.has_bcast_batch ? std::make_optional(false) : std::nullopt;
    request.run_batched = eligibility.input_b_batched;
    request.activation_op = eligibility.has_activation ? std::make_optional(0U) : std::nullopt;
    return resolve(mode, request, eligibility);
}

TEST(MatmulConfigRegistry, ModeFreezesAtFirstUse) {
    const auto original = ttnn::CONFIG.get<"matmul_registry_mode">();
    reset_startup_mode_for_testing();
    ttnn::CONFIG.set<"matmul_registry_mode">(Mode::Shadow);
    EXPECT_EQ(current_mode(), Mode::Shadow);
    ttnn::CONFIG.set<"matmul_registry_mode">(Mode::On);
    EXPECT_EQ(current_mode(), Mode::Shadow);

    ttnn::CONFIG.set<"matmul_registry_mode">(original);
    reset_startup_mode_for_testing();
    const auto result =
        resolve_with(Mode::Off, Eligibility{.call = CallSemantics{.domain = OperationDomain::DenseMatmul}});
    EXPECT_EQ(result.reason, ResolutionReason::Disabled);
    EXPECT_FALSE(result.recipe.has_value());
}

TEST(MatmulConfigRegistry, EmptyOnTableFallsBack) {
    const auto result =
        resolve_with(Mode::On, Eligibility{.call = CallSemantics{.domain = OperationDomain::DenseMatmul}});
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

    auto invalid_recipe = basic_recipe();
    invalid_recipe.untilize_out = true;
    EXPECT_EQ(
        execution_action(Mode::On, Resolution{.reason = ResolutionReason::CertifiedMatch, .recipe = invalid_recipe}),
        ExecutionAction::Fallback);
}

TEST(MatmulConfigRegistry, ExactRequestDoesNotCrossMatchKeyAxes) {
    const auto original = exact_request();

    auto changed = original;
    changed.workload.logical_m++;
    EXPECT_NE(changed, original);
    changed = original;
    changed.workload.logical_k++;
    EXPECT_NE(changed, original);
    changed = original;
    changed.workload.logical_n++;
    EXPECT_NE(changed, original);
    changed = original;
    changed.workload.padded_k++;
    EXPECT_NE(changed, original);
    changed = original;
    changed.input_a.dtype = DataType::FLOAT32;
    EXPECT_NE(changed, original);
    changed = original;
    changed.input_b.layout = tt::tt_metal::Layout::ROW_MAJOR;
    EXPECT_NE(changed, original);
    changed = original;
    changed.output.buffer_type = BufferType::L1;
    EXPECT_NE(changed, original);
    changed = original;
    changed.output.memory_layout = TensorMemoryLayout::WIDTH_SHARDED;
    EXPECT_NE(changed, original);
    changed = original;
    changed.output.tile_width = 16;
    EXPECT_NE(changed, original);
    changed = original;
    changed.device.architecture++;
    EXPECT_NE(changed, original);
    changed = original;
    changed.device.compute_grid_x++;
    EXPECT_NE(changed, original);
    changed = original;
    changed.device.mesh_cols++;
    EXPECT_NE(changed, original);
    changed = original;
    changed.call.domain = OperationDomain::Linear;
    EXPECT_NE(changed, original);
}

TEST(MatmulConfigRegistry, RequestAndEligibilitySemanticsMustAgree) {
    const auto result = resolve(
        Mode::On,
        exact_request(OperationDomain::DenseMatmul),
        Eligibility{.call = CallSemantics{.domain = OperationDomain::Linear}});
    EXPECT_EQ(result.reason, ResolutionReason::InconsistentRequest);
}

TEST(MatmulConfigRegistry, TelemetryIsBoundedAndResettable) {
    reset_stats_for_testing();
    const Resolution hit{.reason = ResolutionReason::CertifiedMatch, .recipe = basic_recipe()};
    record_resolution(Mode::Shadow, OperationDomain::DenseMatmul, hit, ExecutionAction::ObserveOnly);
    record_resolution(
        Mode::On,
        OperationDomain::Linear,
        Resolution{.reason = ResolutionReason::EmptyRegistry},
        ExecutionAction::Fallback);

    auto snapshot = stats_snapshot();
    const auto& dense = snapshot.domains[static_cast<std::size_t>(OperationDomain::DenseMatmul)];
    EXPECT_EQ(dense.resolution_attempts, 1);
    EXPECT_EQ(dense.certified_hits, 1);
    EXPECT_EQ(dense.shadow_would_hits, 1);
    EXPECT_EQ(dense.reasons[static_cast<std::size_t>(ResolutionReason::CertifiedMatch)], 1);
    const auto& linear = snapshot.domains[static_cast<std::size_t>(OperationDomain::Linear)];
    EXPECT_EQ(linear.resolution_attempts, 1);
    EXPECT_EQ(linear.fallbacks, 1);

    reset_stats_for_testing();
    snapshot = stats_snapshot();
    EXPECT_EQ(snapshot.domains[static_cast<std::size_t>(OperationDomain::DenseMatmul)].resolution_attempts, 0);
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
        resolve_with(Mode::On, Eligibility{.call = CallSemantics{.domain = OperationDomain::DenseMatmul}}).reason,
        ResolutionReason::EmptyRegistry);
    EXPECT_EQ(
        resolve_with(
            Mode::On,
            Eligibility{
                .call = CallSemantics{.domain = OperationDomain::Linear},
                .has_bias = true,
                .has_activation = true,
                .transpose_b = true})
            .reason,
        ResolutionReason::EmptyRegistry);
    EXPECT_EQ(
        resolve_with(
            Mode::On,
            Eligibility{
                .call =
                    CallSemantics{.domain = OperationDomain::Addmm, .alpha_f32_bits = 0x3f800000, .beta_f32_bits = 0}})
            .reason,
        ResolutionReason::EmptyRegistry);
}

TEST(MatmulConfigRegistry, SharedCallersAreNeverEligible) {
    const auto result =
        resolve_with(Mode::On, Eligibility{.call = CallSemantics{.domain = OperationDomain::IneligibleSharedCaller}});
    EXPECT_EQ(result.reason, ResolutionReason::IneligibleOperationDomain);
}

TEST(MatmulConfigRegistry, AddmmRequiresExactScalarBits) {
    EXPECT_EQ(
        resolve_with(Mode::On, Eligibility{.call = CallSemantics{.domain = OperationDomain::Addmm}}).reason,
        ResolutionReason::MalformedOperationSemantics);
    EXPECT_EQ(
        resolve_with(
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
        resolve_with(
            Mode::On,
            Eligibility{.call = CallSemantics{.domain = OperationDomain::DenseMatmul}, .has_program_config = true})
            .reason,
        ResolutionReason::ExplicitOverride);
    EXPECT_EQ(
        resolve_with(
            Mode::On,
            Eligibility{.call = CallSemantics{.domain = OperationDomain::Linear}, .has_compute_kernel_config = true})
            .reason,
        ResolutionReason::ExplicitOverride);
    EXPECT_EQ(
        resolve_with(
            Mode::On,
            Eligibility{
                .call =
                    CallSemantics{.domain = OperationDomain::Addmm, .alpha_f32_bits = 0x3f800000, .beta_f32_bits = 0},
                .has_user_core_grid = true})
            .reason,
        ResolutionReason::ExplicitOverride);
}

TEST(MatmulConfigRegistry, InconsistentIoContractIsNeverLookedUp) {
    const auto result = resolve_with(
        Mode::On,
        Eligibility{
            .call = CallSemantics{.domain = OperationDomain::DenseMatmul},
            .io_contract_status = IoContractStatus::OutputTileConflict});
    EXPECT_EQ(result.reason, ResolutionReason::InconsistentIoContract);
    EXPECT_FALSE(result.recipe.has_value());
}

TEST(MatmulConfigRegistry, UnsupportedV1SemanticsFallBack) {
    EXPECT_EQ(
        resolve_with(
            Mode::On, Eligibility{.call = CallSemantics{.domain = OperationDomain::DenseMatmul}, .has_bias = true})
            .reason,
        ResolutionReason::UnsupportedSemantics);
    EXPECT_EQ(
        resolve_with(
            Mode::On,
            Eligibility{.call = CallSemantics{.domain = OperationDomain::DenseMatmul}, .input_a_sharded = true})
            .reason,
        ResolutionReason::UnsupportedSemantics);
    EXPECT_EQ(
        resolve_with(
            Mode::On,
            Eligibility{.call = CallSemantics{.domain = OperationDomain::DenseMatmul}, .input_b_batched = true})
            .reason,
        ResolutionReason::UnsupportedSemantics);
    EXPECT_EQ(
        resolve_with(
            Mode::On, Eligibility{.call = CallSemantics{.domain = OperationDomain::DenseMatmul}, .transpose_b = true})
            .reason,
        ResolutionReason::UnsupportedSemantics);
}

}  // namespace
}  // namespace ttnn::operations::matmul::registry
