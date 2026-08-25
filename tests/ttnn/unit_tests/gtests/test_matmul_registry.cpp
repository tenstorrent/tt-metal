// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <thread>
#include <variant>

#include <tt_stl/reflection.hpp>

#include "ttnn/operations/matmul/device/config/matmul_config_registry.hpp"

namespace ttnn::operations::matmul::registry {
namespace {

template <typename T>
concept HasRegistryEntryId = requires(T value) { value.registry_entry_id; };

template <typename T>
concept HasRegistryMode = requires(T value) { value.registry_mode; };

static_assert(!HasRegistryEntryId<ttnn::prim::MatmulParams>);
static_assert(!HasRegistryMode<ttnn::prim::MatmulParams>);

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

std::size_t resolver_invocations = 0;
CallSemantics resolver_observed_call;

Resolution counting_certified_resolver(
    const Mode /*mode*/, const MatmulRegistryRequest& request, const Eligibility& eligibility) noexcept {
    static const auto recipe = basic_recipe();
    ++resolver_invocations;
    resolver_observed_call = eligibility.call;
    if (request.call != eligibility.call) {
        return {.reason = ResolutionReason::InconsistentRequest};
    }
    return {.reason = ResolutionReason::CertifiedMatch, .recipe = &recipe};
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
    EXPECT_EQ(result.recipe, nullptr);
}

TEST(MatmulConfigRegistry, InvalidConfiguredModeFreezesFailClosed) {
    const auto original = ttnn::CONFIG.get<"matmul_registry_mode">();
    reset_startup_mode_for_testing();
    ttnn::CONFIG.set<"matmul_registry_mode">(static_cast<Mode>(0xff));
    EXPECT_EQ(current_mode(), Mode::Off);
    EXPECT_EQ(stats_snapshot().frozen_mode, Mode::Off);

    ttnn::CONFIG.set<"matmul_registry_mode">(original);
    reset_startup_mode_for_testing();
}

TEST(MatmulConfigRegistry, ConcurrentFirstUseFreezesOneMode) {
    const auto original = ttnn::CONFIG.get<"matmul_registry_mode">();
    reset_startup_mode_for_testing();
    ttnn::CONFIG.set<"matmul_registry_mode">(Mode::Shadow);

    constexpr std::size_t thread_count = 16;
    std::array<Mode, thread_count> observed{};
    std::array<std::thread, thread_count> threads;
    for (std::size_t index = 0; index < threads.size(); ++index) {
        threads[index] = std::thread([index, &observed] { observed[index] = current_mode(); });
    }
    for (auto& thread : threads) {
        thread.join();
    }
    for (const auto mode : observed) {
        EXPECT_EQ(mode, Mode::Shadow);
    }

    ttnn::CONFIG.set<"matmul_registry_mode">(original);
    reset_startup_mode_for_testing();
}

TEST(MatmulConfigRegistry, EmptyOnTableFallsBack) {
    const auto result =
        resolve_with(Mode::On, Eligibility{.call = CallSemantics{.domain = OperationDomain::DenseMatmul}});
    EXPECT_EQ(result.reason, ResolutionReason::EmptyRegistry);
    EXPECT_EQ(result.recipe, nullptr);
}

TEST(MatmulConfigRegistry, PublicOperationCallSemanticsAreDisjointAndExact) {
    EXPECT_EQ(dense_matmul_call_semantics().domain, OperationDomain::DenseMatmul);
    EXPECT_FALSE(dense_matmul_call_semantics().alpha_f32_bits.has_value());
    EXPECT_FALSE(dense_matmul_call_semantics().beta_f32_bits.has_value());

    EXPECT_EQ(linear_call_semantics().domain, OperationDomain::Linear);
    EXPECT_FALSE(linear_call_semantics().alpha_f32_bits.has_value());
    EXPECT_FALSE(linear_call_semantics().beta_f32_bits.has_value());

    const auto addmm = addmm_call_semantics(1.0F, -0.0F);
    EXPECT_EQ(addmm.domain, OperationDomain::Addmm);
    EXPECT_EQ(addmm.alpha_f32_bits, 0x3f800000U);
    EXPECT_EQ(addmm.beta_f32_bits, 0x80000000U);
}

TEST(MatmulConfigRegistry, DispatchCardinalityAndCacheIdentityAreModeSafe) {
    const auto request = exact_request();
    const auto eligibility = Eligibility{.call = request.call};
    ttnn::prim::MatmulParams legacy_parameters;
    legacy_parameters.output_dtype = DataType::FLOAT32;
    const auto legacy_hash = ttsl::hash::hash_objects_with_default_seed(legacy_parameters);

    resolver_invocations = 0;
    const auto off =
        resolve_for_dispatch(Mode::Off, request, eligibility, legacy_parameters, &counting_certified_resolver);
    EXPECT_EQ(resolver_invocations, 0);
    EXPECT_EQ(off.resolution.reason, ResolutionReason::Disabled);
    EXPECT_EQ(off.action, ExecutionAction::Fallback);
    EXPECT_FALSE(off.materialized_parameters.has_value());
    EXPECT_EQ(ttsl::hash::hash_objects_with_default_seed(legacy_parameters), legacy_hash);

    resolver_invocations = 0;
    const auto shadow =
        resolve_for_dispatch(Mode::Shadow, request, eligibility, legacy_parameters, &counting_certified_resolver);
    EXPECT_EQ(resolver_invocations, 1);
    EXPECT_EQ(resolver_observed_call, dense_matmul_call_semantics());
    EXPECT_EQ(shadow.action, ExecutionAction::ObserveOnly);
    EXPECT_FALSE(shadow.materialized_parameters.has_value());
    EXPECT_EQ(ttsl::hash::hash_objects_with_default_seed(legacy_parameters), legacy_hash);

    resolver_invocations = 0;
    const auto on =
        resolve_for_dispatch(Mode::On, request, eligibility, legacy_parameters, &counting_certified_resolver);
    EXPECT_EQ(resolver_invocations, 1);
    EXPECT_EQ(on.action, ExecutionAction::ApplyRecipe);
    ASSERT_TRUE(on.materialized_parameters.has_value());
    EXPECT_NE(ttsl::hash::hash_objects_with_default_seed(on.materialized_parameters.value()), legacy_hash);
    EXPECT_EQ(ttsl::hash::hash_objects_with_default_seed(legacy_parameters), legacy_hash);
}

TEST(MatmulConfigRegistry, DispatchDoesNotResolveAnIncompleteRequest) {
    const auto eligibility = Eligibility{.call = dense_matmul_call_semantics()};
    const ttnn::prim::MatmulParams legacy_parameters;
    resolver_invocations = 0;

    const auto result =
        resolve_for_dispatch(Mode::On, std::nullopt, eligibility, legacy_parameters, &counting_certified_resolver);

    EXPECT_EQ(resolver_invocations, 0);
    EXPECT_EQ(result.resolution.reason, ResolutionReason::IncompleteRequest);
    EXPECT_EQ(result.action, ExecutionAction::Fallback);
    EXPECT_FALSE(result.materialized_parameters.has_value());
}

TEST(MatmulConfigRegistry, OnTraceCaptureRejectsBeforeResolver) {
    const auto request = exact_request();
    const ttnn::prim::MatmulParams legacy_parameters;
    const auto eligibility = Eligibility{.call = request.call, .trace_capture_active = true};
    resolver_invocations = 0;

    const auto off =
        resolve_for_dispatch(Mode::Off, request, eligibility, legacy_parameters, &counting_certified_resolver);
    EXPECT_EQ(resolver_invocations, 0);
    EXPECT_EQ(off.resolution.reason, ResolutionReason::Disabled);
    EXPECT_EQ(off.action, ExecutionAction::Fallback);

    const auto result =
        resolve_for_dispatch(Mode::On, request, eligibility, legacy_parameters, &counting_certified_resolver);

    EXPECT_EQ(resolver_invocations, 0);
    EXPECT_EQ(result.resolution.reason, ResolutionReason::TraceCaptureUnsupported);
    EXPECT_EQ(result.action, ExecutionAction::Fallback);
    EXPECT_FALSE(result.materialized_parameters.has_value());
    EXPECT_EQ(resolve(Mode::On, request, eligibility).reason, ResolutionReason::TraceCaptureUnsupported);
}

TEST(MatmulConfigRegistry, ShadowTraceCaptureObservesWithoutMutation) {
    const auto request = exact_request();
    const auto eligibility = Eligibility{.call = request.call, .trace_capture_active = true};
    ttnn::prim::MatmulParams legacy_parameters;
    legacy_parameters.output_dtype = DataType::FLOAT32;
    const auto legacy_hash = ttsl::hash::hash_objects_with_default_seed(legacy_parameters);
    resolver_invocations = 0;

    const auto result =
        resolve_for_dispatch(Mode::Shadow, request, eligibility, legacy_parameters, &counting_certified_resolver);

    EXPECT_EQ(resolver_invocations, 1);
    EXPECT_EQ(result.resolution.reason, ResolutionReason::CertifiedMatch);
    EXPECT_EQ(result.action, ExecutionAction::ObserveOnly);
    EXPECT_FALSE(result.materialized_parameters.has_value());
    EXPECT_EQ(ttsl::hash::hash_objects_with_default_seed(legacy_parameters), legacy_hash);
}

TEST(MatmulConfigRegistry, TraceCaptureRejectionHasBoundedTelemetry) {
    reset_stats_for_testing();
    const auto rejection = Resolution{.reason = ResolutionReason::TraceCaptureUnsupported};
    record_resolution(Mode::On, OperationDomain::DenseMatmul, rejection, ExecutionAction::Fallback);

    const auto snapshot = stats_snapshot();
    const auto& dense = snapshot.domains[static_cast<std::size_t>(OperationDomain::DenseMatmul)];
    EXPECT_EQ(dense.resolution_attempts, 1);
    EXPECT_EQ(dense.fallbacks, 1);
    EXPECT_EQ(dense.reasons[static_cast<std::size_t>(ResolutionReason::TraceCaptureUnsupported)], 1);
    EXPECT_EQ(dense.certified_hits, 0);
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
    const auto recipe = basic_recipe();
    const Resolution certified_hit{.reason = ResolutionReason::CertifiedMatch, .recipe = &recipe};

    EXPECT_EQ(execution_action(Mode::Off, certified_hit), ExecutionAction::Fallback);
    EXPECT_EQ(execution_action(Mode::Shadow, certified_hit), ExecutionAction::ObserveOnly);
    EXPECT_EQ(execution_action(Mode::On, certified_hit), ExecutionAction::ApplyRecipe);
    EXPECT_EQ(
        execution_action(Mode::On, Resolution{.reason = ResolutionReason::EmptyRegistry}), ExecutionAction::Fallback);

    auto invalid_recipe = basic_recipe();
    invalid_recipe.untilize_out = true;
    EXPECT_EQ(
        execution_action(Mode::On, Resolution{.reason = ResolutionReason::CertifiedMatch, .recipe = &invalid_recipe}),
        ExecutionAction::Fallback);
}

TEST(MatmulConfigRegistry, SyntheticHitIsExactAndMaterializesOnlyInOn) {
    const auto request = exact_request();
    const auto eligibility = Eligibility{.call = request.call};
    auto recipe = basic_recipe();
    recipe.compute_kernel_config.math_approx_mode = false;

    auto candidate_request = request;
    const auto shadow =
        resolve_with_synthetic_candidate_for_testing(Mode::Shadow, request, eligibility, candidate_request, recipe);
    ASSERT_EQ(shadow.reason, ResolutionReason::CertifiedMatch);
    ASSERT_EQ(shadow.recipe, &recipe);

    ttnn::prim::MatmulParams legacy;
    legacy.output_dtype = DataType::FLOAT32;
    legacy.transpose_a = true;
    const auto shadow_parameters = materialize_parameters_for_execution(Mode::Shadow, shadow, legacy);
    EXPECT_FALSE(shadow_parameters.has_value());
    EXPECT_FALSE(legacy.program_config.has_value());
    EXPECT_FALSE(legacy.compute_kernel_config.has_value());
    EXPECT_TRUE(legacy.transpose_a);

    const auto on =
        resolve_with_synthetic_candidate_for_testing(Mode::On, request, eligibility, candidate_request, recipe);
    auto on_parameters = materialize_parameters_for_execution(Mode::On, on, legacy);
    ASSERT_TRUE(on_parameters.has_value());
    ASSERT_TRUE(on_parameters->program_config.has_value());
    EXPECT_TRUE(std::holds_alternative<MatmulMultiCoreProgramConfig>(*on_parameters->program_config));
    ASSERT_TRUE(on_parameters->compute_kernel_config.has_value());
    EXPECT_FALSE(on_parameters->compute_kernel_config->math_approx_mode);
    EXPECT_EQ(on_parameters->untilize_out, recipe.untilize_out);
    EXPECT_EQ(on_parameters->output_dtype, legacy.output_dtype);
    EXPECT_EQ(on_parameters->transpose_a, legacy.transpose_a);
    EXPECT_FALSE(legacy.program_config.has_value());
    EXPECT_FALSE(legacy.compute_kernel_config.has_value());

    candidate_request.workload.logical_m++;
    const auto miss =
        resolve_with_synthetic_candidate_for_testing(Mode::On, request, eligibility, candidate_request, recipe);
    EXPECT_EQ(miss.reason, ResolutionReason::EmptyRegistry);
    EXPECT_EQ(miss.recipe, nullptr);
    EXPECT_FALSE(materialize_parameters_for_execution(Mode::On, miss, legacy).has_value());
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
    const auto recipe = basic_recipe();
    const Resolution hit{.reason = ResolutionReason::CertifiedMatch, .recipe = &recipe};
    record_resolution(Mode::Off, OperationDomain::DenseMatmul, hit, ExecutionAction::Fallback);
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

TEST(MatmulConfigRegistry, TelemetryCountersAreConcurrent) {
    reset_stats_for_testing();
    constexpr std::size_t thread_count = 8;
    constexpr std::size_t iterations = 1000;
    std::array<std::thread, thread_count> threads;
    for (auto& thread : threads) {
        thread = std::thread([] {
            for (std::size_t iteration = 0; iteration < iterations; ++iteration) {
                record_resolution(
                    Mode::On,
                    OperationDomain::DenseMatmul,
                    Resolution{.reason = ResolutionReason::EmptyRegistry},
                    ExecutionAction::Fallback);
            }
        });
    }
    for (auto& thread : threads) {
        thread.join();
    }

    const auto snapshot = stats_snapshot();
    const auto& dense = snapshot.domains[static_cast<std::size_t>(OperationDomain::DenseMatmul)];
    EXPECT_EQ(dense.resolution_attempts, thread_count * iterations);
    EXPECT_EQ(dense.fallbacks, thread_count * iterations);
    EXPECT_EQ(dense.reasons[static_cast<std::size_t>(ResolutionReason::EmptyRegistry)], thread_count * iterations);
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
    EXPECT_EQ(result.recipe, nullptr);
}

TEST(MatmulConfigRegistry, UnsupportedV1SemanticsFallBack) {
    const auto dense_call = CallSemantics{.domain = OperationDomain::DenseMatmul};
    const auto expect_unsupported = [](const Eligibility& eligibility) {
        EXPECT_EQ(resolve_with(Mode::On, eligibility).reason, ResolutionReason::UnsupportedSemantics);
    };

    expect_unsupported(Eligibility{.call = dense_call, .has_bias = true});
    expect_unsupported(Eligibility{.call = dense_call, .has_activation = true});
    expect_unsupported(Eligibility{.call = dense_call, .has_optional_output = true});
    expect_unsupported(Eligibility{.call = dense_call, .has_output_tile = true});
    expect_unsupported(Eligibility{.call = dense_call, .has_global_cb = true});
    expect_unsupported(Eligibility{.call = dense_call, .has_sub_device = true});
    expect_unsupported(Eligibility{.call = dense_call, .has_bcast_batch = true});
    expect_unsupported(Eligibility{.call = dense_call, .untilize_out = true});
    expect_unsupported(Eligibility{.call = dense_call, .input_a_sharded = true});
    expect_unsupported(Eligibility{.call = dense_call, .input_b_sharded = true});
    expect_unsupported(Eligibility{.call = dense_call, .output_sharded = true});
    expect_unsupported(Eligibility{.call = dense_call, .input_b_batched = true});
    expect_unsupported(Eligibility{.call = dense_call, .transpose_a = true});
    expect_unsupported(Eligibility{.call = dense_call, .transpose_b = true});
}

}  // namespace
}  // namespace ttnn::operations::matmul::registry
