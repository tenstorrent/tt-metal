// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <optional>
#include <variant>

#include <umd/device/types/arch.hpp>

#include "ttnn/operations/matmul/device/config/matmul_config_registry.hpp"

namespace ttnn::operations::matmul::registry {
namespace {

using tt::tt_metal::BufferType;
using tt::tt_metal::DataType;
using tt::tt_metal::Layout;
using tt::tt_metal::TensorMemoryLayout;

compact::Sha256 digest(const std::uint8_t byte) {
    compact::Sha256 result{};
    result.fill(byte);
    return result;
}

TensorRequest tensor_request(const DataType dtype = DataType::BFLOAT16) {
    return TensorRequest{
        .dtype = dtype,
        .layout = Layout::TILE,
        .memory_layout = TensorMemoryLayout::INTERLEAVED,
        .buffer_type = BufferType::DRAM,
        .tile_height = 32,
        .tile_width = 32};
}

CallSemantics semantics(const OperationDomain domain) {
    return domain == OperationDomain::Addmm ? addmm_call_semantics(1.0F, 0.0F) : CallSemantics{.domain = domain};
}

MatmulRegistryRequest request(const OperationDomain domain = OperationDomain::DenseMatmul) {
    return MatmulRegistryRequest{
        .schema_version = 1,
        .call = semantics(domain),
        .workload = {.logical_m = 64, .logical_k = 64, .logical_n = 64, .padded_m = 64, .padded_k = 64, .padded_n = 64},
        .input_a = tensor_request(),
        .input_b = tensor_request(),
        .output = tensor_request(),
        .device = {
            .architecture = static_cast<std::uint32_t>(tt::ARCH::BLACKHOLE),
            .device_count = 1,
            .mesh_rows = 1,
            .mesh_cols = 1,
            .compute_grid_x = 13,
            .compute_grid_y = 10}};
}

Eligibility eligibility(const OperationDomain domain = OperationDomain::DenseMatmul) {
    return Eligibility{.call = semantics(domain)};
}

compact::ComputeKernelDescriptor kernel(const compact::ThrottleLevel throttle = compact::ThrottleLevel::NoThrottle) {
    return compact::ComputeKernelDescriptor{
        .math_fidelity = compact::MathFidelity::HiFi2,
        .throttle_level = throttle,
        .math_approx_mode = false,
        .fp32_dest_acc_en = false,
        .packer_l1_acc = true,
        .dst_full_sync_en = false};
}

compact::ProgramConfigDescriptor reuse_program(const std::uint16_t grid_x = 2) {
    return compact::ProgramConfigDescriptor{
        .family = compact::ProgramFamily::MultiCoreReuse,
        .compute_grid_x = grid_x,
        .compute_grid_y = 2,
        .in0_block_w = 1,
        .out_subblock_h = 1,
        .out_subblock_w = 1,
        .per_core_m = 1,
        .per_core_n = 2};
}

compact::ProgramConfigDescriptor multicast_1d_program() {
    return compact::ProgramConfigDescriptor{
        .family = compact::ProgramFamily::MultiCast1D,
        .compute_grid_x = 2,
        .compute_grid_y = 2,
        .in0_block_w = 1,
        .out_subblock_h = 1,
        .out_subblock_w = 1,
        .per_core_m = 1,
        .per_core_n = 2,
        .out_block_h = 1,
        .out_block_w = 2,
        .num_global_cb_receivers = 1,
        .fuse_batch = true,
        .mcast_in0 = false};
}

compact::ProgramConfigDescriptor multicast_2d_program() {
    return compact::ProgramConfigDescriptor{
        .family = compact::ProgramFamily::MultiCast2D,
        .compute_grid_x = 2,
        .compute_grid_y = 2,
        .in0_block_w = 1,
        .out_subblock_h = 1,
        .out_subblock_w = 1,
        .per_core_m = 1,
        .per_core_n = 1,
        .out_block_h = 1,
        .out_block_w = 1,
        .fuse_batch = true,
        .mcast_in0 = false,
        .transpose_mcast = false};
}

compact::TableMetadata metadata(const bool exact = true) {
    return compact::TableMetadata{
        .lock_schema_version = 2,
        .key_schema_version = 1,
        .exact_recipe_evidence_schema_version = exact ? std::uint16_t{2} : std::uint16_t{0},
        .matmul_kernel_equivalence_schema_version = exact ? std::uint16_t{1} : std::uint16_t{0},
        .content_sha256 = digest(1),
        .semantic_source_sha256 = digest(2)};
}

compact::ProgramConfigExactEntry exact_entry(
    const MatmulRegistryRequest& runtime_request,
    const compact::ProgramConfigDescriptor& program = reuse_program(),
    const compact::ComputeKernelDescriptor& ckc = kernel(),
    const std::optional<std::uint16_t> grid_x = std::nullopt) {
    auto key = *compact_registry_key(runtime_request);
    key.compute_grid_x = grid_x.value_or(key.compute_grid_x);
    return compact::ProgramConfigExactEntry{
        .entry_id = digest(11), .key = key, .program_config = program, .compute_kernel_config = ckc};
}

TEST(MatmulConfigRegistry, BlackholeKeyUsesNativeArchitectureAndPortablePhysicalFields) {
    const auto key = compact_registry_key(request());
    ASSERT_TRUE(key.has_value());
    EXPECT_EQ(key->architecture, static_cast<std::uint32_t>(tt::ARCH::BLACKHOLE));
    EXPECT_EQ(key->board_capability_class, 0U);
    EXPECT_EQ(key->topology_sha256, compact::Sha256{});
    EXPECT_EQ(key->compute_grid_x, 13);
}

TEST(MatmulConfigRegistry, ExactMatchPreservesHarvestedGridCohorts) {
    auto live = request();
    const std::array entries{
        exact_entry(live, reuse_program(2), kernel(compact::ThrottleLevel::Throttle1), 11),
        exact_entry(live, reuse_program(2), kernel(compact::ThrottleLevel::Throttle2), 12),
        exact_entry(live, reuse_program(2), kernel(compact::ThrottleLevel::Throttle3), 13)};
    for (const std::uint32_t grid_x : {11U, 12U, 13U}) {
        live.device.compute_grid_x = grid_x;
        const auto result = resolve_with_compact_table_for_testing(live, eligibility(), metadata(), entries);
        EXPECT_EQ(result.reason, ResolutionReason::CertifiedMatch);
        ASSERT_TRUE(result.program_config.has_value());
        EXPECT_EQ(result.program_config->compute_grid_x, 2);
        ASSERT_TRUE(result.compute_kernel_config.has_value());
        EXPECT_EQ(result.compute_kernel_config->throttle_level, static_cast<compact::ThrottleLevel>(grid_x - 10));
    }
    live.device.compute_grid_x = 10;
    EXPECT_EQ(
        resolve_with_compact_table_for_testing(live, eligibility(), metadata(), entries).reason,
        ResolutionReason::EmptyRegistry);
}

TEST(MatmulConfigRegistry, ExactArchitectureNeverCrossMatches) {
    const auto dense = request();
    const auto entry = exact_entry(dense);
    auto wrong_arch = dense;
    wrong_arch.device.architecture = static_cast<std::uint32_t>(tt::ARCH::WORMHOLE_B0);
    EXPECT_EQ(
        resolve_with_compact_table_for_testing(wrong_arch, eligibility(), metadata(), {&entry, 1}).reason,
        ResolutionReason::EmptyRegistry);
}

TEST(MatmulConfigRegistry, KernelEquivalentPublicWrappersReuseDenseMeasurements) {
    const auto dense = request();
    const auto entry = exact_entry(dense, reuse_program(2), kernel(compact::ThrottleLevel::Throttle3));
    for (const auto domain : {OperationDomain::Linear, OperationDomain::Addmm}) {
        const auto result = resolve_with_compact_table_for_testing(
            request(domain), eligibility(domain), metadata(), std::span{&entry, std::size_t{1}});
        EXPECT_EQ(result.reason, ResolutionReason::CertifiedMatch);
        ASSERT_TRUE(result.compute_kernel_config.has_value());
        EXPECT_EQ(result.compute_kernel_config->throttle_level, compact::ThrottleLevel::Throttle3);
    }
}

TEST(MatmulConfigRegistry, KernelEquivalentWrapperFallbackRequiresBoundEvidence) {
    const auto dense = request();
    const auto entry = exact_entry(dense);
    auto unproven = metadata();
    unproven.matmul_kernel_equivalence_schema_version = 0;
    EXPECT_EQ(
        resolve_with_compact_table_for_testing(
            request(OperationDomain::Linear), eligibility(OperationDomain::Linear), unproven, {&entry, 1})
            .reason,
        ResolutionReason::EmptyRegistry);
    EXPECT_EQ(
        resolve_with_compact_table_for_testing(dense, eligibility(), unproven, {&entry, 1}).reason,
        ResolutionReason::CertifiedMatch);
}

TEST(MatmulConfigRegistry, KernelEquivalentWrapperFallbackStillFailsClosedBeforeLookup) {
    const auto dense = request();
    const auto entry = exact_entry(dense);

    auto linear = request(OperationDomain::Linear);
    auto linear_eligibility = eligibility(OperationDomain::Linear);
    linear_eligibility.has_bias = true;
    EXPECT_EQ(
        resolve_with_compact_table_for_testing(linear, linear_eligibility, metadata(), {&entry, 1}).reason,
        ResolutionReason::UnsupportedSemantics);
    linear_eligibility = eligibility(OperationDomain::Linear);
    linear_eligibility.has_activation = true;
    EXPECT_EQ(
        resolve_with_compact_table_for_testing(linear, linear_eligibility, metadata(), {&entry, 1}).reason,
        ResolutionReason::UnsupportedSemantics);
    linear_eligibility = eligibility(OperationDomain::Linear);
    linear_eligibility.transpose_b = true;
    linear.transpose_b = true;
    EXPECT_EQ(
        resolve_with_compact_table_for_testing(linear, linear_eligibility, metadata(), {&entry, 1}).reason,
        ResolutionReason::UnsupportedSemantics);

    auto addmm = request(OperationDomain::Addmm);
    auto addmm_eligibility = eligibility(OperationDomain::Addmm);
    addmm.call = addmm_call_semantics(2.0F, 0.0F);
    addmm_eligibility.call = addmm.call;
    EXPECT_EQ(
        resolve_with_compact_table_for_testing(addmm, addmm_eligibility, metadata(), {&entry, 1}).reason,
        ResolutionReason::MalformedOperationSemantics);
    addmm.call = addmm_call_semantics(1.0F, 1.0F);
    addmm_eligibility.call = addmm.call;
    EXPECT_EQ(
        resolve_with_compact_table_for_testing(addmm, addmm_eligibility, metadata(), {&entry, 1}).reason,
        ResolutionReason::UnsupportedSemantics);
}

TEST(MatmulConfigRegistry, OperationSpecificRecipePrecedesDenseWrapperFallback) {
    const auto dense = request();
    const auto linear = request(OperationDomain::Linear);
    const std::array entries{
        exact_entry(dense, reuse_program(2), kernel(compact::ThrottleLevel::Throttle1)),
        exact_entry(linear, reuse_program(2), kernel(compact::ThrottleLevel::Throttle4))};
    const auto result =
        resolve_with_compact_table_for_testing(linear, eligibility(OperationDomain::Linear), metadata(), entries);
    EXPECT_EQ(result.reason, ResolutionReason::CertifiedMatch);
    ASSERT_TRUE(result.compute_kernel_config.has_value());
    EXPECT_EQ(result.compute_kernel_config->throttle_level, compact::ThrottleLevel::Throttle4);

    auto unproven = metadata();
    unproven.matmul_kernel_equivalence_schema_version = 0;
    const auto direct_result =
        resolve_with_compact_table_for_testing(linear, eligibility(OperationDomain::Linear), unproven, entries);
    EXPECT_EQ(direct_result.reason, ResolutionReason::CertifiedMatch);
    ASSERT_TRUE(direct_result.compute_kernel_config.has_value());
    EXPECT_EQ(direct_result.compute_kernel_config->throttle_level, compact::ThrottleLevel::Throttle4);
}

TEST(MatmulConfigRegistry, ExactKeyBindsBothInputAndOutputDtypes) {
    const auto dense = request();
    const auto entry = exact_entry(dense);
    for (const auto dtype : {DataType::BFLOAT8_B, DataType::FLOAT32}) {
        auto changed = dense;
        changed.input_a.dtype = dtype;
        EXPECT_EQ(
            resolve_with_compact_table_for_testing(changed, eligibility(), metadata(), {&entry, 1}).reason,
            ResolutionReason::EmptyRegistry);
        changed = dense;
        changed.input_b.dtype = dtype;
        EXPECT_EQ(
            resolve_with_compact_table_for_testing(changed, eligibility(), metadata(), {&entry, 1}).reason,
            ResolutionReason::EmptyRegistry);
        changed = dense;
        changed.output.dtype = dtype;
        EXPECT_EQ(
            resolve_with_compact_table_for_testing(changed, eligibility(), metadata(), {&entry, 1}).reason,
            ResolutionReason::EmptyRegistry);
    }
}

TEST(MatmulConfigRegistry, ExactCarriesPairedRecipe) {
    const auto req = request();
    const auto entry = exact_entry(req, reuse_program(2), kernel(compact::ThrottleLevel::Throttle3));
    const auto result = resolve_with_compact_table_for_testing(req, eligibility(), metadata(), {&entry, 1});
    EXPECT_EQ(result.reason, ResolutionReason::CertifiedMatch);
    ASSERT_TRUE(result.compute_kernel_config.has_value());
    EXPECT_EQ(result.compute_kernel_config->throttle_level, compact::ThrottleLevel::Throttle3);
}

TEST(MatmulConfigRegistry, EmptyAndMalformedArtifactsFallBack) {
    const auto req = request();
    EXPECT_EQ(
        resolve_with_compact_table_for_testing(req, eligibility(), metadata(false)).reason,
        ResolutionReason::EmptyRegistry);
    auto bad_metadata = metadata();
    bad_metadata.lock_schema_version = 99;
    const auto entry = exact_entry(req);
    EXPECT_EQ(
        resolve_with_compact_table_for_testing(req, eligibility(), bad_metadata, {&entry, 1}).reason,
        ResolutionReason::UnsupportedArtifact);
}

TEST(MatmulConfigRegistry, EveryExplicitTuningAxisBypassesBeforeLookup) {
    const auto req = request();
    ttnn::prim::MatmulParams legacy;
    for (const auto axis : {0, 1, 2}) {
        auto eligible = eligibility();
        eligible.has_program_config = axis == 0;
        eligible.has_compute_kernel_config = axis == 1;
        eligible.has_user_core_grid = axis == 2;
        EXPECT_EQ(preflight_v1_eligibility(eligible), ResolutionReason::ExplicitOverride);
        const auto dispatched = resolve_for_dispatch(req, eligible, legacy);
        EXPECT_EQ(dispatched.resolution.reason, ResolutionReason::ExplicitOverride);
        EXPECT_FALSE(dispatched.materialized_parameters.has_value());
    }
}

TEST(MatmulConfigRegistry, MaterializationSupportsEveryNativeFamily) {
    const auto key = *compact_registry_key(request());
    const auto ckc = kernel();
    const auto reuse = materialize_registry_program_config(key, reuse_program(), ckc);
    const auto one_d = materialize_registry_program_config(key, multicast_1d_program(), ckc);
    const auto two_d = materialize_registry_program_config(key, multicast_2d_program(), ckc);
    ASSERT_TRUE(reuse.has_value());
    ASSERT_TRUE(one_d.has_value());
    ASSERT_TRUE(two_d.has_value());
    EXPECT_TRUE(std::holds_alternative<MatmulMultiCoreReuseProgramConfig>(*reuse));
    EXPECT_TRUE(std::holds_alternative<MatmulMultiCoreReuseMultiCast1DProgramConfig>(*one_d));
    EXPECT_TRUE(std::holds_alternative<MatmulMultiCoreReuseMultiCastProgramConfig>(*two_d));
}

TEST(MatmulConfigRegistry, MaterializationRejectsCandidateOutsideLiveGrid) {
    auto small = request();
    small.device.compute_grid_x = 2;
    const auto key = *compact_registry_key(small);
    EXPECT_FALSE(materialize_registry_program_config(key, reuse_program(3), kernel()).has_value());
}

TEST(MatmulConfigRegistry, ComputeKernelMaterializationCoversThrottleZeroThroughFive) {
    const std::array compact_levels{
        compact::ThrottleLevel::NoThrottle,
        compact::ThrottleLevel::Throttle1,
        compact::ThrottleLevel::Throttle2,
        compact::ThrottleLevel::Throttle3,
        compact::ThrottleLevel::Throttle4,
        compact::ThrottleLevel::Throttle5};
    const std::array native_levels{
        compute_throttle_utils::ThrottleLevel::NO_THROTTLE,
        compute_throttle_utils::ThrottleLevel::LEVEL_1,
        compute_throttle_utils::ThrottleLevel::LEVEL_2,
        compute_throttle_utils::ThrottleLevel::LEVEL_3,
        compute_throttle_utils::ThrottleLevel::LEVEL_4,
        compute_throttle_utils::ThrottleLevel::LEVEL_5};
    for (std::size_t index = 0; index < compact_levels.size(); ++index) {
        const auto result = materialize_registry_compute_kernel_config(kernel(compact_levels[index]));
        ASSERT_TRUE(result.has_value());
        EXPECT_EQ(result->throttle_level, native_levels[index]);
    }
}

TEST(MatmulConfigRegistry, PairedMaterializationPreservesAllCallerOwnedState) {
    const auto req = request();
    const auto entry = exact_entry(req, multicast_2d_program(), kernel(compact::ThrottleLevel::Throttle4));
    const auto selected = resolve_with_compact_table_for_testing(req, eligibility(), metadata(), {&entry, 1});
    ttnn::prim::MatmulParams legacy;
    legacy.output_dtype = DataType::FLOAT32;
    legacy.user_run_batched = false;
    const auto materialized = materialize_parameters_for_execution(selected, legacy);
    ASSERT_TRUE(materialized.has_value());
    ASSERT_TRUE(materialized->program_config.has_value());
    ASSERT_TRUE(materialized->compute_kernel_config.has_value());
    EXPECT_EQ(materialized->output_dtype, legacy.output_dtype);
    EXPECT_EQ(materialized->output_mem_config, legacy.output_mem_config);
    EXPECT_EQ(materialized->compute_kernel_config->throttle_level, compute_throttle_utils::ThrottleLevel::LEVEL_4);
}

TEST(MatmulConfigRegistry, EligibilityFailsClosedOnUnsupportedAndInconsistentCalls) {
    auto eligible = eligibility();
    eligible.has_bias = true;
    EXPECT_EQ(preflight_v1_eligibility(eligible), ResolutionReason::UnsupportedSemantics);
    eligible = eligibility();
    eligible.trace_capture_active = true;
    EXPECT_EQ(preflight_v1_eligibility(eligible), ResolutionReason::TraceCaptureUnsupported);
    auto req = request();
    req.run_batched = true;
    EXPECT_EQ(validate_v1_request_envelope(req, eligibility()), ResolutionReason::InconsistentRequest);
}

TEST(MatmulConfigRegistry, AddmmRequiresExactSafeScalarSemantics) {
    EXPECT_EQ(preflight_v1_eligibility(eligibility(OperationDomain::Addmm)), ResolutionReason::CertifiedMatch);
    auto bad = eligibility(OperationDomain::Addmm);
    bad.call = addmm_call_semantics(2.0F, 0.0F);
    EXPECT_EQ(preflight_v1_eligibility(bad), ResolutionReason::MalformedOperationSemantics);
    bad.call = addmm_call_semantics(1.0F, 1.0F);
    EXPECT_EQ(preflight_v1_eligibility(bad), ResolutionReason::UnsupportedSemantics);
}

}  // namespace
}  // namespace ttnn::operations::matmul::registry
