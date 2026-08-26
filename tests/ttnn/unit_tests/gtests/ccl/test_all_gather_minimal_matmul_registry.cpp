// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <array>
#include <atomic>
#include <cstdint>
#include <stdexcept>

#include <gtest/gtest.h>

#include "ttnn/operations/compute_throttle_utils.hpp"
#include "ttnn/operations/experimental/ccl/all_gather_minimal_matmul_async/registry/agmm_config_registry.hpp"
#include "ttnn/operations/experimental/ccl/all_gather_minimal_matmul_async/registry/agmm_registry_data.hpp"

namespace {

namespace registry = ttnn::experimental::all_gather_minimal_matmul_registry;
namespace compact = registry::compact;
using Mode = registry::Mode;

compact::Sha256 digest(const std::uint8_t value) {
    compact::Sha256 result{};
    result.fill(value);
    return result;
}

compact::TensorDescriptor tensor_descriptor(
    const std::array<std::uint64_t, 4>& logical, const std::array<std::uint64_t, 4>& padded) {
    compact::TensorDescriptor tensor{
        .rank = 4,
        .logical_shape = {logical[0], logical[1], logical[2], logical[3]},
        .padded_shape = {padded[0], padded[1], padded[2], padded[3]},
        .dtype = 1,
        .layout = 1,
        .memory_layout = 0,
        .buffer_type = 1,
        .tile_height = 32,
        .tile_width = 32,
        .memory_config_sha256 = digest(8),
        .tensor_topology_sha256 = digest(9)};
    return tensor;
}

compact::KeyDescriptor valid_key() {
    auto key = compact::KeyDescriptor{};
    key.device = compact::DeviceDescriptor{
        .architecture = 1,
        .board_capability_class = 1,
        .device_count = 8,
        .mesh_rows = 2,
        .mesh_cols = 4,
        .compute_grid_x = 8,
        .compute_grid_y = 8,
        .ordered_mesh_sha256 = digest(4),
        .topology_sha256 = digest(5),
        .runtime_capability_sha256 = digest(3)};
    key.workload = compact::WorkloadDescriptor{
        .logical_m = 64,
        .logical_k = 256,
        .logical_n = 128,
        .padded_m = 64,
        .padded_k = 256,
        .padded_n = 128,
        .batch = 1};
    key.operation.topology = 2;
    key.operation.fsdp_topology = 2;
    key.operation.num_links = 1;
    key.operation.ring_size = 8;
    key.operation.fsdp_ring_size = 1;
    key.operation.chunks = 1;
    key.operation.dim = -1;
    key.operation.output_dtype_present = true;
    key.operation.output_dtype = 1;
    key.operation.output_memory_config_present = true;
    key.operation.output_layout = 1;
    key.operation.output_tile_height = 32;
    key.operation.output_tile_width = 32;
    key.operation.output_memory_config_sha256 = digest(10);
    key.operation.output_tensor_topology_sha256 = digest(11);
    key.input = tensor_descriptor({1, 1, 64, 32}, {1, 1, 64, 32});
    key.weight = tensor_descriptor({1, 1, 256, 128}, {1, 1, 256, 128});
    return key;
}

compact::EntryDescriptor valid_entry() {
    auto entry = compact::EntryDescriptor{};
    entry.entry_id = digest(7);
    entry.key = valid_key();
    entry.replay.config = compact::MinimalMatmulConfigDescriptor{
        .m_block_size = 2,
        .k_block_size = 1,
        .n_block_size = 2,
        .subblock_h = 1,
        .subblock_w = 2,
        .compute_grid_x = 2,
        .compute_grid_y = 2};
    entry.replay.compute_kernel_config = compact::ComputeKernelDescriptor{
        .math_fidelity = static_cast<std::uint32_t>(tt::tt_metal::MathFidelity::HiFi2),
        .math_approx_mode = false,
        .fp32_dest_acc_en = true,
        .packer_l1_acc = true,
        .dst_full_sync_en = false,
        .throttle_level =
            static_cast<std::uint32_t>(ttnn::operations::compute_throttle_utils::ThrottleLevel::NO_THROTTLE)};
    return entry;
}

compact::TableLock valid_lock(const std::size_t entry_count = 1) {
    return compact::TableLock{
        .schema_version = compact::kTableLockSchemaVersion,
        .codegen_recipe_abi = compact::kCodegenRecipeAbi,
        .entry_count = entry_count,
        .metadata = compact::TableMetadata{
            .key_schema_version = compact::kKeySchemaVersion,
            .replay_schema_version = compact::kReplaySchemaVersion,
            .content_sha256 = digest(12),
            .semantic_source_sha256 = digest(1),
            .build_identity_sha256 = digest(2),
            .runtime_capability_sha256 = digest(3)},
        .evidence_manifest_sha256 = digest(13),
        .predictor_sha256 = digest(14),
        .exporter_sha256 = digest(15)};
}

registry::CompatibilityDigests valid_compatibility() {
    return registry::CompatibilityDigests{
        .semantic_source_sha256 = digest(1),
        .build_identity_sha256 = digest(2),
        .runtime_capability_sha256 = digest(3)};
}

registry::RegistryRequestFacts valid_request_facts() {
    const auto key = valid_key();
    return registry::RegistryRequestFacts{
        .attestation = registry::AttestationResult{
            .status = registry::AttestationStatus::Success, .device = key.device},
        .workload = key.workload,
        .operation = key.operation,
        .input = key.input,
        .weight = key.weight,
        .bias = key.bias,
        .ternary_input_a = key.ternary_input_a,
        .ternary_input_b = key.ternary_input_b,
        .persistent_output = key.persistent_output,
        .persistent_weight = key.persistent_weight};
}

std::atomic<std::uint32_t> resolver_calls{0};
std::atomic<std::uint32_t> materializer_calls{0};
compact::EntryDescriptor selected_descriptor = valid_entry();

registry::Resolution selected_resolver(
    Mode,
    const std::optional<registry::RegistryRequest>&,
    registry::AttestationStatus,
    const registry::Eligibility&) noexcept {
    resolver_calls.fetch_add(1, std::memory_order_relaxed);
    return {.reason = registry::ResolutionReason::CertifiedMatch, .descriptor = &selected_descriptor};
}

registry::MaterializationResult counting_materializer(const compact::EntryDescriptor& descriptor) {
    materializer_calls.fetch_add(1, std::memory_order_relaxed);
    return registry::materialize_recipe(descriptor);
}

registry::MaterializationResult throwing_materializer(const compact::EntryDescriptor&) {
    materializer_calls.fetch_add(1, std::memory_order_relaxed);
    throw std::runtime_error("injected materialization failure");
}

void throw_during_selected_execution(std::atomic<std::uint32_t>& launches) {
    const bool selected = true;
    const registry::SelectedExecutionGuard guard(&selected);
    registry::execute_selected_call_once(guard, [&]() -> int {
        launches.fetch_add(1, std::memory_order_relaxed);
        throw std::runtime_error("injected public execution failure");
    });
}

class AgmmRegistryTest : public ::testing::Test {
protected:
    void SetUp() override {
        registry::reset_circuit_breaker_for_testing();
        registry::reset_stats_for_testing();
        resolver_calls.store(0, std::memory_order_relaxed);
        materializer_calls.store(0, std::memory_order_relaxed);
    }
};

TEST_F(AgmmRegistryTest, EmptyProductionTablePreservesOffShadowAndOn) {
    const auto eligibility = registry::Eligibility{};
    auto off = registry::resolve_for_dispatch(
        Mode::Off,
        std::nullopt,
        registry::AttestationStatus::UnsupportedAttestation,
        eligibility,
        &selected_resolver,
        &counting_materializer);
    EXPECT_EQ(off.resolution.reason, registry::ResolutionReason::Disabled);
    EXPECT_EQ(off.action, registry::ExecutionAction::Fallback);
    EXPECT_EQ(resolver_calls.load(), 0U);
    EXPECT_EQ(materializer_calls.load(), 0U);

    const auto shadow =
        registry::resolve(Mode::Shadow, std::nullopt, registry::AttestationStatus::UnsupportedAttestation, eligibility);
    const auto on =
        registry::resolve(Mode::On, std::nullopt, registry::AttestationStatus::UnsupportedAttestation, eligibility);
    EXPECT_EQ(shadow.reason, registry::ResolutionReason::EmptyRegistry);
    EXPECT_EQ(on.reason, registry::ResolutionReason::EmptyRegistry);
}

TEST_F(AgmmRegistryTest, ModeIsFrozenAtFirstRead) {
    const auto original = ttnn::CONFIG.get<"agmm_registry_mode">();
    ttnn::CONFIG.set<"agmm_registry_mode">(Mode::Shadow);
    registry::reset_startup_mode_for_testing();
    EXPECT_EQ(registry::current_mode(), Mode::Shadow);
    ttnn::CONFIG.set<"agmm_registry_mode">(Mode::On);
    EXPECT_EQ(registry::current_mode(), Mode::Shadow);
    ttnn::CONFIG.set<"agmm_registry_mode">(original);
    registry::reset_startup_mode_for_testing();
}

TEST_F(AgmmRegistryTest, UnknownTraceStateFailsClosedInShadowAndOnOnly) {
    EXPECT_FALSE(registry::fail_closed_trace_capture_active(Mode::Off, std::nullopt));
    EXPECT_TRUE(registry::fail_closed_trace_capture_active(Mode::Shadow, std::nullopt));
    EXPECT_TRUE(registry::fail_closed_trace_capture_active(Mode::On, std::nullopt));
    EXPECT_FALSE(registry::fail_closed_trace_capture_active(Mode::Shadow, false));
}

TEST_F(AgmmRegistryTest, PreflightReasonsPrecedeEmptyTableAndAreNamed) {
    const auto trace = registry::resolve(
        Mode::Shadow,
        std::nullopt,
        registry::AttestationStatus::UnsupportedAttestation,
        {.trace_capture_active = true, .has_explicit_program_config = true});
    EXPECT_EQ(trace.reason, registry::ResolutionReason::TraceCaptureUnsupported);
    EXPECT_EQ(registry::resolution_reason_name(trace.reason), "trace_capture_unsupported");

    const auto explicit_config = registry::resolve(
        Mode::Shadow,
        std::nullopt,
        registry::AttestationStatus::UnsupportedAttestation,
        {.has_explicit_program_config = true});
    EXPECT_EQ(explicit_config.reason, registry::ResolutionReason::ExplicitProgramConfig);

    const auto explicit_kernel = registry::resolve(
        Mode::Shadow,
        std::nullopt,
        registry::AttestationStatus::UnsupportedAttestation,
        {.has_explicit_compute_kernel_config = true});
    EXPECT_EQ(explicit_kernel.reason, registry::ResolutionReason::ExplicitComputeKernelConfig);
}

TEST_F(AgmmRegistryTest, ExactLookupRequiresFullKeyAndCompatibility) {
    const auto entry = valid_entry();
    const std::array entries{entry};
    const auto request = registry::RegistryRequest{.key = entry.key};
    auto result = registry::resolve_with_table_for_testing(
        Mode::On, request, registry::AttestationStatus::Success, {}, valid_lock(), entries, valid_compatibility());
    ASSERT_EQ(result.reason, registry::ResolutionReason::CertifiedMatch);
    EXPECT_EQ(result.descriptor, &entries.front());

    auto miss = request;
    miss.key.operation.scalar_present = true;
    miss.key.operation.scalar_f32_bits = 0x80000000U;
    result = registry::resolve_with_table_for_testing(
        Mode::On, miss, registry::AttestationStatus::Success, {}, valid_lock(), entries, valid_compatibility());
    EXPECT_EQ(result.reason, registry::ResolutionReason::ExactMiss);

    result = registry::resolve_with_table_for_testing(
        Mode::On,
        request,
        registry::AttestationStatus::UnsupportedAttestation,
        {},
        valid_lock(),
        entries,
        valid_compatibility());
    EXPECT_EQ(result.reason, registry::ResolutionReason::UnsupportedAttestation);
}

TEST_F(AgmmRegistryTest, GeneratedEmptyLockIsCanonicalAndFailClosed) {
    namespace generated = registry::generated;
    EXPECT_TRUE(generated::entries().empty());
    EXPECT_EQ(
        compact::validate_table_lock(generated::lock(), generated::entries()),
        compact::TableValidationStatus::Empty);
    EXPECT_EQ(
        registry::validate_compatibility(
            generated::lock(), generated::entries(), valid_compatibility(), valid_key().device),
        registry::CompatibilityStatus::EmptyRegistry);
}

TEST_F(AgmmRegistryTest, SyntheticGeneratedLockRequiresTypedProvenanceAndStrictKeyOrder) {
    auto first = valid_entry();
    auto second = valid_entry();
    second.entry_id = digest(6);
    second.key.workload.logical_n += 32;
    const std::array sorted_entries{first, second};
    auto lock = valid_lock(sorted_entries.size());
    EXPECT_EQ(
        compact::validate_table_lock(lock, sorted_entries), compact::TableValidationStatus::Valid);

    auto missing_provenance = lock;
    missing_provenance.predictor_sha256 = {};
    EXPECT_EQ(
        compact::validate_table_lock(missing_provenance, sorted_entries),
        compact::TableValidationStatus::MissingLockDigest);

    auto wrong_count = lock;
    wrong_count.entry_count -= 1;
    EXPECT_EQ(
        compact::validate_table_lock(wrong_count, sorted_entries),
        compact::TableValidationStatus::EntryCountMismatch);

    auto wrong_abi = lock;
    wrong_abi.codegen_recipe_abi += 1;
    EXPECT_EQ(
        compact::validate_table_lock(wrong_abi, sorted_entries),
        compact::TableValidationStatus::LockSchemaMismatch);

    auto zero_entry_id = sorted_entries;
    zero_entry_id.front().entry_id = {};
    EXPECT_EQ(
        compact::validate_table_lock(lock, zero_entry_id), compact::TableValidationStatus::MissingEntryId);

    auto wrong_capability = sorted_entries;
    wrong_capability.back().key.device.runtime_capability_sha256 = digest(16);
    EXPECT_EQ(
        compact::validate_table_lock(lock, wrong_capability),
        compact::TableValidationStatus::RuntimeCapabilityMismatch);

    const std::array reversed_entries{second, first};
    EXPECT_EQ(
        compact::validate_table_lock(lock, reversed_entries),
        compact::TableValidationStatus::EntriesNotStrictlySorted);
    const std::array duplicate_entries{first, first};
    EXPECT_EQ(
        compact::validate_table_lock(lock, duplicate_entries),
        compact::TableValidationStatus::EntriesNotStrictlySorted);
}

TEST_F(AgmmRegistryTest, MalformedSyntheticLockCannotSelectARecipe) {
    const auto entry = valid_entry();
    const std::array entries{entry};
    auto lock = valid_lock();
    lock.evidence_manifest_sha256 = {};
    const auto result = registry::resolve_with_table_for_testing(
        Mode::On,
        registry::RegistryRequest{.key = entry.key},
        registry::AttestationStatus::Success,
        {},
        lock,
        entries,
        valid_compatibility());
    EXPECT_EQ(result.reason, registry::ResolutionReason::CompatibilityMismatch);
    EXPECT_EQ(result.descriptor, nullptr);
}

TEST_F(AgmmRegistryTest, PureRequestBuilderBindsEveryResolvedDescriptor) {
    const auto facts = valid_request_facts();
    const auto built = registry::build_registry_request(facts);
    ASSERT_EQ(built.status, registry::RequestBuildStatus::Success);
    ASSERT_TRUE(built.request.has_value());
    EXPECT_EQ(built.request->key, valid_key());
}

TEST_F(AgmmRegistryTest, PureRequestBuilderFailsClosedWithoutExactAttestation) {
    auto facts = valid_request_facts();
    facts.attestation = {};
    auto built = registry::build_registry_request(facts);
    EXPECT_EQ(built.status, registry::RequestBuildStatus::UnsupportedAttestation);
    EXPECT_FALSE(built.request.has_value());

    facts.attestation.status = registry::AttestationStatus::QueryFailed;
    built = registry::build_registry_request(facts);
    EXPECT_EQ(built.status, registry::RequestBuildStatus::AttestationQueryFailed);
    EXPECT_FALSE(built.request.has_value());
}

TEST_F(AgmmRegistryTest, PureRequestBuilderRejectsExplicitOverridesBeforeAttestation) {
    auto facts = valid_request_facts();
    facts.eligibility.has_explicit_program_config = true;
    facts.attestation = {};
    auto built = registry::build_registry_request(facts);
    EXPECT_EQ(built.status, registry::RequestBuildStatus::ExplicitProgramConfig);

    facts.eligibility = {.has_explicit_compute_kernel_config = true};
    built = registry::build_registry_request(facts);
    EXPECT_EQ(built.status, registry::RequestBuildStatus::ExplicitComputeKernelConfig);
}

TEST_F(AgmmRegistryTest, PureRequestBuilderRejectsMissingAndInconsistentFacts) {
    auto facts = valid_request_facts();
    facts.input.tensor_topology_sha256 = {};
    auto built = registry::build_registry_request(facts);
    EXPECT_EQ(built.status, registry::RequestBuildStatus::IncompleteDescriptor);
    EXPECT_FALSE(built.request.has_value());

    facts = valid_request_facts();
    facts.attestation.device.device_count += 1;
    built = registry::build_registry_request(facts);
    EXPECT_EQ(built.status, registry::RequestBuildStatus::InconsistentDescriptor);

    facts = valid_request_facts();
    facts.workload.logical_k += 32;
    built = registry::build_registry_request(facts);
    EXPECT_EQ(built.status, registry::RequestBuildStatus::InconsistentDescriptor);

    facts = valid_request_facts();
    facts.bias.tensor = tensor_descriptor({1, 1, 1, 128}, {1, 1, 32, 128});
    built = registry::build_registry_request(facts);
    EXPECT_EQ(built.status, registry::RequestBuildStatus::InconsistentDescriptor);
}

TEST_F(AgmmRegistryTest, ShadowObservesAndOnMaterializesAtMostOnce) {
    const auto request = registry::RegistryRequest{.key = selected_descriptor.key};
    auto dispatch = registry::resolve_for_dispatch(
        Mode::Shadow, request, registry::AttestationStatus::Success, {}, &selected_resolver, &counting_materializer);
    EXPECT_EQ(dispatch.action, registry::ExecutionAction::ObserveOnly);
    EXPECT_EQ(resolver_calls.load(), 1U);
    EXPECT_EQ(materializer_calls.load(), 0U);

    dispatch = registry::resolve_for_dispatch(
        Mode::On, request, registry::AttestationStatus::Success, {}, &selected_resolver, &counting_materializer);
    EXPECT_EQ(dispatch.action, registry::ExecutionAction::ApplyRecipe);
    EXPECT_TRUE(dispatch.recipe.has_value());
    EXPECT_EQ(resolver_calls.load(), 2U);
    EXPECT_EQ(materializer_calls.load(), 1U);
}

TEST_F(AgmmRegistryTest, MaterializationExceptionFallsBackAndCircuitBreaks) {
    const auto dispatch = registry::resolve_for_dispatch(
        Mode::On,
        registry::RegistryRequest{.key = selected_descriptor.key},
        registry::AttestationStatus::Success,
        {},
        &selected_resolver,
        &throwing_materializer);
    EXPECT_EQ(dispatch.action, registry::ExecutionAction::Fallback);
    EXPECT_EQ(dispatch.resolution.reason, registry::ResolutionReason::MaterializationRejected);
    EXPECT_EQ(materializer_calls.load(), 1U);
    EXPECT_TRUE(registry::is_circuit_broken());
}

TEST_F(AgmmRegistryTest, SelectedExecutionExceptionIsNotRetriedAndCircuitBreaks) {
    std::atomic<std::uint32_t> launches{0};
    EXPECT_THROW(throw_during_selected_execution(launches), std::runtime_error);
    EXPECT_EQ(launches.load(), 1U);
    EXPECT_TRUE(registry::is_circuit_broken());
    EXPECT_EQ(registry::stats_snapshot().launch_completed_hits, 0U);
}

TEST_F(AgmmRegistryTest, MaterializerRejectsShapeTileGridAndBlockTampering) {
    auto entry = valid_entry();
    EXPECT_EQ(registry::materialize_recipe(entry).status, registry::MaterializationStatus::Success);

    entry.key.input.tile_width = 16;
    EXPECT_EQ(registry::materialize_recipe(entry).status, registry::MaterializationStatus::InvalidProgramConfig);
    entry = valid_entry();
    entry.key.workload.logical_k += 32;
    EXPECT_EQ(registry::materialize_recipe(entry).status, registry::MaterializationStatus::InvalidProgramConfig);
    entry = valid_entry();
    entry.replay.config.compute_grid_x = 1;
    EXPECT_EQ(registry::materialize_recipe(entry).status, registry::MaterializationStatus::InvalidProgramConfig);
    entry = valid_entry();
    entry.replay.config.k_block_size = 2;
    EXPECT_EQ(registry::materialize_recipe(entry).status, registry::MaterializationStatus::InvalidProgramConfig);
}

TEST_F(AgmmRegistryTest, MaterializerRejectsOddSwiGluNBlockBeforeLaunch) {
    auto entry = valid_entry();
    entry.key.operation.fuse_swiglu = true;
    entry.replay.config.n_block_size = 3;
    entry.replay.config.subblock_w = 1;

    EXPECT_EQ(registry::materialize_recipe(entry).status, registry::MaterializationStatus::InvalidProgramConfig);
}

TEST_F(AgmmRegistryTest, MaterializerUsesFullDestinationRegisterCapacity) {
    auto entry = valid_entry();
    entry.replay.config.m_block_size = 2;
    entry.replay.config.n_block_size = 4;
    entry.replay.config.subblock_h = 2;
    entry.replay.config.subblock_w = 4;

    // fp32 accumulation has four 32x32 destination tiles in half-sync mode.
    entry.replay.compute_kernel_config.dst_full_sync_en = false;
    EXPECT_EQ(registry::materialize_recipe(entry).status, registry::MaterializationStatus::InvalidProgramConfig);

    // Full-sync uses the entire destination register and therefore admits all eight tiles.
    entry.replay.compute_kernel_config.dst_full_sync_en = true;
    EXPECT_EQ(registry::materialize_recipe(entry).status, registry::MaterializationStatus::Success);

    // Non-fp32 full-sync exposes sixteen tiles through the shared hardware contract.
    entry.replay.config.m_block_size = 4;
    entry.replay.config.subblock_h = 4;
    entry.replay.compute_kernel_config.fp32_dest_acc_en = false;
    EXPECT_EQ(registry::materialize_recipe(entry).status, registry::MaterializationStatus::Success);
    entry.replay.compute_kernel_config.dst_full_sync_en = false;
    EXPECT_EQ(registry::materialize_recipe(entry).status, registry::MaterializationStatus::InvalidProgramConfig);
}

TEST_F(AgmmRegistryTest, TelemetrySeparatesSelectedFromLaunchCompletedAndUsesNamedReasons) {
    const auto resolution =
        registry::Resolution{.reason = registry::ResolutionReason::CertifiedMatch, .descriptor = &selected_descriptor};
    registry::record_resolution(Mode::On, resolution, registry::ExecutionAction::ApplyRecipe);
    auto snapshot = registry::stats_snapshot();
    EXPECT_EQ(snapshot.selected_hits, 1U);
    EXPECT_EQ(snapshot.launch_completed_hits, 0U);
    EXPECT_EQ(snapshot.reasons[static_cast<std::size_t>(registry::ResolutionReason::CertifiedMatch)], 1U);
    EXPECT_EQ(registry::resolution_reason_name(registry::ResolutionReason::CertifiedMatch), "certified_match");

    const bool selected = true;
    {
        const registry::SelectedExecutionGuard guard(&selected);
    }
    snapshot = registry::stats_snapshot();
    EXPECT_EQ(snapshot.launch_completed_hits, 1U);
}

}  // namespace
