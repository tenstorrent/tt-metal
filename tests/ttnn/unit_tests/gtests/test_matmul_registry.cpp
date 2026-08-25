// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdlib>
#include <stdexcept>
#include <thread>
#include <variant>

#include <tt_stl/reflection.hpp>

#include "ttnn/operations/matmul/device/config/matmul_config_registry.hpp"
#include "ttnn/operations/matmul/device/config/registry/matmul_registry_descriptor.hpp"

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

compact::Sha256 repeated_digest(const std::uint8_t value) {
    compact::Sha256 digest{};
    digest.fill(value);
    return digest;
}

static_assert(std::is_trivially_copyable_v<compact::KeyDescriptor>);
static_assert(std::is_trivially_copyable_v<compact::ReplayDescriptor>);
static_assert(std::is_trivially_copyable_v<compact::EntryDescriptor>);

compact::KeyDescriptor compact_key(const std::uint64_t logical_m) {
    compact::KeyDescriptor key{};
    key.architecture = 2;
    key.board_capability_class = 1;
    key.codegen_recipe_abi = compact::kCodegenRecipeAbi;
    key.compute_grid_x = 8;
    key.compute_grid_y = 8;
    key.device_count = 1;
    key.logical_k = 256;
    key.logical_m = logical_m;
    key.logical_n = 512;
    key.mesh_cols = 1;
    key.mesh_rows = 1;
    key.padded_k = 256;
    key.padded_m = logical_m;
    key.padded_n = 512;
    key.schema_version = 1;
    return key;
}

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

TEST(MatmulConfigRegistry, CompactLookupIsExactNonOwningAndSidebandIndependent) {
    std::array<compact::EntryDescriptor, 3> entries{};
    entries[0].key = compact_key(64);
    entries[1].key = compact_key(128);
    entries[2].key = compact_key(256);
    entries[0].entry_id[0] = 0xff;
    entries[1].entry_id[0] = 0x01;
    entries[2].entry_id[0] = 0x00;

    const auto requested = compact_key(128);
    const auto index = compact::ExactIndex{entries};
    EXPECT_EQ(index.size(), entries.size());
    const auto* hit = index.lookup(requested);
    ASSERT_EQ(hit, &entries[1]);
    EXPECT_EQ(hit->entry_id[0], 0x01);

    auto shape_miss = requested;
    shape_miss.logical_n += 1;
    EXPECT_EQ(index.lookup(shape_miss), nullptr);

    auto topology_miss = requested;
    topology_miss.topology_sha256.back() = 1;
    EXPECT_EQ(index.lookup(topology_miss), nullptr);

    auto domain_miss = requested;
    domain_miss.domain = compact::Domain::DenseLinear;
    EXPECT_EQ(index.lookup(domain_miss), nullptr);
}

MatmulRegistryRequest exact_request(const OperationDomain domain = OperationDomain::DenseMatmul) {
    auto request = MatmulRegistryRequest{
        .schema_version = 1,
        .call = domain == OperationDomain::Addmm ? addmm_call_semantics(1.0F, 0.0F) : CallSemantics{.domain = domain},
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
                .attestation_status = DeviceAttestationStatus::Success,
                .architecture = 2,
                .board_capability_class = 1,
                .device_count = 1,
                .mesh_rows = 1,
                .mesh_cols = 1,
                .compute_grid_x = 8,
                .compute_grid_y = 8,
                .topology_sha256 = repeated_digest(0x44),
                .runtime_capability_sha256 = repeated_digest(0x33),
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
        .activation_param_count = 0,
    };
    return request;
}

compact::EntryDescriptor compact_entry(const OperationDomain domain = OperationDomain::DenseMatmul) {
    const auto request = exact_request(domain);
    auto entry = compact::EntryDescriptor{};
    entry.key = compact::KeyDescriptor{
        .architecture = request.device.architecture,
        .bcast_batch_present = false,
        .bcast_batch = false,
        .board_capability_class = request.device.board_capability_class,
        .codegen_recipe_abi = compact::kCodegenRecipeAbi,
        .compute_grid_x = static_cast<std::uint16_t>(request.device.compute_grid_x),
        .compute_grid_y = static_cast<std::uint16_t>(request.device.compute_grid_y),
        .device_count = static_cast<std::uint16_t>(request.device.device_count),
        .has_activation = false,
        .has_bias = false,
        .input_a =
            compact::TensorDescriptor{
                .buffer_type = compact::BufferType::Dram,
                .dtype = compact::DataType::BFloat16,
                .layout = compact::Layout::Tile,
                .memory_layout = compact::MemoryLayout::Interleaved,
                .tile_height = 32,
                .tile_width = 32},
        .input_b =
            compact::TensorDescriptor{
                .buffer_type = compact::BufferType::Dram,
                .dtype = compact::DataType::BFloat16,
                .layout = compact::Layout::Tile,
                .memory_layout = compact::MemoryLayout::Interleaved,
                .tile_height = 32,
                .tile_width = 32},
        .logical_k = request.workload.logical_k,
        .logical_m = request.workload.logical_m,
        .logical_n = request.workload.logical_n,
        .mesh_cols = static_cast<std::uint16_t>(request.device.mesh_cols),
        .mesh_rows = static_cast<std::uint16_t>(request.device.mesh_rows),
        .output =
            compact::TensorDescriptor{
                .buffer_type = compact::BufferType::Dram,
                .dtype = compact::DataType::BFloat16,
                .layout = compact::Layout::Tile,
                .memory_layout = compact::MemoryLayout::Interleaved,
                .tile_height = 32,
                .tile_width = 32},
        .padded_k = request.workload.padded_k,
        .padded_m = request.workload.padded_m,
        .padded_n = request.workload.padded_n,
        .run_batched = false,
        .schema_version = 1,
        .topology_sha256 = request.device.topology_sha256,
        .transpose_a = false,
        .transpose_b = false,
        .untilize_out = false,
        .domain = domain == OperationDomain::DenseMatmul ? compact::Domain::DenseMatmul
                  : domain == OperationDomain::Linear    ? compact::Domain::DenseLinear
                                                         : compact::Domain::DenseAddmm,
        .alpha_f32_bits = request.call.alpha_f32_bits.value_or(0),
        .beta_f32_bits = request.call.beta_f32_bits.value_or(0)};
    entry.replay = compact::ReplayDescriptor{
        .schema_version = 2,
        .family = compact::ProgramFamily::MultiCoreReuse,
        .program_config =
            compact::MultiCoreReuseDescriptor{
                .compute_grid_x = 8,
                .compute_grid_y = 8,
                .in0_block_w = 2,
                .out_subblock_h = 1,
                .out_subblock_w = 2,
                .per_core_m = 4,
                .per_core_n = 8,
                .allowed_worker_cores_present = false},
        .compute_kernel_config =
            compact::ComputeKernelDescriptor{
                .math_fidelity = compact::MathFidelity::HiFi2,
                .throttle_level = compact::ThrottleLevel::NoThrottle,
                .math_approx_mode = true,
                .fp32_dest_acc_en = false,
                .packer_l1_acc = false,
                .dst_full_sync_en = false},
        .call_state = compact::CallStateDescriptor{
            .output = entry.key.output,
            .untilize_out = false,
            .bcast_batch_is_null = true,
            .user_core_coord_is_null = true,
            .user_fused_activation_is_null = true,
            .user_run_batched_is_false = true,
            .transpose_a_is_false = true,
            .transpose_b_is_false = true,
            .output_tile_is_null = true,
            .global_cb_is_null = true,
            .sub_device_id_is_null = true}};
    return entry;
}

TEST(MatmulConfigRegistry, RuntimeRequestConvertsToTheExactDisjointCompactKey) {
    for (const auto domain : {OperationDomain::DenseMatmul, OperationDomain::Linear, OperationDomain::Addmm}) {
        const auto request = exact_request(domain);
        const auto key = compact_registry_key(request);
        ASSERT_TRUE(key.has_value());
        EXPECT_EQ(key.value(), compact_entry(domain).key);
    }

    auto unsupported = exact_request();
    unsupported.output.memory_layout = TensorMemoryLayout::HEIGHT_SHARDED;
    EXPECT_FALSE(compact_registry_key(unsupported).has_value());
}

TEST(MatmulConfigRegistry, SharedDefaultCallStateProjectionAdmitsEveryPublicDomainWithoutAliasing) {
    const ttnn::prim::MatmulParams default_parameters;
    const std::array calls{dense_matmul_call_semantics(), linear_call_semantics(), addmm_call_semantics(1.0F, 0.0F)};
    for (const auto& call : calls) {
        const auto eligibility = v1_eligibility_from_call_state(
            call, IoContractStatus::Resolved, false, false, default_parameters, false, false, false, false);
        EXPECT_EQ(eligibility.call, call);
        EXPECT_EQ(preflight_v1_eligibility(eligibility), ResolutionReason::CertifiedMatch);
    }

    auto explicit_parameters = default_parameters;
    explicit_parameters.program_config = MatmulMultiCoreProgramConfig{};
    const auto explicit_eligibility = v1_eligibility_from_call_state(
        dense_matmul_call_semantics(),
        IoContractStatus::Resolved,
        false,
        false,
        explicit_parameters,
        false,
        false,
        false,
        false);
    EXPECT_EQ(preflight_v1_eligibility(explicit_eligibility), ResolutionReason::ExplicitOverride);
}

compact::TableMetadata compact_metadata() {
    return compact::TableMetadata{
        .lock_schema_version = 1,
        .key_schema_version = 1,
        .replay_schema_version = 2,
        .semantic_source_sha256 = repeated_digest(0x11),
        .build_identity_sha256 = repeated_digest(0x22),
        .runtime_capability_sha256 = repeated_digest(0x33)};
}

CompatibilityDigests compatible_digests() {
    return CompatibilityDigests{
        .semantic_source_sha256 = repeated_digest(0x11),
        .build_identity_sha256 = repeated_digest(0x22),
        .runtime_capability_sha256 = repeated_digest(0x33)};
}

DeviceAttestationFacts valid_attestation_facts() {
    return DeviceAttestationFacts{
        .architecture = AttestationArchitecture::Blackhole,
        .board_class = AttestationBoardClass::BlackholeGalaxy,
        .cluster_class = AttestationClusterClass::BlackholeGalaxy,
        .device_initialized = true,
        .remote_only = false,
        .active_sub_device_manager_is_default = true,
        .device_count = 1,
        .mesh_rows = 1,
        .mesh_cols = 1,
        .system_mesh_id = 0,
        .compute_grid_x = 13,
        .compute_grid_y = 10,
        .physical_grid_x = 17,
        .physical_grid_y = 12,
        .logical_grid_x = 13,
        .logical_grid_y = 10,
        .dram_grid_x = 8,
        .dram_grid_y = 1,
        .tensix_harvesting_mask = 0,
        .num_hw_cqs = 1,
        .num_dram_channels = 8,
        .l1_size_per_core = 1464320,
        .dram_size_per_channel = 4278190080ULL,
        .firmware_bundle_present = true,
        .firmware_bundle_major = 18,
        .firmware_bundle_minor = 10,
        .firmware_bundle_patch = 0,
        .ethernet_firmware_present = true,
        .ethernet_firmware_major = 6,
        .ethernet_firmware_minor = 8,
        .ethernet_firmware_patch = 1};
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

MaterializationResult throwing_materializer(const compact::EntryDescriptor&) {
    throw std::runtime_error("injected materialization failure");
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
    reset_startup_compatibility_for_testing();
    for (const auto mode : {Mode::Shadow, Mode::On}) {
        const auto result =
            resolve_with(mode, Eligibility{.call = CallSemantics{.domain = OperationDomain::DenseMatmul}});
        EXPECT_EQ(result.reason, ResolutionReason::EmptyRegistry);
        EXPECT_EQ(result.recipe, nullptr);
        EXPECT_EQ(result.descriptor, nullptr);
    }
    EXPECT_EQ(
        resolve_with(Mode::Off, Eligibility{.call = dense_matmul_call_semantics()}).reason, ResolutionReason::Disabled);
    EXPECT_EQ(startup_compatibility_status(), CompatibilityStatus::Uninitialized);
}

TEST(MatmulConfigRegistry, CompatibilityValidationIsExactAndFailClosed) {
    const auto metadata = compact_metadata();
    auto actual = compatible_digests();
    EXPECT_EQ(validate_registry_compatibility(metadata, 1, actual), CompatibilityStatus::Compatible);
    EXPECT_EQ(validate_registry_compatibility(metadata, 0, actual), CompatibilityStatus::EmptyRegistry);

    auto changed_metadata = metadata;
    changed_metadata.key_schema_version++;
    EXPECT_EQ(validate_registry_compatibility(changed_metadata, 1, actual), CompatibilityStatus::SchemaMismatch);
    actual = compatible_digests();
    actual.semantic_source_sha256.back() ^= 1;
    EXPECT_EQ(validate_registry_compatibility(metadata, 1, actual), CompatibilityStatus::SemanticSourceMismatch);
    actual = compatible_digests();
    actual.build_identity_sha256.back() ^= 1;
    EXPECT_EQ(validate_registry_compatibility(metadata, 1, actual), CompatibilityStatus::BuildIdentityMismatch);
    actual = compatible_digests();
    actual.runtime_capability_sha256.back() ^= 1;
    EXPECT_EQ(validate_registry_compatibility(metadata, 1, actual), CompatibilityStatus::RuntimeCapabilityMismatch);
}

TEST(MatmulConfigRegistry, DeviceAttestationMatchesFrozenExporterContract) {
    const std::array<std::uint8_t, 3> abc{'a', 'b', 'c'};
    EXPECT_EQ(registry_sha256(abc), (compact::Sha256{{0xba, 0x78, 0x16, 0xbf, 0x8f, 0x01, 0xcf, 0xea, 0x41, 0x41, 0x40,
                                                      0xde, 0x5d, 0xae, 0x22, 0x23, 0xb0, 0x03, 0x61, 0xa3, 0x96, 0x17,
                                                      0x7a, 0x9c, 0xb4, 0x10, 0xff, 0x61, 0xf2, 0x00, 0x15, 0xad}}));

    const auto result = derive_device_attestation(valid_attestation_facts());
    ASSERT_EQ(result.status, DeviceAttestationStatus::Success);
    EXPECT_EQ(result.attestation.board_capability_class, 3);
    EXPECT_EQ(result.attestation.topology_sha256, (compact::Sha256{{0xfb, 0xe6, 0x47, 0x00, 0xcb, 0x31, 0x63, 0xcc,
                                                                    0x7d, 0xfd, 0xbb, 0xa6, 0x53, 0xd9, 0x29, 0xee,
                                                                    0x24, 0x76, 0x8f, 0xd9, 0xa0, 0xfb, 0xc5, 0xfa,
                                                                    0xbc, 0xb1, 0x6c, 0x17, 0x57, 0xe8, 0x57, 0x9e}}));
    EXPECT_EQ(
        result.attestation.runtime_capability_sha256,
        (compact::Sha256{{0x33, 0x4e, 0x50, 0x71, 0x1e, 0xde, 0x66, 0xbf, 0x5b, 0x97, 0x3b,
                          0x9d, 0x5a, 0xb0, 0xb0, 0xeb, 0xfa, 0x47, 0x00, 0x5f, 0x6e, 0x8d,
                          0xb8, 0x85, 0x82, 0xa9, 0x41, 0xbb, 0x43, 0x55, 0x2e, 0xe8}}));

    const auto compiled = compiled_registry_compatibility_digests(result.attestation.runtime_capability_sha256);
    EXPECT_EQ(compiled.runtime_capability_sha256, result.attestation.runtime_capability_sha256);
    EXPECT_NE(compiled.semantic_source_sha256, compact::Sha256{});
    EXPECT_NE(compiled.build_identity_sha256, compact::Sha256{});

    reset_startup_compatibility_for_testing();
    EXPECT_EQ(
        initialize_registry_compatibility_from_attestation(
            DeviceAttestationResult{.status = DeviceAttestationStatus::FirmwareUnavailable}),
        CompatibilityStatus::Uninitialized);
    EXPECT_EQ(initialize_registry_compatibility_from_attestation(result), CompatibilityStatus::EmptyRegistry);
    EXPECT_EQ(startup_compatibility_status(), CompatibilityStatus::EmptyRegistry);
    reset_startup_compatibility_for_testing();
}

TEST(MatmulConfigRegistry, CompatibilityAttestationReportIsReadOnlyAndFailClosed) {
    const auto device = derive_device_attestation(valid_attestation_facts());
    const auto report = registry_compatibility_attestation(device);
    EXPECT_EQ(report.schema_version, kCompatibilityAttestationSchemaVersion);
    EXPECT_EQ(device_attestation_status_name(report.device_attestation_status), "success");
    EXPECT_EQ(report.codegen_recipe_abi, compact::kCodegenRecipeAbi);
    EXPECT_EQ(report.board_capability_class, device.attestation.board_capability_class);
    EXPECT_EQ(report.actual_topology_sha256, device.attestation.topology_sha256);
    EXPECT_EQ(report.actual_runtime_capability_sha256, device.attestation.runtime_capability_sha256);
    EXPECT_NE(report.actual_semantic_source_sha256, compact::Sha256{});
    EXPECT_NE(report.actual_build_identity_sha256, compact::Sha256{});

    const auto rejected = registry_compatibility_attestation(
        DeviceAttestationResult{.status = DeviceAttestationStatus::UnsupportedArchitecture});
    EXPECT_EQ(device_attestation_status_name(rejected.device_attestation_status), "unsupported_architecture");
    EXPECT_EQ(rejected.board_capability_class, 0);
    EXPECT_EQ(rejected.actual_topology_sha256, compact::Sha256{});
    EXPECT_EQ(rejected.actual_runtime_capability_sha256, compact::Sha256{});
    EXPECT_NE(rejected.actual_semantic_source_sha256, compact::Sha256{});
    EXPECT_NE(rejected.actual_build_identity_sha256, compact::Sha256{});
}

TEST(MatmulConfigRegistry, DeviceAttestationFailsClosedOnEveryRequiredEnvelope) {
    const auto expect_status = [](const DeviceAttestationFacts& facts, const DeviceAttestationStatus expected) {
        EXPECT_EQ(derive_device_attestation(facts).status, expected);
    };
    auto facts = valid_attestation_facts();
    facts.device_initialized = false;
    expect_status(facts, DeviceAttestationStatus::DeviceUninitialized);
    facts = valid_attestation_facts();
    facts.remote_only = true;
    expect_status(facts, DeviceAttestationStatus::RemoteDevice);
    facts = valid_attestation_facts();
    facts.device_count = 2;
    expect_status(facts, DeviceAttestationStatus::NotOneChip);
    facts = valid_attestation_facts();
    facts.active_sub_device_manager_is_default = false;
    expect_status(facts, DeviceAttestationStatus::ActiveSubDeviceManager);
    facts = valid_attestation_facts();
    facts.architecture = static_cast<AttestationArchitecture>(0xff);
    expect_status(facts, DeviceAttestationStatus::UnsupportedArchitecture);
    facts = valid_attestation_facts();
    facts.board_class = static_cast<AttestationBoardClass>(0xffffffffU);
    expect_status(facts, DeviceAttestationStatus::UnsupportedBoard);
    facts = valid_attestation_facts();
    facts.cluster_class = static_cast<AttestationClusterClass>(0xff);
    expect_status(facts, DeviceAttestationStatus::UnsupportedCluster);
    facts = valid_attestation_facts();
    facts.board_class = AttestationBoardClass::BlackholeP150;
    expect_status(facts, DeviceAttestationStatus::BoardClusterMismatch);
    facts = valid_attestation_facts();
    facts.firmware_bundle_present = false;
    expect_status(facts, DeviceAttestationStatus::FirmwareUnavailable);
    facts = valid_attestation_facts();
    facts.compute_grid_x = 12;
    expect_status(facts, DeviceAttestationStatus::InvalidCapability);

    facts = valid_attestation_facts();
    const auto original = derive_device_attestation(facts).attestation;
    facts.firmware_bundle_patch++;
    const auto changed = derive_device_attestation(facts).attestation;
    EXPECT_EQ(changed.topology_sha256, original.topology_sha256);
    EXPECT_NE(changed.runtime_capability_sha256, original.runtime_capability_sha256);
}

TEST(MatmulConfigRegistry, EmptyRegistryStartupCompatibilityFreezesConcurrently) {
    reset_startup_compatibility_for_testing();
    constexpr std::size_t thread_count = 16;
    std::array<CompatibilityStatus, thread_count> observed{};
    std::array<std::thread, thread_count> threads;
    for (std::size_t index = 0; index < threads.size(); ++index) {
        threads[index] = std::thread([index, &observed] {
            auto actual = compatible_digests();
            actual.semantic_source_sha256[index % actual.semantic_source_sha256.size()] ^= 1;
            observed[index] = initialize_registry_compatibility(actual);
        });
    }
    for (auto& thread : threads) {
        thread.join();
    }
    for (const auto status : observed) {
        EXPECT_EQ(status, CompatibilityStatus::EmptyRegistry);
    }
    EXPECT_EQ(startup_compatibility_status(), CompatibilityStatus::EmptyRegistry);
    EXPECT_EQ(stats_snapshot().compatibility_status, CompatibilityStatus::EmptyRegistry);
    EXPECT_EQ(stats_snapshot().entry_count, 0);
    reset_startup_compatibility_for_testing();
}

TEST(MatmulConfigRegistry, FirstNonemptyCompatibilityResultIsFrozenFailClosed) {
    reset_startup_compatibility_for_testing();
    auto incompatible = compatible_digests();
    incompatible.build_identity_sha256.back() ^= 1;
    EXPECT_EQ(
        initialize_registry_compatibility_for_testing(compact_metadata(), 1, incompatible),
        CompatibilityStatus::BuildIdentityMismatch);
    EXPECT_EQ(
        initialize_registry_compatibility_for_testing(compact_metadata(), 1, compatible_digests()),
        CompatibilityStatus::BuildIdentityMismatch);
    EXPECT_EQ(startup_compatibility_status(), CompatibilityStatus::BuildIdentityMismatch);
    reset_startup_compatibility_for_testing();
}

TEST(MatmulConfigRegistry, EmptyRegistryStartupIsDeterministicInSpawnedProcess) {
    EXPECT_EXIT(
        {
            reset_startup_mode_for_testing();
            reset_startup_compatibility_for_testing();
            ttnn::CONFIG.set<"matmul_registry_mode">(Mode::Shadow);
            const auto status = initialize_registry_compatibility(compatible_digests());
            const bool valid = status == CompatibilityStatus::EmptyRegistry && current_mode() == Mode::Shadow;
            std::_Exit(valid ? 0 : 1);
        },
        ::testing::ExitedWithCode(0),
        "");
}

TEST(MatmulConfigRegistry, CompactTableLookupAndNativeMaterializationAreExact) {
    reset_circuit_breakers_for_testing();
    const auto request = exact_request();
    const auto eligibility = Eligibility{.call = request.call};
    const auto entry = compact_entry();
    const std::array entries{entry};

    const auto off = resolve_with_compact_table_for_testing(
        Mode::Off, request, eligibility, compact_metadata(), entries, compatible_digests());
    EXPECT_EQ(off.reason, ResolutionReason::Disabled);
    EXPECT_EQ(off.descriptor, nullptr);

    const auto shadow = resolve_with_compact_table_for_testing(
        Mode::Shadow, request, eligibility, compact_metadata(), entries, compatible_digests());
    EXPECT_EQ(shadow.reason, ResolutionReason::CertifiedMatch);
    EXPECT_EQ(shadow.descriptor, &entries[0]);
    EXPECT_EQ(execution_action(Mode::Shadow, shadow), ExecutionAction::ObserveOnly);

    const auto on = resolve_with_compact_table_for_testing(
        Mode::On, request, eligibility, compact_metadata(), entries, compatible_digests());
    ASSERT_EQ(on.reason, ResolutionReason::CertifiedMatch);
    ASSERT_EQ(on.descriptor, &entries[0]);
    const auto native = materialize_matmul_registry_recipe(*on.descriptor);
    ASSERT_EQ(native.status, MaterializationStatus::Success);
    ASSERT_TRUE(native.recipe.has_value());
    const auto* program = std::get_if<MatmulMultiCoreReuseProgramConfig>(&native.recipe->program_config);
    ASSERT_NE(program, nullptr);
    EXPECT_EQ(program->compute_with_storage_grid_size, tt::tt_metal::CoreCoord(8, 8));
    EXPECT_EQ(program->in0_block_w, 2);
    EXPECT_EQ(program->out_subblock_h, 1);
    EXPECT_EQ(program->out_subblock_w, 2);
    EXPECT_EQ(program->per_core_M, 4);
    EXPECT_EQ(program->per_core_N, 8);
    EXPECT_FALSE(program->allowed_worker_cores.has_value());
    EXPECT_EQ(native.recipe->compute_kernel_config.math_fidelity, tt::tt_metal::MathFidelity::HiFi2);
    EXPECT_EQ(native.recipe->compute_kernel_config.throttle_level, compute_throttle_utils::ThrottleLevel::NO_THROTTLE);

    auto miss = request;
    miss.workload.logical_m++;
    EXPECT_EQ(
        resolve_with_compact_table_for_testing(
            Mode::On, miss, eligibility, compact_metadata(), entries, compatible_digests())
            .reason,
        ResolutionReason::EmptyRegistry);
    miss = request;
    miss.input_a.dtype = DataType::FLOAT32;
    EXPECT_EQ(
        resolve_with_compact_table_for_testing(
            Mode::On, miss, eligibility, compact_metadata(), entries, compatible_digests())
            .reason,
        ResolutionReason::EmptyRegistry);
    miss = request;
    miss.device.topology_sha256.back() ^= 1;
    EXPECT_EQ(
        resolve_with_compact_table_for_testing(
            Mode::On, miss, eligibility, compact_metadata(), entries, compatible_digests())
            .reason,
        ResolutionReason::EmptyRegistry);
    miss = request;
    miss.device.board_capability_class++;
    EXPECT_EQ(
        resolve_with_compact_table_for_testing(
            Mode::On, miss, eligibility, compact_metadata(), entries, compatible_digests())
            .reason,
        ResolutionReason::EmptyRegistry);

    auto invalid_linear = compact_entry(OperationDomain::Linear);
    invalid_linear.key.alpha_f32_bits = 0x3F800000;
    EXPECT_EQ(materialize_matmul_registry_recipe(invalid_linear).status, MaterializationStatus::InvalidCallState);
    auto invalid_addmm = compact_entry(OperationDomain::Addmm);
    invalid_addmm.key.alpha_f32_bits = 0x80000000;
    EXPECT_EQ(materialize_matmul_registry_recipe(invalid_addmm).status, MaterializationStatus::InvalidCallState);
    invalid_addmm = compact_entry(OperationDomain::Addmm);
    invalid_addmm.key.beta_f32_bits = 0x3F800000;
    EXPECT_EQ(materialize_matmul_registry_recipe(invalid_addmm).status, MaterializationStatus::InvalidCallState);
}

TEST(MatmulConfigRegistry, CompactMaterializationRejectsEveryTypedBoundary) {
    const auto expect_rejection = [](const compact::EntryDescriptor& descriptor, const MaterializationStatus status) {
        const auto result = materialize_matmul_registry_recipe(descriptor);
        EXPECT_EQ(result.status, status);
        EXPECT_FALSE(result.recipe.has_value());
    };

    auto descriptor = compact_entry();
    descriptor.key.schema_version++;
    expect_rejection(descriptor, MaterializationStatus::UnsupportedSchema);

    descriptor = compact_entry();
    descriptor.replay.family = static_cast<compact::ProgramFamily>(0xff);
    expect_rejection(descriptor, MaterializationStatus::UnsupportedReplay);

    descriptor = compact_entry();
    descriptor.replay.program_config.compute_grid_x = 0;
    expect_rejection(descriptor, MaterializationStatus::InvalidProgramConfig);

    for (const auto mutate : {
             +[](compact::EntryDescriptor& item) { item.key.input_a.tile_height = 16; },
             +[](compact::EntryDescriptor& item) { item.key.logical_m = 0; },
             +[](compact::EntryDescriptor& item) { item.key.padded_m = item.key.logical_m - 1; },
             +[](compact::EntryDescriptor& item) { item.key.padded_m = 129; },
             +[](compact::EntryDescriptor& item) { item.key.input_b.tile_height = 16; },
             +[](compact::EntryDescriptor& item) { item.replay.program_config.in0_block_w = 3; },
             +[](compact::EntryDescriptor& item) { item.replay.program_config.per_core_m = 3; },
             +[](compact::EntryDescriptor& item) { item.replay.program_config.per_core_n = 3; },
             +[](compact::EntryDescriptor& item) { item.replay.program_config.out_subblock_h = 3; },
             +[](compact::EntryDescriptor& item) { item.replay.program_config.out_subblock_w = 3; },
             +[](compact::EntryDescriptor& item) { item.replay.program_config.compute_grid_x = 9; },
             +[](compact::EntryDescriptor& item) {
                 item.replay.program_config.out_subblock_h = 4;
                 item.replay.program_config.out_subblock_w = 4;
             },
         }) {
        descriptor = compact_entry();
        mutate(descriptor);
        expect_rejection(descriptor, MaterializationStatus::InvalidProgramConfig);
    }

    descriptor = compact_entry();
    descriptor.replay.compute_kernel_config.math_fidelity = static_cast<compact::MathFidelity>(0xff);
    expect_rejection(descriptor, MaterializationStatus::InvalidComputeKernelConfig);

    descriptor = compact_entry();
    descriptor.replay.call_state.output_tile_is_null = false;
    expect_rejection(descriptor, MaterializationStatus::InvalidCallState);
}

TEST(MatmulConfigRegistry, CompatibilityAndGuardsPrecedeCompactLookup) {
    reset_circuit_breakers_for_testing();
    const auto request = exact_request();
    const auto eligibility = Eligibility{.call = request.call};
    const std::array entries{compact_entry()};
    auto incompatible = compatible_digests();
    incompatible.build_identity_sha256.back() ^= 1;
    EXPECT_EQ(
        resolve_with_compact_table_for_testing(
            Mode::On, request, eligibility, compact_metadata(), entries, incompatible)
            .reason,
        ResolutionReason::BuildIdentityMismatch);

    auto unattested_request = request;
    unattested_request.device.attestation_status = DeviceAttestationStatus::FirmwareUnavailable;
    EXPECT_EQ(
        resolve_with_compact_table_for_testing(
            Mode::On, unattested_request, eligibility, compact_metadata(), entries, compatible_digests())
            .reason,
        ResolutionReason::DeviceAttestationUnavailable);

    EXPECT_EQ(resolve(Mode::On, unattested_request, eligibility).reason, ResolutionReason::EmptyRegistry);

    auto wrong_runtime_request = request;
    wrong_runtime_request.device.runtime_capability_sha256.back() ^= 1;
    EXPECT_EQ(
        resolve_with_compact_table_for_testing(
            Mode::On, wrong_runtime_request, eligibility, compact_metadata(), entries, compatible_digests())
            .reason,
        ResolutionReason::RuntimeCapabilityMismatch);

    EXPECT_EQ(
        resolve_with_compact_table_for_testing(
            Mode::On,
            request,
            Eligibility{.call = request.call, .trace_capture_active = true},
            compact_metadata(),
            entries,
            compatible_digests())
            .reason,
        ResolutionReason::TraceCaptureUnsupported);
    EXPECT_EQ(
        resolve_with_compact_table_for_testing(
            Mode::On,
            request,
            Eligibility{.call = request.call, .has_program_config = true},
            compact_metadata(),
            entries,
            incompatible)
            .reason,
        ResolutionReason::ExplicitOverride);
}

TEST(MatmulConfigRegistry, CompactLookupIsConcurrentAndReadOnly) {
    reset_circuit_breakers_for_testing();
    const auto request = exact_request();
    const auto eligibility = Eligibility{.call = request.call};
    const std::array entries{compact_entry()};
    constexpr std::size_t thread_count = 16;
    constexpr std::size_t iterations = 1000;
    std::array<bool, thread_count> correct{};
    std::array<std::thread, thread_count> threads;
    for (std::size_t index = 0; index < threads.size(); ++index) {
        threads[index] = std::thread([&, index] {
            correct[index] = true;
            for (std::size_t iteration = 0; iteration < iterations; ++iteration) {
                const auto result = resolve_with_compact_table_for_testing(
                    Mode::Shadow, request, eligibility, compact_metadata(), entries, compatible_digests());
                correct[index] = correct[index] && result.reason == ResolutionReason::CertifiedMatch &&
                                 result.descriptor == &entries[0];
            }
        });
    }
    for (auto& thread : threads) {
        thread.join();
    }
    EXPECT_TRUE(std::all_of(correct.begin(), correct.end(), [](const bool value) { return value; }));
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

TEST(MatmulConfigRegistry, CallerKnownIneligibilitySkipsRequestAndResolverInShadowAndOn) {
    const ttnn::prim::MatmulParams legacy_parameters;
    for (const auto mode : {Mode::Shadow, Mode::On}) {
        for (const auto& [eligibility, expected] : std::array{
                 std::pair{
                     Eligibility{.call = dense_matmul_call_semantics(), .has_program_config = true},
                     ResolutionReason::ExplicitOverride},
                 std::pair{
                     Eligibility{.call = dense_matmul_call_semantics(), .has_bias = true},
                     ResolutionReason::UnsupportedSemantics},
                 std::pair{
                     Eligibility{.call = CallSemantics{.domain = OperationDomain::IneligibleSharedCaller}},
                     ResolutionReason::IneligibleOperationDomain},
             }) {
            resolver_invocations = 0;
            const auto result =
                resolve_for_dispatch(mode, std::nullopt, eligibility, legacy_parameters, &counting_certified_resolver);
            EXPECT_EQ(resolver_invocations, 0);
            EXPECT_EQ(result.resolution.reason, expected);
            EXPECT_EQ(result.action, ExecutionAction::Fallback);
            EXPECT_FALSE(result.materialized_parameters.has_value());
        }
    }
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

TEST(MatmulConfigRegistry, ShadowTraceCaptureMirrorsOnIneligibilityWithoutMutation) {
    const auto request = exact_request();
    const auto eligibility = Eligibility{.call = request.call, .trace_capture_active = true};
    ttnn::prim::MatmulParams legacy_parameters;
    legacy_parameters.output_dtype = DataType::FLOAT32;
    const auto legacy_hash = ttsl::hash::hash_objects_with_default_seed(legacy_parameters);
    resolver_invocations = 0;

    const auto result =
        resolve_for_dispatch(Mode::Shadow, request, eligibility, legacy_parameters, &counting_certified_resolver);

    EXPECT_EQ(resolver_invocations, 0);
    EXPECT_EQ(result.resolution.reason, ResolutionReason::TraceCaptureUnsupported);
    EXPECT_EQ(result.action, ExecutionAction::Fallback);
    EXPECT_FALSE(result.materialized_parameters.has_value());
    EXPECT_EQ(ttsl::hash::hash_objects_with_default_seed(legacy_parameters), legacy_hash);
    EXPECT_EQ(resolve(Mode::Shadow, request, eligibility).reason, ResolutionReason::TraceCaptureUnsupported);
}

TEST(MatmulConfigRegistry, UnknownTraceCaptureStateFailsClosedInShadowAndOn) {
    EXPECT_FALSE(fail_closed_trace_capture_active(Mode::Off, std::nullopt));
    EXPECT_FALSE(fail_closed_trace_capture_active(Mode::Shadow, false));
    EXPECT_FALSE(fail_closed_trace_capture_active(Mode::On, false));
    EXPECT_TRUE(fail_closed_trace_capture_active(Mode::Shadow, true));
    EXPECT_TRUE(fail_closed_trace_capture_active(Mode::On, true));
    EXPECT_TRUE(fail_closed_trace_capture_active(Mode::Shadow, std::nullopt));
    EXPECT_TRUE(fail_closed_trace_capture_active(Mode::On, std::nullopt));
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

TEST(MatmulConfigRegistry, CompactDescriptorDispatchMaterializesOnlyInOn) {
    reset_circuit_breakers_for_testing();
    const auto request = exact_request();
    const auto eligibility = Eligibility{.call = request.call};
    const std::array entries{compact_entry()};
    const auto descriptor_hit = resolve_with_compact_table_for_testing(
        Mode::On, request, eligibility, compact_metadata(), entries, compatible_digests());
    ASSERT_EQ(descriptor_hit.reason, ResolutionReason::CertifiedMatch);

    const auto descriptor_resolver =
        +[](const Mode, const MatmulRegistryRequest&, const Eligibility&) noexcept -> Resolution {
        static auto descriptor = compact_entry();
        return {.reason = ResolutionReason::CertifiedMatch, .descriptor = &descriptor};
    };
    ttnn::prim::MatmulParams legacy;
    legacy.output_dtype = DataType::FLOAT32;
    const auto legacy_hash = ttsl::hash::hash_objects_with_default_seed(legacy);

    const auto shadow = resolve_for_dispatch(Mode::Shadow, request, eligibility, legacy, descriptor_resolver);
    EXPECT_EQ(shadow.action, ExecutionAction::ObserveOnly);
    EXPECT_FALSE(shadow.materialized_parameters.has_value());
    EXPECT_EQ(ttsl::hash::hash_objects_with_default_seed(legacy), legacy_hash);

    const auto on = resolve_for_dispatch(Mode::On, request, eligibility, legacy, descriptor_resolver);
    EXPECT_EQ(on.action, ExecutionAction::ApplyRecipe);
    ASSERT_TRUE(on.materialized_parameters.has_value());
    EXPECT_NE(ttsl::hash::hash_objects_with_default_seed(*on.materialized_parameters), legacy_hash);
    EXPECT_EQ(ttsl::hash::hash_objects_with_default_seed(legacy), legacy_hash);
}

TEST(MatmulConfigRegistry, CompactLookupAndMaterializationKeepAllPublicDomainsDisjoint) {
    reset_circuit_breakers_for_testing();
    for (const auto domain : {OperationDomain::DenseMatmul, OperationDomain::Linear, OperationDomain::Addmm}) {
        const auto request = exact_request(domain);
        const auto eligibility = Eligibility{.call = request.call};
        const std::array entries{compact_entry(domain)};
        const auto hit = resolve_with_compact_table_for_testing(
            Mode::On, request, eligibility, compact_metadata(), entries, compatible_digests());
        ASSERT_EQ(hit.reason, ResolutionReason::CertifiedMatch);
        ASSERT_NE(hit.descriptor, nullptr);
        EXPECT_EQ(materialize_matmul_registry_recipe(*hit.descriptor).status, MaterializationStatus::Success);

        const auto other_domain = domain == OperationDomain::DenseMatmul ? OperationDomain::Linear
                                  : domain == OperationDomain::Linear    ? OperationDomain::Addmm
                                                                         : OperationDomain::DenseMatmul;
        const auto other_request = exact_request(other_domain);
        EXPECT_EQ(
            resolve_with_compact_table_for_testing(
                Mode::On,
                other_request,
                Eligibility{.call = other_request.call},
                compact_metadata(),
                entries,
                compatible_digests())
                .reason,
            ResolutionReason::EmptyRegistry);
    }

    auto addmm_request = exact_request(OperationDomain::Addmm);
    const std::array addmm_entries{compact_entry(OperationDomain::Addmm)};
    addmm_request.call = addmm_call_semantics(2.0F, 0.0F);
    EXPECT_EQ(
        resolve_with_compact_table_for_testing(
            Mode::On,
            addmm_request,
            Eligibility{.call = addmm_request.call},
            compact_metadata(),
            addmm_entries,
            compatible_digests())
            .reason,
        ResolutionReason::EmptyRegistry);
}

TEST(MatmulConfigRegistry, TypedMaterializationRejectionFallsBackAndCircuitBreaksOnlyItsDomain) {
    reset_stats_for_testing();
    reset_circuit_breakers_for_testing();
    const auto request = exact_request();
    const auto eligibility = Eligibility{.call = request.call};
    const auto invalid_resolver =
        +[](const Mode, const MatmulRegistryRequest&, const Eligibility&) noexcept -> Resolution {
        static auto descriptor = [] {
            auto value = compact_entry();
            value.replay.program_config.out_subblock_h = 9;
            return value;
        }();
        return {.reason = ResolutionReason::CertifiedMatch, .descriptor = &descriptor};
    };
    ttnn::prim::MatmulParams legacy;
    legacy.output_dtype = DataType::FLOAT32;
    const auto legacy_hash = ttsl::hash::hash_objects_with_default_seed(legacy);

    const auto result = resolve_for_dispatch(Mode::On, request, eligibility, legacy, invalid_resolver);
    EXPECT_EQ(result.resolution.reason, ResolutionReason::MaterializationRejected);
    EXPECT_EQ(result.action, ExecutionAction::Fallback);
    EXPECT_FALSE(result.materialized_parameters.has_value());
    EXPECT_EQ(ttsl::hash::hash_objects_with_default_seed(legacy), legacy_hash);
    EXPECT_TRUE(is_domain_circuit_broken(OperationDomain::DenseMatmul));
    EXPECT_FALSE(is_domain_circuit_broken(OperationDomain::Linear));
    EXPECT_FALSE(circuit_break_domain(OperationDomain::DenseMatmul));

    const auto snapshot = stats_snapshot();
    const auto& dense = snapshot.domains[static_cast<std::size_t>(OperationDomain::DenseMatmul)];
    EXPECT_EQ(dense.circuit_breaker_activations, 1);
    EXPECT_TRUE(dense.circuit_broken);
    reset_circuit_breakers_for_testing();
}

TEST(MatmulConfigRegistry, UnexpectedMaterializationExceptionCircuitBreaksAndFallsBackBeforeDispatch) {
    reset_circuit_breakers_for_testing();
    const auto request = exact_request(OperationDomain::Linear);
    const auto eligibility = Eligibility{.call = request.call};
    const auto descriptor_resolver =
        +[](const Mode, const MatmulRegistryRequest&, const Eligibility&) noexcept -> Resolution {
        static auto descriptor = compact_entry(OperationDomain::Linear);
        return {.reason = ResolutionReason::CertifiedMatch, .descriptor = &descriptor};
    };

    const auto result = resolve_for_dispatch(
        Mode::On, request, eligibility, ttnn::prim::MatmulParams{}, descriptor_resolver, &throwing_materializer);
    EXPECT_EQ(result.resolution.reason, ResolutionReason::MaterializationRejected);
    EXPECT_EQ(result.action, ExecutionAction::Fallback);
    EXPECT_FALSE(result.materialized_parameters.has_value());
    EXPECT_TRUE(is_domain_circuit_broken(OperationDomain::Linear));
    EXPECT_FALSE(is_domain_circuit_broken(OperationDomain::DenseMatmul));
    reset_circuit_breakers_for_testing();
}

TEST(MatmulConfigRegistry, InconsistentInjectedRecipeCircuitBreaksAndFallsBackBeforeDispatch) {
    reset_circuit_breakers_for_testing();
    const auto request = exact_request(OperationDomain::Addmm);
    const auto eligibility = Eligibility{.call = request.call};
    const auto inconsistent_resolver =
        +[](const Mode, const MatmulRegistryRequest&, const Eligibility&) noexcept -> Resolution {
        static auto recipe = [] {
            auto value = basic_recipe();
            value.untilize_out = true;
            return value;
        }();
        return {.reason = ResolutionReason::CertifiedMatch, .recipe = &recipe};
    };

    const auto result =
        resolve_for_dispatch(Mode::On, request, eligibility, ttnn::prim::MatmulParams{}, inconsistent_resolver);
    EXPECT_EQ(result.resolution.reason, ResolutionReason::MaterializationRejected);
    EXPECT_EQ(result.action, ExecutionAction::Fallback);
    EXPECT_FALSE(result.materialized_parameters.has_value());
    EXPECT_TRUE(is_domain_circuit_broken(OperationDomain::Addmm));
    reset_circuit_breakers_for_testing();
}

TEST(MatmulConfigRegistry, CircuitBreakerActivationIsConcurrentAndDomainIsolated) {
    reset_stats_for_testing();
    reset_circuit_breakers_for_testing();
    constexpr std::size_t thread_count = 16;
    std::array<bool, thread_count> activated{};
    std::array<std::thread, thread_count> threads;
    for (std::size_t index = 0; index < threads.size(); ++index) {
        threads[index] =
            std::thread([index, &activated] { activated[index] = circuit_break_domain(OperationDomain::DenseMatmul); });
    }
    for (auto& thread : threads) {
        thread.join();
    }
    EXPECT_EQ(std::count(activated.begin(), activated.end(), true), 1);
    EXPECT_TRUE(is_domain_circuit_broken(OperationDomain::DenseMatmul));
    EXPECT_FALSE(is_domain_circuit_broken(OperationDomain::Linear));
    EXPECT_EQ(
        stats_snapshot().domains[static_cast<std::size_t>(OperationDomain::DenseMatmul)].circuit_breaker_activations,
        1);

    const auto dense_request = exact_request();
    const std::array dense_entries{compact_entry()};
    EXPECT_EQ(
        resolve_with_compact_table_for_testing(
            Mode::On,
            dense_request,
            Eligibility{.call = dense_request.call},
            compact_metadata(),
            dense_entries,
            compatible_digests())
            .reason,
        ResolutionReason::CircuitBroken);

    EXPECT_TRUE(circuit_break_domain(OperationDomain::Linear));
    const auto linear_request = exact_request(OperationDomain::Linear);
    EXPECT_EQ(
        resolve_with(Mode::Shadow, Eligibility{.call = linear_request.call}).reason, ResolutionReason::CircuitBroken);
    reset_circuit_breakers_for_testing();
}

TEST(MatmulConfigRegistry, CompactEntryIdentityStaysOutsideProgramCacheIdentity) {
    auto first = compact_entry();
    auto second = first;
    first.entry_id[0] = 1;
    second.entry_id[0] = 2;

    const auto first_recipe = materialize_matmul_registry_recipe(first);
    const auto second_recipe = materialize_matmul_registry_recipe(second);
    ASSERT_TRUE(first_recipe.recipe.has_value());
    ASSERT_TRUE(second_recipe.recipe.has_value());
    ttnn::prim::MatmulParams legacy;
    const auto first_parameters = materialize_parameters_for_execution(
        Mode::On, Resolution{.reason = ResolutionReason::CertifiedMatch, .recipe = &*first_recipe.recipe}, legacy);
    const auto second_parameters = materialize_parameters_for_execution(
        Mode::On, Resolution{.reason = ResolutionReason::CertifiedMatch, .recipe = &*second_recipe.recipe}, legacy);
    ASSERT_TRUE(first_parameters.has_value());
    ASSERT_TRUE(second_parameters.has_value());
    EXPECT_EQ(
        ttsl::hash::hash_objects_with_default_seed(*first_parameters),
        ttsl::hash::hash_objects_with_default_seed(*second_parameters));

    second.replay.compute_kernel_config.math_approx_mode = false;
    const auto changed_recipe = materialize_matmul_registry_recipe(second);
    ASSERT_TRUE(changed_recipe.recipe.has_value());
    const auto changed_parameters = materialize_parameters_for_execution(
        Mode::On, Resolution{.reason = ResolutionReason::CertifiedMatch, .recipe = &*changed_recipe.recipe}, legacy);
    ASSERT_TRUE(changed_parameters.has_value());
    EXPECT_NE(
        ttsl::hash::hash_objects_with_default_seed(*first_parameters),
        ttsl::hash::hash_objects_with_default_seed(*changed_parameters));
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
    record_resolution(Mode::On, OperationDomain::Addmm, hit, ExecutionAction::ApplyRecipe);
    record_completed_hit(OperationDomain::Addmm);
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
    const auto& addmm = snapshot.domains[static_cast<std::size_t>(OperationDomain::Addmm)];
    EXPECT_EQ(addmm.selected_hits, 1);
    EXPECT_EQ(addmm.completed_hits, 1);

    reset_stats_for_testing();
    snapshot = stats_snapshot();
    EXPECT_EQ(snapshot.domains[static_cast<std::size_t>(OperationDomain::DenseMatmul)].resolution_attempts, 0);
}

TEST(MatmulConfigRegistry, EveryTelemetryReasonHasAStableUniqueName) {
    std::array<std::string_view, kResolutionReasonCount> names;
    for (std::size_t index = 0; index < names.size(); ++index) {
        names[index] = resolution_reason_name(static_cast<ResolutionReason>(index));
        EXPECT_FALSE(names[index].empty());
        EXPECT_NE(names[index], "unknown");
    }
    std::ranges::sort(names);
    EXPECT_EQ(std::ranges::adjacent_find(names), names.end());
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

TEST(MatmulConfigRegistry, EachPublicOperationHasADistinctSafeEmptyDomain) {
    EXPECT_EQ(
        resolve_with(Mode::On, Eligibility{.call = CallSemantics{.domain = OperationDomain::DenseMatmul}}).reason,
        ResolutionReason::EmptyRegistry);
    EXPECT_EQ(
        resolve_with(Mode::On, Eligibility{.call = linear_call_semantics()}).reason, ResolutionReason::EmptyRegistry);
    EXPECT_EQ(
        resolve_with(
            Mode::On,
            Eligibility{
                .call =
                    CallSemantics{.domain = OperationDomain::Addmm, .alpha_f32_bits = 0x3f800000, .beta_f32_bits = 0}})
            .reason,
        ResolutionReason::EmptyRegistry);
    EXPECT_EQ(
        resolve_with(
            Mode::On,
            Eligibility{.call = linear_call_semantics(), .has_bias = true, .has_activation = true, .transpose_b = true})
            .reason,
        ResolutionReason::UnsupportedSemantics);
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
        resolve_with(Mode::On, Eligibility{.call = addmm_call_semantics(-0.0F, 1.0F)}).reason,
        ResolutionReason::MalformedOperationSemantics);
    EXPECT_EQ(
        resolve_with(Mode::On, Eligibility{.call = addmm_call_semantics(1.0F, 1.0F)}).reason,
        ResolutionReason::UnsupportedSemantics);
    EXPECT_EQ(
        resolve_with(
            Mode::On,
            Eligibility{
                .call =
                    CallSemantics{
                        .domain = OperationDomain::Linear, .alpha_f32_bits = 0x3f800000, .beta_f32_bits = 0x3f800000}})
            .reason,
        ResolutionReason::MalformedOperationSemantics);
    EXPECT_EQ(
        resolve_with(
            Mode::On,
            Eligibility{.call = CallSemantics{.domain = OperationDomain::Linear, .alpha_f32_bits = 0x3f800000}})
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
    expect_unsupported(Eligibility{.call = linear_call_semantics(), .has_bias = true});
    expect_unsupported(Eligibility{.call = linear_call_semantics(), .has_activation = true});
}

}  // namespace
}  // namespace ttnn::operations::matmul::registry
