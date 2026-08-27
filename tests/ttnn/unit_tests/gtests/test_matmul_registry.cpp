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
#include <tuple>
#include <variant>
#include <vector>

#include <tt_stl/reflection.hpp>

#include "ttnn/operations/matmul/device/config/matmul_config_registry.hpp"
#include "ttnn/operations/matmul/device/config/registry/matmul_program_config_model.hpp"
#include "ttnn/operations/matmul/device/config/registry/matmul_registry_descriptor.hpp"
#include "ttnn/operations/matmul/device/matmul_device_operation.hpp"

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

compact::Domain compact_domain(const OperationDomain domain) {
    switch (domain) {
        case OperationDomain::DenseMatmul: return compact::Domain::DenseMatmul;
        case OperationDomain::Linear: return compact::Domain::DenseLinear;
        case OperationDomain::Addmm: return compact::Domain::DenseAddmm;
        case OperationDomain::IneligibleSharedCaller: break;
    }
    throw std::invalid_argument("operation domain has no compact registry representation");
}

OperationDomain next_public_domain(const OperationDomain domain) {
    switch (domain) {
        case OperationDomain::DenseMatmul: return OperationDomain::Linear;
        case OperationDomain::Linear: return OperationDomain::Addmm;
        case OperationDomain::Addmm: return OperationDomain::DenseMatmul;
        case OperationDomain::IneligibleSharedCaller: break;
    }
    throw std::invalid_argument("operation domain is not a public registry domain");
}

static_assert(std::is_trivially_copyable_v<compact::KeyDescriptor>);
static_assert(std::is_trivially_copyable_v<compact::ReplayDescriptor>);
static_assert(std::is_trivially_copyable_v<compact::EntryDescriptor>);

template <typename T>
concept HasComputeKernelConfig = requires(T value) { value.compute_kernel_config; };

static_assert(!HasComputeKernelConfig<compact::ProgramConfigCandidate>);

TEST(MatmulConfigRegistry, EffectiveComputeKernelConfigUsesExecutionDefaultSourceOfTruth) {
    const auto explicit_program_bf16 = ttnn::prim::resolve_matmul_effective_compute_kernel_config(
        tt::ARCH::BLACKHOLE,
        DataType::BFLOAT16,
        DataType::BFLOAT16,
        DataType::BFLOAT16,
        /*has_program_config=*/true,
        /*has_user_core_coord=*/false,
        std::nullopt);
    EXPECT_EQ(explicit_program_bf16.math_fidelity, tt::tt_metal::MathFidelity::LoFi);
    EXPECT_FALSE(explicit_program_bf16.math_approx_mode);
    EXPECT_FALSE(explicit_program_bf16.fp32_dest_acc_en);
    EXPECT_TRUE(explicit_program_bf16.packer_l1_acc);
    EXPECT_FALSE(explicit_program_bf16.dst_full_sync_en);
    EXPECT_EQ(
        explicit_program_bf16.throttle_level, ttnn::operations::compute_throttle_utils::ThrottleLevel::NO_THROTTLE);

    const auto implicit_program_bf16 = ttnn::prim::resolve_matmul_effective_compute_kernel_config(
        tt::ARCH::BLACKHOLE,
        DataType::BFLOAT16,
        DataType::BFLOAT16,
        DataType::BFLOAT16,
        /*has_program_config=*/false,
        /*has_user_core_coord=*/false,
        std::nullopt);
    EXPECT_EQ(implicit_program_bf16.math_fidelity, tt::tt_metal::MathFidelity::HiFi2);

    const auto explicit_program_fp32_wh = ttnn::prim::resolve_matmul_effective_compute_kernel_config(
        tt::ARCH::WORMHOLE_B0,
        DataType::FLOAT32,
        DataType::FLOAT32,
        DataType::FLOAT32,
        /*has_program_config=*/true,
        /*has_user_core_coord=*/false,
        std::nullopt);
    EXPECT_EQ(explicit_program_fp32_wh.math_fidelity, tt::tt_metal::MathFidelity::HiFi3);
    EXPECT_TRUE(explicit_program_fp32_wh.fp32_dest_acc_en);
    EXPECT_FALSE(explicit_program_fp32_wh.packer_l1_acc);

    auto caller_config = DeviceComputeKernelConfig{
        .math_fidelity = tt::tt_metal::MathFidelity::HiFi2,
        .math_approx_mode = true,
        .fp32_dest_acc_en = false,
        .packer_l1_acc = false,
        .dst_full_sync_en = true,
        .throttle_level = ttnn::operations::compute_throttle_utils::ThrottleLevel::LEVEL_3};
    const auto explicit_caller = ttnn::prim::resolve_matmul_effective_compute_kernel_config(
        tt::ARCH::BLACKHOLE,
        DataType::BFLOAT16,
        DataType::BFLOAT16,
        DataType::BFLOAT16,
        /*has_program_config=*/true,
        /*has_user_core_coord=*/false,
        caller_config);
    EXPECT_EQ(explicit_caller.math_fidelity, caller_config.math_fidelity);
    EXPECT_EQ(explicit_caller.math_approx_mode, caller_config.math_approx_mode);
    EXPECT_EQ(explicit_caller.fp32_dest_acc_en, caller_config.fp32_dest_acc_en);
    EXPECT_EQ(explicit_caller.packer_l1_acc, caller_config.packer_l1_acc);
    EXPECT_EQ(explicit_caller.dst_full_sync_en, caller_config.dst_full_sync_en);
    EXPECT_EQ(explicit_caller.throttle_level, caller_config.throttle_level);
}

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

using Placement = tt::tt_metal::distributed::MeshMapperConfig::Placement;
using Replicate = tt::tt_metal::distributed::MeshMapperConfig::Replicate;
using Shard = tt::tt_metal::distributed::MeshMapperConfig::Shard;
using MeshCoordinate = tt::tt_metal::distributed::MeshCoordinate;

struct DistributedTensorFixture {
    std::vector<std::uint32_t> logical_shape;
    DataType dtype = DataType::BFLOAT16;
    tt::tt_metal::Layout layout = tt::tt_metal::Layout::TILE;
    TensorMemoryLayout memory_layout = TensorMemoryLayout::INTERLEAVED;
    BufferType buffer_type = BufferType::DRAM;
    std::vector<std::uint32_t> distribution_shape;
    std::vector<Placement> placements;
    std::vector<MeshCoordinate> mesh_coordinates;
    std::vector<MeshCoordinate> storage_coordinates;

    DistributedTensorView view() const {
        return {
            .logical_shape = logical_shape,
            .dtype = dtype,
            .layout = layout,
            .memory_layout = memory_layout,
            .buffer_type = buffer_type,
            .distribution_shape = distribution_shape,
            .placements = placements,
            .mesh_coordinates = mesh_coordinates,
            .storage_coordinates = storage_coordinates,
        };
    }
};

std::vector<MeshCoordinate> bh32_coordinates() {
    std::vector<MeshCoordinate> coordinates;
    coordinates.reserve(32);
    for (std::uint32_t row = 0; row < 8; ++row) {
        for (std::uint32_t col = 0; col < 4; ++col) {
            coordinates.emplace_back(row, col);
        }
    }
    return coordinates;
}

struct DistributedObservationFixture {
    std::array<std::uint32_t, 2> device_mesh_shape{8, 4};
    DistributedTensorFixture a;
    DistributedTensorFixture b;

    DistributedMatmulObservation observation() const {
        return {
            .domain = OperationDomain::DenseMatmul,
            .tensors_share_device = true,
            .device_count = 32,
            .device_mesh_shape = device_mesh_shape,
            .input_a = a.view(),
            .input_b = b.view(),
            .output_is_dram_interleaved = true,
        };
    }
};

DistributedObservationFixture dp_observation_fixture() {
    const auto coordinates = bh32_coordinates();
    return {
        .a =
            DistributedTensorFixture{
                .logical_shape = {1, 1, 9472, 5120},
                .distribution_shape = {8, 4},
                .placements = {Shard{0}, Shard{1}},
                .mesh_coordinates = coordinates,
                .storage_coordinates = coordinates,
            },
        .b =
            DistributedTensorFixture{
                .logical_shape = {5120, 3840},
                .distribution_shape = {32},
                .placements = {Replicate{}},
                .mesh_coordinates = coordinates,
                .storage_coordinates = coordinates,
            },
    };
}

DistributedObservationFixture tpn_observation_fixture() {
    const auto coordinates = bh32_coordinates();
    return {
        .a =
            DistributedTensorFixture{
                .logical_shape = {9472, 5120},
                .distribution_shape = {8, 4},
                .placements = {Shard{0}, Replicate{}},
                .mesh_coordinates = coordinates,
                .storage_coordinates = coordinates,
            },
        .b =
            DistributedTensorFixture{
                .logical_shape = {5120, 3840},
                .distribution_shape = {8, 4},
                .placements = {Replicate{}, Shard{1}},
                .mesh_coordinates = coordinates,
                .storage_coordinates = coordinates,
            },
    };
}

TEST(MatmulConfigRegistry, DistributedClassifierNamesAreStableAndBounded) {
    EXPECT_EQ(distributed_matmul_class_name(DistributedMatmulClass::NotDistributed), "not_distributed");
    EXPECT_EQ(
        distributed_matmul_class_name(DistributedMatmulClass::DpRankDistinctV1), "distributed.dp.rank_distinct_v1");
    EXPECT_EQ(distributed_matmul_class_name(DistributedMatmulClass::TpnSpMTpNV1), "distributed.tpn.sp_m_tp_n_v1");
    EXPECT_EQ(distributed_matmul_class_name(DistributedMatmulClass::Unknown), "distributed.unknown");
    EXPECT_EQ(kDistributedMatmulClassCount, 4);
}

TEST(MatmulConfigRegistry, DistributedClassifierRecognizesOnlyExactExploredBh32Layouts) {
    auto dp = dp_observation_fixture();
    auto tpn = tpn_observation_fixture();
    EXPECT_EQ(classify_distributed_matmul(dp.observation()), DistributedMatmulClass::DpRankDistinctV1);
    EXPECT_EQ(classify_distributed_matmul(tpn.observation()), DistributedMatmulClass::TpnSpMTpNV1);

    for (const auto dtype : {DataType::BFLOAT16, DataType::BFLOAT8_B, DataType::BFLOAT4_B}) {
        dp.b.dtype = dtype;
        tpn.b.dtype = dtype;
        EXPECT_EQ(classify_distributed_matmul(dp.observation()), DistributedMatmulClass::DpRankDistinctV1);
        EXPECT_EQ(classify_distributed_matmul(tpn.observation()), DistributedMatmulClass::TpnSpMTpNV1);
    }

    auto one_chip = dp.observation();
    one_chip.device_count = 1;
    EXPECT_EQ(classify_distributed_matmul(one_chip), DistributedMatmulClass::NotDistributed);
    one_chip.device_count = 0;
    EXPECT_EQ(classify_distributed_matmul(one_chip), DistributedMatmulClass::NotDistributed);
}

TEST(MatmulConfigRegistry, DistributedClassifierRejectsEveryNonBareCallAxis) {
    const auto fixture = dp_observation_fixture();
    const auto expect_unknown = [](const DistributedMatmulObservation& observation) {
        EXPECT_EQ(classify_distributed_matmul(observation), DistributedMatmulClass::Unknown);
    };

    auto changed = fixture.observation();
    changed.domain = OperationDomain::Linear;
    expect_unknown(changed);
    changed = fixture.observation();
    changed.tensors_share_device = false;
    expect_unknown(changed);
    changed = fixture.observation();
    changed.device_count = 31;
    expect_unknown(changed);
    changed = fixture.observation();
    changed.transpose_a = true;
    expect_unknown(changed);
    changed = fixture.observation();
    changed.transpose_b = true;
    expect_unknown(changed);
    changed = fixture.observation();
    changed.has_bias = true;
    expect_unknown(changed);
    changed = fixture.observation();
    changed.has_activation = true;
    expect_unknown(changed);
    changed = fixture.observation();
    changed.has_program_config = true;
    expect_unknown(changed);
    changed = fixture.observation();
    changed.has_compute_kernel_config = true;
    expect_unknown(changed);
    changed = fixture.observation();
    changed.has_user_core_grid = true;
    expect_unknown(changed);
    changed = fixture.observation();
    changed.has_output_dtype = true;
    expect_unknown(changed);
    changed = fixture.observation();
    changed.has_optional_output = true;
    expect_unknown(changed);
    changed = fixture.observation();
    changed.has_output_tile = true;
    expect_unknown(changed);
    changed = fixture.observation();
    changed.has_global_cb = true;
    expect_unknown(changed);
    changed = fixture.observation();
    changed.has_sub_device = true;
    expect_unknown(changed);
    changed = fixture.observation();
    changed.has_bcast_batch = true;
    expect_unknown(changed);
    changed = fixture.observation();
    changed.untilize_out = true;
    expect_unknown(changed);
    changed = fixture.observation();
    changed.run_batched = true;
    expect_unknown(changed);
    changed = fixture.observation();
    changed.output_is_dram_interleaved = false;
    expect_unknown(changed);
}

TEST(MatmulConfigRegistry, DistributedClassifierRejectsUnexploredTensorContracts) {
    const auto expect_dp_unknown = [](const auto& mutate) {
        auto fixture = dp_observation_fixture();
        mutate(fixture);
        EXPECT_EQ(classify_distributed_matmul(fixture.observation()), DistributedMatmulClass::Unknown);
    };
    const auto expect_tpn_unknown = [](const auto& mutate) {
        auto fixture = tpn_observation_fixture();
        mutate(fixture);
        EXPECT_EQ(classify_distributed_matmul(fixture.observation()), DistributedMatmulClass::Unknown);
    };

    expect_dp_unknown([](auto& f) { f.device_mesh_shape = {4, 8}; });
    expect_dp_unknown([](auto& f) { f.a.logical_shape = {1, 9472, 5120}; });
    expect_dp_unknown([](auto& f) { f.a.logical_shape[0] = 2; });
    expect_dp_unknown([](auto& f) { f.b.logical_shape[0]++; });
    expect_dp_unknown([](auto& f) { f.a.distribution_shape = {32}; });
    expect_dp_unknown([](auto& f) { f.a.placements = {Shard{0}, Replicate{}}; });
    expect_dp_unknown([](auto& f) { f.b.placements = {Shard{1}}; });
    expect_tpn_unknown([](auto& f) { f.a.logical_shape = {1, 1, 9472, 5120}; });
    expect_tpn_unknown([](auto& f) { f.b.logical_shape[0]++; });
    expect_tpn_unknown([](auto& f) { f.a.placements = {Replicate{}, Shard{0}}; });
    expect_tpn_unknown([](auto& f) { f.b.placements = {Shard{1}, Replicate{}}; });
    expect_tpn_unknown([](auto& f) { f.b.placements = {Replicate{}, Shard{0}}; });

    for (const auto bad_dtype : {DataType::FLOAT32, DataType::UINT32}) {
        expect_dp_unknown([bad_dtype](auto& f) { f.a.dtype = bad_dtype; });
        expect_tpn_unknown([bad_dtype](auto& f) { f.b.dtype = bad_dtype; });
    }
    expect_dp_unknown([](auto& f) { f.a.layout = tt::tt_metal::Layout::ROW_MAJOR; });
    expect_dp_unknown([](auto& f) { f.b.memory_layout = TensorMemoryLayout::WIDTH_SHARDED; });
    expect_dp_unknown([](auto& f) { f.b.buffer_type = BufferType::L1; });
}

TEST(MatmulConfigRegistry, DistributedClassifierRejectsPartialMismatchedOrReorderedCoordinates) {
    const auto expect_unknown = [](const auto& mutate) {
        auto fixture = tpn_observation_fixture();
        mutate(fixture);
        EXPECT_EQ(classify_distributed_matmul(fixture.observation()), DistributedMatmulClass::Unknown);
    };
    expect_unknown([](auto& f) { f.a.mesh_coordinates.pop_back(); });
    expect_unknown([](auto& f) { f.a.storage_coordinates.pop_back(); });
    expect_unknown([](auto& f) { std::swap(f.a.mesh_coordinates[0], f.a.mesh_coordinates[1]); });
    expect_unknown([](auto& f) { std::swap(f.b.mesh_coordinates[0], f.b.mesh_coordinates[1]); });
    expect_unknown([](auto& f) { std::swap(f.b.storage_coordinates[0], f.b.storage_coordinates[1]); });
    expect_unknown([](auto& f) { f.a.mesh_coordinates[31] = MeshCoordinate(0, 0); });
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
        .domain = compact_domain(domain),
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
                .per_core_n = 16,
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

compact::ProgramConfigModelSupport compact_model_support(const compact::KeyDescriptor& key) {
    return {
        .architecture = key.architecture,
        .board_capability_class = key.board_capability_class,
        .device_count = key.device_count,
        .mesh_rows = key.mesh_rows,
        .mesh_cols = key.mesh_cols,
        .topology_sha256 = key.topology_sha256,
        .domain = key.domain,
        .input_a = key.input_a,
        .input_b = key.input_b,
        .output = key.output,
        .shape_scale = compact::shape_scale_class(key.logical_m),
        .shape_geometry = compact::shape_geometry_class(key.logical_k, key.logical_n),
        .minimum_m = 1,
        .maximum_m = 16384,
        .minimum_k = 1,
        .maximum_k = 32768,
        .minimum_n = 1,
        .maximum_n = 32768,
    };
}

compact::TableMetadata compact_metadata();
CompatibilityDigests compatible_digests();

TEST(MatmulConfigRegistry, OnlineProgramConfigLookupUsesExactBeforeGbdt) {
    auto exact = compact_entry();
    const std::array entries{compact::ProgramConfigExactEntry{
        .entry_id = exact.entry_id,
        .key = exact.key,
        .program_config = compact::exact_program_config(exact.replay),
    }};
    const compact::ProgramConfigGbdtModel malformed_model{};

    const auto result = compact::lookup_program_config(
        exact.key, entries, malformed_model, compact_metadata().online_model_bundle_binding_sha256);

    EXPECT_EQ(result.source, compact::ProgramConfigLookupSource::Exact);
    ASSERT_TRUE(result.program_config.has_value());
    EXPECT_EQ(result.program_config->family, compact::ProgramFamily::MultiCoreReuse);
    EXPECT_EQ(result.program_config->compute_grid_x, entries[0].program_config.compute_grid_x);
    EXPECT_EQ(result.identity, &entries[0].entry_id);

    auto malformed_entries = entries;
    malformed_entries[0].program_config.compute_grid_x = 0;
    const auto malformed = compact::lookup_program_config(
        exact.key, malformed_entries, malformed_model, compact_metadata().online_model_bundle_binding_sha256);
    EXPECT_EQ(malformed.source, compact::ProgramConfigLookupSource::None);
    EXPECT_FALSE(malformed.program_config.has_value());
}

TEST(MatmulConfigRegistry, OnlineGbdtScoresOnlyLegalProgramConfigsAndFallsBackOnNoCandidate) {
    const auto training_key = compact_entry().key;
    auto key = training_key;
    key.logical_m -= 1;

    std::array candidates{
        compact::ProgramConfigCandidate{
            .program_config =
                compact::ProgramConfigDescriptor{
                    .family = compact::ProgramFamily::MultiCoreReuse,
                    .compute_grid_x = 4,
                    .compute_grid_y = 4,
                    .in0_block_w = 2,
                    .out_subblock_h = 1,
                    .out_subblock_w = 2,
                    .per_core_m = 2,
                    .per_core_n = 16,
                    .allowed_worker_cores_present = false},
            .candidate_id = repeated_digest(1)},
        compact::ProgramConfigCandidate{
            .program_config =
                compact::ProgramConfigDescriptor{
                    .family = compact::ProgramFamily::MultiCoreReuse,
                    .compute_grid_x = 8,
                    .compute_grid_y = 8,
                    .in0_block_w = 2,
                    .out_subblock_h = 1,
                    .out_subblock_w = 2,
                    .per_core_m = 4,
                    .per_core_n = 16,
                    .allowed_worker_cores_present = false},
            .candidate_id = repeated_digest(2)},
    };
    // One stump: larger grid gets the lower (better) score.
    const std::array nodes{
        compact::GbdtNode{.feature = compact::ProgramConfigFeature::GridX, .threshold = 4, .left = 1, .right = 2},
        compact::GbdtNode{.feature = compact::ProgramConfigFeature::Count, .leaf_value = 10},
        compact::GbdtNode{.feature = compact::ProgramConfigFeature::Count, .leaf_value = -10},
    };
    const std::array trees{compact::GbdtTree{.node_offset = 0, .node_count = 3}};
    const std::array training_shapes{compact::TrainingShapeLandmark{
        .logical_m = training_key.logical_m, .logical_k = training_key.logical_k, .logical_n = training_key.logical_n}};
    const compact::ProgramConfigGbdtModel model{
        .schema_version = 1,
        .enabled = true,
        .score_orientation = compact::GbdtScoreOrientation::LowerIsBetterNegatedPairwiseMargin,
        .feature_schema_sha256 = repeated_digest(2),
        .model_sha256 = repeated_digest(3),
        .training_table_sha256 = repeated_digest(4),
        .safety_evidence_sha256 = repeated_digest(6),
        .candidate_policy_sha256 = repeated_digest(7),
        .lineage_sha256 = repeated_digest(8),
        .evaluation_model_payload_sha256 = repeated_digest(11),
        .quality_evaluation_sha256 = repeated_digest(9),
        .unseen_abstention_policy_sha256 = repeated_digest(10),
        .support_sha256 = repeated_digest(5),
        .bundle_binding_sha256 = compact_metadata().online_model_bundle_binding_sha256,
        .support = compact_model_support(key),
        .base_score = 0,
        .score_scale = 1,
        .minimum_score_margin = 1,
        .maximum_normalized_shape_distance_ppm = 250'000,
        .training_shapes = training_shapes,
        .candidates = candidates,
        .trees = trees,
        .nodes = nodes,
    };

    const auto selected =
        compact::lookup_program_config(key, {}, model, compact_metadata().online_model_bundle_binding_sha256);
    EXPECT_EQ(selected.source, compact::ProgramConfigLookupSource::Gbdt);
    ASSERT_TRUE(selected.program_config.has_value());
    EXPECT_EQ(selected.program_config->compute_grid_x, 8);

    const auto missing_exact =
        compact::lookup_program_config(training_key, {}, model, compact_metadata().online_model_bundle_binding_sha256);
    EXPECT_EQ(missing_exact.source, compact::ProgramConfigLookupSource::None);
    EXPECT_FALSE(missing_exact.program_config.has_value());

    const auto exact_descriptor = compact_entry();
    const std::array exact_entries{compact::ProgramConfigExactEntry{
        .entry_id = exact_descriptor.entry_id,
        .key = training_key,
        .program_config = compact::exact_program_config(exact_descriptor.replay),
    }};
    const auto exact = compact::lookup_program_config(
        training_key, exact_entries, model, compact_metadata().online_model_bundle_binding_sha256);
    EXPECT_EQ(exact.source, compact::ProgramConfigLookupSource::Exact);
    ASSERT_TRUE(exact.program_config.has_value());
    EXPECT_EQ(*exact.program_config, exact_entries[0].program_config);

    auto insufficient_margin = model;
    insufficient_margin.minimum_score_margin = 21;
    const auto low_confidence = compact::lookup_program_config(
        key, {}, insufficient_margin, compact_metadata().online_model_bundle_binding_sha256);
    EXPECT_EQ(low_confidence.source, compact::ProgramConfigLookupSource::None);

    auto tie_nodes = nodes;
    tie_nodes[1].leaf_value = -10;
    auto tied = model;
    tied.nodes = tie_nodes;
    const auto tied_result =
        compact::lookup_program_config(key, {}, tied, compact_metadata().online_model_bundle_binding_sha256);
    EXPECT_EQ(tied_result.source, compact::ProgramConfigLookupSource::None);

    key.padded_n += 32;
    const auto fallback =
        compact::lookup_program_config(key, {}, model, compact_metadata().online_model_bundle_binding_sha256);
    EXPECT_EQ(fallback.source, compact::ProgramConfigLookupSource::None);
    EXPECT_FALSE(fallback.program_config.has_value());
}

TEST(MatmulConfigRegistry, OnlineModelRejectsDuplicateProgramConfigsFromHiddenKnobVariants) {
    auto key = compact_entry().key;
    key.logical_m -= 1;
    auto candidate = compact::ProgramConfigCandidate{
        .program_config = compact::exact_program_config(compact_entry().replay),
        .candidate_id = repeated_digest(1),
    };
    std::array candidates{candidate, candidate};
    candidates[1].candidate_id = repeated_digest(2);
    const std::array nodes{
        compact::GbdtNode{.feature = compact::ProgramConfigFeature::Count, .leaf_value = 0},
    };
    const std::array trees{compact::GbdtTree{.node_offset = 0, .node_count = 1}};
    const std::array training_shapes{compact::TrainingShapeLandmark{
        .logical_m = key.logical_m + 1, .logical_k = key.logical_k, .logical_n = key.logical_n}};
    const compact::ProgramConfigGbdtModel model{
        .schema_version = 1,
        .enabled = true,
        .score_orientation = compact::GbdtScoreOrientation::LowerIsBetterNegatedPairwiseMargin,
        .feature_schema_sha256 = repeated_digest(2),
        .model_sha256 = repeated_digest(3),
        .training_table_sha256 = repeated_digest(4),
        .safety_evidence_sha256 = repeated_digest(6),
        .candidate_policy_sha256 = repeated_digest(7),
        .lineage_sha256 = repeated_digest(8),
        .evaluation_model_payload_sha256 = repeated_digest(11),
        .quality_evaluation_sha256 = repeated_digest(9),
        .unseen_abstention_policy_sha256 = repeated_digest(10),
        .support_sha256 = repeated_digest(5),
        .bundle_binding_sha256 = compact_metadata().online_model_bundle_binding_sha256,
        .support = compact_model_support(key),
        .base_score = 0,
        .score_scale = 1,
        .minimum_score_margin = 1,
        .maximum_normalized_shape_distance_ppm = 250'000,
        .training_shapes = training_shapes,
        .candidates = candidates,
        .trees = trees,
        .nodes = nodes,
    };

    // The online candidate ABI has no CKC field, and duplicate program config
    // identities fail closed rather than allowing hidden CKC rows to compete.
    const auto result =
        compact::lookup_program_config(key, {}, model, compact_metadata().online_model_bundle_binding_sha256);
    EXPECT_EQ(result.source, compact::ProgramConfigLookupSource::None);
    EXPECT_FALSE(result.program_config.has_value());
}

TEST(MatmulConfigRegistry, OnlineFeatureVectorDistinguishesEveryProgramConfigBoolean) {
    const auto key = compact_entry().key;
    auto candidate = compact::ProgramConfigCandidate{
        .program_config = compact::exact_program_config(compact_entry().replay),
        .candidate_id = repeated_digest(1),
    };
    EXPECT_EQ(compact::feature_value(key, candidate, compact::ProgramConfigFeature::FuseBatch), 0);
    EXPECT_EQ(compact::feature_value(key, candidate, compact::ProgramConfigFeature::McastIn0), 0);
    EXPECT_EQ(compact::feature_value(key, candidate, compact::ProgramConfigFeature::TransposeMcast), 0);

    candidate.program_config.fuse_batch = true;
    candidate.program_config.mcast_in0 = true;
    candidate.program_config.transpose_mcast = true;
    EXPECT_EQ(compact::feature_value(key, candidate, compact::ProgramConfigFeature::FuseBatch), 1);
    EXPECT_EQ(compact::feature_value(key, candidate, compact::ProgramConfigFeature::McastIn0), 1);
    EXPECT_EQ(compact::feature_value(key, candidate, compact::ProgramConfigFeature::TransposeMcast), 1);
}

TEST(MatmulConfigRegistry, OnlineModelSupportRejectsNoncanonicalAndOverflowingPadding) {
    auto key = compact_entry().key;
    const auto support = compact_model_support(key);
    const std::array training_shapes{compact::TrainingShapeLandmark{
        .logical_m = key.logical_m, .logical_k = key.logical_k, .logical_n = key.logical_n}};
    const auto model = compact::ProgramConfigGbdtModel{
        .schema_version = 1,
        .enabled = true,
        .feature_schema_sha256 = repeated_digest(1),
        .model_sha256 = repeated_digest(2),
        .training_table_sha256 = repeated_digest(3),
        .safety_evidence_sha256 = repeated_digest(4),
        .candidate_policy_sha256 = repeated_digest(5),
        .lineage_sha256 = repeated_digest(7),
        .evaluation_model_payload_sha256 = repeated_digest(10),
        .quality_evaluation_sha256 = repeated_digest(8),
        .unseen_abstention_policy_sha256 = repeated_digest(9),
        .support_sha256 = repeated_digest(6),
        .bundle_binding_sha256 = compact_metadata().online_model_bundle_binding_sha256,
        .support = support,
        .minimum_score_margin = 1,
        .maximum_normalized_shape_distance_ppm = 250'000,
        .training_shapes = training_shapes,
    };
    EXPECT_TRUE(compact::model_shape_is_training_landmark(key, model));
    EXPECT_FALSE(compact::model_supports(key, model, compact_metadata().online_model_bundle_binding_sha256));

    --key.logical_m;
    EXPECT_FALSE(compact::model_shape_is_training_landmark(key, model));
    EXPECT_TRUE(compact::model_supports(key, model, compact_metadata().online_model_bundle_binding_sha256));

    key.padded_n += key.input_b.tile_width;
    EXPECT_FALSE(compact::model_supports(key, model, compact_metadata().online_model_bundle_binding_sha256));

    EXPECT_FALSE(compact::is_canonical_tile_padding(
        std::numeric_limits<std::uint64_t>::max(), std::numeric_limits<std::uint64_t>::max(), key.input_b.tile_width));

    const std::array sparse_landmarks{
        compact::TrainingShapeLandmark{.logical_m = 1, .logical_k = 1, .logical_n = 1},
        compact::TrainingShapeLandmark{.logical_m = 100, .logical_k = 100, .logical_n = 100},
    };
    auto sparse_model = model;
    sparse_model.support.minimum_m = sparse_model.support.minimum_k = sparse_model.support.minimum_n = 1;
    sparse_model.support.maximum_m = sparse_model.support.maximum_k = sparse_model.support.maximum_n = 100;
    sparse_model.maximum_normalized_shape_distance_ppm = 100'000;
    sparse_model.training_shapes = sparse_landmarks;
    // Mirrors the Python ceil(delta * 1e6 / span) boundary: a 99-wide
    // support at 100k ppm permits delta 9, while delta 10 must abstain.
    EXPECT_EQ(compact::calibrated_axis_delta(99, 100'000), 9);
    auto near = key;
    near.logical_m = 10;
    near.logical_k = 1;
    near.logical_n = 1;
    EXPECT_TRUE(compact::model_shape_is_near_training_data(near, sparse_model));
    near.logical_m = 11;
    EXPECT_FALSE(compact::model_shape_is_near_training_data(near, sparse_model));
    sparse_model.maximum_normalized_shape_distance_ppm = compact::MAX_NORMALIZED_SHAPE_DISTANCE_PPM + 1;
    EXPECT_FALSE(compact::model_shape_is_near_training_data(key, sparse_model));
    sparse_model.maximum_normalized_shape_distance_ppm = 100'000;
    auto sparse_interior = near;
    sparse_interior.logical_m = 50;
    sparse_interior.logical_n = 100;
    EXPECT_FALSE(compact::model_shape_is_near_training_data(sparse_interior, sparse_model));
}

TEST(MatmulConfigRegistry, OnlineProgramConfigMaterializerSupportsBankedWinnerFamilies) {
    const auto key = compact_entry().key;
    const auto common = compact::ProgramConfigDescriptor{
        .family = compact::ProgramFamily::MultiCoreReuse,
        .compute_grid_x = 8,
        .compute_grid_y = 8,
        .in0_block_w = 2,
        .out_subblock_h = 1,
        .out_subblock_w = 2,
        .per_core_m = 4,
        .per_core_n = 16,
        .allowed_worker_cores_present = false,
        .fuse_batch = false,
        .mcast_in0 = false,
        .transpose_mcast = false,
    };
    const auto basic = materialize_registry_program_config(key, common);
    ASSERT_TRUE(basic.has_value());
    EXPECT_TRUE(std::holds_alternative<MatmulMultiCoreReuseProgramConfig>(*basic));

    auto mm1d_descriptor = common;
    mm1d_descriptor.family = compact::ProgramFamily::MultiCast1D;
    mm1d_descriptor.per_core_n = 2;
    mm1d_descriptor.out_block_h = mm1d_descriptor.per_core_m;
    mm1d_descriptor.out_block_w = mm1d_descriptor.per_core_n;
    mm1d_descriptor.num_global_cb_receivers = 1;
    mm1d_descriptor.fuse_batch = true;
    mm1d_descriptor.mcast_in0 = true;
    const auto mm1d = materialize_registry_program_config(key, mm1d_descriptor);
    ASSERT_TRUE(mm1d.has_value());
    const auto* mm1d_native = std::get_if<MatmulMultiCoreReuseMultiCast1DProgramConfig>(&*mm1d);
    ASSERT_NE(mm1d_native, nullptr);
    EXPECT_TRUE(mm1d_native->mcast_in0);
    EXPECT_TRUE(mm1d_native->fuse_batch);
    EXPECT_EQ(mm1d_native->out_block_h, mm1d_descriptor.per_core_m);
    EXPECT_EQ(mm1d_native->out_block_w, mm1d_descriptor.per_core_n);
    EXPECT_EQ(mm1d_native->num_global_cb_receivers, 1);
    EXPECT_FALSE(mm1d_native->fused_activation.has_value());

    auto mm2d_descriptor = common;
    mm2d_descriptor.family = compact::ProgramFamily::MultiCast2D;
    mm2d_descriptor.per_core_m = 1;
    mm2d_descriptor.per_core_n = 2;
    mm2d_descriptor.out_block_h = mm2d_descriptor.per_core_m;
    mm2d_descriptor.out_block_w = mm2d_descriptor.per_core_n;
    mm2d_descriptor.fuse_batch = true;
    mm2d_descriptor.transpose_mcast = true;
    const auto mm2d = materialize_registry_program_config(key, mm2d_descriptor);
    ASSERT_TRUE(mm2d.has_value());
    const auto* mm2d_native = std::get_if<MatmulMultiCoreReuseMultiCastProgramConfig>(&*mm2d);
    ASSERT_NE(mm2d_native, nullptr);
    EXPECT_TRUE(mm2d_native->transpose_mcast);
    EXPECT_TRUE(mm2d_native->fuse_batch);
    EXPECT_EQ(mm2d_native->out_block_h, mm2d_descriptor.per_core_m);
    EXPECT_EQ(mm2d_native->out_block_w, mm2d_descriptor.per_core_n);
    EXPECT_FALSE(mm2d_native->fused_activation.has_value());
}

TEST(MatmulConfigRegistry, ProgramConfigOnlyExactResolverCoversEveryBankedWinnerFamily) {
    const auto request = exact_request();
    const auto key = compact_registry_key(request);
    ASSERT_TRUE(key.has_value());
    auto basic = compact::exact_program_config(compact_entry().replay);
    auto mm1d = basic;
    mm1d.family = compact::ProgramFamily::MultiCast1D;
    mm1d.per_core_n = 2;
    mm1d.out_block_h = mm1d.per_core_m;
    mm1d.out_block_w = mm1d.per_core_n;
    mm1d.num_global_cb_receivers = 1;
    mm1d.fuse_batch = true;
    mm1d.mcast_in0 = true;
    auto mm2d = basic;
    mm2d.family = compact::ProgramFamily::MultiCast2D;
    mm2d.per_core_m = 1;
    mm2d.per_core_n = 2;
    mm2d.out_block_h = mm2d.per_core_m;
    mm2d.out_block_w = mm2d.per_core_n;
    mm2d.fuse_batch = true;
    mm2d.transpose_mcast = true;

    const std::array programs{basic, mm1d, mm2d};
    for (std::size_t index = 0; index < programs.size(); ++index) {
        const std::array exact_entries{compact::ProgramConfigExactEntry{
            .entry_id = repeated_digest(static_cast<std::uint8_t>(index + 1)),
            .key = *key,
            .program_config = programs[index],
        }};
        const auto resolution = resolve_with_compact_table_for_testing(
            Mode::On,
            request,
            Eligibility{.call = request.call},
            compact_metadata(),
            {},
            compatible_digests(),
            {},
            exact_entries);
        ASSERT_EQ(resolution.reason, ResolutionReason::CertifiedMatch);
        ASSERT_TRUE(resolution.predicted_program_config.has_value());
        EXPECT_EQ(*resolution.predicted_program_config, programs[index]);
        EXPECT_TRUE(materialize_registry_program_config(*key, *resolution.predicted_program_config).has_value());
    }
}

TEST(MatmulConfigRegistry, ResolverRunsExactThenGbdtAndAbstainsOutsideModelContext) {
    auto request = exact_request();
    request.workload.logical_m -= 1;
    const auto key = compact_registry_key(request);
    ASSERT_TRUE(key.has_value());
    const auto descriptor = compact::ProgramConfigDescriptor{
        .family = compact::ProgramFamily::MultiCast1D,
        .compute_grid_x = 8,
        .compute_grid_y = 8,
        .in0_block_w = 2,
        .out_subblock_h = 1,
        .out_subblock_w = 2,
        .per_core_m = 4,
        .per_core_n = 2,
        .out_block_h = 4,
        .out_block_w = 2,
        .num_global_cb_receivers = 1,
        .allowed_worker_cores_present = false,
        .fuse_batch = true,
        .mcast_in0 = true,
        .transpose_mcast = false,
    };
    const std::array candidates{
        compact::ProgramConfigCandidate{
            .program_config = compact::exact_program_config(compact_entry().replay),
            .candidate_id = repeated_digest(1),
        },
        compact::ProgramConfigCandidate{
            .program_config = descriptor,
            .candidate_id = repeated_digest(2),
        },
    };
    const std::array nodes{
        compact::GbdtNode{.feature = compact::ProgramConfigFeature::Family, .threshold = 0, .left = 1, .right = 2},
        compact::GbdtNode{.feature = compact::ProgramConfigFeature::Count, .leaf_value = 10},
        compact::GbdtNode{.feature = compact::ProgramConfigFeature::Count, .leaf_value = -10},
    };
    const std::array trees{compact::GbdtTree{.node_offset = 0, .node_count = 3}};
    const std::array training_shapes{compact::TrainingShapeLandmark{
        .logical_m = key->logical_m + 1, .logical_k = key->logical_k, .logical_n = key->logical_n}};
    auto model = compact::ProgramConfigGbdtModel{
        .schema_version = 1,
        .enabled = true,
        .score_orientation = compact::GbdtScoreOrientation::LowerIsBetterNegatedPairwiseMargin,
        .feature_schema_sha256 = repeated_digest(2),
        .model_sha256 = repeated_digest(3),
        .training_table_sha256 = repeated_digest(4),
        .safety_evidence_sha256 = repeated_digest(6),
        .candidate_policy_sha256 = repeated_digest(7),
        .lineage_sha256 = repeated_digest(8),
        .evaluation_model_payload_sha256 = repeated_digest(11),
        .quality_evaluation_sha256 = repeated_digest(9),
        .unseen_abstention_policy_sha256 = repeated_digest(10),
        .support_sha256 = repeated_digest(5),
        .bundle_binding_sha256 = compact_metadata().online_model_bundle_binding_sha256,
        .support = compact_model_support(*key),
        .base_score = 0,
        .score_scale = 1,
        .minimum_score_margin = 1,
        .maximum_normalized_shape_distance_ppm = 250'000,
        .training_shapes = training_shapes,
        .candidates = candidates,
        .trees = trees,
        .nodes = nodes,
    };
    const std::array nonmatching_exact_entries{compact_entry()};
    const auto predicted = resolve_with_compact_table_for_testing(
        Mode::On,
        request,
        Eligibility{.call = request.call},
        compact_metadata(),
        nonmatching_exact_entries,
        compatible_digests(),
        std::span<const compact::ProgramConfigGbdtModel>{&model, 1});
    EXPECT_EQ(predicted.reason, ResolutionReason::PredictedMatch);
    EXPECT_EQ(execution_action(Mode::On, predicted), ExecutionAction::ApplyRecipe);
    ASSERT_TRUE(predicted.predicted_program_config.has_value());
    EXPECT_EQ(predicted.predicted_program_config->family, compact::ProgramFamily::MultiCast1D);

    auto missing_model_proof = compact_metadata();
    missing_model_proof.online_program_config_model_evidence_schema_version = 0;
    const auto unproved = resolve_with_compact_table_for_testing(
        Mode::On,
        request,
        Eligibility{.call = request.call},
        missing_model_proof,
        nonmatching_exact_entries,
        compatible_digests(),
        std::span<const compact::ProgramConfigGbdtModel>{&model, 1});
    EXPECT_EQ(unproved.reason, ResolutionReason::EmptyRegistry);
    EXPECT_EQ(execution_action(Mode::On, unproved), ExecutionAction::Fallback);

    const std::array overlapping_models{model, model};
    const auto ambiguous = resolve_with_compact_table_for_testing(
        Mode::On,
        request,
        Eligibility{.call = request.call},
        compact_metadata(),
        nonmatching_exact_entries,
        compatible_digests(),
        overlapping_models);
    EXPECT_EQ(ambiguous.reason, ResolutionReason::EmptyRegistry);
    EXPECT_EQ(execution_action(Mode::On, ambiguous), ExecutionAction::Fallback);

    model.support.domain = compact::Domain::DenseLinear;
    const auto abstained = resolve_with_compact_table_for_testing(
        Mode::On,
        request,
        Eligibility{.call = request.call},
        compact_metadata(),
        nonmatching_exact_entries,
        compatible_digests(),
        std::span<const compact::ProgramConfigGbdtModel>{&model, 1});
    EXPECT_EQ(abstained.reason, ResolutionReason::EmptyRegistry);
    EXPECT_EQ(execution_action(Mode::On, abstained), ExecutionAction::Fallback);

    model.support.domain = key->domain;
    model.bundle_binding_sha256 = repeated_digest(0x99);
    const auto mixed_bundle = resolve_with_compact_table_for_testing(
        Mode::On,
        request,
        Eligibility{.call = request.call},
        compact_metadata(),
        nonmatching_exact_entries,
        compatible_digests(),
        std::span<const compact::ProgramConfigGbdtModel>{&model, 1});
    EXPECT_EQ(mixed_bundle.reason, ResolutionReason::EmptyRegistry);
}

TEST(MatmulConfigRegistry, LegacyExplicitCkcExactEvidenceCannotActivateProgramConfigOnlyRuntime) {
    const auto request = exact_request();
    const std::array entries{compact_entry()};
    const auto authorized = resolve_with_compact_table_for_testing(
        Mode::On, request, Eligibility{.call = request.call}, compact_metadata(), entries, compatible_digests(), {});
    ASSERT_EQ(authorized.reason, ResolutionReason::CertifiedMatch);
    ASSERT_TRUE(authorized.predicted_program_config.has_value());
    ttnn::prim::MatmulParams caller_parameters;
    caller_parameters.compute_kernel_config = DeviceComputeKernelConfig{};
    const auto exact_parameters = materialize_parameters_for_execution(Mode::On, authorized, caller_parameters);
    ASSERT_TRUE(exact_parameters.has_value());
    EXPECT_TRUE(exact_parameters->program_config.has_value());
    ASSERT_TRUE(exact_parameters->compute_kernel_config.has_value());
    const auto& exact_ckc = exact_parameters->compute_kernel_config.value();
    const auto& caller_ckc = caller_parameters.compute_kernel_config.value();
    EXPECT_EQ(
        std::tie(
            exact_ckc.math_fidelity,
            exact_ckc.math_approx_mode,
            exact_ckc.fp32_dest_acc_en,
            exact_ckc.packer_l1_acc,
            exact_ckc.dst_full_sync_en,
            exact_ckc.throttle_level),
        std::tie(
            caller_ckc.math_fidelity,
            caller_ckc.math_approx_mode,
            caller_ckc.fp32_dest_acc_en,
            caller_ckc.packer_l1_acc,
            caller_ckc.dst_full_sync_en,
            caller_ckc.throttle_level));

    auto legacy_metadata = compact_metadata();
    legacy_metadata.program_config_only_evidence_schema_version = 0;

    const auto skipped = resolve_with_compact_table_for_testing(
        Mode::On, request, Eligibility{.call = request.call}, legacy_metadata, entries, compatible_digests(), {});
    EXPECT_EQ(skipped.reason, ResolutionReason::EmptyRegistry);
    EXPECT_EQ(execution_action(Mode::On, skipped), ExecutionAction::Fallback);
    EXPECT_EQ(skipped.descriptor, nullptr);

    auto nonmatching_pc_key = entries[0].key;
    ++nonmatching_pc_key.logical_m;
    const std::array pc_only_entries{compact::ProgramConfigExactEntry{
        .entry_id = repeated_digest(0x7a),
        .key = nonmatching_pc_key,
        .program_config = compact::exact_program_config(entries[0].replay),
    }};
    const auto mixed_lock_miss = resolve_with_compact_table_for_testing(
        Mode::On,
        request,
        Eligibility{.call = request.call},
        compact_metadata(),
        entries,
        compatible_digests(),
        {},
        pc_only_entries);
    EXPECT_EQ(mixed_lock_miss.reason, ResolutionReason::EmptyRegistry);
    EXPECT_EQ(mixed_lock_miss.descriptor, nullptr);
    EXPECT_FALSE(mixed_lock_miss.predicted_program_config.has_value());

    const std::array candidates{compact::ProgramConfigCandidate{
        .program_config = compact::exact_program_config(entries[0].replay),
        .candidate_id = repeated_digest(1),
    }};
    const std::array nodes{
        compact::GbdtNode{.feature = compact::ProgramConfigFeature::Count, .leaf_value = 0},
    };
    const std::array trees{compact::GbdtTree{.node_offset = 0, .node_count = 1}};
    const std::array training_shapes{compact::TrainingShapeLandmark{
        .logical_m = entries[0].key.logical_m + 1,
        .logical_k = entries[0].key.logical_k,
        .logical_n = entries[0].key.logical_n}};
    const compact::ProgramConfigGbdtModel model{
        .schema_version = 1,
        .enabled = true,
        .score_orientation = compact::GbdtScoreOrientation::LowerIsBetterNegatedPairwiseMargin,
        .feature_schema_sha256 = repeated_digest(2),
        .model_sha256 = repeated_digest(3),
        .training_table_sha256 = repeated_digest(4),
        .safety_evidence_sha256 = repeated_digest(5),
        .candidate_policy_sha256 = repeated_digest(6),
        .lineage_sha256 = repeated_digest(8),
        .evaluation_model_payload_sha256 = repeated_digest(11),
        .quality_evaluation_sha256 = repeated_digest(9),
        .unseen_abstention_policy_sha256 = repeated_digest(10),
        .support_sha256 = repeated_digest(7),
        .bundle_binding_sha256 = legacy_metadata.online_model_bundle_binding_sha256,
        .support = compact_model_support(entries[0].key),
        .base_score = 0,
        .score_scale = 1,
        .minimum_score_margin = 1,
        .maximum_normalized_shape_distance_ppm = 250'000,
        .training_shapes = training_shapes,
        .candidates = candidates,
        .trees = trees,
        .nodes = nodes,
    };
    const auto abstained = resolve_with_compact_table_for_testing(
        Mode::On,
        request,
        Eligibility{.call = request.call},
        legacy_metadata,
        entries,
        compatible_digests(),
        std::span<const compact::ProgramConfigGbdtModel>{&model, 1});
    EXPECT_EQ(abstained.reason, ResolutionReason::EmptyRegistry);
    EXPECT_FALSE(abstained.predicted_program_config.has_value());
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
        .program_config_only_evidence_schema_version = 1,
        .online_program_config_model_evidence_schema_version = 1,
        .content_sha256 = repeated_digest(0x10),
        .semantic_source_sha256 = repeated_digest(0x11),
        .build_identity_sha256 = repeated_digest(0x22),
        .runtime_capability_sha256 = repeated_digest(0x33),
        .online_model_bundle_binding_sha256 = repeated_digest(0x44)};
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

TEST(MatmulConfigRegistry, DirectBankExactIgnoresUnbankablePhysicalSessionFields) {
    auto request = exact_request();
    request.device.attestation_status = DeviceAttestationStatus::QueryFailed;
    request.device.board_capability_class += 17;
    request.device.topology_sha256 = repeated_digest(0xa5);
    request.device.runtime_capability_sha256 = repeated_digest(0xb6);

    const auto legacy = compact_entry();
    auto bank_key = legacy.key;
    bank_key.board_capability_class = 0;
    bank_key.topology_sha256 = {};
    const std::array bank_entries{compact::ProgramConfigExactEntry{
        .entry_id = repeated_digest(0x7a),
        .key = bank_key,
        .program_config = compact::exact_program_config(legacy.replay),
    }};
    auto metadata = compact_metadata();
    metadata.program_config_only_evidence_schema_version = 2;
    metadata.online_program_config_model_evidence_schema_version = 0;
    metadata.runtime_capability_sha256 = {};
    auto actual = compatible_digests();
    actual.runtime_capability_sha256 = repeated_digest(0xc7);

    const auto result = resolve_with_compact_table_for_testing(
        Mode::On, request, Eligibility{.call = request.call}, metadata, {}, actual, {}, bank_entries);
    EXPECT_EQ(result.reason, ResolutionReason::CertifiedMatch);
    EXPECT_TRUE(result.predicted_program_config.has_value());

    request.device.architecture += 1;
    const auto wrong_architecture = resolve_with_compact_table_for_testing(
        Mode::On, request, Eligibility{.call = request.call}, metadata, {}, actual, {}, bank_entries);
    EXPECT_EQ(wrong_architecture.reason, ResolutionReason::EmptyRegistry);
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
    EXPECT_EQ(shadow.descriptor, entries.data());
    EXPECT_EQ(execution_action(Mode::Shadow, shadow), ExecutionAction::ObserveOnly);

    const auto on = resolve_with_compact_table_for_testing(
        Mode::On, request, eligibility, compact_metadata(), entries, compatible_digests());
    ASSERT_EQ(on.reason, ResolutionReason::CertifiedMatch);
    ASSERT_EQ(on.descriptor, entries.data());
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
    EXPECT_EQ(program->per_core_N, 16);
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
    invalid_addmm.key.alpha_f32_bits = 0x40000000;
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
             +[](compact::EntryDescriptor& item) { item.key.input_a.tile_width = 16; },
             +[](compact::EntryDescriptor& item) { item.key.logical_m = 0; },
             +[](compact::EntryDescriptor& item) { item.key.padded_m = item.key.logical_m - 1; },
             +[](compact::EntryDescriptor& item) { item.key.padded_m = 129; },
             +[](compact::EntryDescriptor& item) { item.key.input_b.tile_height = 16; },
             +[](compact::EntryDescriptor& item) { item.key.input_b.tile_width = 16; },
             +[](compact::EntryDescriptor& item) { item.key.output.tile_height = 16; },
             +[](compact::EntryDescriptor& item) { item.key.output.tile_width = 16; },
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
    descriptor.replay.compute_kernel_config.fp32_dest_acc_en = true;
    descriptor.replay.program_config.out_subblock_h = 2;
    descriptor.replay.program_config.out_subblock_w = 4;
    expect_rejection(descriptor, MaterializationStatus::InvalidProgramConfig);

    descriptor = compact_entry();
    descriptor.replay.call_state.output_tile_is_null = false;
    expect_rejection(descriptor, MaterializationStatus::InvalidCallState);

    descriptor = compact_entry();
    descriptor.key.input_a.layout = compact::Layout::RowMajor;
    expect_rejection(descriptor, MaterializationStatus::InvalidCallState);

    descriptor = compact_entry();
    descriptor.key.input_b.layout = compact::Layout::RowMajor;
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
                                 result.descriptor == entries.data();
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

TEST(MatmulConfigRegistry, StatelessConstraintQueryRequiresStateOnlyForExactOnHit) {
    EXPECT_TRUE(registry_constraint_state_required(Mode::On, ExecutionAction::ApplyRecipe, true));

    EXPECT_FALSE(registry_constraint_state_required(Mode::Off, ExecutionAction::ApplyRecipe, true));
    EXPECT_FALSE(registry_constraint_state_required(Mode::Shadow, ExecutionAction::ObserveOnly, true));
    EXPECT_FALSE(registry_constraint_state_required(Mode::On, ExecutionAction::Fallback, true));
    EXPECT_FALSE(registry_constraint_state_required(Mode::On, ExecutionAction::ApplyRecipe, false));

    const auto request = exact_request();
    resolver_invocations = 0;
    const auto decision = resolve_for_dispatch_decision(
        Mode::On, request, Eligibility{.call = request.call}, &counting_certified_resolver);
    EXPECT_EQ(resolver_invocations, 1);
    EXPECT_EQ(decision.resolution.reason, ResolutionReason::CertifiedMatch);
    EXPECT_EQ(decision.action, ExecutionAction::ApplyRecipe);
    EXPECT_FALSE(decision.materialized_parameters.has_value());
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

TEST(MatmulConfigRegistry, NondefaultTileFaceMetadataNeverReachesLookup) {
    const Tile transposed_tile({32, 32}, true);
    ASSERT_TRUE(has_nondefault_v1_tile_transpose(transposed_tile));

    const ttnn::prim::MatmulParams parameters;
    const auto eligibility = v1_eligibility_from_call_state(
        dense_matmul_call_semantics(),
        IoContractStatus::Resolved,
        false,
        false,
        parameters,
        false,
        false,
        false,
        false,
        has_nondefault_v1_tile_transpose(transposed_tile));
    resolver_invocations = 0;
    const auto dispatch =
        resolve_for_dispatch(Mode::On, exact_request(), eligibility, parameters, &counting_certified_resolver);
    EXPECT_EQ(resolver_invocations, 0);
    EXPECT_EQ(dispatch.resolution.reason, ResolutionReason::UnsupportedSemantics);
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
                     Eligibility{.call = dense_matmul_call_semantics(), .has_unsupported_tile_metadata = true},
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
    EXPECT_FALSE(on_parameters->compute_kernel_config.has_value());
    EXPECT_EQ(on_parameters->untilize_out, legacy.untilize_out);
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

        const auto other_domain = next_public_domain(domain);
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
    addmm_request.call = addmm_call_semantics(1.0F, -0.0F);
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

    second.replay.program_config.out_subblock_w = 1;
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
    record_distributed_observation(Mode::Off, DistributedMatmulClass::DpRankDistinctV1);
    record_distributed_observation(Mode::Shadow, DistributedMatmulClass::DpRankDistinctV1);
    record_distributed_observation(Mode::On, DistributedMatmulClass::TpnSpMTpNV1);
    record_distributed_observation(Mode::Shadow, DistributedMatmulClass::Unknown);
    record_distributed_observation(Mode::Shadow, DistributedMatmulClass::NotDistributed);
    record_distributed_observation(Mode::On, DistributedMatmulClass::Count);

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
    EXPECT_EQ(snapshot.distributed_observations[static_cast<std::size_t>(DistributedMatmulClass::DpRankDistinctV1)], 1);
    EXPECT_EQ(snapshot.distributed_observations[static_cast<std::size_t>(DistributedMatmulClass::TpnSpMTpNV1)], 1);
    EXPECT_EQ(snapshot.distributed_observations[static_cast<std::size_t>(DistributedMatmulClass::Unknown)], 1);
    EXPECT_EQ(snapshot.distributed_observations[static_cast<std::size_t>(DistributedMatmulClass::NotDistributed)], 1);

    reset_stats_for_testing();
    snapshot = stats_snapshot();
    EXPECT_EQ(snapshot.domains[static_cast<std::size_t>(OperationDomain::DenseMatmul)].resolution_attempts, 0);
    EXPECT_TRUE(std::ranges::all_of(snapshot.distributed_observations, [](const auto count) { return count == 0; }));
}

TEST(MatmulConfigRegistry, TypedExactHitRecordsCertifiedTelemetry) {
    reset_stats_for_testing();
    const auto request = exact_request();
    const auto descriptor = compact_entry();
    const std::array exact_entries{compact::ProgramConfigExactEntry{
        .entry_id = descriptor.entry_id,
        .key = descriptor.key,
        .program_config = compact::exact_program_config(descriptor.replay),
    }};
    const auto selected = resolve_with_compact_table_for_testing(
        Mode::On,
        request,
        Eligibility{.call = request.call},
        compact_metadata(),
        {},
        compatible_digests(),
        {},
        exact_entries);
    ASSERT_EQ(selected.reason, ResolutionReason::CertifiedMatch);
    ASSERT_TRUE(selected.predicted_program_config.has_value());
    ASSERT_TRUE(selected.predicted_key.has_value());
    record_resolution(Mode::On, request.call.domain, selected, ExecutionAction::ApplyRecipe);

    const auto snapshot = stats_snapshot();
    const auto& dense = snapshot.domains[static_cast<std::size_t>(OperationDomain::DenseMatmul)];
    EXPECT_EQ(dense.resolution_attempts, 1);
    EXPECT_EQ(dense.certified_hits, 1);
    EXPECT_EQ(dense.selected_hits, 1);
    reset_stats_for_testing();
}

TEST(MatmulConfigRegistry, SelectedPublicExecutionErrorCircuitBreaksWithoutCompletionOrRetry) {
    reset_stats_for_testing();
    reset_circuit_breakers_for_testing();
    const auto request = exact_request();
    const auto eligibility = Eligibility{.call = request.call};
    const auto recipe = basic_recipe();
    const auto selected = resolve_with_synthetic_candidate_for_testing(Mode::On, request, eligibility, request, recipe);
    ASSERT_EQ(selected.reason, ResolutionReason::CertifiedMatch);
    ASSERT_EQ(execution_action(Mode::On, selected), ExecutionAction::ApplyRecipe);
    ttnn::prim::MatmulParams legacy;
    const auto materialized = materialize_parameters_for_execution(Mode::On, selected, legacy);
    ASSERT_TRUE(materialized.has_value());
    record_resolution(Mode::On, request.call.domain, selected, ExecutionAction::ApplyRecipe);

    bool applied = materialized.has_value();
    std::size_t execution_attempts = 0;
    const auto public_wrapper = [&]() -> int {
        const SelectedExecutionGuard guard(OperationDomain::DenseMatmul, &applied);
        return execute_selected_call_once(guard, [&]() -> int {
            ++execution_attempts;
            throw std::runtime_error("injected public execution failure");
        });
    };
    EXPECT_THROW(public_wrapper(), std::runtime_error);
    EXPECT_EQ(execution_attempts, 1);

    const auto snapshot = stats_snapshot();
    const auto& dense = snapshot.domains[static_cast<std::size_t>(OperationDomain::DenseMatmul)];
    EXPECT_EQ(dense.resolution_attempts, 1);
    EXPECT_EQ(dense.selected_hits, 1);
    EXPECT_EQ(dense.completed_hits, 0);
    EXPECT_EQ(dense.circuit_breaker_activations, 1);
    EXPECT_TRUE(dense.circuit_broken);
    EXPECT_EQ(
        resolve_with(Mode::On, Eligibility{.call = dense_matmul_call_semantics()}).reason,
        ResolutionReason::CircuitBroken);

    reset_circuit_breakers_for_testing();
    reset_stats_for_testing();
}

TEST(MatmulConfigRegistry, SelectedPublicExecutionSucceedsExactlyOnceAndRecordsCompletion) {
    reset_stats_for_testing();
    reset_circuit_breakers_for_testing();
    const auto request = exact_request();
    const auto eligibility = Eligibility{.call = request.call};
    const auto recipe = basic_recipe();
    const auto selected = resolve_with_synthetic_candidate_for_testing(Mode::On, request, eligibility, request, recipe);
    ASSERT_EQ(selected.reason, ResolutionReason::CertifiedMatch);
    ASSERT_EQ(execution_action(Mode::On, selected), ExecutionAction::ApplyRecipe);
    ttnn::prim::MatmulParams legacy;
    const auto materialized = materialize_parameters_for_execution(Mode::On, selected, legacy);
    ASSERT_TRUE(materialized.has_value());
    record_resolution(Mode::On, request.call.domain, selected, ExecutionAction::ApplyRecipe);

    bool applied = materialized.has_value();
    std::size_t execution_attempts = 0;
    const auto public_wrapper = [&] {
        const SelectedExecutionGuard guard(OperationDomain::DenseMatmul, &applied);
        return execute_selected_call_once(guard, [&] {
            ++execution_attempts;
            return 17;
        });
    };
    EXPECT_EQ(public_wrapper(), 17);
    EXPECT_EQ(execution_attempts, 1);

    const auto snapshot = stats_snapshot();
    const auto& dense = snapshot.domains[static_cast<std::size_t>(OperationDomain::DenseMatmul)];
    EXPECT_EQ(dense.resolution_attempts, 1);
    EXPECT_EQ(dense.selected_hits, 1);
    EXPECT_EQ(dense.completed_hits, 1);
    EXPECT_EQ(dense.circuit_breaker_activations, 0);
    EXPECT_FALSE(dense.circuit_broken);

    reset_circuit_breakers_for_testing();
    reset_stats_for_testing();
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
        resolve_with(Mode::On, Eligibility{.call = addmm_call_semantics(2.0F, 0.0F)}).reason,
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
