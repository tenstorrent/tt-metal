// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Cache-key regression tests for #59914.
//
// Each op below compiles a value into its program that its compute_program_hash did not cover, so two calls
// that differ only in that value shared one cached program. Every *SeparatesKeys test builds such a pair and
// expects distinct keys; every *SharesKey test checks the key still ignores per-call state like buffer
// addresses, so the fix does not cost reuse.
//
// These stop at the hash on purpose. Most of these ops need Blackhole or a multi-device ring to dispatch, but
// none of their hashes touches the hardware, so the keys can be checked on any single device.

#include <gtest/gtest.h>

#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/global_circular_buffer.hpp>
#include <tt-metalium/sub_device_types.hpp>

#include "ttnn/operations/experimental/ccl/llama_reduce_scatter_create_heads/device/llama_reduce_scatter_create_heads_device_op.hpp"
#include "ttnn/operations/experimental/ccl/strided_reduce_scatter_async/device/strided_reduce_scatter_async_op_device_operation.hpp"
#include "ttnn/operations/experimental/deepseek_prefill/pack_scaled_fp8_kv_cache/device/pack_scaled_fp8_kv_cache_device_operation.hpp"
#include "ttnn/operations/experimental/deepseek_prefill/per_token_cast_to_fp8/device/per_token_cast_to_fp8_device_operation.hpp"
#include "ttnn/operations/experimental/test/prefetcher_consumer/dram_prefetcher_consumer.hpp"
#include "ttnn/operations/pool/generic/device/pool_op.hpp"
#include "ttnn/operations/transformer/sdpa/device/exp_ring_joint_sdpa_device_operation.hpp"
#include "ttnn/operations/transformer/sdpa/device/ring_joint_sdpa_device_operation.hpp"
#include "ttnn/tensor/layout/tensor_layout.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/tensor_ops.hpp"
#include "ttnn/tensor/tensor_spec.hpp"
#include "ttnn/types.hpp"
#include "ttnn_test_fixtures.hpp"

namespace ttnn::test::stale_program_hash {
namespace {

using ::tt::tt_metal::Alignment;
using ::tt::tt_metal::BufferType;
using ::tt::tt_metal::CoreCoord;
using ::tt::tt_metal::CoreRange;
using ::tt::tt_metal::CoreRangeSet;
using ::tt::tt_metal::DataType;
using ::tt::tt_metal::Layout;
using ::tt::tt_metal::MemoryConfig;
using ::tt::tt_metal::PageConfig;
using ::tt::tt_metal::ShardSpec;
using ::tt::tt_metal::TensorLayout;
using ::tt::tt_metal::TensorMemoryLayout;
using ::tt::tt_metal::TensorSpec;
using MeshDevice = ::tt::tt_metal::distributed::MeshDevice;

Tensor make_tensor(
    MeshDevice* device,
    const ttnn::Shape& shape,
    DataType dtype,
    Layout layout,
    const MemoryConfig& memory_config = ttnn::DRAM_MEMORY_CONFIG,
    const Alignment& alignment = {}) {
    return ttnn::create_device_tensor(
        TensorSpec(shape, TensorLayout(dtype, PageConfig(layout), memory_config, alignment)), device);
}

Tensor make_tile(
    MeshDevice* device, const ttnn::Shape& shape, const MemoryConfig& memory_config = ttnn::DRAM_MEMORY_CONFIG) {
    return make_tensor(device, shape, DataType::BFLOAT16, Layout::TILE, memory_config);
}

// A row-major tensor's accessor bakes in its aligned page size, which follows the padded width. Over-aligning
// the width keeps the logical shape, dtype, layout and memory config but changes that page size, as untilize's
// fromPaddedShape output does.
std::pair<Tensor, Tensor> make_row_major_alignment_pair(MeshDevice* device, const ttnn::Shape& shape, DataType dtype) {
    auto plain = make_tensor(device, shape, dtype, Layout::ROW_MAJOR);
    auto over_aligned =
        make_tensor(device, shape, dtype, Layout::ROW_MAJOR, ttnn::DRAM_MEMORY_CONFIG, Alignment({2 * shape[-1]}));
    EXPECT_EQ(plain.logical_shape(), over_aligned.logical_shape());
    EXPECT_EQ(plain.memory_config(), over_aligned.memory_config());
    EXPECT_NE(plain.buffer()->aligned_page_size(), over_aligned.buffer()->aligned_page_size());
    return {std::move(plain), std::move(over_aligned)};
}

MemoryConfig height_sharded_l1(const CoreRangeSet& grid, const std::array<uint32_t, 2>& shard_shape) {
    return MemoryConfig(TensorMemoryLayout::HEIGHT_SHARDED, BufferType::L1, ShardSpec(grid, shard_shape));
}

class StaleProgramHashFixture : public TTNNFixtureWithSuiteDevice<StaleProgramHashFixture> {};

// ---------------------------------------------------------------------------------------------------------------
// 1. per_token_cast_to_fp8: row-major page size
// ---------------------------------------------------------------------------------------------------------------

namespace per_token_cast = ::ttnn::experimental::prim::per_token_cast_to_fp8;

ttsl::hash::hash_t per_token_cast_hash(const Tensor& input) {
    const per_token_cast::PerTokenCastToFp8Params attrs{
        .output_memory_config = ttnn::DRAM_MEMORY_CONFIG, .round_scale_to_power_of_two = false};
    return per_token_cast::PerTokenCastToFp8DeviceOperation::compute_program_hash(attrs, {input});
}

const ttnn::Shape kPerTokenCastShape{1, 1, 32, 128};

TEST_F(StaleProgramHashFixture, PerTokenCastToFp8SameSpecSharesKey) {
    const auto first = make_tensor(device_, kPerTokenCastShape, DataType::BFLOAT16, Layout::ROW_MAJOR);
    const auto second = make_tensor(device_, kPerTokenCastShape, DataType::BFLOAT16, Layout::ROW_MAJOR);
    EXPECT_EQ(per_token_cast_hash(first), per_token_cast_hash(second));
}

TEST_F(StaleProgramHashFixture, PerTokenCastToFp8RowMajorAlignmentSeparatesKeys) {
    const auto [plain, over_aligned] = make_row_major_alignment_pair(device_, kPerTokenCastShape, DataType::BFLOAT16);
    EXPECT_NE(per_token_cast_hash(plain), per_token_cast_hash(over_aligned));
}

// ---------------------------------------------------------------------------------------------------------------
// 2. pack_scaled_fp8_kv_cache: row-major page size of latent, scales and rope
// ---------------------------------------------------------------------------------------------------------------

namespace pack_kv = ::ttnn::experimental::prim::pack_scaled_fp8_kv_cache;

ttsl::hash::hash_t pack_kv_hash(const Tensor& latent, const Tensor& scales, const Tensor& rope) {
    const pack_kv::PackScaledFp8KvCacheParams attrs{.output_memory_config = ttnn::DRAM_MEMORY_CONFIG};
    return pack_kv::PackScaledFp8KvCacheDeviceOperation::compute_program_hash(attrs, {latent, scales, rope});
}

const ttnn::Shape kLatentShape{1, 1, 32, 512};
const ttnn::Shape kScalesShape{1, 1, 32, 32};
const ttnn::Shape kRopeShape{1, 1, 32, 64};

struct PackKvInputs {
    Tensor latent;
    Tensor scales;
    Tensor rope;
};

PackKvInputs make_pack_kv_inputs(MeshDevice* device) {
    return {
        make_tensor(device, kLatentShape, DataType::BFLOAT16, Layout::ROW_MAJOR),
        make_tensor(device, kScalesShape, DataType::FLOAT32, Layout::ROW_MAJOR),
        make_tensor(device, kRopeShape, DataType::BFLOAT16, Layout::ROW_MAJOR)};
}

TEST_F(StaleProgramHashFixture, PackScaledFp8KvCacheSameSpecSharesKey) {
    const auto first = make_pack_kv_inputs(device_);
    const auto second = make_pack_kv_inputs(device_);
    EXPECT_EQ(
        pack_kv_hash(first.latent, first.scales, first.rope), pack_kv_hash(second.latent, second.scales, second.rope));
}

TEST_F(StaleProgramHashFixture, PackScaledFp8KvCacheLatentAlignmentSeparatesKeys) {
    const auto base = make_pack_kv_inputs(device_);
    const auto [plain, over_aligned] = make_row_major_alignment_pair(device_, kLatentShape, DataType::BFLOAT16);
    EXPECT_NE(pack_kv_hash(plain, base.scales, base.rope), pack_kv_hash(over_aligned, base.scales, base.rope));
}

TEST_F(StaleProgramHashFixture, PackScaledFp8KvCacheScalesAlignmentSeparatesKeys) {
    const auto base = make_pack_kv_inputs(device_);
    const auto [plain, over_aligned] = make_row_major_alignment_pair(device_, kScalesShape, DataType::FLOAT32);
    EXPECT_NE(pack_kv_hash(base.latent, plain, base.rope), pack_kv_hash(base.latent, over_aligned, base.rope));
}

TEST_F(StaleProgramHashFixture, PackScaledFp8KvCacheRopeAlignmentSeparatesKeys) {
    const auto base = make_pack_kv_inputs(device_);
    const auto [plain, over_aligned] = make_row_major_alignment_pair(device_, kRopeShape, DataType::BFLOAT16);
    EXPECT_NE(pack_kv_hash(base.latent, base.scales, plain), pack_kv_hash(base.latent, base.scales, over_aligned));
}

// ---------------------------------------------------------------------------------------------------------------
// 3. strided_reduce_scatter_async: caller-supplied intermediate and output buffers
// ---------------------------------------------------------------------------------------------------------------

namespace strided_rs = ::ttnn::operations::experimental::ccl::strided_reduce_scatter_async::detail;

strided_rs::operation_attributes_t make_strided_rs_attributes() {
    return strided_rs::operation_attributes_t{
        .dim = 3,
        .num_links = 1,
        .ring_size = 2,
        .output_mem_config = ttnn::DRAM_MEMORY_CONFIG,
        .optional_intermediate_mem_config = std::nullopt,
        .topology = ttnn::ccl::Topology::Ring,
        .semaphore = {},
        .barrier_semaphore = std::nullopt,
        .using_persistent_buffers = true,
        .sub_device_id = std::nullopt,
        .cluster_axis = std::nullopt,
        .num_workers_per_link = std::nullopt,
        .num_buffers_per_channel = std::nullopt,
        .mm_cores_y = std::nullopt,
        .mm_block_ht = 1,
        .mm_block_wt = 1,
        .mm_N_full_block_wt = std::nullopt,
        .chunk_width_in_mm_blocks = std::nullopt,
    };
}

ttsl::hash::hash_t strided_rs_hash(
    const Tensor& input, const std::optional<Tensor>& intermediate, const std::optional<Tensor>& output) {
    return strided_rs::StridedReduceScatterAsyncDeviceOperation::compute_program_hash(
        make_strided_rs_attributes(), {input, intermediate, output});
}

const ttnn::Shape kStridedRsInputShape{1, 1, 64, 256};
const ttnn::Shape kStridedRsOutputShape{1, 1, 64, 128};

TEST_F(StaleProgramHashFixture, StridedReduceScatterAsyncSameSpecSharesKey) {
    const auto input = make_tile(device_, kStridedRsInputShape);
    EXPECT_EQ(
        strided_rs_hash(input, make_tile(device_, kStridedRsInputShape), make_tile(device_, kStridedRsOutputShape)),
        strided_rs_hash(input, make_tile(device_, kStridedRsInputShape), make_tile(device_, kStridedRsOutputShape)));
}

TEST_F(StaleProgramHashFixture, StridedReduceScatterAsyncIntermediateMemoryConfigSeparatesKeys) {
    const auto input = make_tile(device_, kStridedRsInputShape);
    const auto output = make_tile(device_, kStridedRsOutputShape);
    EXPECT_NE(
        strided_rs_hash(input, make_tile(device_, kStridedRsInputShape, ttnn::DRAM_MEMORY_CONFIG), output),
        strided_rs_hash(input, make_tile(device_, kStridedRsInputShape, ttnn::L1_MEMORY_CONFIG), output));
}

TEST_F(StaleProgramHashFixture, StridedReduceScatterAsyncOutputMemoryConfigSeparatesKeys) {
    const auto input = make_tile(device_, kStridedRsInputShape);
    const auto intermediate = make_tile(device_, kStridedRsInputShape);
    EXPECT_NE(
        strided_rs_hash(input, intermediate, make_tile(device_, kStridedRsOutputShape, ttnn::DRAM_MEMORY_CONFIG)),
        strided_rs_hash(input, intermediate, make_tile(device_, kStridedRsOutputShape, ttnn::L1_MEMORY_CONFIG)));
}

// ---------------------------------------------------------------------------------------------------------------
// 4. llama_reduce_scatter_create_heads: input shape, packet buffer grid, qkv_memory_config, subdevice_id
// ---------------------------------------------------------------------------------------------------------------

using LlamaRsCreateHeads = ::ttnn::operations::experimental::ccl::LlamaReduceScatterCreateHeadsDeviceOperation;

const CoreRangeSet kGridA(CoreRange({0, 0}, {1, 0}));
const CoreRangeSet kGridB(CoreRange({0, 1}, {1, 1}));
const ttnn::Shape kLlamaInputShape{1, 1, 32, 1024};
const ttnn::Shape kLlamaPacketShape{1, 1, 64, 32};

LlamaRsCreateHeads::operation_attributes_t make_llama_attributes(
    const std::optional<MemoryConfig>& qkv_memory_config, ::tt::tt_metal::SubDeviceId subdevice_id) {
    return LlamaRsCreateHeads::operation_attributes_t{
        .dim = 3,
        .cross_device_semaphore = std::nullopt,
        .subdevice_id = subdevice_id,
        .cluster_axis = 1,
        .output_mem_config = std::nullopt,
        .ring_devices = 4,
        .topology = ttnn::ccl::Topology::Linear,
        .num_links = 1,
        .num_heads = 8,
        .num_kv_heads = 1,
        .head_dim = 128,
        .slice_size = 32,
        .qkv_memory_config = qkv_memory_config,
        .use_noc1_only = false,
        .use_optimal_ccl_for_llama = false,
    };
}

const MemoryConfig kQkvOnGridA = height_sharded_l1(kGridA, {32, 128});
const MemoryConfig kQkvOnGridB = height_sharded_l1(kGridB, {32, 128});
const ::tt::tt_metal::SubDeviceId kSubDevice0{0};
const ::tt::tt_metal::SubDeviceId kSubDevice1{1};

ttsl::hash::hash_t llama_hash(
    const LlamaRsCreateHeads::operation_attributes_t& attributes, const Tensor& input, const Tensor& packet_buffer) {
    return LlamaRsCreateHeads::compute_program_hash(attributes, {input, packet_buffer});
}

Tensor make_packet_buffer(MeshDevice* device, const CoreRangeSet& grid) {
    return make_tile(device, kLlamaPacketShape, height_sharded_l1(grid, {32, 32}));
}

TEST_F(StaleProgramHashFixture, LlamaReduceScatterCreateHeadsSameSpecSharesKey) {
    const auto attributes = make_llama_attributes(kQkvOnGridA, kSubDevice0);
    EXPECT_EQ(
        llama_hash(attributes, make_tile(device_, kLlamaInputShape), make_packet_buffer(device_, kGridA)),
        llama_hash(attributes, make_tile(device_, kLlamaInputShape), make_packet_buffer(device_, kGridA)));
}

TEST_F(StaleProgramHashFixture, LlamaReduceScatterCreateHeadsInputShapeSeparatesKeys) {
    const auto attributes = make_llama_attributes(kQkvOnGridA, kSubDevice0);
    const auto packet_buffer = make_packet_buffer(device_, kGridA);
    EXPECT_NE(
        llama_hash(attributes, make_tile(device_, kLlamaInputShape), packet_buffer),
        llama_hash(attributes, make_tile(device_, ttnn::Shape{1, 1, 32, 2048}), packet_buffer));
}

TEST_F(StaleProgramHashFixture, LlamaReduceScatterCreateHeadsPacketBufferGridSeparatesKeys) {
    const auto attributes = make_llama_attributes(kQkvOnGridA, kSubDevice0);
    const auto input = make_tile(device_, kLlamaInputShape);
    EXPECT_NE(
        llama_hash(attributes, input, make_packet_buffer(device_, kGridA)),
        llama_hash(attributes, input, make_packet_buffer(device_, kGridB)));
}

TEST_F(StaleProgramHashFixture, LlamaReduceScatterCreateHeadsQkvMemoryConfigSeparatesKeys) {
    const auto input = make_tile(device_, kLlamaInputShape);
    const auto packet_buffer = make_packet_buffer(device_, kGridA);
    EXPECT_NE(
        llama_hash(make_llama_attributes(kQkvOnGridA, kSubDevice0), input, packet_buffer),
        llama_hash(make_llama_attributes(kQkvOnGridB, kSubDevice0), input, packet_buffer));
}

TEST_F(StaleProgramHashFixture, LlamaReduceScatterCreateHeadsSubdeviceIdSeparatesKeys) {
    const auto input = make_tile(device_, kLlamaInputShape);
    const auto packet_buffer = make_packet_buffer(device_, kGridA);
    EXPECT_NE(
        llama_hash(make_llama_attributes(kQkvOnGridA, kSubDevice0), input, packet_buffer),
        llama_hash(make_llama_attributes(kQkvOnGridA, kSubDevice1), input, packet_buffer));
}

// ---------------------------------------------------------------------------------------------------------------
// 5. Pool2D: explicit ceil_pad_hw from the DRAM slicer
// ---------------------------------------------------------------------------------------------------------------

using ::ttnn::operations::pool::Pool2D;
using ::ttnn::operations::sliding_window::SlidingWindowConfig;

Pool2D::operation_attributes_t make_pool_attributes(const std::optional<std::pair<uint32_t, uint32_t>>& ceil_pad_hw) {
    return Pool2D::operation_attributes_t{
        .sliding_window_config_ =
            SlidingWindowConfig{
                .batch_size = 1,
                .channels = 32,
                .input_hw = {32, 32},
                .window_hw = {3, 3},
                .stride_hw = {2, 2},
                .padding = {1, 1, 1, 1},
                .ceil_pad_hw = ceil_pad_hw,
                .num_cores_nhw = 1,
                .num_cores_c = 1,
                .core_range_set = CoreRangeSet(CoreRange({0, 0}, {0, 0})),
                .ceil_mode = true},
        .pool_type_ = ::ttnn::operations::pool::Pool2DType::MAX_POOL2D,
        .output_dtype_ = DataType::BFLOAT16,
        .output_layout_ = Layout::ROW_MAJOR,
        .memory_config_ = ttnn::L1_MEMORY_CONFIG,
    };
}

TEST_F(StaleProgramHashFixture, Pool2DExplicitCeilPadSeparatesKeys) {
    const auto input = make_tensor(device_, ttnn::Shape{1, 1, 32 * 32, 32}, DataType::BFLOAT16, Layout::ROW_MAJOR);
    const auto computed = make_pool_attributes(std::nullopt);
    ASSERT_EQ(computed.sliding_window_config_.get_ceil_pad_hw(), std::make_pair(1u, 1u));
    EXPECT_NE(
        Pool2D::compute_program_hash(computed, {input}),
        Pool2D::compute_program_hash(make_pool_attributes(std::make_pair(0u, 0u)), {input}));
}

TEST_F(StaleProgramHashFixture, Pool2DCeilPadEqualToComputedSharesKey) {
    const auto input = make_tensor(device_, ttnn::Shape{1, 1, 32 * 32, 32}, DataType::BFLOAT16, Layout::ROW_MAJOR);
    EXPECT_EQ(
        Pool2D::compute_program_hash(make_pool_attributes(std::nullopt), {input}),
        Pool2D::compute_program_hash(make_pool_attributes(std::make_pair(1u, 1u)), {input}));
}

// ---------------------------------------------------------------------------------------------------------------
// 6. RingJointSDPA: logical_n / logical_l metadata tensors and the CCL core allocation strategy
// ---------------------------------------------------------------------------------------------------------------

using ::ttnn::prim::RingJointSDPADeviceOperation;
using ::ttnn::prim::RingJointSDPAInputs;
using ::ttnn::prim::RingJointSDPAParams;

constexpr uint32_t kRingSize = 2;
const ttnn::Shape kQShape{1, 1, 64, 64};
const ttnn::Shape kGatheredKShape{1, 1, 64 * kRingSize, 64};

using ::ttnn::ccl::CoreAllocationStrategy;

RingJointSDPAParams make_ring_joint_params(CoreAllocationStrategy strategy = CoreAllocationStrategy::ROW_MAJOR) {
    return RingJointSDPAParams(
        /*joint_strategy=*/"rear",
        /*scale=*/std::nullopt,
        /*is_causal=*/false,
        /*is_balanced=*/false,
        /*is_cross=*/false,
        /*logical_n=*/64 * kRingSize,
        /*logical_l=*/0,
        /*ring_size=*/kRingSize,
        /*output_memory_config=*/ttnn::DRAM_MEMORY_CONFIG,
        /*program_config=*/std::nullopt,
        /*compute_kernel_config=*/DeviceComputeKernelConfig{},
        ::ttnn::experimental::prim::RingAttentionAllGatherAsyncParams(
            /*dim=*/2,
            /*num_links=*/1,
            /*ring_size=*/kRingSize,
            ttnn::DRAM_MEMORY_CONFIG,
            ttnn::ccl::Topology::Ring,
            /*semaphore=*/{},
            /*sub_device_id=*/std::nullopt,
            /*cluster_axis=*/std::nullopt,
            strategy),
        ::ttnn::experimental::prim::RingAttentionAllGatherAsyncInputs{},
        /*ccl_core_grid_offset=*/CoreCoord(0, 0));
}

Tensor make_scalar_u32(MeshDevice* device, const MemoryConfig& memory_config) {
    return make_tensor(device, ttnn::Shape{1}, DataType::UINT32, Layout::ROW_MAJOR, memory_config);
}

ttsl::hash::hash_t ring_joint_hash(
    MeshDevice* device,
    const std::optional<Tensor>& logical_n_tensor,
    const std::optional<Tensor>& logical_l_tensor,
    CoreAllocationStrategy strategy = CoreAllocationStrategy::ROW_MAJOR) {
    RingJointSDPAInputs inputs{
        .input_q = make_tile(device, kQShape),
        .input_k = make_tile(device, kQShape),
        .input_v = make_tile(device, kQShape),
        .gathered_k = make_tile(device, kGatheredKShape),
        .gathered_v = make_tile(device, kGatheredKShape),
        .logical_n_tensor = logical_n_tensor,
        .logical_l_tensor = logical_l_tensor,
    };
    return RingJointSDPADeviceOperation::compute_program_hash(make_ring_joint_params(strategy), inputs);
}

TEST_F(StaleProgramHashFixture, RingJointSdpaSameSpecSharesKey) {
    EXPECT_EQ(
        ring_joint_hash(device_, make_scalar_u32(device_, ttnn::DRAM_MEMORY_CONFIG), std::nullopt),
        ring_joint_hash(device_, make_scalar_u32(device_, ttnn::DRAM_MEMORY_CONFIG), std::nullopt));
}

TEST_F(StaleProgramHashFixture, RingJointSdpaLogicalNTensorMemoryConfigSeparatesKeys) {
    EXPECT_NE(
        ring_joint_hash(device_, make_scalar_u32(device_, ttnn::DRAM_MEMORY_CONFIG), std::nullopt),
        ring_joint_hash(device_, make_scalar_u32(device_, ttnn::L1_MEMORY_CONFIG), std::nullopt));
}

TEST_F(StaleProgramHashFixture, RingJointSdpaLogicalLTensorMemoryConfigSeparatesKeys) {
    EXPECT_NE(
        ring_joint_hash(device_, std::nullopt, make_scalar_u32(device_, ttnn::DRAM_MEMORY_CONFIG)),
        ring_joint_hash(device_, std::nullopt, make_scalar_u32(device_, ttnn::L1_MEMORY_CONFIG)));
}

// use_column_major_ccl picks the all-gather and halo worker cores column- rather than row-major.
TEST_F(StaleProgramHashFixture, RingJointSdpaCoreAllocationStrategySeparatesKeys) {
    EXPECT_NE(
        ring_joint_hash(device_, std::nullopt, std::nullopt, CoreAllocationStrategy::ROW_MAJOR),
        ring_joint_hash(device_, std::nullopt, std::nullopt, CoreAllocationStrategy::COL_MAJOR));
}

// ---------------------------------------------------------------------------------------------------------------
// 7. ExpRingJointSDPA: topology and num_buffers_per_channel
// ---------------------------------------------------------------------------------------------------------------

using ::ttnn::prim::ExpRingJointSDPADeviceOperation;
using ::ttnn::prim::ExpRingJointSDPAInputs;
using ::ttnn::prim::ExpRingJointSDPAParams;

// ExpRingJointSDPA has no custom hash, so this goes through the default one the program cache uses.
ttsl::hash::hash_t exp_ring_joint_hash(
    MeshDevice* device, ttnn::ccl::Topology topology, uint32_t num_buffers_per_channel) {
    const ExpRingJointSDPAParams params(
        /*joint_strategy=*/"rear",
        /*scale=*/std::nullopt,
        /*logical_n=*/64 * kRingSize,
        /*ring_size=*/kRingSize,
        /*output_memory_config=*/ttnn::DRAM_MEMORY_CONFIG,
        /*program_config=*/std::nullopt,
        /*compute_kernel_config=*/DeviceComputeKernelConfig{},
        /*dim=*/2,
        /*num_links=*/1,
        topology,
        /*semaphore=*/{},
        /*sub_device_id=*/std::nullopt,
        /*cluster_axis=*/1,
        /*num_workers_per_link=*/1,
        num_buffers_per_channel);
    const ExpRingJointSDPAInputs inputs{
        .input_q = make_tile(device, kQShape),
        .input_k = make_tile(device, kQShape),
        .input_v = make_tile(device, kQShape),
        .gathered_k = make_tile(device, kGatheredKShape),
        .gathered_v = make_tile(device, kGatheredKShape),
    };
    return ttnn::device_operation::detail::compute_program_hash<ExpRingJointSDPADeviceOperation>(params, inputs);
}

TEST_F(StaleProgramHashFixture, ExpRingJointSdpaSameAttributesShareKey) {
    EXPECT_EQ(
        exp_ring_joint_hash(device_, ttnn::ccl::Topology::Ring, 8),
        exp_ring_joint_hash(device_, ttnn::ccl::Topology::Ring, 8));
}

TEST_F(StaleProgramHashFixture, ExpRingJointSdpaTopologySeparatesKeys) {
    EXPECT_NE(
        exp_ring_joint_hash(device_, ttnn::ccl::Topology::Ring, 8),
        exp_ring_joint_hash(device_, ttnn::ccl::Topology::Linear, 8));
}

TEST_F(StaleProgramHashFixture, ExpRingJointSdpaNumBuffersPerChannelSeparatesKeys) {
    EXPECT_NE(
        exp_ring_joint_hash(device_, ttnn::ccl::Topology::Ring, 8),
        exp_ring_joint_hash(device_, ttnn::ccl::Topology::Ring, 16));
}

// ---------------------------------------------------------------------------------------------------------------
// 8. DramPrefetcherConsumer: global CB structure
// ---------------------------------------------------------------------------------------------------------------

using ::tt::tt_metal::experimental::GlobalCircularBuffer;
using ::ttnn::operations::experimental::test::DramPrefetcherConsumerDeviceOperation;

constexpr uint32_t kGcbSize = 3200;
const CoreCoord kGcbSender(0, 0);
// Same receiver count, so the two GCBs have the same footprint.
const CoreRangeSet kGcbReceiversA(CoreRange({1, 1}, {2, 1}));
const CoreRangeSet kGcbReceiversB(CoreRange({1, 2}, {2, 2}));

GlobalCircularBuffer make_gcb(MeshDevice* device, const CoreRangeSet& receivers) {
    return ::tt::tt_metal::experimental::CreateGlobalCircularBuffer(
        *device, {{kGcbSender, receivers}}, kGcbSize, BufferType::L1);
}

ttsl::hash::hash_t prefetcher_consumer_hash(MeshDevice* device, const GlobalCircularBuffer& global_cb) {
    const DramPrefetcherConsumerDeviceOperation::operation_attributes_t attrs{
        .num_iters = 1, .page_size_bytes = kGcbSize, .global_cb = global_cb, .mesh_device = device};
    return DramPrefetcherConsumerDeviceOperation::compute_program_hash(attrs, {});
}

TEST_F(StaleProgramHashFixture, DramPrefetcherConsumerSameGlobalCbSharesKey) {
    const auto global_cb = make_gcb(device_, kGcbReceiversA);
    EXPECT_EQ(prefetcher_consumer_hash(device_, global_cb), prefetcher_consumer_hash(device_, global_cb));
}

// A freed GCB's config address is handed to the next GCB of the same footprint, so a GCB with other receiver
// cores can land on it. The receiver kernel and remote CB are placed on those cores, so the key must differ.
TEST_F(StaleProgramHashFixture, DramPrefetcherConsumerReallocatedGlobalCbSeparatesKeys) {
    std::optional<GlobalCircularBuffer> first = make_gcb(device_, kGcbReceiversA);
    const auto first_config_address = first->config_address();
    const auto first_hash = prefetcher_consumer_hash(device_, *first);
    first.reset();

    const auto second = make_gcb(device_, kGcbReceiversB);
    ASSERT_EQ(second.config_address(), first_config_address);
    EXPECT_NE(first_hash, prefetcher_consumer_hash(device_, second));
}

}  // namespace
}  // namespace ttnn::test::stale_program_hash
