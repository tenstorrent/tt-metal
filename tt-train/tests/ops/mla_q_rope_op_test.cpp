// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <cstdint>
#include <string>
#include <vector>

#include "autograd/auto_context.hpp"
#include "core/compute_kernel_config.hpp"
#include "core/tt_tensor_utils.hpp"
#include "metal/operations.hpp"
#include "ops/rope_op.hpp"
#include "test_utils/random_data.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/data_movement/concat/concat.hpp"
#include "ttnn/operations/data_movement/slice/slice.hpp"
#include "ttnn/operations/experimental/transformer/rotary_embedding_llama/rotary_embedding_llama.hpp"
#include "ttnn/types.hpp"

namespace {

struct MLA_QRopeShape {
    std::string name;
    uint32_t batch = 0;
    uint32_t seq_len = 0;
    uint32_t n_heads = 0;
    uint32_t qk_nope_dim = 0;
    uint32_t qk_rope_dim = 0;
};

ttnn::Tensor make_bf16_4d(uint32_t d0, uint32_t d1, uint32_t d2, uint32_t d3, uint32_t seed) {
    auto* device = &ttml::autograd::ctx().get_device();
    const size_t count = static_cast<size_t>(d0) * d1 * d2 * d3;
    const auto host = ttml::test_utils::make_uniform_vector<float>(count, -1.0F, 1.0F, seed);
    return ttml::core::from_vector<float, ttnn::DataType::BFLOAT16>(
        host, ttnn::Shape{d0, d1, d2, d3}, device, ttnn::Layout::TILE);
}

ttml::ops::RotaryEmbeddingParams build_params(uint32_t seq_len, uint32_t qk_rope_dim) {
    return ttml::ops::build_rope_params(seq_len, qk_rope_dim, /*theta=*/10000.0F);
}

ttnn::Tensor slice_head_dim(
    const ttnn::Tensor& tensor, uint32_t B, uint32_t H, uint32_t S, uint32_t start_w, uint32_t end_w) {
    ttsl::SmallVector<uint32_t> step = {1, 1, 1, 1};
    return ttnn::slice(
        tensor, ttsl::SmallVector<uint32_t>{0, 0, 0, start_w}, ttsl::SmallVector<uint32_t>{B, H, S, end_w}, step);
}

ttnn::Tensor reference_q_rope(
    const ttnn::Tensor& q_in,
    const ttml::ops::RotaryEmbeddingParams& params,
    uint32_t qk_nope_dim,
    uint32_t qk_rope_dim) {
    const auto shape = q_in.logical_shape();
    const uint32_t B = shape[0];
    const uint32_t H = shape[1];
    const uint32_t S = shape[2];
    const uint32_t qk_head = qk_nope_dim + qk_rope_dim;

    ttsl::SmallVector<uint32_t> step = {1, 1, 1, 1};
    auto q_nope = ttnn::slice(
        q_in, ttsl::SmallVector<uint32_t>{0, 0, 0, 0}, ttsl::SmallVector<uint32_t>{B, H, S, qk_nope_dim}, step);
    auto q_pe = ttnn::slice(
        q_in, ttsl::SmallVector<uint32_t>{0, 0, 0, qk_nope_dim}, ttsl::SmallVector<uint32_t>{B, H, S, qk_head}, step);

    auto q_pe_rot = ttnn::experimental::rotary_embedding_llama(
        q_pe,
        params.cos_cache,
        params.sin_cache,
        params.trans_mat,
        /*is_decode_mode=*/false,
        /*memory_config=*/std::nullopt,
        ttml::core::ComputeKernelConfig::precise());

    return ttnn::concat(std::vector<ttnn::Tensor>{q_nope, q_pe_rot}, /*dim=*/3);
}

void expect_q_rope_matches_reference(
    const ttnn::Tensor& fused, const ttnn::Tensor& ref, const MLA_QRopeShape& shape, const std::string& label_prefix) {
    const uint32_t qk_head = shape.qk_nope_dim + shape.qk_rope_dim;
    const uint32_t B = shape.batch;
    const uint32_t H = shape.n_heads;
    const uint32_t S = shape.seq_len;

    {
        const auto actual_q_nope = ttml::core::to_xtensor(slice_head_dim(fused, B, H, S, 0U, shape.qk_nope_dim));
        const auto expected_q_nope = ttml::core::to_xtensor(slice_head_dim(ref, B, H, S, 0U, shape.qk_nope_dim));
        ASSERT_EQ(actual_q_nope.shape(), expected_q_nope.shape())
            << label_prefix << shape.name << " q_nope: shape mismatch";
        EXPECT_TRUE(xt::allclose(actual_q_nope, expected_q_nope, 0.0, 0.0))
            << label_prefix << shape.name << " q_nope: value mismatch";
    }
    {
        const auto actual_q_pe = ttml::core::to_xtensor(slice_head_dim(fused, B, H, S, shape.qk_nope_dim, qk_head));
        const auto expected_q_pe = ttml::core::to_xtensor(slice_head_dim(ref, B, H, S, shape.qk_nope_dim, qk_head));
        ASSERT_EQ(actual_q_pe.shape(), expected_q_pe.shape()) << label_prefix << shape.name << " q_pe: shape mismatch";
        EXPECT_TRUE(xt::allclose(actual_q_pe, expected_q_pe, 1e-2, 1e-2))
            << label_prefix << shape.name << " q_pe: value mismatch";
    }
}

}  // namespace

class MLA_QRopeParamTest : public ::testing::TestWithParam<MLA_QRopeShape> {
protected:
    static void SetUpTestSuite() {
        ttml::autograd::ctx().open_device();
    }

    static void TearDownTestSuite() {
        ttml::autograd::ctx().close_device();
    }
};

TEST_P(MLA_QRopeParamTest, FusedMatchesReference) {
    const MLA_QRopeShape shape = GetParam();
    ASSERT_LE(shape.qk_rope_dim, 128U) << shape.name << ": mla_q_rope requires qk_rope_dim <= 128";

    const uint32_t qk_head = shape.qk_nope_dim + shape.qk_rope_dim;
    auto q_in = make_bf16_4d(shape.batch, shape.n_heads, shape.seq_len, qk_head, /*seed=*/7U);
    auto params = build_params(shape.seq_len, shape.qk_rope_dim);

    const auto fused = ttml::metal::mla_q_rope(
        q_in, params.cos_cache, params.sin_cache, params.trans_mat, shape.qk_nope_dim, shape.qk_rope_dim);
    const auto ref = reference_q_rope(q_in, params, shape.qk_nope_dim, shape.qk_rope_dim);

    expect_q_rope_matches_reference(fused, ref, shape, /*label_prefix=*/"");
}

INSTANTIATE_TEST_SUITE_P(
    MLA_QRopeShapes,
    MLA_QRopeParamTest,
    ::testing::Values(
        MLA_QRopeShape{.name = "small", .batch = 1, .seq_len = 32, .n_heads = 4, .qk_nope_dim = 64, .qk_rope_dim = 32},
        MLA_QRopeShape{
            .name = "mla_like", .batch = 1, .seq_len = 32, .n_heads = 8, .qk_nope_dim = 128, .qk_rope_dim = 64},
        MLA_QRopeShape{.name = "batch2", .batch = 2, .seq_len = 32, .n_heads = 4, .qk_nope_dim = 64, .qk_rope_dim = 32},
        MLA_QRopeShape{
            .name = "square_st1", .batch = 2, .seq_len = 32, .n_heads = 2, .qk_nope_dim = 32, .qk_rope_dim = 32},
        MLA_QRopeShape{
            .name = "square_st2", .batch = 2, .seq_len = 64, .n_heads = 2, .qk_nope_dim = 32, .qk_rope_dim = 32},
        MLA_QRopeShape{
            .name = "asym_rope2", .batch = 2, .seq_len = 64, .n_heads = 4, .qk_nope_dim = 128, .qk_rope_dim = 64},
        MLA_QRopeShape{
            .name = "heads8_s96", .batch = 1, .seq_len = 96, .n_heads = 8, .qk_nope_dim = 64, .qk_rope_dim = 32},
        MLA_QRopeShape{
            .name = "large_tr4",
            .batch = 1,
            .seq_len = 32,
            .n_heads = 4,
            .qk_nope_dim = 128,
            .qk_rope_dim = 128},  // Tr = 4, Th = 8
        // Large Tn: q_nope bypasses compute (reader -> writer); only Tr rope tiles hit DST.
        MLA_QRopeShape{
            .name = "chunked_nope_tn35",
            .batch = 1,
            .seq_len = 32,
            .n_heads = 2,
            .qk_nope_dim = 1120,
            .qk_rope_dim = 64},  // Tn = 35, Tr = 2
        // B * Ts = 8 blocks: exercises multi-core split and repeated batch-boundary jumps.
        MLA_QRopeShape{
            .name = "batch4_st2", .batch = 4, .seq_len = 64, .n_heads = 2, .qk_nope_dim = 32, .qk_rope_dim = 32}),
    [](const ::testing::TestParamInfo<MLA_QRopeShape>& info) { return info.param.name; });

class MLA_QRopeCacheTest : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        ttml::autograd::ctx().open_device();
    }

    static void TearDownTestSuite() {
        ttml::autograd::ctx().close_device();
    }
};

// Buffer placement is a compile-time property: TensorAccessorArgs bake it into the
// kernel binary, so the program cache must key on every input's memory config. Runs
// the op with q_in interleaved in DRAM and again with q_in in L1 at identical shapes,
// asserting the second run compiles a distinct program and still matches the
// reference — a cache key blind to placement reuses the DRAM program for the L1
// buffer and reads through the wrong accessor tables.
TEST_F(MLA_QRopeCacheTest, DistinctProgramPerInputPlacement) {
    auto* device = &ttml::autograd::ctx().get_device();
    device->enable_program_cache();

    const MLA_QRopeShape shape{
        .name = "cache_small", .batch = 1, .seq_len = 32, .n_heads = 4, .qk_nope_dim = 64, .qk_rope_dim = 32};
    const uint32_t qk_head = shape.qk_nope_dim + shape.qk_rope_dim;
    auto q_in = make_bf16_4d(shape.batch, shape.n_heads, shape.seq_len, qk_head, /*seed=*/11U);
    auto params = build_params(shape.seq_len, shape.qk_rope_dim);
    const auto ref = reference_q_rope(q_in, params, shape.qk_nope_dim, shape.qk_rope_dim);

    const auto entries_start = device->num_program_cache_entries();
    const auto fused_dram = ttml::metal::mla_q_rope(
        q_in, params.cos_cache, params.sin_cache, params.trans_mat, shape.qk_nope_dim, shape.qk_rope_dim);
    const auto entries_after_dram = device->num_program_cache_entries();
    // Guard against a vacuous test: the first call must actually populate the cache,
    // else the delta check below would pass trivially with the cache disabled.
    ASSERT_GT(entries_after_dram, entries_start) << "program cache not populated by the DRAM-placement run";
    expect_q_rope_matches_reference(fused_dram, ref, shape, /*label_prefix=*/"dram: ");

    auto q_in_l1 = ttnn::to_memory_config(q_in, ttnn::L1_MEMORY_CONFIG);
    const auto entries_before_l1 = device->num_program_cache_entries();
    const auto fused_l1 = ttml::metal::mla_q_rope(
        q_in_l1, params.cos_cache, params.sin_cache, params.trans_mat, shape.qk_nope_dim, shape.qk_rope_dim);
    const auto entries_after_l1 = device->num_program_cache_entries();
    EXPECT_GT(entries_after_l1, entries_before_l1)
        << "mla_q_rope reused the DRAM-placement program for an L1 input (stale cache key)";
    expect_q_rope_matches_reference(fused_l1, ref, shape, /*label_prefix=*/"l1: ");
}
