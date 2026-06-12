// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <cstdint>
#include <string>
#include <vector>

#include "autograd/auto_context.hpp"
#include "core/tt_tensor_utils.hpp"
#include "metal/operations.hpp"
#include "ops/rope_op.hpp"
#include "test_utils/random_data.hpp"
#include "ttnn/operations/data_movement/concat/concat.hpp"
#include "ttnn/operations/data_movement/slice/slice.hpp"
#include "ttnn/operations/experimental/transformer/rotary_embedding_llama/rotary_embedding_llama.hpp"

namespace {

struct QRopeShape {
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

void expect_allclose(
    const xt::xarray<float>& actual,
    const xt::xarray<float>& expected,
    double rtol,
    double atol,
    const std::string& tag) {
    ASSERT_EQ(actual.shape(), expected.shape()) << tag << ": shape mismatch";
    EXPECT_TRUE(xt::allclose(actual, expected, rtol, atol)) << tag << ": value mismatch";
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
        q_pe, params.cos_cache, params.sin_cache, params.trans_mat, /*is_decode_mode=*/false);

    return ttnn::concat(std::vector<ttnn::Tensor>{q_nope, q_pe_rot}, /*dim=*/3);
}

}  // namespace

class QRopeParamTest : public ::testing::TestWithParam<QRopeShape> {
protected:
    static void SetUpTestSuite() {
        ttml::autograd::ctx().open_device();
    }

    static void TearDownTestSuite() {
        ttml::autograd::ctx().close_device();
    }
};

TEST_P(QRopeParamTest, FusedMatchesReference) {
    const QRopeShape shape = GetParam();
    const uint32_t qk_head = shape.qk_nope_dim + shape.qk_rope_dim;

    auto q_in = make_bf16_4d(shape.batch, shape.n_heads, shape.seq_len, qk_head, /*seed=*/7U);
    auto params = build_params(shape.seq_len, shape.qk_rope_dim);

    const auto fused = ttml::metal::q_rope_fw(
        q_in, params.cos_cache, params.sin_cache, params.trans_mat, shape.qk_nope_dim, shape.qk_rope_dim);
    const auto ref = reference_q_rope(q_in, params, shape.qk_nope_dim, shape.qk_rope_dim);

    const uint32_t B = shape.batch;
    const uint32_t H = shape.n_heads;
    const uint32_t S = shape.seq_len;

    // q_nope is a straight copy in the fused kernel -> bit-exact vs reference; q_pe is RoPE math -> BF16 tolerance.
    expect_allclose(
        ttml::core::to_xtensor(slice_head_dim(fused, B, H, S, 0U, shape.qk_nope_dim)),
        ttml::core::to_xtensor(slice_head_dim(ref, B, H, S, 0U, shape.qk_nope_dim)),
        0.0,
        0.0,
        shape.name + " q_nope");
    expect_allclose(
        ttml::core::to_xtensor(slice_head_dim(fused, B, H, S, shape.qk_nope_dim, qk_head)),
        ttml::core::to_xtensor(slice_head_dim(ref, B, H, S, shape.qk_nope_dim, qk_head)),
        1e-4,
        1e-4,
        shape.name + " q_pe");
}

INSTANTIATE_TEST_SUITE_P(
    QRopeShapes,
    QRopeParamTest,
    ::testing::Values(
        QRopeShape{.name = "small", .batch = 1, .seq_len = 32, .n_heads = 4, .qk_nope_dim = 64, .qk_rope_dim = 32},
        QRopeShape{.name = "mla_like", .batch = 1, .seq_len = 32, .n_heads = 8, .qk_nope_dim = 128, .qk_rope_dim = 64},
        QRopeShape{.name = "batch2", .batch = 2, .seq_len = 32, .n_heads = 4, .qk_nope_dim = 64, .qk_rope_dim = 32},
        QRopeShape{.name = "square_st1", .batch = 2, .seq_len = 32, .n_heads = 2, .qk_nope_dim = 32, .qk_rope_dim = 32},
        QRopeShape{.name = "square_st2", .batch = 2, .seq_len = 64, .n_heads = 2, .qk_nope_dim = 32, .qk_rope_dim = 32},
        QRopeShape{
            .name = "asym_rope2", .batch = 2, .seq_len = 64, .n_heads = 4, .qk_nope_dim = 128, .qk_rope_dim = 64},
        QRopeShape{.name = "heads8_s96", .batch = 1, .seq_len = 96, .n_heads = 8, .qk_nope_dim = 64, .qk_rope_dim = 32},
        // Near-limit coverage (fp32 dest acc off when qk_rope_dim > 128, same as rotary_embedding_llama).
        QRopeShape{
            .name = "large_tr4",
            .batch = 1,
            .seq_len = 32,
            .n_heads = 4,
            .qk_nope_dim = 128,
            .qk_rope_dim = 128},  // Tr = 4, Th = 8
        QRopeShape{
            .name = "large_tr5",
            .batch = 1,
            .seq_len = 32,
            .n_heads = 4,
            .qk_nope_dim = 128,
            .qk_rope_dim = 160},  // Tr = 5
        QRopeShape{
            .name = "max_th15",
            .batch = 1,
            .seq_len = 32,
            .n_heads = 2,
            .qk_nope_dim = 256,
            .qk_rope_dim = 224},  // Tn = 8, Tr = 7, Th = 15
        // Tn > 4 (fp32 on): chunked nope in 4-tile DST batches.
        QRopeShape{
            .name = "chunked_nope_tn35",
            .batch = 1,
            .seq_len = 32,
            .n_heads = 2,
            .qk_nope_dim = 1120,
            .qk_rope_dim = 64},  // Tn = 35, Tr = 2, fp32 on
        // Tn > 8 (fp32 off): chunked nope in 8-tile DST batches.
        QRopeShape{
            .name = "chunked_nope_tn35_fp32_off",
            .batch = 1,
            .seq_len = 32,
            .n_heads = 2,
            .qk_nope_dim = 1120,
            .qk_rope_dim = 160},  // Tn = 35, Tr = 5, fp32 off
        // B * Ts = 8 blocks: exercises multi-core split and repeated batch-boundary jumps.
        QRopeShape{
            .name = "batch4_st2", .batch = 4, .seq_len = 64, .n_heads = 2, .qk_nope_dim = 32, .qk_rope_dim = 32}),
    [](const ::testing::TestParamInfo<QRopeShape>& info) { return info.param.name; });
