// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "autograd/auto_context.hpp"
#include "core/tt_tensor_utils.hpp"
#include "ops/multi_head_utils.hpp"
#include "test_utils/random_data.hpp"

class SplitHeadsTest : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        ttml::autograd::ctx().open_device();
    }

    static void TearDownTestSuite() {
        ttml::autograd::ctx().close_device();
    }

    void TearDown() override {
        ttml::autograd::ctx().reset_graph();
    }
};

namespace {

// One split_heads case. head_dim is a multiple of TILE_WIDTH and seq_len a multiple of TILE_HEIGHT,
// which is the whole domain the op accepts.
struct SplitShape {
    std::string name;
    uint32_t batch = 0;
    uint32_t seq_len = 0;
    uint32_t num_heads = 0;
    uint32_t head_dim = 0;

    uint32_t embedding_dim() const {
        return num_heads * head_dim;
    }
};

// Deterministic uniform BF16 device tensor of the given 4D shape.
ttnn::Tensor make_input(uint32_t d0, uint32_t d1, uint32_t d2, uint32_t d3, uint32_t seed) {
    auto* device = &ttml::autograd::ctx().get_device();
    const std::size_t count = static_cast<std::size_t>(d0) * d1 * d2 * d3;
    const auto host = ttml::test_utils::make_uniform_vector<float>(count, -1.0F, 1.0F, seed);
    return ttml::core::from_vector<float, ttnn::DataType::BFLOAT16>(
        host, ttnn::Shape({d0, d1, d2, d3}), device, ttnn::Layout::TILE);
}

ttml::autograd::TensorPtr make_activation(const SplitShape& s, uint32_t seed) {
    return ttml::autograd::create_tensor(
        make_input(s.batch, 1U, s.seq_len, s.embedding_dim(), seed), /*requires_grad=*/true);
}

// Forward reference: (B, 1, S, H*D) -> (B, S, H, D) -> (B, H, S, D).
xt::xarray<float> reference_fw(const ttnn::Tensor& x, const SplitShape& s) {
    const std::size_t B = s.batch;
    const std::size_t S = s.seq_len;
    const std::size_t H = s.num_heads;
    const std::size_t D = s.head_dim;

    xt::xarray<float> x_bf = ttml::core::to_xtensor(x);
    x_bf.reshape({B, S, H, D});
    return xt::transpose(x_bf, {0, 2, 1, 3});
}

// Backward reference: the forward run in reverse, (B, H, S, D) -> (B, 1, S, H*D).
xt::xarray<float> reference_bw(const ttnn::Tensor& grad, const SplitShape& s) {
    const std::size_t B = s.batch;
    const std::size_t S = s.seq_len;
    const std::size_t H = s.num_heads;
    const std::size_t D = s.head_dim;

    const xt::xarray<float> grad_bf = ttml::core::to_xtensor(grad);
    xt::xarray<float> ref = xt::transpose(grad_bf, {0, 2, 1, 3});
    ref.reshape({B, 1U, S, H * D});
    return ref;
}

void run_fw(const SplitShape& s) {
    const auto x_value = make_input(s.batch, 1U, s.seq_len, s.embedding_dim(), 1001U);
    auto x = ttml::autograd::create_tensor(x_value, /*requires_grad=*/true);

    auto out = ttml::ops::split_heads(x, s.num_heads);

    EXPECT_EQ(out->get_value().logical_shape(), ttnn::Shape({s.batch, s.num_heads, s.seq_len, s.head_dim}))
        << s.name << " fw/shape";
    // Pure data movement → bit-exact.
    EXPECT_TRUE(xt::allclose(ttml::core::to_xtensor(out->get_value()), reference_fw(x_value, s), 0.0, 0.0))
        << s.name << " fw/values";
}

void run_bw(const SplitShape& s) {
    auto x = make_activation(s, 2002U);
    auto out = ttml::ops::split_heads(x, s.num_heads);

    // Seed an arbitrary (not uniform) gradient: a mean-reduction loss would give every element the
    // same gradient and so could not tell a correct un-split from a transposed one.
    const auto grad = make_input(s.batch, s.num_heads, s.seq_len, s.head_dim, 3003U);
    out->set_grad(grad);
    out->backward();

    EXPECT_EQ(x->get_grad().logical_shape(), ttnn::Shape({s.batch, 1U, s.seq_len, s.embedding_dim()}))
        << s.name << " bw/shape";
    EXPECT_TRUE(xt::allclose(ttml::core::to_xtensor(x->get_grad()), reference_bw(grad, s), 0.0, 0.0))
        << s.name << " bw/values";
}

const std::vector<SplitShape>& shapes() {
    static const std::vector<SplitShape> cases = {
        {"h8_d64_s128", 1, 128, 8, 64},
        {"b2_h4_d32_s64", 2, 64, 4, 32},
        // S = 96 is a multiple of TILE_HEIGHT but not a power of two.
        {"h8_d32_s96", 1, 96, 8, 32},
        {"h1_d64_s32", 1, 32, 1, 64},
        {"b2_h16_d32_s32", 2, 32, 16, 32},
    };
    return cases;
}

}  // namespace

TEST_F(SplitHeadsTest, ForwardMatchesReference) {
    for (const auto& shape : shapes()) {
        run_fw(shape);
    }
}

TEST_F(SplitHeadsTest, BackwardMatchesReference) {
    for (const auto& shape : shapes()) {
        run_bw(shape);
    }
}

TEST_F(SplitHeadsTest, RoundTripsThroughHeadsFusion) {
    // heads_fusion is the inverse, so fusing a split must return the original tensor unchanged.
    for (const auto& shape : shapes()) {
        const auto x_value = make_input(shape.batch, 1U, shape.seq_len, shape.embedding_dim(), 4004U);
        auto x = ttml::autograd::create_tensor(x_value, /*requires_grad=*/true);

        auto fused = ttml::ops::heads_fusion(ttml::ops::split_heads(x, shape.num_heads));

        EXPECT_EQ(fused->get_value().logical_shape(), x_value.logical_shape()) << shape.name << " round_trip/shape";
        EXPECT_TRUE(xt::allclose(ttml::core::to_xtensor(fused->get_value()), ttml::core::to_xtensor(x_value), 0.0, 0.0))
            << shape.name << " round_trip/values";
        ttml::autograd::ctx().reset_graph();
    }
}

TEST_F(SplitHeadsTest, SplitsQueryAndKeyValueOfDifferentSequenceLengths) {
    // The cross-attention case that heads_creation and grouped_heads_creation cannot express:
    // Q, K and V are projected separately and Q's sequence length differs from K/V's.
    const SplitShape q_shape{"cross_q", 1, 128, 8, 64};
    const SplitShape kv_shape{"cross_kv", 1, 96, 8, 64};

    const auto q_value = make_input(q_shape.batch, 1U, q_shape.seq_len, q_shape.embedding_dim(), 5005U);
    const auto k_value = make_input(kv_shape.batch, 1U, kv_shape.seq_len, kv_shape.embedding_dim(), 6006U);

    auto q = ttml::ops::split_heads(ttml::autograd::create_tensor(q_value, /*requires_grad=*/true), 8U);
    auto k = ttml::ops::split_heads(ttml::autograd::create_tensor(k_value, /*requires_grad=*/true), 8U);

    EXPECT_EQ(q->get_value().logical_shape(), ttnn::Shape({1U, 8U, 128U, 64U}));
    EXPECT_EQ(k->get_value().logical_shape(), ttnn::Shape({1U, 8U, 96U, 64U}));
    EXPECT_TRUE(xt::allclose(ttml::core::to_xtensor(q->get_value()), reference_fw(q_value, q_shape), 0.0, 0.0));
    EXPECT_TRUE(xt::allclose(ttml::core::to_xtensor(k->get_value()), reference_fw(k_value, kv_shape), 0.0, 0.0));
}

TEST_F(SplitHeadsTest, RejectsNonUnitHeadDimension) {
    auto x = ttml::autograd::create_tensor(make_input(1U, 2U, 32U, 64U, 7007U), /*requires_grad=*/true);
    EXPECT_THROW(ttml::ops::split_heads(x, 2U), std::invalid_argument);
}

TEST_F(SplitHeadsTest, RejectsZeroHeads) {
    auto x = ttml::autograd::create_tensor(make_input(1U, 1U, 32U, 512U, 8008U), /*requires_grad=*/true);
    EXPECT_THROW(ttml::ops::split_heads(x, 0U), std::invalid_argument);
}

TEST_F(SplitHeadsTest, RejectsIndivisibleEmbeddingDim) {
    auto x = ttml::autograd::create_tensor(make_input(1U, 1U, 32U, 512U, 9009U), /*requires_grad=*/true);
    EXPECT_THROW(ttml::ops::split_heads(x, 7U), std::invalid_argument);
}

TEST_F(SplitHeadsTest, RejectsSubTileHeadDim) {
    // head_dim = 96 / 4 = 24 < TILE_WIDTH. The kernel would compute head_dim / TILE_WIDTH = 0 tiles,
    // write nothing, and hand back a tile-padded (1, 4, 32, 32) tensor of uninitialised memory.
    auto x = ttml::autograd::create_tensor(make_input(1U, 1U, 32U, 96U, 1111U), /*requires_grad=*/true);
    EXPECT_THROW(ttml::ops::split_heads(x, 4U), std::invalid_argument);

    // ... and the same for a tile-aligned embedding dim that still splits below a tile.
    auto y = ttml::autograd::create_tensor(make_input(1U, 1U, 32U, 128U, 2222U), /*requires_grad=*/true);
    EXPECT_THROW(ttml::ops::split_heads(y, 8U), std::invalid_argument);
}
