// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ops/embedding_op.hpp"

#include <gtest/gtest.h>

#include <algorithm>

#include "autograd/auto_context.hpp"
#include "autograd/tensor.hpp"
#include "core/system_utils.hpp"
#include "core/tt_tensor_utils.hpp"
#include "ops/losses.hpp"
#include "ttnn/operations/data_movement/fill_pad/fill_pad.hpp"

class EmbeddingOpTest : public ::testing::Test {
protected:
    void SetUp() override {
        ttml::autograd::ctx().open_device();
    }

    void TearDown() override {
        ttml::autograd::ctx().close_device();
    }
};

TEST_F(EmbeddingOpTest, EmbeddingForwardBackward) {
    using namespace ttml;

    auto* device = &autograd::ctx().get_device();
    uint32_t num_embeddings = 32;
    uint32_t embedding_dim = 32;
    auto weight_tensor = ttml::core::zeros(ttnn::Shape({1, 1, num_embeddings, embedding_dim}), device);
    autograd::TensorPtr weight = autograd::create_tensor(weight_tensor, /* requires_grad */ true);

    uint32_t batch_size = 1;
    uint32_t sentence_size = 32;
    std::vector<uint32_t> input_data((size_t)batch_size * sentence_size);
    std::iota(input_data.begin(), input_data.end(), 0U);
    auto input_tensor = ttml::core::from_vector<uint32_t, ttnn::DataType::UINT32>(
        input_data, ttnn::Shape({batch_size, 1, 1, sentence_size}), device, ttnn::Layout::ROW_MAJOR);
    autograd::TensorPtr input = autograd::create_tensor(input_tensor);

    autograd::TensorPtr embeddings = ops::embedding_op(input, weight);

    std::vector<float> target_vector((size_t)batch_size * sentence_size * embedding_dim);
    for (uint32_t i = 0; i < batch_size * sentence_size; i++) {
        for (uint32_t j = 0; j < embedding_dim; j++) {
            target_vector[embedding_dim * i + j] = static_cast<float>(i);
        }
    }
    auto target_tensor = autograd::create_tensor(
        ttml::core::from_vector(target_vector, ttnn::Shape({batch_size, 1, sentence_size, embedding_dim}), device));
    auto result = ttml::ops::mse_loss(embeddings, target_tensor);
    result->backward();

    auto weight_grad_tensor = weight->get_grad();
    auto weight_grad_data = ttml::core::to_vector(weight_grad_tensor);
    for (uint32_t i = 0; i < num_embeddings; i++) {
        for (uint32_t j = 0; j < embedding_dim; j++) {
            EXPECT_NEAR(
                weight_grad_data[embedding_dim * i + j],
                -static_cast<float>(i) / sentence_size / embedding_dim / batch_size * 2.F,
                1e-2);
        }
    }
}

TEST_F(EmbeddingOpTest, EmbeddingNumEmbeddingsEmbeddingDimNotDivisibleBy32) {
    using namespace ttnn;
    using namespace ttml;

    auto* device = &autograd::ctx().get_device();
    uint32_t num_embeddings = 13;
    uint32_t embedding_dim = 26;
    auto weight_tensor = ttml::core::zeros(ttnn::Shape({1, 1, num_embeddings, embedding_dim}), device);
    autograd::TensorPtr weight = autograd::create_tensor(weight_tensor);

    uint32_t batch_size = 1;
    uint32_t sentence_size = 32;
    std::vector<uint32_t> input_data((size_t)batch_size * sentence_size);
    std::iota(input_data.begin(), input_data.end(), 0U);
    auto input_tensor = ttml::core::from_vector<uint32_t, DataType::UINT32>(
        input_data, ttnn::Shape({batch_size, 1, 1, sentence_size}), device, Layout::ROW_MAJOR);
    autograd::TensorPtr input = autograd::create_tensor(input_tensor);

    EXPECT_NO_THROW(ops::embedding_op(input, weight));
}

TEST_F(EmbeddingOpTest, EmbeddingSentenceDimNotDivisibleBy32) {
    using namespace ttnn;
    using namespace ttml;

    auto* device = &autograd::ctx().get_device();
    uint32_t num_embeddings = 32;
    uint32_t embedding_dim = 32;
    std::vector<float> weight_data((size_t)num_embeddings * embedding_dim);
    for (uint32_t row = 0; row < num_embeddings; ++row) {
        std::fill_n(weight_data.begin() + row * embedding_dim, embedding_dim, static_cast<float>(row));
    }
    auto weight_tensor =
        ttml::core::from_vector(weight_data, ttnn::Shape({1, 1, num_embeddings, embedding_dim}), device);
    autograd::TensorPtr weight = autograd::create_tensor(weight_tensor, /* requires_grad */ true);

    uint32_t batch_size = 2;
    uint32_t sentence_size = 13;
    std::vector<uint32_t> input_data = {1, 3, 3, 7,  12, 3, 5,  1,  31, 7, 2, 2, 9,
                                        4, 3, 1, 12, 12, 8, 31, 31, 6,  3, 5, 1, 1};
    auto input_tensor = ttml::core::from_vector<uint32_t, DataType::UINT32>(
        input_data, ttnn::Shape({batch_size, 1, 1, sentence_size}), device, Layout::ROW_MAJOR);
    autograd::TensorPtr input = autograd::create_tensor(input_tensor, /* requires_grad */ false);

    auto embeddings = ops::embedding_op(input, weight);
    EXPECT_EQ(embeddings->get_value().logical_shape(), ttnn::Shape({batch_size, 1, sentence_size, embedding_dim}));

    const auto embeddings_data = ttml::core::to_vector(embeddings->get_value());
    for (uint32_t token = 0; token < input_data.size(); ++token) {
        for (uint32_t column = 0; column < embedding_dim; ++column) {
            EXPECT_NEAR(embeddings_data[token * embedding_dim + column], input_data[token], 1e-2F);
        }
    }

    auto upstream_grad =
        ttml::core::ones(ttnn::Shape({batch_size, 1, sentence_size, embedding_dim}), device, ttnn::DataType::BFLOAT16);
    upstream_grad = ttnn::fill_implicit_tile_padding(upstream_grad, 64.0F);
    embeddings->set_grad(upstream_grad);
    embeddings->backward();

    std::vector<uint32_t> token_counts(num_embeddings, 0U);
    for (const auto token : input_data) {
        ++token_counts[token];
    }
    const auto weight_grad_data = ttml::core::to_vector(weight->get_grad());
    for (uint32_t row = 0; row < num_embeddings; ++row) {
        for (uint32_t column = 0; column < embedding_dim; ++column) {
            EXPECT_NEAR(weight_grad_data[row * embedding_dim + column], token_counts[row], 1e-2F);
        }
    }
}

// This test was previously throwing an exception, but now it just freezes
// The main reason that we are passing input_tensor as tiled, but it should be row major
// We will uncomment it once the issue is fixed at ttnn side
// TEST_F(EmbeddingOpTest, EmbeddingBadLayout_BROKEN) {
//     using namespace ttml;

//     auto* device = &autograd::ctx().get_device();
//     uint32_t num_embeddings = 32;
//     uint32_t embedding_dim = 32;
//     auto weight_tensor = core::zeros(ttnn::Shape({1, 1, num_embeddings, embedding_dim}), device);
//     autograd::TensorPtr weight = autograd::create_tensor(weight_tensor);

//     uint32_t batch_size = 1;
//     uint32_t sentence_size = 32;
//     std::vector<uint32_t> input_data((size_t)batch_size * sentence_size);
//     std::iota(input_data.begin(), input_data.end(), 0U);
//     auto input_tensor =
//         core::from_vector<uint32_t>(input_data, ttnn::Shape({batch_size, 1, 1, sentence_size}), device);
//     autograd::TensorPtr input = autograd::create_tensor(input_tensor);

//     EXPECT_ANY_THROW(ops::embedding_op(input, weight));
// }
