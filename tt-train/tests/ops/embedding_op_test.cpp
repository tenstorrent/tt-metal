// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ops/embedding_op.hpp"

#include <gtest/gtest.h>

#include <core/ttnn_all_includes.hpp>

#include "autograd/auto_context.hpp"
#include "autograd/tensor.hpp"
#include "core/tt_tensor_utils.hpp"
#include "ops/losses.hpp"

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
    autograd::TensorPtr weight = autograd::create_tensor(weight_tensor);

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
    auto weight_tensor = ttml::core::zeros(ttnn::Shape({1, 1, num_embeddings, embedding_dim}), device);
    autograd::TensorPtr weight = autograd::create_tensor(weight_tensor);

    uint32_t batch_size = 1;
    uint32_t sentence_size = 13;
    std::vector<uint32_t> input_data((size_t)batch_size * sentence_size);
    std::iota(input_data.begin(), input_data.end(), 0U);
    auto input_tensor = ttml::core::from_vector<uint32_t, DataType::UINT32>(
        input_data, ttnn::Shape({batch_size, 1, 1, sentence_size}), device, Layout::ROW_MAJOR);
    autograd::TensorPtr input = autograd::create_tensor(input_tensor);

    EXPECT_NO_THROW(ops::embedding_op(input, weight));
}

// NEW TEST: This test was previously commented out because it would freeze.
// With the layout conversion fix in embedding_op.cpp, it should now work correctly.
TEST_F(EmbeddingOpTest, EmbeddingTileLayoutForward) {
    using namespace ttml;

    auto* device = &autograd::ctx().get_device();
    uint32_t num_embeddings = 32;
    uint32_t embedding_dim = 32;
    auto weight_tensor = core::zeros(ttnn::Shape({1, 1, num_embeddings, embedding_dim}), device);
    autograd::TensorPtr weight = autograd::create_tensor(weight_tensor);

    uint32_t batch_size = 1;
    uint32_t sentence_size = 32;
    std::vector<uint32_t> input_data((size_t)batch_size * sentence_size);
    std::iota(input_data.begin(), input_data.end(), 0U);

    // Create input with TILE layout (default when layout not specified)
    auto input_tensor = core::from_vector<uint32_t, ttnn::DataType::UINT32>(
        input_data, ttnn::Shape({batch_size, 1, 1, sentence_size}), device, ttnn::Layout::TILE);
    autograd::TensorPtr input = autograd::create_tensor(input_tensor);

    // Verify input is in TILE layout
    EXPECT_EQ(input->get_value().layout(), ttnn::Layout::TILE);

    // Forward pass should work with TILE layout input
    EXPECT_NO_THROW(ops::embedding_op(input, weight));
}

// NEW TEST: Test backward pass with TILE layout input
// This is the critical test that was failing before our fix
TEST_F(EmbeddingOpTest, EmbeddingTileLayoutBackward) {
    using namespace ttml;

    auto* device = &autograd::ctx().get_device();
    uint32_t num_embeddings = 32;
    uint32_t embedding_dim = 32;
    auto weight_tensor = core::zeros(ttnn::Shape({1, 1, num_embeddings, embedding_dim}), device);
    autograd::TensorPtr weight = autograd::create_tensor(weight_tensor);

    uint32_t batch_size = 1;
    uint32_t sentence_size = 32;
    std::vector<uint32_t> input_data((size_t)batch_size * sentence_size);
    std::iota(input_data.begin(), input_data.end(), 0U);

    // Create input with TILE layout (default when layout not specified)
    auto input_tensor = core::from_vector<uint32_t, ttnn::DataType::UINT32>(
        input_data, ttnn::Shape({batch_size, 1, 1, sentence_size}), device, ttnn::Layout::TILE);
    autograd::TensorPtr input = autograd::create_tensor(input_tensor);

    // Verify input is in TILE layout
    EXPECT_EQ(input->get_value().layout(), ttnn::Layout::TILE);

    autograd::TensorPtr embeddings = ops::embedding_op(input, weight);

    std::vector<float> target_vector((size_t)batch_size * sentence_size * embedding_dim);
    for (uint32_t i = 0; i < batch_size * sentence_size; i++) {
        for (uint32_t j = 0; j < embedding_dim; j++) {
            target_vector[embedding_dim * i + j] = static_cast<float>(i);
        }
    }
    auto target_tensor = autograd::create_tensor(
        core::from_vector(target_vector, ttnn::Shape({batch_size, 1, sentence_size, embedding_dim}), device));
    auto result = ops::mse_loss(embeddings, target_tensor);

    // Backward pass should work correctly with TILE layout input
    // This would have failed with "TT_FATAL: index_tensor.layout() == Layout::ROW_MAJOR" before the fix
    EXPECT_NO_THROW(result->backward());

    // Verify gradients are computed correctly
    auto weight_grad_tensor = weight->get_grad();
    EXPECT_TRUE(core::is_tensor_initialized(weight_grad_tensor));

    auto weight_grad_data = core::to_vector(weight_grad_tensor);
    for (uint32_t i = 0; i < num_embeddings; i++) {
        for (uint32_t j = 0; j < embedding_dim; j++) {
            EXPECT_NEAR(
                weight_grad_data[embedding_dim * i + j],
                -static_cast<float>(i) / sentence_size / embedding_dim / batch_size * 2.F,
                1e-2);
        }
    }
}

// NEW TEST: Compare gradients from TILE vs ROW_MAJOR layouts
// This ensures our layout conversion doesn't affect gradient correctness
TEST_F(EmbeddingOpTest, EmbeddingLayoutConsistency) {
    using namespace ttml;

    auto* device = &autograd::ctx().get_device();
    uint32_t num_embeddings = 32;
    uint32_t embedding_dim = 32;
    uint32_t batch_size = 1;
    uint32_t sentence_size = 32;

    std::vector<uint32_t> input_data((size_t)batch_size * sentence_size);
    std::iota(input_data.begin(), input_data.end(), 0U);

    // Test with ROW_MAJOR layout
    auto weight_row = autograd::create_tensor(core::zeros(ttnn::Shape({1, 1, num_embeddings, embedding_dim}), device));
    auto input_row = autograd::create_tensor(core::from_vector<uint32_t, ttnn::DataType::UINT32>(
        input_data, ttnn::Shape({batch_size, 1, 1, sentence_size}), device, ttnn::Layout::ROW_MAJOR));

    auto embeddings_row = ops::embedding_op(input_row, weight_row);
    std::vector<float> target_vector((size_t)batch_size * sentence_size * embedding_dim);
    for (uint32_t i = 0; i < batch_size * sentence_size; i++) {
        for (uint32_t j = 0; j < embedding_dim; j++) {
            target_vector[embedding_dim * i + j] = static_cast<float>(i);
        }
    }
    auto target_row = autograd::create_tensor(
        core::from_vector(target_vector, ttnn::Shape({batch_size, 1, sentence_size, embedding_dim}), device));
    auto loss_row = ops::mse_loss(embeddings_row, target_row);
    loss_row->backward();
    auto grad_row = core::to_vector(weight_row->get_grad());

    // Reset context for second test
    autograd::ctx().reset_graph();

    // Test with TILE layout
    auto weight_tile = autograd::create_tensor(core::zeros(ttnn::Shape({1, 1, num_embeddings, embedding_dim}), device));
    auto input_tile = autograd::create_tensor(core::from_vector<uint32_t, ttnn::DataType::UINT32>(
        input_data, ttnn::Shape({batch_size, 1, 1, sentence_size}), device, ttnn::Layout::TILE));

    auto embeddings_tile = ops::embedding_op(input_tile, weight_tile);
    auto target_tile = autograd::create_tensor(
        core::from_vector(target_vector, ttnn::Shape({batch_size, 1, sentence_size, embedding_dim}), device));
    auto loss_tile = ops::mse_loss(embeddings_tile, target_tile);
    loss_tile->backward();
    auto grad_tile = core::to_vector(weight_tile->get_grad());

    // Gradients should be identical regardless of input layout
    ASSERT_EQ(grad_row.size(), grad_tile.size());
    for (size_t i = 0; i < grad_row.size(); i++) {
        EXPECT_NEAR(grad_row[i], grad_tile[i], 1e-2);
    }
}
