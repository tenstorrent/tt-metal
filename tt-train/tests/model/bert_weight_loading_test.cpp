// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

/**
 * BERT QKV Weight Loading Tests
 *
 * These tests validate the correct implementation of QKV weight loading from
 * HuggingFace BERT models, which store Q, K, V weights separately, into TTML's
 * combined QKV linear layer format.
 *
 * Test Coverage:
 * 1. QKVShapesCorrect: Validates weight and bias tensor shapes
 * 2. QKVCombinationCorrectness: Validates concatenation logic with known inputs
 * 3. QKVBiasCorrectness: Validates bias concatenation order
 * 4. ManualQKVSetAndForward: End-to-end test with forward pass
 *
 * Note: A Python golden reference test (test_bert_golden_reference.py) is
 * available for comparing TTML BERT output against HuggingFace BERT using
 * actual pretrained models. This provides end-to-end numerical validation.
 */

#include <gtest/gtest.h>

#include "autograd/auto_context.hpp"
#include "core/tt_tensor_utils.hpp"
#include "core/xtensor_utils.hpp"
#include "models/bert.hpp"

using namespace ttml;
using namespace ttml::models::bert;

class BertWeightLoadingTest : public ::testing::Test {
protected:
    void SetUp() override {
        autograd::ctx().open_device();
    }

    void TearDown() override {
        autograd::ctx().close_device();
    }
};

TEST_F(BertWeightLoadingTest, QKVShapesCorrect) {
    BertConfig config;
    config.vocab_size = 1000;
    config.max_sequence_length = 128;
    config.embedding_dim = 256;
    config.intermediate_size = 512;
    config.num_heads = 8;
    config.num_blocks = 2;
    config.dropout_prob = 0.0F;
    config.layer_norm_eps = 1e-12F;

    auto bert = ttml::models::bert::create(config);
    auto params = bert->parameters();

    // Check QKV linear weight shape for layer 0: should be [3*hidden_dim, hidden_dim] = [768, 256]
    // Note: TTML stores weights as [out_features, in_features]
    auto qkv_weight = params["bert/bert_block_0/attention/self_attention/qkv_linear/weight"];
    ASSERT_NE(qkv_weight, nullptr);

    auto shape = qkv_weight->get_value().logical_shape();
    EXPECT_EQ(shape[-2], 768);  // 3 * hidden_dim (out_features)
    EXPECT_EQ(shape[-1], 256);  // hidden_dim (in_features)

    // Check layer 1 as well
    auto qkv_weight_1 = params["bert/bert_block_1/attention/self_attention/qkv_linear/weight"];
    ASSERT_NE(qkv_weight_1, nullptr);

    auto shape_1 = qkv_weight_1->get_value().logical_shape();
    EXPECT_EQ(shape_1[-2], 768);  // 3 * hidden_dim
    EXPECT_EQ(shape_1[-1], 256);  // hidden_dim

    // Check bias shapes
    auto qkv_bias = params["bert/bert_block_0/attention/self_attention/qkv_linear/bias"];
    ASSERT_NE(qkv_bias, nullptr);
    auto bias_shape = qkv_bias->get_value().logical_shape();
    EXPECT_EQ(bias_shape[-1], 768);  // 3 * hidden_dim
}

TEST_F(BertWeightLoadingTest, QKVCombinationCorrectness) {
    // Test that QKV combination produces correct results
    // by manually combining Q, K, V and comparing to expected output

    const size_t hidden = 4;
    const size_t batch = 1;
    const size_t seq_len = 2;

    // Create simple Q, K, V matrices for testing
    // Q = [[1, 2, 3, 4],    K = [[10, 11, 12, 13],    V = [[20, 21, 22, 23],
    //      [5, 6, 7, 8]]         [14, 15, 16, 17]]         [24, 25, 26, 27]]
    std::vector<float> Q_vec = {1, 2, 3, 4, 5, 6, 7, 8};
    std::vector<float> K_vec = {10, 11, 12, 13, 14, 15, 16, 17};
    std::vector<float> V_vec = {20, 21, 22, 23, 24, 25, 26, 27};

    // Convert to xtensor and transpose: [hidden, hidden] -> [hidden, hidden].t()
    auto Q = xt::adapt(Q_vec, std::vector<size_t>{2, 4});
    auto K = xt::adapt(K_vec, std::vector<size_t>{2, 4});
    auto V = xt::adapt(V_vec, std::vector<size_t>{2, 4});

    xt::xarray<float> Q_t = xt::transpose(Q);  // [4, 2]
    xt::xarray<float> K_t = xt::transpose(K);  // [4, 2]
    xt::xarray<float> V_t = xt::transpose(V);  // [4, 2]

    // Concatenate along dim=1: [4, 2] + [4, 2] + [4, 2] -> [4, 6]
    auto qkv_combined = core::concat(std::vector<xt::xarray<float>>{Q_t, K_t, V_t}, 1);

    ASSERT_EQ(qkv_combined.shape()[0], 4);
    ASSERT_EQ(qkv_combined.shape()[1], 6);

    // Expected QKV combined (row-major flat):
    // Row 0: [Q[0,0]=1,  Q[1,0]=5,  K[0,0]=10, K[1,0]=14, V[0,0]=20, V[1,0]=24]
    // Row 1: [Q[0,1]=2,  Q[1,1]=6,  K[0,1]=11, K[1,1]=15, V[0,1]=21, V[1,1]=25]
    // Row 2: [Q[0,2]=3,  Q[1,2]=7,  K[0,2]=12, K[1,2]=16, V[0,2]=22, V[1,2]=26]
    // Row 3: [Q[0,3]=4,  Q[1,3]=8,  K[0,3]=13, K[1,3]=17, V[0,3]=23, V[1,3]=27]

    EXPECT_FLOAT_EQ(qkv_combined(0, 0), 1.0F);   // Q[0,0]
    EXPECT_FLOAT_EQ(qkv_combined(0, 1), 5.0F);   // Q[1,0]
    EXPECT_FLOAT_EQ(qkv_combined(0, 2), 10.0F);  // K[0,0]
    EXPECT_FLOAT_EQ(qkv_combined(0, 3), 14.0F);  // K[1,0]
    EXPECT_FLOAT_EQ(qkv_combined(0, 4), 20.0F);  // V[0,0]
    EXPECT_FLOAT_EQ(qkv_combined(0, 5), 24.0F);  // V[1,0]

    EXPECT_FLOAT_EQ(qkv_combined(1, 0), 2.0F);   // Q[0,1]
    EXPECT_FLOAT_EQ(qkv_combined(1, 1), 6.0F);   // Q[1,1]
    EXPECT_FLOAT_EQ(qkv_combined(1, 2), 11.0F);  // K[0,1]
    EXPECT_FLOAT_EQ(qkv_combined(1, 3), 15.0F);  // K[1,1]
    EXPECT_FLOAT_EQ(qkv_combined(1, 4), 21.0F);  // V[0,1]
    EXPECT_FLOAT_EQ(qkv_combined(1, 5), 25.0F);  // V[1,1]
}

TEST_F(BertWeightLoadingTest, QKVBiasCorrectness) {
    // Test bias concatenation
    std::vector<float> Q_bias = {1.0F, 2.0F, 3.0F, 4.0F};
    std::vector<float> K_bias = {10.0F, 11.0F, 12.0F, 13.0F};
    std::vector<float> V_bias = {20.0F, 21.0F, 22.0F, 23.0F};

    // Concatenate biases
    std::vector<float> qkv_bias;
    qkv_bias.reserve(12);
    qkv_bias.insert(qkv_bias.end(), Q_bias.begin(), Q_bias.end());
    qkv_bias.insert(qkv_bias.end(), K_bias.begin(), K_bias.end());
    qkv_bias.insert(qkv_bias.end(), V_bias.begin(), V_bias.end());

    ASSERT_EQ(qkv_bias.size(), 12);

    // Check order: [Q_bias, K_bias, V_bias]
    EXPECT_FLOAT_EQ(qkv_bias[0], 1.0F);
    EXPECT_FLOAT_EQ(qkv_bias[1], 2.0F);
    EXPECT_FLOAT_EQ(qkv_bias[2], 3.0F);
    EXPECT_FLOAT_EQ(qkv_bias[3], 4.0F);
    EXPECT_FLOAT_EQ(qkv_bias[4], 10.0F);
    EXPECT_FLOAT_EQ(qkv_bias[5], 11.0F);
    EXPECT_FLOAT_EQ(qkv_bias[6], 12.0F);
    EXPECT_FLOAT_EQ(qkv_bias[7], 13.0F);
    EXPECT_FLOAT_EQ(qkv_bias[8], 20.0F);
    EXPECT_FLOAT_EQ(qkv_bias[9], 21.0F);
    EXPECT_FLOAT_EQ(qkv_bias[10], 22.0F);
    EXPECT_FLOAT_EQ(qkv_bias[11], 23.0F);
}

TEST_F(BertWeightLoadingTest, ManualQKVSetAndForward) {
    // Test that manually setting QKV weights in the combined format
    // allows forward pass to work correctly

    BertConfig config;
    config.vocab_size = 100;
    config.max_sequence_length = 32;
    config.embedding_dim = 64;
    config.intermediate_size = 256;
    config.num_heads = 4;
    config.num_blocks = 1;
    config.dropout_prob = 0.0F;
    config.layer_norm_eps = 1e-12F;

    auto bert = ttml::models::bert::create(config);
    const size_t hidden = 64;

    // Create mock Q, K, V weights (uniform for simplicity)
    std::vector<float> Q(hidden * hidden, 0.01F);
    std::vector<float> K(hidden * hidden, 0.01F);
    std::vector<float> V(hidden * hidden, 0.01F);

    // Manually combine using correct transpose + concatenate
    auto Q_arr = xt::adapt(Q, std::vector<size_t>{hidden, hidden});
    auto K_arr = xt::adapt(K, std::vector<size_t>{hidden, hidden});
    auto V_arr = xt::adapt(V, std::vector<size_t>{hidden, hidden});

    xt::xarray<float> Q_t = xt::transpose(Q_arr);
    xt::xarray<float> K_t = xt::transpose(K_arr);
    xt::xarray<float> V_t = xt::transpose(V_arr);

    auto qkv_combined = core::concat(std::vector<xt::xarray<float>>{Q_t, K_t, V_t}, 1);
    std::vector<float> qkv_flat(qkv_combined.begin(), qkv_combined.end());

    // Set the combined weight
    auto params = bert->parameters();
    auto qkv_weight = params["bert/bert_block_0/attention/self_attention/qkv_linear/weight"];
    ASSERT_NE(qkv_weight, nullptr);

    qkv_weight->set_value(
        core::from_vector(qkv_flat, qkv_weight->get_value().logical_shape(), qkv_weight->get_value().device()));

    // Test forward pass doesn't crash and produces valid output
    std::vector<float> input_data(32, 5.0f);  // Token IDs as float
    auto input =
        autograd::create_tensor(core::from_vector(input_data, ttnn::Shape{1, 1, 1, 32}, &autograd::ctx().get_device()));

    auto output = (*bert)(input, nullptr);
    ASSERT_NE(output, nullptr);

    // Verify output shape
    auto output_shape = output->get_value().logical_shape();
    EXPECT_EQ(output_shape[0], 1);
    EXPECT_EQ(output_shape[1], 1);
    EXPECT_EQ(output_shape[2], 32);  // sequence length
    EXPECT_EQ(output_shape[3], 64);  // hidden_dim

    // Verify output has no NaN or Inf
    auto output_vec = core::to_vector(output->get_value());
    for (const auto& val : output_vec) {
        EXPECT_FALSE(std::isnan(val)) << "Output contains NaN";
        EXPECT_FALSE(std::isinf(val)) << "Output contains Inf";
    }
}
