// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <core/ttnn_all_includes.hpp>

#include "autograd/tensor.hpp"
#include "core/tt_tensor_utils.hpp"
#include "modules/positional_embeddings.hpp"

class PositionalEmbeddingTest : public ::testing::Test {
protected:
    void SetUp() override {
        ttml::autograd::ctx().open_device();
    }

    void TearDown() override {
        ttml::autograd::ctx().close_device();
    }
};

TEST_F(PositionalEmbeddingTest, NonTrainableEmbedding) {
    using namespace ttml;

    auto* device = &autograd::ctx().get_device();

    uint32_t batch_size = 2;
    uint32_t sentence_size = 2;
    uint32_t embedding_dim = 4;

    auto x =
        autograd::create_tensor(core::zeros(core::create_shape({batch_size, 1, sentence_size, embedding_dim}), device));
    auto pos_emb = modules::PositionalEmbedding(modules::PositionalEmbeddingConfig{
        .embedding_dim = embedding_dim,
        .sequence_length = sentence_size,
        .dropout_prob = 0.F,
        .use_dropout_seed_per_device = true});
    auto y = pos_emb(x);

    auto y_vector = core::to_vector(y->get_value());
    std::vector<float> target_vector{
        0.0000F,
        1.0000F,
        0.0000F,
        1.0000F,
        0.8415F,
        0.5403F,
        0.0100F,
        0.9999F,
        0.0000F,
        1.0000F,
        0.0000F,
        1.0000F,
        0.8415F,
        0.5403F,
        0.0100F,
        0.9999F,

    };

    EXPECT_EQ(y_vector.size(), target_vector.size());
    const float eps = 4e-3F;
    for (size_t i = 0; i < y_vector.size(); i++) {
        EXPECT_NEAR(y_vector[i], target_vector[i], eps);
    }
}
