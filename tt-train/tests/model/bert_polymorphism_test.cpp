// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <memory>

#include "autograd/auto_context.hpp"
#include "autograd/tensor.hpp"
#include "core/tt_tensor_utils.hpp"
#include "models/base_transformer.hpp"
#include "models/bert.hpp"

namespace ttml::models::bert::tests {

class BertPolymorphismTest : public ::testing::Test {
protected:
    void SetUp() override {
        autograd::ctx().open_device();
    }

    void TearDown() override {
        autograd::ctx().close_device();
    }
};

// Test that BERT properly implements BaseTransformer interface
TEST_F(BertPolymorphismTest, BertIsBaseTransformer) {
    // Create a BERT model
    BertConfig config;
    config.vocab_size = 1000;
    config.max_sequence_length = 128;
    config.embedding_dim = 256;
    config.intermediate_size = 512;
    config.num_heads = 8;
    config.num_blocks = 2;
    config.use_pooler = false;

    auto bert = std::make_shared<Bert>(config);

    // Verify it can be used as BaseTransformer
    std::shared_ptr<BaseTransformer> transformer = bert;
    ASSERT_NE(transformer, nullptr);

    // The dynamic cast should succeed
    auto* bert_ptr = dynamic_cast<Bert*>(transformer.get());
    ASSERT_NE(bert_ptr, nullptr);
}

// Test the BaseTransformer operator() interface
TEST_F(BertPolymorphismTest, BaseTransformerOperatorCall) {
    auto* device = &autograd::ctx().get_device();

    BertConfig config;
    config.vocab_size = 1000;
    config.max_sequence_length = 128;
    config.embedding_dim = 256;
    config.intermediate_size = 512;
    config.num_heads = 8;
    config.num_blocks = 2;
    config.use_pooler = false;

    auto bert = std::make_shared<Bert>(config);

    // Use it through BaseTransformer interface
    std::shared_ptr<BaseTransformer> transformer = bert;

    // Create input tensors
    std::vector<float> input_ids_data(128);
    for (size_t i = 0; i < 128; ++i) {
        input_ids_data[i] = static_cast<float>(i % config.vocab_size);
    }

    auto input_ids = autograd::create_tensor(core::from_vector(input_ids_data, ttnn::Shape{1, 1, 1, 128}, device));

    // Create attention mask
    std::vector<float> mask_data(128, 1.0f);
    // Last 20 tokens are padding
    for (size_t i = 108; i < 128; ++i) {
        mask_data[i] = 0.0f;
    }
    auto attention_mask = autograd::create_tensor(core::from_vector(mask_data, ttnn::Shape{1, 1, 1, 128}, device));

    // Call through BaseTransformer interface
    auto output = (*transformer)(input_ids, attention_mask);

    // Verify output shape
    EXPECT_EQ(output->get_shape()[0], 1);    // batch
    EXPECT_EQ(output->get_shape()[1], 1);    // channels
    EXPECT_EQ(output->get_shape()[2], 128);  // seq_len
    EXPECT_EQ(output->get_shape()[3], 256);  // embedding_dim
}

// Test the BERT-specific forward() method
TEST_F(BertPolymorphismTest, BertSpecificForward) {
    auto* device = &autograd::ctx().get_device();

    BertConfig config;
    config.vocab_size = 1000;
    config.max_sequence_length = 128;
    config.embedding_dim = 256;
    config.intermediate_size = 512;
    config.num_heads = 8;
    config.num_blocks = 2;
    config.use_token_type_embeddings = true;
    config.use_pooler = false;

    auto bert = create(config);

    // Create input tensors
    std::vector<float> input_ids_data(128);
    std::vector<float> token_type_data(128);
    for (size_t i = 0; i < 128; ++i) {
        input_ids_data[i] = static_cast<float>(i % config.vocab_size);
        token_type_data[i] = (i < 64) ? 0.0f : 1.0f;  // First half is sentence A, second is B
    }

    auto input_ids = autograd::create_tensor(core::from_vector(input_ids_data, ttnn::Shape{1, 1, 1, 128}, device));
    auto token_type_ids =
        autograd::create_tensor(core::from_vector(token_type_data, ttnn::Shape{1, 1, 1, 128}, device));

    // Create attention mask
    std::vector<float> mask_data(128, 1.0f);
    for (size_t i = 108; i < 128; ++i) {
        mask_data[i] = 0.0f;  // Last 20 are padding
    }
    auto attention_mask = autograd::create_tensor(core::from_vector(mask_data, ttnn::Shape{1, 1, 1, 128}, device));

    // Call the BERT-specific forward method
    auto output = bert->forward(input_ids, attention_mask, token_type_ids);

    // Verify output shape
    EXPECT_EQ(output->get_shape()[0], 1);    // batch
    EXPECT_EQ(output->get_shape()[1], 1);    // channels
    EXPECT_EQ(output->get_shape()[2], 128);  // seq_len
    EXPECT_EQ(output->get_shape()[3], 256);  // embedding_dim
}

// Test compatibility with polymorphic container
TEST_F(BertPolymorphismTest, PolymorphicContainer) {
    // Create a container of BaseTransformer pointers
    std::vector<std::shared_ptr<BaseTransformer>> models;

    // Add BERT model
    BertConfig bert_config;
    bert_config.vocab_size = 1000;
    bert_config.max_sequence_length = 128;
    bert_config.embedding_dim = 256;
    bert_config.intermediate_size = 512;
    bert_config.num_heads = 8;
    bert_config.num_blocks = 2;

    models.push_back(std::make_shared<Bert>(bert_config));

    // Process through polymorphic interface
    auto* device = &autograd::ctx().get_device();

    std::vector<float> input_data(128);
    for (size_t i = 0; i < 128; ++i) {
        input_data[i] = static_cast<float>(i % bert_config.vocab_size);
    }
    auto input = autograd::create_tensor(core::from_vector(input_data, ttnn::Shape{1, 1, 1, 128}, device));

    for (const auto& model : models) {
        auto output = (*model)(input, nullptr);
        // Each model processes the input according to its implementation
        ASSERT_NE(output, nullptr);
    }
}

// Test backward compatibility with three-argument operator()
TEST_F(BertPolymorphismTest, BackwardCompatibleOperator) {
    auto* device = &autograd::ctx().get_device();

    BertConfig config;
    config.vocab_size = 1000;
    config.max_sequence_length = 64;
    config.embedding_dim = 128;
    config.intermediate_size = 256;
    config.num_heads = 4;
    config.num_blocks = 1;
    config.dropout_prob = 0.0F;  // Disable dropout for deterministic comparison
    config.use_token_type_embeddings = true;

    auto bert = create(config);

    // Create minimal tensors
    std::vector<float> input_data(64, 101.0f);  // [CLS] token ID
    auto input_ids = autograd::create_tensor(core::from_vector(input_data, ttnn::Shape{1, 1, 1, 64}, device));

    std::vector<float> mask_data(64, 1.0f);
    auto attention_mask = autograd::create_tensor(core::from_vector(mask_data, ttnn::Shape{1, 1, 1, 64}, device));

    std::vector<float> type_data(64, 0.0f);
    auto token_type_ids = autograd::create_tensor(core::from_vector(type_data, ttnn::Shape{1, 1, 1, 64}, device));

    // Test three-argument operator() still works
    auto output = (*bert)(input_ids, attention_mask, token_type_ids);
    ASSERT_NE(output, nullptr);

    // Compare with forward() - should give same result
    auto output2 = bert->forward(input_ids, attention_mask, token_type_ids);

    auto out_data1 = core::to_vector(output->get_value());
    auto out_data2 = core::to_vector(output2->get_value());

    ASSERT_EQ(out_data1.size(), out_data2.size());
    for (size_t i = 0; i < out_data1.size(); ++i) {
        EXPECT_FLOAT_EQ(out_data1[i], out_data2[i]);
    }
}

// Test with pooler enabled through polymorphic interface
TEST_F(BertPolymorphismTest, PolymorphicWithPooler) {
    auto* device = &autograd::ctx().get_device();

    BertConfig config;
    config.vocab_size = 1000;
    config.max_sequence_length = 64;
    config.embedding_dim = 128;
    config.intermediate_size = 256;
    config.num_heads = 4;
    config.num_blocks = 1;
    config.use_pooler = true;  // Enable pooler

    auto bert = create(config);
    std::shared_ptr<BaseTransformer> transformer = bert;

    // Create input
    std::vector<float> input_data(64);
    input_data[0] = 101.0f;  // [CLS] token
    for (size_t i = 1; i < 64; ++i) {
        input_data[i] = static_cast<float>(100 + (i % 100));
    }

    auto input_ids = autograd::create_tensor(core::from_vector(input_data, ttnn::Shape{1, 1, 1, 64}, device));

    // Call through polymorphic interface
    auto output = (*transformer)(input_ids, nullptr);

    // With pooler enabled, output should be [batch, 1, 1, embedding_dim]
    EXPECT_EQ(output->get_shape()[0], 1);    // batch
    EXPECT_EQ(output->get_shape()[1], 1);    // channels
    EXPECT_EQ(output->get_shape()[2], 1);    // pooled to single token
    EXPECT_EQ(output->get_shape()[3], 128);  // embedding_dim

    // Verify we can query if pooler is enabled
    EXPECT_TRUE(bert->is_pooler_enabled());
}

// Test error handling for mismatched shapes
TEST_F(BertPolymorphismTest, ErrorHandlingMismatchedShapes) {
    auto* device = &autograd::ctx().get_device();

    BertConfig config;
    config.vocab_size = 1000;
    config.max_sequence_length = 128;
    config.embedding_dim = 256;
    config.num_heads = 8;
    config.num_blocks = 1;
    config.use_token_type_embeddings = true;

    auto bert = create(config);

    // Create mismatched input tensors
    auto input_ids =
        autograd::create_tensor(core::from_vector(std::vector<float>(128, 0.0f), ttnn::Shape{1, 1, 1, 128}, device));

    // Token type IDs with wrong shape
    auto token_type_ids = autograd::create_tensor(
        core::from_vector(std::vector<float>(64, 0.0f), ttnn::Shape{1, 1, 1, 64}, device));  // Wrong shape!

    // This should throw an error
    EXPECT_THROW([&]() { std::ignore = bert->forward(input_ids, nullptr, token_type_ids); }(), std::logic_error);
}

// Test gradient flow through polymorphic interface
TEST_F(BertPolymorphismTest, GradientFlowPolymorphic) {
    auto* device = &autograd::ctx().get_device();

    BertConfig config;
    config.vocab_size = 100;
    config.max_sequence_length = 32;
    config.embedding_dim = 64;
    config.intermediate_size = 256;  // Must be valid multiple (4x embedding_dim)
    config.num_heads = 4;
    config.num_blocks = 1;
    config.dropout_prob = 0.0F;  // Disable dropout for deterministic testing
    config.use_pooler = false;

    auto bert = create(config);
    std::shared_ptr<BaseTransformer> transformer = bert;

    // Create input with requires_grad
    std::vector<float> input_data(32);
    for (size_t i = 0; i < 32; ++i) {
        input_data[i] = static_cast<float>(i % config.vocab_size);
    }

    auto input_ids = autograd::create_tensor(core::from_vector(input_data, ttnn::Shape{1, 1, 1, 32}, device));

    // Forward through polymorphic interface
    auto output = (*transformer)(input_ids, nullptr);

    // Set gradient on output
    output->set_grad(core::ones_like(output->get_value()));

    // Backward pass should work
    output->backward();

    // Check that parameters have gradients
    auto params = bert->parameters();
    bool has_gradients = false;
    for (const auto& [name, param] : params) {
        if (core::is_tensor_initialized(param->get_grad())) {
            has_gradients = true;
            break;
        }
    }
    EXPECT_TRUE(has_gradients);
}

// Test config access through base and derived classes
TEST_F(BertPolymorphismTest, ConfigAccess) {
    BertConfig config;
    config.vocab_size = 5000;
    config.max_sequence_length = 256;
    config.embedding_dim = 512;
    config.num_heads = 8;
    config.num_blocks = 6;

    auto bert = create(config);

    // Access config through BERT interface
    const auto& retrieved_config = bert->get_config();
    EXPECT_EQ(retrieved_config.vocab_size, 5000);
    EXPECT_EQ(retrieved_config.max_sequence_length, 256);
    EXPECT_EQ(retrieved_config.embedding_dim, 512);
    EXPECT_EQ(retrieved_config.num_heads, 8);
    EXPECT_EQ(retrieved_config.num_blocks, 6);

    // Verify polymorphic usage still works
    std::shared_ptr<BaseTransformer> transformer = bert;
    ASSERT_NE(transformer, nullptr);
}

}  // namespace ttml::models::bert::tests
