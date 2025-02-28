// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <vector>

#include "autograd/auto_context.hpp"
#include "models/gpt2.hpp"
#include "models/mlp.hpp"
class MultiLayerPerceptronParametersTest : public ::testing::Test {
protected:
    void SetUp() override {
        ttml::autograd::ctx().open_device();
    }

    void TearDown() override {
        ttml::autograd::ctx().close_device();
    }
};

class TransformerConfigTest : public ::testing::Test {
protected:
    void SetUp() override {
        ttml::autograd::ctx().open_device();
    }

    void TearDown() override {
        ttml::autograd::ctx().close_device();
    }
};

TEST_F(MultiLayerPerceptronParametersTest, BasicReadWrite) {
    // Original configuration
    ttml::modules::MultiLayerPerceptronParameters original_config;
    original_config.input_features = 16;
    original_config.hidden_features = {32, 64, 32};
    original_config.output_features = 8;

    // Write to YAML
    YAML::Node yaml_node = ttml::models::mlp::write_config(original_config);

    // Read from YAML
    ttml::modules::MultiLayerPerceptronParameters read_config_result = ttml::models::mlp::read_config(yaml_node);

    // Assertions to verify correctness
    EXPECT_EQ(original_config.input_features, read_config_result.input_features);
    EXPECT_EQ(original_config.hidden_features, read_config_result.hidden_features);
    EXPECT_EQ(original_config.output_features, read_config_result.output_features);
}

TEST_F(MultiLayerPerceptronParametersTest, MissingFields) {
    // YAML configuration with missing 'hidden_features'
    YAML::Node yaml_node;
    yaml_node["input_features"] = 16;
    // yaml_node["hidden_features"] is intentionally omitted
    yaml_node["output_features"] = 8;

    EXPECT_THROW(
        {
            ttml::modules::MultiLayerPerceptronParameters read_config_result =
                ttml::models::mlp::read_config(yaml_node);
        },
        YAML::Exception);
}

// Test 3: Handling of Invalid Data Types in YAML Configuration
TEST_F(MultiLayerPerceptronParametersTest, InvalidDataTypes) {
    // YAML configuration with invalid data types
    YAML::Node yaml_node;
    yaml_node["input_features"] = "sixteen";        // Should be uint32_t
    yaml_node["hidden_features"] = "invalid_type";  // Should be std::vector<uint32_t>
    yaml_node["output_features"] = 8;

    EXPECT_THROW(
        {
            ttml::modules::MultiLayerPerceptronParameters read_config_result =
                ttml::models::mlp::read_config(yaml_node);
        },
        YAML::Exception);
}

TEST_F(TransformerConfigTest, BasicReadWrite) {
    // Original configuration
    ttml::models::gpt2::TransformerConfig original_config;
    original_config.num_heads = 8;
    original_config.embedding_dim = 512;
    original_config.dropout_prob = 0.1f;
    original_config.num_blocks = 6;
    original_config.vocab_size = 10000;
    original_config.max_sequence_length = 512;

    // Write to YAML
    YAML::Node yaml_node = ttml::models::gpt2::write_config(original_config);

    // Read from YAML
    auto read_config_result = ttml::models::gpt2::read_config(yaml_node);

    // Assertions to verify correctness
    EXPECT_EQ(original_config.num_heads, read_config_result.num_heads);
    EXPECT_EQ(original_config.embedding_dim, read_config_result.embedding_dim);
    EXPECT_FLOAT_EQ(original_config.dropout_prob, read_config_result.dropout_prob);
    EXPECT_EQ(original_config.num_blocks, read_config_result.num_blocks);
    EXPECT_EQ(original_config.vocab_size, read_config_result.vocab_size);
    EXPECT_EQ(original_config.max_sequence_length, read_config_result.max_sequence_length);
}

TEST_F(TransformerConfigTest, MissingFields) {
    // YAML configuration with missing 'dropout_prob'
    YAML::Node yaml_node;
    yaml_node["num_heads"] = 8;
    yaml_node["embedding_dim"] = 512;
    // yaml_node["dropout_prob"] is intentionally omitted
    yaml_node["num_blocks"] = 6;
    yaml_node["vocab_size"] = 10000;
    yaml_node["max_sequence_length"] = 512;

    EXPECT_THROW({ auto read_config_result = ttml::models::gpt2::read_config(yaml_node); }, YAML::Exception);
}

TEST_F(TransformerConfigTest, InvalidDataTypes) {
    // YAML configuration with invalid data types
    YAML::Node yaml_node;
    yaml_node["num_heads"] = "eight";                    // Should be uint32_t
    yaml_node["embedding_dim"] = "five hundred twelve";  // Should be uint32_t
    yaml_node["dropout_prob"] = "zero point one";        // Should be float
    yaml_node["num_blocks"] = 6;
    yaml_node["vocab_size"] = 10000;
    yaml_node["max_sequence_length"] = 512;

    EXPECT_THROW({ auto read_config_result = ttml::models::gpt2::read_config(yaml_node); }, YAML::Exception);
}
