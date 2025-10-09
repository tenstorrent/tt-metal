// Debug test to isolate PolymorphicContainer segfault
#include <gtest/gtest.h>

#include <memory>

#include "autograd/auto_context.hpp"
#include "autograd/tensor.hpp"
#include "core/tt_tensor_utils.hpp"
#include "models/base_transformer.hpp"
#include "models/bert.hpp"

namespace ttml::models::bert::tests {

class BertDebugTest : public ::testing::Test {
protected:
    void SetUp() override {
        autograd::ctx().open_device();
    }

    void TearDown() override {
        autograd::ctx().close_device();
    }
};

// Simplified test to debug the segfault
TEST_F(BertDebugTest, DebugPolymorphicCall) {
    auto* device = &autograd::ctx().get_device();

    // Create BERT with minimal config
    BertConfig config;
    config.vocab_size = 100;          // Smaller vocab
    config.max_sequence_length = 32;  // Shorter sequence
    config.embedding_dim = 64;        // Smaller embeddings
    config.intermediate_size = 128;
    config.num_heads = 4;
    config.num_blocks = 1;                     // Just one block
    config.use_token_type_embeddings = false;  // Disable to simplify
    config.use_pooler = false;

    fmt::print("Creating BERT model...\n");
    auto bert = std::make_shared<Bert>(config);

    fmt::print("Casting to BaseTransformer...\n");
    std::shared_ptr<BaseTransformer> transformer = bert;

    // Create valid input token IDs
    std::vector<float> input_data(32);
    for (size_t i = 0; i < 32; ++i) {
        input_data[i] = static_cast<float>(i % config.vocab_size);
    }

    fmt::print("Creating input tensor with data: {}, {}, ..., {}\n", input_data[0], input_data[1], input_data[31]);

    auto input = autograd::create_tensor(core::from_vector(input_data, ttnn::Shape{1, 1, 1, 32}, device));

    fmt::print("Input tensor shape: {}\n", input->get_shape());
    fmt::print("Input tensor dtype check...\n");

    // Try to access dtype to see if tensor is valid
    try {
        auto dtype = input->get_value().dtype();
        fmt::print("Input tensor dtype: OK\n");
    } catch (const std::exception& e) {
        fmt::print("Input tensor dtype error: {}\n", e.what());
    }

    fmt::print("Calling BERT through polymorphic interface...\n");

    // Call with no mask first to simplify
    try {
        auto output = (*transformer)(input, nullptr);
        fmt::print("Forward pass succeeded!\n");
        fmt::print("Output shape: {}\n", output->get_shape());
    } catch (const std::exception& e) {
        fmt::print("Forward pass failed with exception: {}\n", e.what());
    }
}

// Test calling directly vs through interface
TEST_F(BertDebugTest, DirectVsPolymorphic) {
    auto* device = &autograd::ctx().get_device();

    BertConfig config;
    config.vocab_size = 100;
    config.max_sequence_length = 32;
    config.embedding_dim = 64;
    config.intermediate_size = 128;
    config.num_heads = 4;
    config.num_blocks = 1;
    config.use_token_type_embeddings = false;
    config.use_pooler = false;

    auto bert = std::make_shared<Bert>(config);

    std::vector<float> input_data(32);
    for (size_t i = 0; i < 32; ++i) {
        input_data[i] = static_cast<float>(i % config.vocab_size);
    }

    auto input = autograd::create_tensor(core::from_vector(input_data, ttnn::Shape{1, 1, 1, 32}, device));

    // Test 1: Direct call to forward()
    fmt::print("Test 1: Direct forward() call...\n");
    try {
        auto output1 = bert->forward(input, nullptr, nullptr);
        fmt::print("  Success! Output shape: {}\n", output1->get_shape());
    } catch (const std::exception& e) {
        fmt::print("  Failed: {}\n", e.what());
    }

    // Test 2: Through three-arg operator()
    fmt::print("Test 2: Three-arg operator() call...\n");
    try {
        auto output2 = (*bert)(input, nullptr, nullptr);
        fmt::print("  Success! Output shape: {}\n", output2->get_shape());
    } catch (const std::exception& e) {
        fmt::print("  Failed: {}\n", e.what());
    }

    // Test 3: Through BaseTransformer interface
    fmt::print("Test 3: BaseTransformer interface call...\n");
    std::shared_ptr<BaseTransformer> transformer = bert;
    try {
        auto output3 = (*transformer)(input, nullptr);
        fmt::print("  Success! Output shape: {}\n", output3->get_shape());
    } catch (const std::exception& e) {
        fmt::print("  Failed: {}\n", e.what());
    }
}

}  // namespace ttml::models::bert::tests
