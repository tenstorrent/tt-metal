// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "autograd/auto_context.hpp"
#include "core/tt_tensor_utils.hpp"
#include "models/bert.hpp"

using namespace ttml;

int main() {
    // Initialize device context
    autograd::ctx().open_device();

    // Create BERT configuration
    models::bert::BertConfig config;
    config.vocab_size = 30522;
    config.max_sequence_length = 128;  // Smaller for example
    config.embedding_dim = 768;
    config.intermediate_size = 3072;
    config.num_heads = 12;
    config.num_blocks = 6;  // Smaller for example
    config.dropout_prob = 0.1F;
    config.use_token_type_embeddings = true;
    config.use_pooler = false;  // Set true for classification tasks

    // Create BERT model
    auto bert_model = models::bert::create(config);

    // Example input (batch_size=1, sequence_length=128 to match config)
    std::vector<uint32_t> input_ids_data(128, 0);  // Initialize with padding
    input_ids_data[0] = 101;                       // [CLS]
    input_ids_data[1] = 2023;                      // this
    input_ids_data[2] = 2003;                      // is
    input_ids_data[3] = 1037;                      // a
    input_ids_data[4] = 2742;                      // test
    input_ids_data[5] = 102;                       // [SEP]
    // Rest are padding (0s)

    std::vector<uint32_t> token_type_ids_data(128, 0);  // All sentence A

    // Attention mask: 1 for real tokens, 0 for padding
    std::vector<float> attention_mask_data(128, 0.0F);
    for (int i = 0; i < 6; ++i) {
        attention_mask_data[i] = 1.0F;  // Mark first 6 tokens as real
    }

    // Create tensors with proper types
    auto device = &autograd::ctx().get_device();

    // Convert input_ids to float for embedding lookup (framework expectation)
    std::vector<float> input_ids_float(input_ids_data.begin(), input_ids_data.end());
    auto input_ids = autograd::create_tensor(core::from_vector<float, ttnn::DataType::BFLOAT16>(
        input_ids_float, ttnn::Shape{1, 1, 1, 128}, device, ttnn::Layout::ROW_MAJOR));

    // Token type IDs also as float
    std::vector<float> token_type_ids_float(token_type_ids_data.begin(), token_type_ids_data.end());
    auto token_type_ids = autograd::create_tensor(core::from_vector<float, ttnn::DataType::BFLOAT16>(
        token_type_ids_float, ttnn::Shape{1, 1, 1, 128}, device, ttnn::Layout::ROW_MAJOR));

    auto attention_mask = autograd::create_tensor(
        core::from_vector<float, ttnn::DataType::BFLOAT16>(attention_mask_data, ttnn::Shape{1, 1, 1, 128}, device));

    // Run forward pass
    bert_model->eval();  // Set to eval mode for inference (train mode for training)
    auto output = (*bert_model)(input_ids, attention_mask, token_type_ids);

    // Print output shape
    auto output_shape = output->get_shape();
    fmt::print(
        "BERT output shape: [{}, {}, {}, {}]\n", output_shape[0], output_shape[1], output_shape[2], output_shape[3]);

    // For classification tasks with pooler enabled:
    if (config.use_pooler) {
        fmt::print(
            "Pooled output shape (for classification): [{}, {}, {}, {}]\n",
            output_shape[0],
            output_shape[1],
            1,
            output_shape[3]);

        // Add classification head example:
        // auto num_classes = 2;  // Binary classification
        // auto classification_head = std::make_shared<modules::LinearLayer>(config.embedding_dim, num_classes);
        // auto logits = (*classification_head)(output);
        // fmt::print("Classification logits shape: [{}, {}, {}, {}]\n", ...);
    }

    autograd::ctx().close_device();

    return 0;
}
