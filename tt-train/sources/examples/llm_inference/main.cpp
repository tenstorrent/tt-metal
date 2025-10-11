// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <CLI/CLI.hpp>
#include <core/ttnn_all_includes.hpp>
#include <cstdint>
#include <iostream>
#include <string>
#include <variant>

#include "autograd/auto_context.hpp"
#include "autograd/tensor.hpp"
#include "models/gpt2.hpp"
#include "models/llama.hpp"
#include "ops/sampling_op.hpp"
#include "tokenizers/bpe_tokenizer.hpp"
#include "ttnn_fixed/trivial_ttnn_ops.hpp"

using MeshShape = tt::tt_metal::distributed::MeshShape;
using Model = std::shared_ptr<ttml::models::BaseTransformer>;

struct EvalConfig {
    float repetition_penalty = 1.0F;
    float temperature = 1.0F;
    int top_k = -1;
    float top_p = 1.0F;
};

EvalConfig parse_eval_config(const YAML::Node &yaml_config) {
    EvalConfig config;
    if (!yaml_config["eval_config"]) {
        return config;
    }
    auto eval_config = yaml_config["eval_config"];
    config.repetition_penalty = eval_config["repetition_penalty"].as<float>(config.repetition_penalty);
    config.temperature = eval_config["temperature"].as<float>(config.temperature);
    config.top_k = eval_config["top_k"].as<int>(config.top_k);
    config.top_p = eval_config["top_p"].as<float>(config.top_p);
    return config;
}

struct TrainingConfig {
    std::string project_name;
    std::string model_type;  // one of "gpt2", "llama"
    uint32_t seed = 5489U;

    std::string tokenizer_type = "char";
    std::string tokenizer_path = "data/gpt2-tokenizer.json";
    ttml::models::gpt2::TransformerConfig transformer_config;
};

TrainingConfig parse_config(const YAML::Node &yaml_config) {
    TrainingConfig config;
    auto training_config = yaml_config["training_config"];
    config.project_name = training_config["project_name"].as<std::string>("tt_train_nano_gpt");
    config.model_type = training_config["model_type"].as<std::string>();
    config.seed = training_config["seed"].as<uint32_t>();
    config.tokenizer_type = training_config["tokenizer_type"].as<std::string>(config.tokenizer_type);
    config.tokenizer_path = training_config["tokenizer_path"].as<std::string>(config.tokenizer_path);

    config.transformer_config = ttml::models::gpt2::read_config(training_config["transformer_config"]);

    return config;
}

int main(int argc, char **argv) {
    CLI::App app{"GPT2S Inference Example"};
    argv = app.ensure_utf8(argv);

    std::string model_path = "configs/training_shakespeare_gpt2s";
    std::string prompt = "";
    std::string safetensors_path = "";
    float temperature = 0.0F;
    uint32_t tokens_to_generate = 256U;
    uint32_t seed = ttml::autograd::ctx().get_generator()();

    app.add_option("-c,--config", model_path, "Path to the LLM config")->default_val(model_path);
    app.add_option("--safetensors", safetensors_path, "Path to the model safetensors file")
        ->default_val(safetensors_path);
    app.add_option("--num_tokens", tokens_to_generate, "Number of tokens to generate")->default_val(tokens_to_generate);
    CLI11_PARSE(app, argc, argv);

    // Set up device and context
    ttml::autograd::ctx().open_device(MeshShape{1, 1}, {});
    auto *device = &ttml::autograd::ctx().get_device();

    // Load model
    auto yaml_config = YAML::LoadFile(model_path + ".yaml");
    auto training_config = parse_config(yaml_config);
    auto eval_config = parse_eval_config(yaml_config);
    auto transformer_config = training_config.transformer_config;

    // Load tokenizer
    auto tokenizer = std::make_shared<ttml::tokenizers::BPETokenizer>(training_config.tokenizer_path);

    // Set eval parameters
    temperature = eval_config.temperature;

    ttml::autograd::ctx().set_seed(training_config.seed);
    fmt::print("Setting random seed to: {}\n", training_config.seed);

    fmt::print("Overriding vocab size to be divisible by 32\n");
    // this is workaround for tensor parallel case, we need to have vocab size divisible by 32 per device

    uint32_t original_vocab_size = tokenizer->get_vocab_size();
    uint32_t padded_vocab_size = round_up_to_tile(original_vocab_size, 32U);

    transformer_config.vocab_size = padded_vocab_size;
    uint32_t max_sequence_length = transformer_config.max_sequence_length;

    Model model = ttml::models::gpt2::create(transformer_config);

    model->load_from_safetensors(safetensors_path);
    model->eval();

    // Build mask (causal) for attention

    std::vector<float> mask;
    mask.reserve(static_cast<size_t>(max_sequence_length * max_sequence_length));
    for (uint32_t i = 0; i < max_sequence_length; ++i) {
        for (uint32_t j = 0; j < max_sequence_length; ++j) {
            mask.push_back(i >= j ? 1.0F : 0.0F);
        }
    }

    auto causal_mask_tensor = ttml::autograd::create_tensor(
        ttml::core::from_vector(mask, ttnn::Shape({1, 1, max_sequence_length, max_sequence_length}), device));

    ttml::autograd::TensorPtr logits_padding_mask_autograd = nullptr;

    if (original_vocab_size != padded_vocab_size) {
        // Create a large negative mask for out-of-vocab logits
        auto vocab_mask = std::vector<float>(padded_vocab_size - original_vocab_size, 1e4F);

        auto argmax_zeros =
            ttml::core::zeros(ttnn::Shape({1U, 1U, 1U, original_vocab_size}), device, tt::tt_metal::DataType::BFLOAT16);
        auto argmax_nonzero = ttml::core::from_vector<float, tt::tt_metal::DataType::BFLOAT16>(
            vocab_mask, ttnn::Shape({1U, 1U, 1U, padded_vocab_size - original_vocab_size}), device, ttnn::Layout::TILE);

        auto logits_padding_mask_vector = std::vector<ttnn::Tensor>{argmax_zeros, argmax_nonzero};
        auto logits_padding_mask = ttnn::concat(logits_padding_mask_vector, 3);
        logits_padding_mask_autograd = ttml::autograd::create_tensor(logits_padding_mask);
    }

    while (true) {
        // Prepare prompt
        std::cout << "Enter a prompt: ";
        std::getline(std::cin, prompt);
        if (prompt.empty())
            prompt = "\n";
        auto prompt_tokens = tokenizer->encode(prompt);

        // Pad prompt tokens
        std::vector<uint32_t> prompt_tokens_padded(max_sequence_length, 0);
        uint32_t start_idx =
            prompt_tokens.size() > max_sequence_length ? prompt_tokens.size() - max_sequence_length : 0;
        for (uint32_t i = start_idx; i < prompt_tokens.size(); ++i) {
            prompt_tokens_padded[i - start_idx] = prompt_tokens[i];
        }

        // Create input tensor
        auto prompt_tensor = ttml::autograd::create_tensor(ttml::core::from_vector<uint32_t, ttnn::DataType::UINT32>(
            prompt_tokens_padded, ttnn::Shape({1U, 1U, 1U, max_sequence_length}), device, ttnn::Layout::ROW_MAJOR));

        // Generate tokens
        std::cout << "Generated text:\n";
        std::cout << "*******************\n";
        std::cout << prompt;

        for (uint32_t token_idx = 0; token_idx < tokens_to_generate; ++token_idx) {
            uint32_t start_idx = 0;
            if (prompt_tokens.size() > max_sequence_length) {
                start_idx = static_cast<uint32_t>(prompt_tokens.size() - max_sequence_length);
            }
            // Fill padded array
            std::fill(prompt_tokens_padded.begin(), prompt_tokens_padded.end(), 0);
            std::copy(prompt_tokens.begin() + start_idx, prompt_tokens.end(), prompt_tokens_padded.begin());

            auto prompt_tokens_padded_size = static_cast<uint32_t>(prompt_tokens_padded.size());
            auto prompt_tensor =
                ttml::autograd::create_tensor(ttml::core::from_vector<uint32_t, ttnn::DataType::UINT32>(
                    prompt_tokens_padded,
                    ttnn::Shape({1U, 1U, 1U, prompt_tokens_padded_size}),
                    device,
                    ttnn::Layout::ROW_MAJOR));

            // Forward pass
            auto output = (*model)(prompt_tensor, causal_mask_tensor);

            // Sample next token
            auto next_token_tensor = ttml::ops::sample_op(output, temperature, seed, logits_padding_mask_autograd);

            uint32_t predicted_token_idx =
                (prompt_tokens.size() > max_sequence_length) ? (max_sequence_length - 1U) : (prompt_tokens.size() - 1U);

            auto next_token_vector = ttml::core::to_vector<uint32_t>(next_token_tensor->get_value());
            uint32_t next_token_id = next_token_vector[predicted_token_idx];

            // Append and print
            prompt_tokens.push_back(next_token_id);
            fmt::print("{}", tokenizer->decode({next_token_id}));
            std::cout.flush();

            ttml::autograd::ctx().reset_graph();
        }

        std::cout << "\n*******************\n";
    }

    return 0;
}
