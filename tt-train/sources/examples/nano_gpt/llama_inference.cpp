// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <CLI/CLI.hpp>
#include <chrono>
#include <core/ttnn_all_includes.hpp>
#include <cstdint>
#include <vector>

#include "autograd/auto_context.hpp"
#include "autograd/tensor.hpp"
#include "core/tt_tensor_utils.hpp"
#include "models/common/transformer_common.hpp"
#include "models/llama.hpp"
#include "utils.hpp"

using ttml::autograd::TensorPtr;

// Configuration for inference
struct InferenceConfig {
    uint32_t max_new_tokens = 50;
    uint32_t seed = 5489;
    std::string prompt = "1,2,3,4,5";
    std::string model_path;
    bool use_kv_cache = true;
};

// Create a causal attention mask for autoregressive generation
TensorPtr create_causal_mask(
    ttnn::distributed::MeshDevice* device, const uint32_t query_seq_len, const uint32_t prompt_len = 0) {
    // Mask shape: [padded_seq_len, padded_whole_seq_len] - query_len x key_len
    const uint32_t whole_seq_len = prompt_len + query_seq_len;
    const uint32_t padded_query_len = round_up_to_tile(query_seq_len);
    const uint32_t padded_whole_len = round_up_to_tile(whole_seq_len);

    std::vector<float> mask_data(padded_query_len * padded_whole_len, 0.0f);

    for (uint32_t i = 0; i < query_seq_len; ++i) {
        for (uint32_t j = 0; j <= i + prompt_len; ++j) {
            mask_data[i * padded_whole_len + j] = 1.0f;
        }
    }

    const auto shape = ttnn::Shape({1, 1, padded_query_len, padded_whole_len});
    const auto mask_tensor = ttml::core::from_vector(mask_data, shape, device);

    return ttml::autograd::create_tensor(mask_tensor);
}

// Sample next token using greedy decoding (argmax)
const uint32_t sample_token(const TensorPtr& logits, const int position) {
    if (position == 0) {
        throw std::invalid_argument("position must be at least 1");
    }
    const auto logits_tensor = logits->get_value();
    const auto logits_host = logits_tensor.to_vector<float>();

    const auto shape = logits_tensor.logical_shape();
    const uint32_t vocab_size = shape[-1];
    const size_t last_token_offset = (position - 1) * vocab_size;
    const auto start_it = logits_host.begin() + last_token_offset;
    const auto end_it = start_it + vocab_size;
    const auto max_it = std::max_element(start_it, end_it);

    return static_cast<uint32_t>(std::distance(start_it, max_it));
}

// Create tensor from token IDs with padding to nearest multiple of 32
TensorPtr tokens_to_tensor(const std::vector<uint32_t>& tokens, ttnn::distributed::MeshDevice* device) {
    const uint32_t actual_len = tokens.size();
    const uint32_t padded_len = round_up_to_tile(actual_len);

    // Pad tokens with zeros to reach padded length
    std::vector<uint32_t> padded_tokens(padded_len, 0);
    std::copy(tokens.begin(), tokens.end(), padded_tokens.begin());

    const auto shape = ttnn::Shape({1, 1, 1, padded_len});
    auto tokens_tensor = ttml::core::from_vector<uint32_t, ttnn::DataType::UINT32>(
        padded_tokens, shape, device, ttnn::Layout::ROW_MAJOR);

    return ttml::autograd::create_tensor(tokens_tensor);
}

void run_inference_with_kv_cache(
    std::shared_ptr<ttml::models::llama::Llama>& model,
    const std::vector<uint32_t>& prompt_tokens,
    const InferenceConfig& inference_config,
    const uint32_t vocab_size,
    const uint32_t max_seq_len,
    const ttml::models::llama::LlamaConfig& llama_config,
    ttnn::distributed::MeshDevice* device) {
    fmt::print("Running Inference with KV Cache\n");

    // Create KV cache using config
    const uint32_t batch_size = 1;
    const uint32_t head_dim = llama_config.embedding_dim / llama_config.num_heads;
    const ttml::models::common::transformer::KvCacheConfig kv_cache_config(
        llama_config.num_blocks, batch_size, llama_config.num_groups, llama_config.max_sequence_length, head_dim);
    auto kv_cache = std::make_shared<ttml::models::common::transformer::KvCache>(kv_cache_config);

    // Reset KV cache for new sequence
    kv_cache->reset();

    std::vector<uint32_t> generated_tokens = prompt_tokens;
    generated_tokens.reserve(
        std::min(prompt_tokens.size() + inference_config.max_new_tokens + 10U, static_cast<size_t>(max_seq_len)));

    const uint32_t prompt_len = prompt_tokens.size();

    fmt::print("Prompt tokens: [");
    for (uint32_t i = 0; i < std::min(10U, prompt_len); ++i) {
        fmt::print("{}", prompt_tokens[i]);
        if (i < std::min(10U, prompt_len) - 1)
            fmt::print(", ");
    }
    if (prompt_len > 10U)
        fmt::print(", ...");
    fmt::print("]\n");
    fmt::print("Prompt length: {}\n\n", prompt_len);

    const auto start_timer = std::chrono::high_resolution_clock::now();

    const uint32_t num_steps = std::min(uint32_t(inference_config.max_new_tokens), max_seq_len - prompt_len);

    // Generate tokens one by one
    for (uint32_t step = 0; step < num_steps; ++step) {
        // For first step (prefill): use all prompt tokens
        // For subsequent steps (decode): use only the last generated token
        std::vector<uint32_t> input_tokens;
        uint32_t processed_tokens = 0;

        if (kv_cache->get_cache_position() == 0) {
            // Prefill: process entire prompt
            input_tokens = generated_tokens;
        } else {
            // Decode: process only last token
            input_tokens = {generated_tokens.back()};
            processed_tokens = generated_tokens.size() - 1;
        }

        auto token_tensor = tokens_to_tensor(input_tokens, device);

        // Create causal mask
        // For prefill: query_len = prompt_len, key_len = prompt_len
        // For decode: query_len = 1, key_len = cache_position + 1
        const auto mask = create_causal_mask(device, input_tokens.size(), processed_tokens);
        const uint32_t new_tokens = input_tokens.size();
        const auto logits = (*model)(token_tensor, mask, kv_cache, /* new_tokens_to_process */ new_tokens);

        // Sample next token from the last position
        const uint32_t next_token = sample_token(logits, input_tokens.size());
        generated_tokens.push_back(next_token);

        if (step % 10 == 0) {
            fmt::print("Step {}: token={}, cache_position={}\n", step, next_token, kv_cache->get_cache_position());
        }
    }

    const auto end_timer = std::chrono::high_resolution_clock::now();
    const auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_timer - start_timer).count();

    kv_cache->reset();
    // ============================================================================
    // Summary
    // ============================================================================
    uint32_t new_tokens = generated_tokens.size() - prompt_len;
    fmt::print("\n=== GENERATION SUMMARY ===\n");
    fmt::print("Total tokens generated: {}\n", generated_tokens.size());
    fmt::print("  Prompt: {} tokens\n", prompt_len);
    fmt::print("  New: {} tokens\n", new_tokens);
    fmt::print("\nGenerated token sequence: [");
    for (size_t i = 0; i < std::min(size_t(20), generated_tokens.size()); ++i) {
        fmt::print("{}", generated_tokens[i]);
        if (i < std::min(size_t(20), generated_tokens.size()) - 1)
            fmt::print(", ");
    }
    if (generated_tokens.size() > 20)
        fmt::print(", ...");
    fmt::print("]\n");

    fmt::print("\nTotal time: {} ms\n", total_duration);
    fmt::print("Average time per token: {} ms\n", new_tokens > 0 ? total_duration / new_tokens : 0);
    fmt::print("Cache entries: {}\n", device->num_program_cache_entries());
    fmt::print("{}\n", std::string(80, '='));
}

void run_inference_no_cache(
    std::shared_ptr<ttml::models::llama::Llama>& model,
    const std::vector<uint32_t>& prompt_tokens,
    const InferenceConfig& inference_config,
    const uint32_t vocab_size,
    const uint32_t max_seq_len,
    ttnn::distributed::MeshDevice* device) {
    fmt::print("\n{}\n", std::string(80, '='));
    fmt::print("Running Inference WITHOUT KV Cache (Full Sequence Forward)\n");
    fmt::print("{}\n\n", std::string(80, '='));

    std::vector<uint32_t> generated_tokens = prompt_tokens;
    generated_tokens.reserve(
        std::min(prompt_tokens.size() + inference_config.max_new_tokens + 10U, static_cast<size_t>(max_seq_len)));
    const uint32_t prompt_len = prompt_tokens.size();

    fmt::print("Prompt tokens: [");
    for (uint32_t i = 0; i < std::min(10U, prompt_len); ++i) {
        fmt::print("{}", prompt_tokens[i]);
        if (i < std::min(10U, prompt_len) - 1)
            fmt::print(", ");
    }
    if (prompt_len > 10U)
        fmt::print(", ...");
    fmt::print("]\n");
    fmt::print("Prompt length: {}\n\n", prompt_len);

    const auto start_timer = std::chrono::high_resolution_clock::now();

    const uint32_t num_steps = std::min(uint32_t(inference_config.max_new_tokens), max_seq_len - prompt_len);
    // Generate tokens one by one, running full forward pass each time
    for (uint32_t step = 0; step < num_steps; ++step) {
        // Create tensor with ALL tokens generated so far (grows each iteration)
        auto current_seq = tokens_to_tensor(generated_tokens, device);

        // Create causal mask for current sequence length
        const auto mask = create_causal_mask(device, generated_tokens.size(), 0);
        const auto logits = (*model)(current_seq, mask);

        // Sample next token from the last actual token position
        const uint32_t next_token = sample_token(logits, generated_tokens.size());
        generated_tokens.push_back(next_token);

        if (step % 10 == 0) {
            fmt::print("Step {}: token={}, seq_len={}\n", step, next_token, generated_tokens.size());
        }
    }

    const auto end_timer = std::chrono::high_resolution_clock::now();
    const auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_timer - start_timer).count();

    // ============================================================================
    // Summary
    // ============================================================================
    fmt::print("\n=== GENERATION SUMMARY ===\n");
    fmt::print("Total tokens generated: {}\n", generated_tokens.size());
    fmt::print("  Prompt: {} tokens\n", prompt_len);
    fmt::print("  New: {} tokens\n", generated_tokens.size() - prompt_len);
    fmt::print("\nGenerated token sequence: [");
    for (size_t i = 0; i < std::min(size_t(20), generated_tokens.size()); ++i) {
        fmt::print("{}", generated_tokens[i]);
        if (i < std::min(size_t(20), generated_tokens.size()) - 1)
            fmt::print(", ");
    }
    if (generated_tokens.size() > 20)
        fmt::print(", ...");
    fmt::print("]\n");

    uint32_t new_tokens = generated_tokens.size() - prompt_len;
    fmt::print("\nTotal time: {} ms\n", total_duration);
    fmt::print("Average time per token: {} ms\n", new_tokens > 0 ? total_duration / new_tokens : 0);
    fmt::print("Cache entries: {}\n", device->num_program_cache_entries());
    fmt::print("{}\n", std::string(80, '='));
}

int main(int argc, char** argv) {
    const auto start_timer = std::chrono::high_resolution_clock::now();

    // Parse command line arguments
    CLI::App app{"LLaMA Inference"};
    argv = app.ensure_utf8(argv);

    InferenceConfig inference_config;

    app.add_option("--max-tokens", inference_config.max_new_tokens, "Maximum new tokens to generate")
        ->default_val(inference_config.max_new_tokens);
    app.add_option("--seed", inference_config.seed, "Random seed")->default_val(inference_config.seed);
    app.add_option("--prompt", inference_config.prompt, "Input prompt (comma-separated token IDs)")
        ->default_val(inference_config.prompt);
    app.add_option("-p,--model-path", inference_config.model_path, "Path to model weights (.msgpack)");
    app.add_flag("--use-kv-cache,!--no-kv-cache", inference_config.use_kv_cache, "Use KV cache (default: true)")
        ->default_val(true);

    CLI11_PARSE(app, argc, argv);

    // Use default LLaMA config path
    std::string model_config_path = "./configs/model_configs/tinyllama.yaml";

    fmt::print("\n{}\n", std::string(80, '='));
    fmt::print("LLaMA Inference {}\n", inference_config.use_kv_cache ? "with KV Cache" : "without KV Cache");
    fmt::print("{}\n\n", std::string(80, '='));

    // Set random seed
    ttml::autograd::ctx().set_seed(inference_config.seed);
    fmt::print("Random seed: {}\n", inference_config.seed);

    // Load model configuration
    fmt::print("Loading model configuration from: {}\n", model_config_path);
    YAML::Node yaml_config = YAML::LoadFile(model_config_path);
    auto llama_config = ttml::models::llama::read_config(yaml_config["transformer_config"]);

    fmt::print("\n1. Model Configuration:\n");
    fmt::print("   Num heads: {}\n", llama_config.num_heads);
    fmt::print("   Num groups: {}\n", llama_config.num_groups);
    fmt::print("   Embedding dim: {}\n", llama_config.embedding_dim);
    fmt::print("   Num blocks: {}\n", llama_config.num_blocks);
    fmt::print("   Vocab size: {}\n", llama_config.vocab_size);
    fmt::print("   Max seq length: {}\n", llama_config.max_sequence_length);
    fmt::print("   Inference mode (KV cache): {}\n", inference_config.use_kv_cache ? "Enabled" : "Disabled");
    if (!inference_config.use_kv_cache) {
        fmt::print("   Method: Full sequence forward pass each step\n");
    }
    fmt::print("\n");

    // Initialize device
    fmt::print("2. Initializing TT device...\n");
    tt::tt_metal::distributed::MeshShape mesh_shape{1, 1};
    std::vector<int> device_ids{};
    initialize_device(mesh_shape, device_ids);
    auto* device = &ttml::autograd::ctx().get_device();
    fmt::print("   Device initialized\n");
    fmt::print("   Num devices: {}\n\n", device->num_devices());

    // Create model
    fmt::print("3. Creating LLaMA model...\n");
    auto model = ttml::models::llama::create(llama_config);

    // Load weights if provided
    if (!inference_config.model_path.empty()) {
        if (!std::filesystem::exists(inference_config.model_path)) {
            fmt::print("   Error: Model path does not exist: {}\n", inference_config.model_path);
            ttml::autograd::ctx().close_device();
            return -1;
        }
        fmt::print("   Loading model weights from {}\n", inference_config.model_path);
        load_model_parameters(inference_config.model_path, model, "llama");
        fmt::print("   Model loaded successfully\n");
    } else {
        fmt::print("   Using randomly initialized weights (no model path provided)\n");
    }

    // Set model to eval mode and disable gradient computation for inference
    model->eval();
    ttml::autograd::ctx().set_gradient_mode(ttml::autograd::GradMode::DISABLED);
    fmt::print("   Gradient mode disabled for inference\n\n");

    // Parse prompt tokens (comma-separated numbers)
    fmt::print("4. Parsing prompt: '{}'\n", inference_config.prompt);
    std::vector<uint32_t> prompt_tokens;
    std::stringstream ss(inference_config.prompt);
    uint32_t token;

    while (ss >> token) {
        prompt_tokens.push_back(token);
        if (ss.peek() == ',')
            ss.ignore();
    }

    if (prompt_tokens.size() > llama_config.max_sequence_length) {
        fmt::print(
            "   Warning: prompt too long ({}), truncating to {}\n",
            prompt_tokens.size(),
            llama_config.max_sequence_length);
        prompt_tokens.resize(llama_config.max_sequence_length);
    }

    fmt::print("   Parsed {} tokens\n\n", prompt_tokens.size());

    // Run inference
    fmt::print("5. Running inference (greedy decoding)...\n");
    fmt::print("   Max new tokens: {}\n\n", inference_config.max_new_tokens);

    if (inference_config.use_kv_cache) {
        run_inference_with_kv_cache(
            model,
            prompt_tokens,
            inference_config,
            llama_config.vocab_size,
            llama_config.max_sequence_length,
            llama_config,
            device);
    } else {
        run_inference_no_cache(
            model, prompt_tokens, inference_config, llama_config.vocab_size, llama_config.max_sequence_length, device);
    }

    // Cleanup
    fmt::print("\n6. Cleaning up...\n");
    const auto end_timer = std::chrono::high_resolution_clock::now();
    const auto total_duration = std::chrono::duration_cast<std::chrono::seconds>(end_timer - start_timer).count();

    fmt::print("   Total execution time: {} s\n", total_duration);
    fmt::print("   Closing device...\n");

    ttml::autograd::ctx().close_device();

    fmt::print("\nInference complete!\n");

    return 0;
}
