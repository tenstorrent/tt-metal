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
#include "models/llama.hpp"
#include "utils.hpp"

using ttml::autograd::TensorPtr;

// Configuration for inference
struct InferenceConfig {
    uint32_t max_new_tokens = 50;      // Number of new tokens to generate
    uint32_t seed = 5489;              // Random seed for reproducibility
    std::string prompt = "1,2,3,4,5";  // Comma-separated token IDs
    std::string model_path;            // Path to pretrained model weights (.msgpack)
    bool use_kv_cache = true;          // Whether to use KV cache
};

// Create a causal attention mask for autoregressive generation
TensorPtr create_causal_mask(uint32_t seq_len, uint32_t max_seq_len, ttnn::distributed::MeshDevice* device) {
    std::vector<float> mask_data(max_seq_len * max_seq_len, 0.0f);

    for (uint32_t i = 0; i < seq_len; ++i) {
        for (uint32_t j = 0; j <= i; ++j) {
            mask_data[i * max_seq_len + j] = 1.0f;
        }
    }

    auto shape = ttnn::Shape({1, 1, max_seq_len, max_seq_len});
    auto mask_tensor = ttml::core::from_vector(mask_data, shape, device);

    return ttml::autograd::create_tensor(mask_tensor);
}

// Sample next token using greedy decoding (argmax)
uint32_t sample_token(const TensorPtr& logits, int position) {
    auto logits_tensor = logits->get_value();
    auto logits_host = logits_tensor.to_vector<float>();

    auto shape = logits_tensor.logical_shape();
    uint32_t vocab_size = shape[-1];
    size_t last_token_offset = (position - 1) * vocab_size;

    // Find token with highest logit value
    uint32_t max_idx = 0;
    float max_val = logits_host[last_token_offset];

    for (uint32_t i = 1; i < vocab_size; ++i) {
        float val = logits_host[last_token_offset + i];
        if (val > max_val) {
            max_val = val;
            max_idx = i;
        }
    }

    return max_idx;
}

// Create tensor from token IDs
TensorPtr tokens_to_tensor(
    const std::vector<uint32_t>& tokens, uint32_t max_seq_len, ttnn::distributed::MeshDevice* device) {
    std::vector<uint32_t> padded_tokens(max_seq_len, 0);
    for (size_t i = 0; i < tokens.size() && i < max_seq_len; ++i) {
        padded_tokens[i] = tokens[i];
    }

    auto shape = ttnn::Shape({1, 1, 1, max_seq_len});
    auto tokens_tensor = ttml::core::from_vector<uint32_t, ttnn::DataType::UINT32>(
        padded_tokens, shape, device, ttnn::Layout::ROW_MAJOR);

    return ttml::autograd::create_tensor(tokens_tensor);
}

void run_inference_with_kv_cache(
    std::shared_ptr<ttml::models::llama::Llama>& model,
    const std::vector<uint32_t>& prompt_tokens,
    const InferenceConfig& inference_config,
    uint32_t vocab_size,
    uint32_t max_seq_len,
    ttnn::distributed::MeshDevice* device) {
    fmt::print("\n{}\n", std::string(80, '='));
    fmt::print("Running Inference with KV Cache\n");
    fmt::print("{}\n\n", std::string(80, '='));

    // Reset KV cache for new sequence
    model->reset_cache();

    std::vector<uint32_t> generated_tokens = prompt_tokens;
    uint32_t prompt_len = prompt_tokens.size();

    fmt::print("Prompt tokens: [");
    for (size_t i = 0; i < std::min(size_t(10), prompt_tokens.size()); ++i) {
        fmt::print("{}", prompt_tokens[i]);
        if (i < std::min(size_t(10), prompt_tokens.size()) - 1)
            fmt::print(", ");
    }
    if (prompt_tokens.size() > 10)
        fmt::print(", ...");
    fmt::print("]\n");
    fmt::print("Prompt length: {}\n", prompt_len);
    fmt::print("Cache position at start: {}\n\n", model->get_cache_position());

    // ============================================================================
    // Phase 1: PREFILL - Process entire prompt at once
    // ============================================================================
    fmt::print("=== PREFILL PHASE ===\n");
    auto start_prefill = std::chrono::high_resolution_clock::now();

    auto prompt_tensor = tokens_to_tensor(prompt_tokens, max_seq_len, device);
    auto prefill_mask = create_causal_mask(prompt_len, max_seq_len, device);

    // Forward pass - fills KV cache with all prompt tokens
    auto logits = (*model)(prompt_tensor, prefill_mask);

    // Update cache position to reflect number of tokens processed
    model->set_cache_position(prompt_len);

    auto end_prefill = std::chrono::high_resolution_clock::now();
    auto prefill_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_prefill - start_prefill).count();

    fmt::print("Prefill completed in {} ms\n", prefill_duration);
    fmt::print("Cache position after prefill: {}\n", model->get_cache_position());

    // Sample first new token (greedy)
    uint32_t next_token = sample_token(logits, prompt_len);
    generated_tokens.push_back(next_token);

    fmt::print("First generated token: {}\n\n", next_token);

    // ============================================================================
    // Phase 2: DECODE - Generate tokens one by one using KV cache
    // ============================================================================
    fmt::print("=== DECODE PHASE ===\n");
    fmt::print("Generating {} more tokens...\n\n", inference_config.max_new_tokens - 1);

    auto start_decode = std::chrono::high_resolution_clock::now();
    std::vector<uint32_t> decode_tokens;
    decode_tokens.push_back(next_token);

    for (uint32_t step = 0; step < inference_config.max_new_tokens - 1; ++step) {
        std::vector<uint32_t> single_token = {next_token};
        auto token_tensor = tokens_to_tensor(single_token, max_seq_len, device);

        // Create decode mask: position 0 can attend to all cached positions
        uint32_t current_cache_pos = model->get_cache_position();
        std::vector<float> decode_mask_data(max_seq_len * max_seq_len, 0.0f);
        for (uint32_t j = 0; j <= current_cache_pos && j < max_seq_len; ++j) {
            decode_mask_data[j] = 1.0f;
        }
        auto decode_mask_shape = ttnn::Shape({1, 1, max_seq_len, max_seq_len});
        auto decode_mask_tensor = ttml::core::from_vector(decode_mask_data, decode_mask_shape, device);
        auto decode_mask = ttml::autograd::create_tensor(decode_mask_tensor);

        logits = (*model)(token_tensor, decode_mask);
        next_token = sample_token(logits, 1);

        generated_tokens.push_back(next_token);
        decode_tokens.push_back(next_token);

        if (step % 10 == 0) {
            fmt::print("Step {}: token={}, cache_pos={}\n", step, next_token, model->get_cache_position());
        }

        if (generated_tokens.size() >= max_seq_len) {
            fmt::print("\nReached max sequence length\n");
            break;
        }
    }

    auto end_decode = std::chrono::high_resolution_clock::now();
    auto decode_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_decode - start_decode).count();

    fmt::print("\nDecode completed in {} ms\n", decode_duration);
    fmt::print(
        "Average time per token: {} ms\n", decode_tokens.size() > 0 ? decode_duration / decode_tokens.size() : 0);
    fmt::print("Final cache position: {}\n\n", model->get_cache_position());

    // ============================================================================
    // Summary
    // ============================================================================
    fmt::print("=== GENERATION SUMMARY ===\n");
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

    fmt::print("\nTotal time: {} ms\n", prefill_duration + decode_duration);
    fmt::print("Cache entries: {}\n", device->num_program_cache_entries());
    fmt::print("{}\n", std::string(80, '='));
}

void run_inference_no_cache(
    std::shared_ptr<ttml::models::llama::Llama>& model,
    const std::vector<uint32_t>& prompt_tokens,
    const InferenceConfig& inference_config,
    uint32_t vocab_size,
    uint32_t max_seq_len,
    ttnn::distributed::MeshDevice* device) {
    fmt::print("\n{}\n", std::string(80, '='));
    fmt::print("Running Inference WITHOUT KV Cache (Full Sequence Forward)\n");
    fmt::print("{}\n\n", std::string(80, '='));

    std::vector<uint32_t> generated_tokens = prompt_tokens;
    uint32_t prompt_len = prompt_tokens.size();

    fmt::print("Prompt tokens: [");
    for (size_t i = 0; i < std::min(size_t(10), prompt_tokens.size()); ++i) {
        fmt::print("{}", prompt_tokens[i]);
        if (i < std::min(size_t(10), prompt_tokens.size()) - 1)
            fmt::print(", ");
    }
    if (prompt_tokens.size() > 10)
        fmt::print(", ...");
    fmt::print("]\n");
    fmt::print("Prompt length: {}\n\n", prompt_len);

    auto start_timer = std::chrono::high_resolution_clock::now();

    // Generate tokens one by one, running full forward pass each time
    for (uint32_t step = 0; step < inference_config.max_new_tokens; ++step) {
        // Create tensor with ALL tokens generated so far (grows each iteration)
        auto current_seq = tokens_to_tensor(generated_tokens, max_seq_len, device);

        // Create causal mask for current sequence length
        auto mask = create_causal_mask(generated_tokens.size(), max_seq_len, device);

        // Full forward pass through entire sequence
        auto logits = (*model)(current_seq, mask);

        // Sample next token from the last actual token position
        uint32_t next_token = sample_token(logits, generated_tokens.size());
        generated_tokens.push_back(next_token);

        if (step % 10 == 0) {
            fmt::print("Step {}: token={}, seq_len={}\n", step, next_token, generated_tokens.size());
        }

        if (generated_tokens.size() >= max_seq_len) {
            fmt::print("\nReached max sequence length\n");
            break;
        }
    }

    auto end_timer = std::chrono::high_resolution_clock::now();
    auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_timer - start_timer).count();

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
    auto start_timer = std::chrono::high_resolution_clock::now();

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

    const char* tt_metal_home = std::getenv("TT_METAL_HOME");
    TT_FATAL(tt_metal_home != nullptr, "TT_METAL_HOME environment variable is not set");

    // Use default LLaMA config path
    std::string model_config_path =
        std::string(tt_metal_home) + "/tt-train/configs/model_configs/llama3_gpt2s_size.yaml";

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

    // Set inference mode based on command-line flag
    llama_config.inference = inference_config.use_kv_cache;

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
    fmt::print(
        "3. Creating LLaMA model{}...\n", inference_config.use_kv_cache ? " with KV cache" : " without KV cache");
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

    // Set model to eval mode
    model->eval();
    fmt::print("   Model set to eval mode\n\n");

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
            model, prompt_tokens, inference_config, llama_config.vocab_size, llama_config.max_sequence_length, device);
    } else {
        run_inference_no_cache(
            model, prompt_tokens, inference_config, llama_config.vocab_size, llama_config.max_sequence_length, device);
    }

    // Cleanup
    fmt::print("\n6. Cleaning up...\n");
    auto end_timer = std::chrono::high_resolution_clock::now();
    auto total_duration = std::chrono::duration_cast<std::chrono::seconds>(end_timer - start_timer).count();

    fmt::print("   Total execution time: {} s\n", total_duration);
    fmt::print("   Closing device...\n");

    ttml::autograd::ctx().close_device();

    fmt::print("\nInference complete!\n");

    return 0;
}
