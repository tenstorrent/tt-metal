// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <CLI/CLI.hpp>
#include <chrono>
#include <core/ttnn_all_includes.hpp>
#include <csignal>
#include <cstdint>
#include <ttnn/tensor/tensor.hpp>
#include <wandbcpp.hpp>

#include "autograd/tensor.hpp"
#include "core/tt_tensor_utils.hpp"
#include "datasets/dataloader.hpp"
#include "datasets/in_memory_token_dataset.hpp"
#include "datasets/utils.hpp"
#include "models.hpp"
#include "ops/binary_ops.hpp"
#include "ops/losses.hpp"
#include "optimizers/adamw.hpp"
#include "optimizers/sgd.hpp"
#include "tokenizers/char_tokenizer.hpp"
#include "ttnn_fixed/trivial_ttnn_ops.hpp"
#include "utils.hpp"
/* WANDB BLocks this signal.
 Control+C didn't work.
*/
void signal_handler(int signum) {
    std::cout << "\nInterrupt signal (" << signum << ") received.\n";
    wandbcpp::finish();
    exit(signum);
}

using ttml::autograd::TensorPtr;

using DatasetSample = std::pair<std::span<const uint32_t>, std::span<const uint32_t>>;
// tokens, targets, mask, positions
using BatchType = std::tuple<TensorPtr, TensorPtr, TensorPtr, TensorPtr>;
using DataLoader = ttml::datasets::DataLoader<
    ttml::datasets::InMemoryTokenDataset,
    std::function<BatchType(std::vector<DatasetSample> &&samples)>,
    BatchType>;

struct DemoConfig {
    // training
    uint32_t batch_size = 64;
    uint32_t sequence_length = 256;
    uint32_t num_epochs = 1;
    uint32_t max_steps = 5000;
    float dropout_prob = 0.2F;
    // model
    uint32_t num_heads = 6;
    uint32_t embedding_dim = 384;
    uint32_t num_blocks = 6;
    // optimizer
    float learning_rate = 3e-4F;
    float weight_decay = 1e-2F;
};
const DemoConfig config;

uint32_t sample(std::span<const float> log_softmax) {
    auto probabilities_vector = std::vector<float>(log_softmax.size());
    std::transform(log_softmax.begin(), log_softmax.end(), probabilities_vector.begin(), [](float value) {
        return std::exp(value);
    });
    auto distribution = std::discrete_distribution<uint32_t>(probabilities_vector.begin(), probabilities_vector.end());
    return distribution(ttml::autograd::ctx().get_generator());
}

template <typename Model, typename Tokenizer>
void generate(
    const std::shared_ptr<Model> &model,
    const Tokenizer &tokenizer,
    uint32_t max_sequence_length,
    uint32_t num_heads,
    uint32_t tokens_to_generate = 1024U) {
    model->eval();

    std::string prompt;
    fmt::print("Enter a prompt: ");
    std::getline(std::cin, prompt);

    if (prompt.empty()) {
        prompt = "\n";
    }

    auto *device = &ttml::autograd::ctx().get_device();

    auto prompt_tokens = tokenizer.encode(prompt);

    auto pad_token_id = 0U;

    auto vocab_size = tokenizer.get_vocab_size();

    auto positions_vector = std::vector<uint32_t>(max_sequence_length);
    std::iota(positions_vector.begin(), positions_vector.end(), 0);
    auto positions_tensor = ttml::autograd::create_tensor(ttml::core::from_vector<uint32_t, DataType::UINT32>(
        positions_vector, ttml::core::create_shape({1, 1, 1, max_sequence_length}), device, Layout::ROW_MAJOR));

    std::vector<float> mask;
    mask.reserve(static_cast<size_t>(max_sequence_length * max_sequence_length * num_heads));
    for (int head = 0; head < num_heads; ++head) {
        for (int i = 0; i < max_sequence_length; ++i) {
            for (int j = 0; j < max_sequence_length; ++j) {
                mask.push_back(i >= j ? 1.0F : 0.0F);
            }
        }
    }
    auto mask_tensor = ttml::autograd::create_tensor(ttml::core::from_vector(
        mask, ttml::core::create_shape({1, num_heads, max_sequence_length, max_sequence_length}), device));

    std::vector<uint32_t> prompt_tokens_padded(max_sequence_length, pad_token_id);
    fmt::print("Generated text:\n");
    fmt::print("*******************\n");
    fmt::print("{}", prompt);
    for (uint32_t token_idx = 0; token_idx < tokens_to_generate; ++token_idx) {
        uint32_t start_idx = 0;
        if (prompt_tokens.size() > max_sequence_length) {
            start_idx = prompt_tokens.size() - max_sequence_length;
        }
        for (uint32_t i = start_idx; i < prompt_tokens.size(); ++i) {
            prompt_tokens_padded[i - start_idx] = prompt_tokens[i];
        }

        auto prompt_tokens_padded_size = static_cast<uint32_t>(prompt_tokens_padded.size());
        auto prompt_tensor = ttml::autograd::create_tensor(ttml::core::from_vector<uint32_t, DataType::UINT32>(
            prompt_tokens_padded,
            ttml::core::create_shape({1, 1, 1, prompt_tokens_padded_size}),
            device,
            Layout::ROW_MAJOR));

        auto output = (*model)(prompt_tensor, positions_tensor, mask_tensor);
        auto output_vector = ttml::core::to_vector(output->get_value());

        uint32_t predicted_token_id = prompt_tokens.size() - 1U;
        if (prompt_tokens.size() > max_sequence_length) {
            predicted_token_id = prompt_tokens_padded_size - 1U;
        }
        auto logits_ptr = output_vector.data() + predicted_token_id * vocab_size;
        auto token_id = sample(std::span<float>(logits_ptr, vocab_size));
        prompt_tokens.push_back(token_id);
        fmt::print("{}", tokenizer.decode({token_id}));
        ttml::autograd::ctx().reset_graph();
    }
    fmt::print("\n*******************\n");

    model->train();
}

int main(int argc, char **argv) {
    auto result = signal(SIGINT, signal_handler);
    if (result == SIG_ERR) {
        std::cerr << "Failed to set signal handler\n";
        return -1;
    }
    wandbcpp::init({.project = "tt_train_nano_gpt"});
    wandbcpp::update_config({
        {"model", "transformer"},
        {"num_heads", static_cast<int>(config.num_heads)},
        {"embedding_dim", static_cast<int>(config.embedding_dim)},
        {"num_blocks", static_cast<int>(config.num_blocks)},
        {"dropout_prob", config.dropout_prob},
        {"learning_rate", config.learning_rate},
        {"weight_decay", config.weight_decay},
        {"batch_size", static_cast<int>(config.batch_size)},
        {"sequence_length", static_cast<int>(config.sequence_length)},
        {"max_steps", static_cast<int>(config.max_steps)},
    });

    auto start_timer = std::chrono::high_resolution_clock::now();
    CLI::App app{"NanoGPT Example"};
    argv = app.ensure_utf8(argv);

    uint32_t seed = 5489U;
    uint32_t model_save_interval = 500;
    uint32_t max_steps = config.max_steps;
    uint32_t batch_size = config.batch_size;
    uint32_t sequence_length = config.sequence_length;
    std::string model_path;
    std::string data_path = std::string(DATA_FOLDER) + "/shakespeare.txt";
    bool is_eval = false;

    app.add_option("-b,--batch_size", batch_size, "Batch size")->default_val(batch_size);
    app.add_option("-i,--model_save_interval", model_save_interval, "Model save interval")
        ->default_val(model_save_interval);
    app.add_option("-p,--model_path", model_path, "Model path")->default_val(model_path);
    app.add_option("-d,--data_path", data_path, "Data path")->default_val(data_path);
    app.add_option("-s,--seed", seed, "Seed")->default_val(seed);
    app.add_option("-m,--max_steps", max_steps, "Max steps")->default_val(max_steps);
    app.add_flag("-e,--eval", is_eval, "Evaluation mode")->default_val(is_eval);
    CLI11_PARSE(app, argc, argv);

    // set seed
    ttml::autograd::ctx().set_seed(seed);

    std::string text;
    try {
        text = read_file_to_str(data_path);
    } catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
        return -1;
    }

    fmt::print("Max steps {}\n", max_steps);
    fmt::print("Batch size {}\n", batch_size);
    fmt::print("Seed {}\n", ttml::autograd::ctx().get_seed());

    auto [dataset, tokenizer] =
        ttml::datasets::create_in_memory_token_dataset<ttml::tokenizers::CharTokenizer>(text, sequence_length);
    fmt::print("Dataset size: {}\n", dataset.get_size());
    fmt::print("Vocab size: {}\n", tokenizer.get_vocab_size());

    auto *device = &ttml::autograd::ctx().get_device();
    device->enable_program_cache();

    // disable for now, unexpected freezes and crashes
    // device->enable_async(true);

    struct CachedHostData {
        std::vector<uint32_t> data;
        std::vector<int32_t> targets;
        ttml::autograd::TensorPtr masks_tensor;
        ttml::autograd::TensorPtr positions_tensor;
    };
    CachedHostData cached_data;
    std::vector<uint32_t> positions;
    std::vector<float> mask;
    positions.reserve((size_t)batch_size * sequence_length);
    for (int sample_idx = 0; sample_idx < batch_size; ++sample_idx) {
        for (int i = 0; i < sequence_length; ++i) {
            positions.push_back(i);
        }
    }

    mask.reserve((size_t)batch_size * sequence_length * sequence_length * config.num_heads);
    for (int sample_idx = 0; sample_idx < batch_size; ++sample_idx) {
        for (int head = 0; head < config.num_heads; ++head) {
            for (int i = 0; i < sequence_length; ++i) {
                for (int j = 0; j < sequence_length; ++j) {
                    mask.push_back(i >= j ? 1.0F : 0.0F);
                }
            }
        }
    }
    cached_data.masks_tensor = ttml::autograd::create_tensor(ttml::core::from_vector(
        mask, ttml::core::create_shape({batch_size, config.num_heads, sequence_length, sequence_length}), device));
    cached_data.positions_tensor = ttml::autograd::create_tensor(ttml::core::from_vector<uint32_t, DataType::UINT32>(
        positions, ttml::core::create_shape({batch_size, 1, 1, sequence_length}), device, Layout::ROW_MAJOR));

    std::function<BatchType(std::vector<DatasetSample> && samples)> collate_fn =
        [sequence_length, num_heads = config.num_heads, vocab_size = tokenizer.get_vocab_size(), device, &cached_data](
            std::vector<DatasetSample> &&samples) {
            auto start_timer = std::chrono::high_resolution_clock::now();
            const uint32_t batch_size = samples.size();
            std::vector<uint32_t> &data = cached_data.data;
            std::vector<int32_t> &targets = cached_data.targets;

            data.clear();
            targets.clear();

            data.reserve((size_t)batch_size * sequence_length);
            targets.reserve((size_t)batch_size * sequence_length);
            for (auto &[features, target_span] : samples) {
                std::copy(features.begin(), features.end(), std::back_inserter(data));
                std::copy(target_span.begin(), target_span.end(), std::back_inserter(targets));
            }
            auto end_timer = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_timer - start_timer).count();
            fmt::print("dataloader host only step time {} ms\n", (double)duration / 1000.);
            auto data_tensor = ttml::autograd::create_tensor(ttml::core::from_vector<uint32_t, DataType::UINT32>(
                data, ttml::core::create_shape({batch_size, 1, 1, sequence_length}), device, Layout::ROW_MAJOR));
            auto targets_tensor = ttml::autograd::create_tensor(
                ttml::core::from_vector<int32_t, DataType::INT32>(targets, {batch_size * sequence_length}, device));
            end_timer = std::chrono::high_resolution_clock::now();
            duration = std::chrono::duration_cast<std::chrono::microseconds>(end_timer - start_timer).count();
            fmt::print("dataloader step time {} ms\n", (double)duration / 1000.);
            return std::make_tuple(data_tensor, targets_tensor, cached_data.masks_tensor, cached_data.positions_tensor);
        };

    LossAverageMeter loss_meter;
    auto train_dataloader = DataLoader(dataset, /* batch_size */ batch_size, /* shuffle */ true, collate_fn);

    auto transformer_config = TransformerConfig();
    transformer_config.num_heads = config.num_heads;
    transformer_config.embedding_dim = config.embedding_dim;
    transformer_config.dropout_prob = config.dropout_prob;
    transformer_config.num_blocks = config.num_blocks;
    transformer_config.vocab_size = round_up_to_tile(tokenizer.get_vocab_size());
    transformer_config.max_sequence_length = sequence_length;
    auto model = std::make_shared<Transformer>(transformer_config);

    auto adamw_params = ttml::optimizers::AdamWConfig();
    adamw_params.lr = config.learning_rate;
    adamw_params.weight_decay = config.weight_decay;
    fmt::print("AdamW configuration:\n");
    fmt::print("    Learning rate: {}\n", adamw_params.lr);
    fmt::print("    Weight decay: {}\n", adamw_params.weight_decay);
    auto optimizer = ttml::optimizers::AdamW(model->parameters(), adamw_params);

    if (!model_path.empty() && std::filesystem::exists(model_path)) {
        fmt::print("Loading model from {}\n", model_path);
        load_model_and_optimizer(model_path, model, optimizer, "transformer", "adamw");
        fmt::print("Model loaded after {} steps\n", optimizer.get_steps());
    }

    if (is_eval) {
        fmt::print("\nEvaluation started\n");
        for (;;) {
            generate(model, tokenizer, sequence_length, config.num_heads);
        }
        fmt::print("\nEvaluation finished\n");
        return 0;
    }

    const uint32_t num_epochs = config.num_epochs;
    for (uint32_t epoch = 0; epoch < num_epochs; ++epoch) {
        for (auto [features, target, masks, positions] : train_dataloader) {
            auto start_timer = std::chrono::high_resolution_clock::now();
            optimizer.zero_grad();
            auto output = (*model)(features, positions, masks);
            auto loss = ttml::ops::nll_loss(output, target);
            auto loss_float = ttml::core::to_vector(loss->get_value())[0];
            loss_meter.update(loss_float, features->get_value().get_shape()[0]);
            loss->backward();
            optimizer.step();
            ttml::autograd::ctx().reset_graph();
            auto global_step = optimizer.get_steps();
            fmt::print("Step: {}, Loss: {}\n", global_step, loss_float);

            if (global_step % 10 == 0) {
                wandbcpp::log({{"Step", (int)global_step}, {"Loss", loss_float}});
            }
            if (!model_path.empty() && global_step % model_save_interval == 0) {
                save_model_and_optimizer(model_path, model, optimizer, "transformer", "adamw");
            }

            if (global_step >= max_steps) {
                break;
            }
            auto end_timer = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_timer - start_timer).count();
            fmt::print(
                "Full step time {} ms, cache entries: {}\n",
                (double)duration / 1000,
                device->num_program_cache_entries());
        }
        if (optimizer.get_steps() >= max_steps) {
            break;
        }
    }

    if (!model_path.empty()) {
        save_model_and_optimizer(model_path, model, optimizer, "transformer", "adamw");
    }

    auto end_timer = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_timer - start_timer).count();
    fmt::print(
        "{} Steps training time: {} s, cache entries: {}\n",
        max_steps,
        (double)duration / 1000000.,
        device->num_program_cache_entries());
    wandbcpp::finish();
    return 0;
}
