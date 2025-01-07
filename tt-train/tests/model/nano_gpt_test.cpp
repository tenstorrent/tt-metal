// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <fmt/format.h>
#include <gtest/gtest.h>

#include <core/ttnn_all_includes.hpp>

#include "autograd/auto_context.hpp"
#include "core/tt_tensor_utils.hpp"
#include "datasets/dataloader.hpp"
#include "datasets/in_memory_token_dataset.hpp"
#include "datasets/utils.hpp"
#include "models/gpt2.hpp"
#include "ops/losses.hpp"
#include "optimizers/adamw.hpp"
#include "optimizers/optimizer_base.hpp"
#include "tokenizers/char_tokenizer.hpp"

class NanoGPTTest : public ::testing::Test {
protected:
    void SetUp() override {
        ttml::autograd::ctx().open_device();
    }

    void TearDown() override {
        ttml::autograd::ctx().close_device();
    }
};

using ttml::autograd::TensorPtr;

using DatasetSample = std::pair<std::span<const uint32_t>, std::span<const uint32_t>>;
// tokens, targets, mask
using BatchType = std::tuple<TensorPtr, TensorPtr, TensorPtr>;
using DataLoader = ttml::datasets::DataLoader<
    ttml::datasets::InMemoryTokenDataset,
    std::function<BatchType(std::vector<DatasetSample> &&samples)>,
    BatchType>;

struct TrainingConfig {
    std::string project_name;
    uint32_t seed = 5489U;
    uint32_t model_save_interval = 500;
    uint32_t batch_size = 64;
    uint32_t num_epochs = 1;
    uint32_t max_steps = 100;
    float learning_rate = 3e-4F;
    float weight_decay = 1e-2F;
    std::string model_path;
    std::string data_path;
    ttml::models::gpt2::TransformerConfig transformer_config;
};

void train_test(bool use_moreh_adamw = false, bool memory_efficient = false) {
    auto config = TrainingConfig();
    config.transformer_config.dropout_prob = 0.0F;
    config.transformer_config.runner_type =
        memory_efficient ? ttml::models::gpt2::RunnerType::MemoryEfficient : ttml::models::gpt2::RunnerType::Default;
    config.data_path = "/shakespeare.txt";

    // set seed
    ttml::autograd::ctx().set_seed(config.seed);

    std::string text;
    // reading training data from txt file
    {
        std::ifstream file(config.data_path);
        if (!file.is_open()) {
            throw std::runtime_error("Failed to open file: " + config.data_path);
        }

        std::stringstream buffer;
        buffer << file.rdbuf();

        text = buffer.str();
    }

    auto *device = &ttml::autograd::ctx().get_device();
    device->enable_program_cache();

    auto sequence_length = config.transformer_config.max_sequence_length;

    auto [dataset, tokenizer] =
        ttml::datasets::create_in_memory_token_dataset<ttml::tokenizers::CharTokenizer>(text, sequence_length);

    struct CachedHostData {
        std::vector<uint32_t> data;
        std::vector<int32_t> targets;
        ttml::autograd::TensorPtr masks_tensor;
    };
    CachedHostData cached_data;

    std::vector<float> mask;
    auto num_heads = config.transformer_config.num_heads;
    mask.reserve((size_t)config.batch_size * sequence_length * sequence_length * num_heads);
    for (int sample_idx = 0; sample_idx < config.batch_size; ++sample_idx) {
        for (int head = 0; head < num_heads; ++head) {
            for (int i = 0; i < sequence_length; ++i) {
                for (int j = 0; j < sequence_length; ++j) {
                    mask.push_back(i >= j ? 1.0F : 0.0F);
                }
            }
        }
    }
    cached_data.masks_tensor = ttml::autograd::create_tensor(ttml::core::from_vector(
        mask, ttml::core::create_shape({config.batch_size, num_heads, sequence_length, sequence_length}), device));

    std::function<BatchType(std::vector<DatasetSample> && samples)> collate_fn =
        [sequence_length, num_heads, device, &cached_data](std::vector<DatasetSample> &&samples) {
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
            return std::make_tuple(data_tensor, targets_tensor, cached_data.masks_tensor);
        };
    auto train_dataloader = DataLoader(dataset, /* batch_size */ config.batch_size, /* shuffle */ true, collate_fn);

    fmt::print("Overriding vocab size to be divisible by 32\n");
    config.transformer_config.vocab_size = (tokenizer->get_vocab_size() + 31) / 32 * 32;
    auto model = ttml::models::gpt2::create(config.transformer_config);

    auto adamw_params = ttml::optimizers::AdamWConfig();
    adamw_params.lr = config.learning_rate;
    adamw_params.weight_decay = config.weight_decay;

    auto create_optimizer = [&]() -> std::shared_ptr<ttml::optimizers::OptimizerBase> {
        if (use_moreh_adamw) {
            return std::make_shared<ttml::optimizers::MorehAdamW>(model->parameters(), adamw_params);
        } else {
            return std::make_shared<ttml::optimizers::AdamW>(model->parameters(), adamw_params);
        }
    };

    auto optimizer = create_optimizer();

    std::vector<double> steps_time;
    std::vector<float> losses;

    for (auto [features, target, masks] : train_dataloader) {
        auto start_timer = std::chrono::high_resolution_clock::now();
        optimizer->zero_grad();
        auto output = (*model)(features, masks);
        auto loss = ttml::ops::nll_loss(output, target);
        auto loss_float = ttml::core::to_vector(loss->get_value())[0];
        loss->backward();
        optimizer->step();
        ttml::autograd::ctx().reset_graph();
        auto global_step = optimizer->get_steps();
        losses.emplace_back(loss_float);
        if (global_step >= config.max_steps) {
            break;
        }
        auto end_timer = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_timer - start_timer).count();
        steps_time.emplace_back((double)duration / 1000.0);
    }

    // verify program cache
    auto program_cache_entries = device->num_program_cache_entries();
    if (!use_moreh_adamw) {
        EXPECT_EQ(program_cache_entries, 123);
    } else {
        EXPECT_EQ(program_cache_entries, 102);
    }

    // verify time per step
    size_t num_steps_below = 0;
    const double expected_default_runner_time_ms = 330.0;
    const double expected_memory_efficient_runner_time_ms = 450.0;
    double expected_time_ms =
        memory_efficient ? expected_memory_efficient_runner_time_ms : expected_default_runner_time_ms;
    for (auto &time : steps_time) {
        num_steps_below += (time < expected_time_ms);
    }
    if (num_steps_below / static_cast<double>(steps_time.size()) < 0.9) {
        EXPECT_TRUE(false);
    }

    // verify loss
    if (!use_moreh_adamw) {
        EXPECT_EQ(losses.size(), config.max_steps);
        EXPECT_EQ(losses[0], 4.6875);
        EXPECT_EQ(losses[9], 2.96875);
        EXPECT_EQ(losses[19], 2.703125);
        EXPECT_EQ(losses[29], 2.59375);
        EXPECT_EQ(losses[39], 2.546875);
        EXPECT_EQ(losses[49], 2.5);
        EXPECT_EQ(losses[59], 2.484375);
        EXPECT_EQ(losses[69], 2.46875);
        EXPECT_EQ(losses[79], 2.453125);
        EXPECT_EQ(losses[89], 2.4375);
        EXPECT_EQ(losses[99], 2.453125);
    } else {
        EXPECT_EQ(losses.size(), config.max_steps);
        EXPECT_EQ(losses[0], 4.6875);
        EXPECT_EQ(losses[9], 2.96875);
        EXPECT_EQ(losses[19], 2.703125);
        EXPECT_EQ(losses[29], 2.59375);
        EXPECT_EQ(losses[39], 2.546875);
        EXPECT_EQ(losses[49], 2.484375);
        EXPECT_EQ(losses[59], 2.484375);
        EXPECT_EQ(losses[69], 2.46875);
        EXPECT_EQ(losses[79], 2.453125);
        EXPECT_EQ(losses[89], 2.4375);
        EXPECT_EQ(losses[99], 2.4375);
    }
}

bool should_run_tests() {
    const char *env_var = std::getenv("ENABLE_CI_ONLY_TT_TRAIN_TESTS");
    return env_var ? true : ENABLE_CI_ONLY_TT_TRAIN_TESTS;
}

/*
This tests are supposed to run only in CI.
Change the value of ENABLE_CI_ONLY_TT_TRAIN_TESTS to true to run them.
If one of these tests fails, it means one (or more) of the following:
- program cache size changed (new ops added/removed silently)
- time per step changed (performance regression)
- loss values changed (regression in ops accuracy)
*/

TEST_F(NanoGPTTest, AdamW) {
    GTEST_SKIP() << "Skipping AdamW";
    return;
    if (should_run_tests()) {
        train_test(/* use_moreh_adamw */ false, /* memory_efficient */ false);
    }
}

TEST_F(NanoGPTTest, MorehAdamW) {
    GTEST_SKIP() << "Skipping MorehAdamW";
    return;

    if (should_run_tests()) {
        train_test(/* use_moreh_adamw */ true, /* memory_efficient */ false);
    }
}

TEST_F(NanoGPTTest, AdamW_MemoryEfficient) {
    GTEST_SKIP() << "Skipping AdamW + MemoryEfficient";
    return;

    if (should_run_tests()) {
        train_test(/* use_moreh_adamw */ false, /* memory_efficient */ true);
    }
}

TEST_F(NanoGPTTest, MorehAdamW_MemoryEfficient) {
    GTEST_SKIP() << "Skipping MorehAdamW + MemoryEfficient";
    return;

    if (should_run_tests()) {
        train_test(/* use_moreh_adamw */ true, /* memory_efficient */ true);
    }
}
