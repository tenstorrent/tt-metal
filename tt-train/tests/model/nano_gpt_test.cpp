// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <fmt/format.h>
#include <gtest/gtest.h>

#include <core/ttnn_all_includes.hpp>

#include "autograd/auto_context.hpp"
#include "core/distributed/distributed.hpp"
#include "core/tt_tensor_utils.hpp"
#include "datasets/dataloader.hpp"
#include "datasets/in_memory_token_dataset.hpp"
#include "datasets/utils.hpp"
#include "models/distributed/llama.hpp"
#include "models/llama.hpp"
#include "ops/losses.hpp"
#include "optimizers/adamw.hpp"
#include "optimizers/optimizer_base.hpp"
#include "tokenizers/char_tokenizer.hpp"

class NanoLlamaTest : public ::testing::Test {
protected:
    void SetUp() override {
        ttml::autograd::ctx().open_device();
    }

    void TearDown() override {
        ttml::autograd::ctx().close_device();
    }
};

class NanoLlamaMultiDeviceTest : public ::testing::Test {
protected:
    void SetUp() override {
        auto shape = tt::tt_metal::distributed::MeshShape(1, 2);
        ttml::autograd::ctx().open_device(shape);
    }

    void TearDown() override {
        ttml::autograd::ctx().close_device();
    }
};

using ttml::autograd::TensorPtr;
using namespace ttml;

struct TrainingConfig {
    uint32_t seed = 5489U;
    uint32_t model_save_interval = 500;
    uint32_t batch_size = 64;
    uint32_t num_epochs = 1;
    uint32_t max_steps = 5000;
    float learning_rate = 3e-4F;
    float weight_decay = 1e-2F;
    bool use_moreh_adamw = false;
    // works only for AdamW
    bool use_kahan_summation = false;
    // accumulate batches for gradient update
    uint32_t gradient_accumulation_steps = 1;
    std::string model_path;
    std::string data_path;
    ttml::models::llama::LlamaConfig transformer_config;

    // mpi config
    bool enable_mpi = false;
    uint32_t num_mh_workers = 0U;
};

using DatasetSample = std::pair<std::span<const uint32_t>, std::span<const uint32_t>>;
// tokens, targets, masks
using BatchType = std::tuple<TensorPtr, TensorPtr, TensorPtr>;
using DataLoader = ttml::datasets::DataLoader<
    ttml::datasets::InMemoryTokenDataset,
    std::function<BatchType(std::vector<DatasetSample> &&samples)>,
    BatchType>;

// test training with MorehAdamW, nanollama config, memory efficient runner
// no clip grad norm
void train_test(bool use_tensor_parallel = false, bool use_ddp = false) {
    if (use_tensor_parallel && use_ddp) {
        throw std::runtime_error("DDP and TP cannot be enabled at the same time. Disable DDP or TP.");
    }

    auto config = TrainingConfig();
    config.transformer_config.runner_type = ttml::models::llama::RunnerType::MemoryEfficient;
    config.data_path = std::string(TEST_DATA_DIR) + "/shakespeare.txt";

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
        mask, ttnn::Shape({config.batch_size, num_heads, sequence_length, sequence_length}), device));

    std::function<BatchType(std::vector<DatasetSample> && samples)> collate_fn =
        [sequence_length, num_heads, device, &cached_data, use_ddp](std::vector<DatasetSample> &&samples) {
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

            auto create_data_and_targets = [&]() -> std::tuple<TensorPtr, TensorPtr> {
                if (use_ddp) {
                    const auto mapper = ttnn::distributed::shard_tensor_to_mesh_mapper(*device, 0);
                    auto data_tensor =
                        ttml::autograd::create_tensor(ttml::core::from_vector<uint32_t, ttnn::DataType::UINT32>(
                            data,
                            ttnn::Shape({batch_size, 1, 1, sequence_length}),
                            device,
                            ttnn::Layout::ROW_MAJOR,
                            mapper.get()));
                    auto targets_tensor =
                        ttml::autograd::create_tensor(ttml::core::from_vector<int32_t, ttnn::DataType::INT32>(
                            targets,
                            ttnn::Shape({batch_size, sequence_length}),
                            device,
                            ttnn::Layout::ROW_MAJOR,
                            mapper.get()));
                    return {data_tensor, targets_tensor};
                }

                const auto mapper = ttnn::distributed::shard_tensor_to_mesh_mapper(*device, 0);
                auto data_tensor =
                    ttml::autograd::create_tensor(ttml::core::from_vector<uint32_t, ttnn::DataType::UINT32>(
                        data,
                        ttnn::Shape({batch_size, 1, 1, sequence_length}),
                        device,
                        ttnn::Layout::ROW_MAJOR,
                        mapper.get()));
                auto targets_tensor =
                    ttml::autograd::create_tensor(ttml::core::from_vector<int32_t, ttnn::DataType::INT32>(
                        targets, ttnn::Shape({batch_size, sequence_length}), device, ttnn::Layout::ROW_MAJOR));
                return {data_tensor, targets_tensor};
            };

            auto [data_tensor, targets_tensor] = create_data_and_targets();
            end_timer = std::chrono::high_resolution_clock::now();
            duration = std::chrono::duration_cast<std::chrono::microseconds>(end_timer - start_timer).count();
            fmt::print("dataloader step time {} ms\n", (double)duration / 1000.);
            return std::make_tuple(data_tensor, targets_tensor, cached_data.masks_tensor);
        };
    auto train_dataloader = DataLoader(dataset, /* batch_size */ config.batch_size, /* shuffle */ true, collate_fn);
    fmt::print("Overriding vocab size to be divisible by 32\n");
    auto num_devices = static_cast<uint32_t>(device->num_devices());
    auto round_up_to_tile = [](uint32_t value, uint32_t tile_size) -> uint32_t {
        return (value + tile_size - 1) / tile_size * tile_size;
    };
    config.transformer_config.vocab_size =
        round_up_to_tile(tokenizer->get_vocab_size(), (use_tensor_parallel ? num_devices : 1U) * 32U);

    std::shared_ptr<ttml::autograd::ModuleBase> model;
    if (use_tensor_parallel) {
        config.transformer_config.num_groups = num_devices;
        config.transformer_config.num_heads = num_devices * 2;
        config.transformer_config.embedding_dim = (384U / 3U) * config.transformer_config.num_heads;

        model = ttml::models::distributed::llama::create(config.transformer_config);
    } else {
        model = ttml::models::llama::create(config.transformer_config);
    }

    auto adamw_params = ttml::optimizers::AdamWConfig();
    adamw_params.lr = config.learning_rate;
    adamw_params.weight_decay = config.weight_decay;

    auto optimizer = std::make_shared<ttml::optimizers::MorehAdamW>(model->parameters(), adamw_params);

    std::vector<double> steps_time;
    std::vector<float> losses;

    for (auto [features, target, masks] : train_dataloader) {
        auto start_timer = std::chrono::high_resolution_clock::now();
        optimizer->zero_grad();
        auto output = (*model)(features, masks);
        auto loss = ttml::ops::nll_loss(output, target);
        auto loss_float = ttml::core::to_vector(loss->get_value())[0];
        loss->backward();

        // synchronize gradients for multi-device case, no-op if single device
        auto parameters = model->parameters();
        if (!use_tensor_parallel) {
            ttml::core::distributed::synchronize_parameters(parameters);
        }

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

    float abs_tol = 1e-4;
    EXPECT_NEAR(program_cache_entries, 79, abs_tol);

    // verify time per step
    size_t num_steps_below = 0;
    const double expected_memory_efficient_runner_time_ms = 450.0;
    double expected_time_ms = expected_memory_efficient_runner_time_ms;
    for (auto &time : steps_time) {
        num_steps_below += (time < expected_time_ms);
    }
    if (num_steps_below / static_cast<double>(steps_time.size()) < 0.9) {
        EXPECT_TRUE(false);
    }

    // verify loss
    EXPECT_NEAR(losses.size(), config.max_steps, abs_tol);

    EXPECT_NEAR(losses[0], 0.024047852, abs_tol);
    EXPECT_NEAR(losses[9], -2.296875, abs_tol);
    EXPECT_NEAR(losses[19], -3.296875, abs_tol);
    EXPECT_NEAR(losses[29], -4.46875, abs_tol);
    EXPECT_NEAR(losses[39], -5.6875, abs_tol);
    EXPECT_NEAR(losses[49], -6.875, abs_tol);
    EXPECT_NEAR(losses[59], -8.0625, abs_tol);
    EXPECT_NEAR(losses[69], -9.3125, abs_tol);
    EXPECT_NEAR(losses[79], -10.5625, abs_tol);
    EXPECT_NEAR(losses[89], -11.9375, abs_tol);
    EXPECT_NEAR(losses[99], -13.1875, abs_tol);
}

bool should_run_nightly_tests() {
    const char *env_var = std::getenv("ENABLE_NIGHTLY_TT_TRAIN_TESTS");
    return env_var ? true : ENABLE_NIGHTLY_TT_TRAIN_TESTS;
}

bool should_run_multi_device_tests() {
    bool enable_nightly = should_run_nightly_tests();
    bool sufficient_devices = tt::tt_metal::GetNumAvailableDevices() >= 2;
    return enable_nightly && sufficient_devices;
}

/*
These tests are meant for nightly CI runs.
Change the value of ENABLE_NIGHTLY_TT_TRAIN_TESTS to true to run them.
If one of these tests fails, it means one (or more) of the following:
- program cache size changed (new ops added/removed silently)
- time per step changed (performance regression)
- loss values changed (regression in ops accuracy)
*/

TEST_F(NanoLlamaTest, NIGHTLY_Default) {
    if (should_run_nightly_tests()) {
        train_test();
    }
}

TEST_F(NanoLlamaMultiDeviceTest, NIGHTLY_TensorParallel) {
    if (should_run_multi_device_tests()) {
        train_test(/*use_tensor_parallel=*/true, /*use_ddp=*/false);
    }
}

TEST_F(NanoLlamaMultiDeviceTest, NIGHTLY_DDP) {
    if (should_run_multi_device_tests()) {
        train_test(/*use_tensor_parallel=*/false, /*use_ddp=*/true);
    }
}
