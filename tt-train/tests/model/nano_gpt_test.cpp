// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <fmt/format.h>
#include <gtest/gtest.h>

#include <core/ttnn_all_includes.hpp>
#include <fstream>

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
#include "tokenizers/char_tokenizer.hpp"
namespace {
/*
Nightly tests could be enabled by setting the environment variable ENABLE_NIGHTLY_TT_TRAIN_TESTS=1
or setting 'is_nigthly_tt_train_tests_enabled' variable to true.
*/
constexpr bool is_nigthly_tt_train_tests_enabled = false;

[[nodiscard]] bool is_wormhole_b0() {
    static bool arch_is_wormhole_b0 = []() {
        auto shape = tt::tt_metal::distributed::MeshShape(1, 1);
        ttml::core::MeshDevice device(shape, {});

        return device.get_device().arch() == tt::ARCH::WORMHOLE_B0;
    }();
    return arch_is_wormhole_b0;
}

[[nodiscard]] bool should_run_nightly_tests() {
    const char *env_var = std::getenv("ENABLE_NIGHTLY_TT_TRAIN_TESTS");
    bool is_whb0 = is_wormhole_b0();
    bool is_ci = env_var && is_nigthly_tt_train_tests_enabled;
    return is_whb0 && is_ci;
}

[[nodiscard]] bool should_run_multi_device_tests() {
    bool enable_nightly = should_run_nightly_tests();
    bool sufficient_devices = tt::tt_metal::GetNumAvailableDevices() >= 2;
    return enable_nightly && sufficient_devices;
}
}  // namespace
class NanoLlamaTest : public ::testing::Test {
protected:
    void SetUp() override {
        if (should_run_nightly_tests()) {
            auto shape = tt::tt_metal::distributed::MeshShape(1, 1);
            ttml::autograd::ctx().open_device(shape);
        }
    }

    void TearDown() override {
        if (should_run_nightly_tests()) {
            ttml::autograd::ctx().close_device();
        }
    }
};

class NanoLlamaMultiDeviceTest : public ::testing::Test {
protected:
    void SetUp() override {
        if (should_run_multi_device_tests()) {
            auto shape = tt::tt_metal::distributed::MeshShape(1, 2);
            ttml::autograd::ctx().open_device(shape);
        }
    }

    void TearDown() override {
        if (should_run_multi_device_tests()) {
            ttml::autograd::ctx().close_device();
        }
    }
};

using ttml::autograd::TensorPtr;
using namespace ttml;

struct TrainingConfig {
    uint32_t seed = 5489U;
    uint32_t model_save_interval = 500;
    uint32_t batch_size = 64;
    uint32_t num_epochs = 1;
    uint32_t max_steps = 100;
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
    device->clear_program_cache();  // we want a fresh program cache count for each run

    auto sequence_length = config.transformer_config.max_sequence_length;

    auto [dataset, tokenizer] =
        ttml::datasets::create_in_memory_token_dataset<ttml::tokenizers::CharTokenizer>(text, sequence_length);

    struct CachedHostData {
        std::vector<uint32_t> data;
        std::vector<uint32_t> targets;
        ttml::autograd::TensorPtr masks_tensor;
    };
    CachedHostData cached_data;

    std::vector<float> mask;
    mask.reserve((size_t)sequence_length * sequence_length);
    for (int i = 0; i < sequence_length; ++i) {
        for (int j = 0; j < sequence_length; ++j) {
            mask.push_back(i >= j ? 1.0F : 0.0F);
        }
    }
    cached_data.masks_tensor = ttml::autograd::create_tensor(
        ttml::core::from_vector(mask, ttnn::Shape({1, 1, sequence_length, sequence_length}), device));

    std::function<BatchType(std::vector<DatasetSample> && samples)> collate_fn =
        [sequence_length, device, &cached_data, use_ddp](std::vector<DatasetSample> &&samples) {
            const uint32_t batch_size = samples.size();
            std::vector<uint32_t> &data = cached_data.data;
            std::vector<uint32_t> &targets = cached_data.targets;

            data.clear();
            targets.clear();

            data.reserve((size_t)batch_size * sequence_length);
            targets.reserve((size_t)batch_size * sequence_length);
            for (auto &[features, target_span] : samples) {
                std::copy(features.begin(), features.end(), std::back_inserter(data));
                std::copy(target_span.begin(), target_span.end(), std::back_inserter(targets));
            }

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
                        ttml::autograd::create_tensor(ttml::core::from_vector<uint32_t, ttnn::DataType::UINT32>(
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
                    ttml::autograd::create_tensor(ttml::core::from_vector<uint32_t, ttnn::DataType::UINT32>(
                        targets, ttnn::Shape({batch_size, sequence_length}), device, ttnn::Layout::ROW_MAJOR));
                return {data_tensor, targets_tensor};
            };

            auto [data_tensor, targets_tensor] = create_data_and_targets();
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

    std::shared_ptr<ttml::modules::ModuleBase> model;
    if (use_tensor_parallel) {
        config.transformer_config.num_groups = num_devices;
        config.transformer_config.num_heads = num_devices * 3;
        config.transformer_config.embedding_dim = (384U / 3U) * config.transformer_config.num_heads;

        model = ttml::models::distributed::llama::create(config.transformer_config);
    } else {
        model = ttml::models::llama::create(config.transformer_config);
    }

    auto adamw_params = ttml::optimizers::AdamWConfig();
    adamw_params.lr = config.learning_rate;
    adamw_params.weight_decay = config.weight_decay;

    auto optimizer = std::make_shared<ttml::optimizers::MorehAdamW>(model->parameters(), adamw_params);

    auto get_loss_value = [](const TensorPtr &loss) {
        auto loss_xtensors = ttml::core::to_xtensor(loss->get_value(), ttml::core::IdentityComposer{});
        // sum of loss xtensors
        float loss_float =
            std::accumulate(loss_xtensors.begin(), loss_xtensors.end(), 0.0F, [](float acc, auto &xtensor) {
                return acc + xtensor(0);
            });

        return loss_float / static_cast<float>(loss_xtensors.size());
    };

    std::vector<double> steps_time;
    std::vector<float> losses;

    for (auto [features, target, masks] : train_dataloader) {
        auto start_timer = std::chrono::high_resolution_clock::now();
        optimizer->zero_grad();
        auto output = (*model)(features, masks);
        auto loss = ttml::ops::cross_entropy_loss(output, target);
        auto loss_float = get_loss_value(loss);
        loss->backward();

        // synchronize gradients for multi-device case, no-op if single device
        auto parameters = model->parameters();
        if (!use_tensor_parallel) {
            ttml::core::distributed::synchronize_gradients(parameters);
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

    float abs_tol = 1e-3F;
    std::string run_type = "SingleDevice";
    if (use_tensor_parallel) {
        run_type = "TP";
    } else if (use_ddp) {
        run_type = "DDP";
    }

    fmt::print("run_type: {}\n", run_type);
    std::unordered_map<std::string, uint32_t> expected_program_cache_entries_map = {
        {"SingleDevice", 79}, {"DDP", 93}, {"TP", 103}};
    EXPECT_NEAR(program_cache_entries, expected_program_cache_entries_map.at(run_type), abs_tol);

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

    std::unordered_map<std::string, std::vector<float>> expected_losses_map = {
        {"SingleDevice", {4.6875, 3.46875, 3.46875, 3.0625, 2.95312, 2.75, 2.625, 2.53125, 2.375, 2.32812, 2.21875}},
        {"DDP", {4.73438, 3.45312, 3.42969, 3.07812, 2.91406, 2.6875, 2.57031, 2.46875, 2.38281, 2.34375, 2.24219}},
        {"TP", {5, 3.46875, 3.32812, 3.29688, 3.0625, 2.90625, 2.70312, 2.57812, 2.4375, 2.34375, 2.25}}};

    auto expected_losses = expected_losses_map.at(run_type);

    auto test_indices = {0, 9, 19, 29, 39, 49, 59, 69, 79, 89, 99};
    auto actual_losses = std::vector<float>(test_indices.size());
    std::transform(test_indices.begin(), test_indices.end(), actual_losses.begin(), [&losses](size_t index) {
        return losses[index];
    });

    for (size_t i = 0; i < test_indices.size(); ++i) {
        EXPECT_NEAR(actual_losses[i], expected_losses[i], abs_tol);
    }
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
    if (!should_run_nightly_tests()) {
        GTEST_SKIP() << "Skipping Nightly test.";
    }
    train_test();
}

TEST_F(NanoLlamaMultiDeviceTest, DISABLED_NIGHTLY_TensorParallel) {
    if (!should_run_multi_device_tests()) {
        GTEST_SKIP() << "Skipping test as we are running on a single device.";
    }
    train_test(/*use_tensor_parallel=*/true, /*use_ddp=*/false);
}

TEST_F(NanoLlamaMultiDeviceTest, NIGHTLY_DDP) {
    if (!should_run_multi_device_tests()) {
        GTEST_SKIP() << "Skipping test as we are running on a single device.";
    }
    train_test(/*use_tensor_parallel=*/false, /*use_ddp=*/true);
}
