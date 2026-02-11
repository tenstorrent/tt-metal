// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <fmt/format.h>

#include <core/ttnn_all_includes.hpp>
#include <ttnn/tensor/tensor.hpp>

#include "autograd/auto_context.hpp"
#include "autograd/tensor.hpp"
#include "core/tt_tensor_utils.hpp"
#include "datasets/dataloader.hpp"
#include "datasets/generators.hpp"
#include "models/linear_regression.hpp"
#include "modules/linear_module.hpp"
#include "ops/losses.hpp"
#include "optimizers/sgd.hpp"
#include "utils/memory_utils.hpp"

using ttml::autograd::TensorPtr;

using DatasetSample = std::pair<std::vector<float>, std::vector<float>>;
using BatchType = std::pair<TensorPtr, TensorPtr>;
using DataLoader = ttml::datasets::DataLoader<
    ttml::datasets::InMemoryFloatVecDataset,
    std::function<BatchType(std::vector<DatasetSample>&& samples)>,
    BatchType>;

int main() {
    const size_t training_samples_count = 100000;
    const size_t num_features = 64;
    const size_t num_targets = 32;
    const float noise = 0.0F;
    const bool bias = true;

    auto training_params = ttml::datasets::MakeRegressionParams{
        .n_samples = training_samples_count,
        .n_features = num_features,
        .n_targets = num_targets,
        .noise = noise,
        .bias = bias,
    };

    auto training_dataset = ttml::datasets::make_regression(training_params);

    // Pass tt::tt_metal::IGraphProcessor::RunMode::NO_DISPATCH to measure memory usage
    // of model that doesn't fit in the memory of the device.
    ttnn::ScopeGuard memory_usage_guard = ttml::utils::MemoryUsageTracker::begin_capture();

    auto* device = &ttml::autograd::ctx().get_device();

    std::function<BatchType(std::vector<DatasetSample> && samples)> collate_fn =
        [device](std::vector<DatasetSample>&& samples) {
            const uint32_t batch_size = samples.size();
            std::vector<float> data;
            std::vector<float> targets;
            data.reserve(batch_size * num_features);
            targets.reserve(batch_size * num_targets);
            for (auto& [features, target] : samples) {
                std::move(features.begin(), features.end(), std::back_inserter(data));
                std::move(target.begin(), target.end(), std::back_inserter(targets));
            }

            auto data_tensor = ttml::autograd::create_tensor(
                ttml::core::from_vector(data, ttnn::Shape({batch_size, 1, 1, num_features}), device));
            auto targets_tensor = ttml::autograd::create_tensor(
                ttml::core::from_vector(targets, ttnn::Shape({batch_size, 1, 1, num_targets}), device));
            return std::make_pair(data_tensor, targets_tensor);
        };

    const uint32_t batch_size = 128;
    auto train_dataloader = DataLoader(training_dataset, batch_size, /* shuffle */ true, collate_fn);

    auto model = ttml::models::linear_regression::create(num_features, num_targets);

    // fmt::print("Model number of parameters: {}\n", get_number_of_parameters(model, device_config.enable_tp));
    ttml::utils::MemoryUsageTracker::snapshot("MODEL_CREATION");

    float learning_rate = 0.1F * num_targets * (batch_size / 128.F);
    auto sgd_config = ttml::optimizers::SGDConfig{.lr = learning_rate, .momentum = 0.0F};
    auto optimizer = ttml::optimizers::SGD(model->parameters(), sgd_config);

    ttml::utils::MemoryUsageTracker::snapshot("OPTIMIZER_CREATION");

    bool is_everything_compiled = false;
    auto memory_snapshot = [&is_everything_compiled](const std::string& name) {
        if (!is_everything_compiled) {
            ttml::utils::MemoryUsageTracker::snapshot(name);
        }
    };

    int training_step = 0;
    const int num_epochs = 10;
    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        for (const auto& [data, targets] : train_dataloader) {
            optimizer.zero_grad();
            auto output = (*model)(data);
            memory_snapshot("FORWARD_PASS");
            auto loss = ttml::ops::mse_loss(output, targets);
            auto loss_float = ttml::core::to_vector(loss->get_value())[0];
            fmt::print("Step: {} Loss: {}\n", training_step++, loss_float);
            loss->backward();
            memory_snapshot("BACKWARD_PASS");
            optimizer.step();
            ttml::autograd::ctx().reset_graph();

            if (!is_everything_compiled) {
                ttml::autograd::ctx().get_profiler().read_results(device, "compilation_finished");
                is_everything_compiled = true;
                ttml::utils::MemoryUsageTracker::end_capture("FIRST_ITERATION_COMPLETE");
                ttml::utils::MemoryUsageTracker::print_memory_usage();
                ttml::utils::MemoryUsageTracker::clear();
            }
        }
    }
}
