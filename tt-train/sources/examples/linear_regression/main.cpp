// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
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

    auto* device = &ttml::autograd::ctx().get_device();

    std::function<BatchType(std::vector<DatasetSample> && samples)> collate_fn =
        [&num_features, &num_targets, device](std::vector<DatasetSample>&& samples) {
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
                ttml::core::from_vector(data, ttml::core::create_shape({batch_size, 1, 1, num_features}), device));
            auto targets_tensor = ttml::autograd::create_tensor(
                ttml::core::from_vector(targets, ttml::core::create_shape({batch_size, 1, 1, num_targets}), device));
            return std::make_pair(data_tensor, targets_tensor);
        };

    const uint32_t batch_size = 128;
    auto train_dataloader = DataLoader(training_dataset, batch_size, /* shuffle */ true, collate_fn);

    auto model = ttml::models::linear_regression::create(num_features, num_targets);

    float learning_rate = 0.1F * num_targets * (batch_size / 128.F);
    auto sgd_config = ttml::optimizers::SGDConfig{.lr = learning_rate, .momentum = 0.0F};
    auto optimizer = ttml::optimizers::SGD(model->parameters(), sgd_config);

    int training_step = 0;
    const int num_epochs = 10;
    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        for (const auto& [data, targets] : train_dataloader) {
            optimizer.zero_grad();
            auto output = (*model)(data);
            auto loss = ttml::ops::mse_loss(output, targets);
            auto loss_float = ttml::core::to_vector(loss->get_value())[0];
            fmt::print("Step: {} Loss: {}\n", training_step++, loss_float);
            loss->backward();
            optimizer.step();
            ttml::autograd::ctx().reset_graph();
        }
    }
}
