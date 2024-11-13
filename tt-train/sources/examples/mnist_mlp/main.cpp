// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <CLI/CLI.hpp>
#include <core/ttnn_all_includes.hpp>
#include <mnist/mnist_reader.hpp>
#include <ttnn/operations/eltwise/ternary/where.hpp>
#include <ttnn/tensor/tensor_utils.hpp>
#include <ttnn/tensor/types.hpp>

#include "autograd/auto_context.hpp"
#include "autograd/tensor.hpp"
#include "core/tt_tensor_utils.hpp"
#include "datasets/dataloader.hpp"
#include "datasets/in_memory_dataset.hpp"
#include "models.hpp"
#include "ops/losses.hpp"
#include "optimizers/sgd.hpp"
#include "utils.hpp"

using ttml::autograd::TensorPtr;

using DatasetSample = std::pair<std::vector<uint8_t>, uint8_t>;
using BatchType = std::pair<TensorPtr, TensorPtr>;
using DataLoader = ttml::datasets::DataLoader<
    ttml::datasets::InMemoryDataset<std::vector<uint8_t>, uint8_t>,
    std::function<BatchType(std::vector<DatasetSample> &&samples)>,
    BatchType>;

constexpr auto model_name = "mlp";
constexpr auto optimizer_name = "optimizer";

template <typename Model>
float evaluate(DataLoader &test_dataloader, Model &model, size_t num_targets) {
    model->eval();
    float num_correct = 0;
    float num_samples = 0;
    for (const auto &[data, target] : test_dataloader) {
        auto output = (*model)(data);
        auto output_vec = ttml::core::to_vector(output->get_value());
        auto target_vec = ttml::core::to_vector(target->get_value());
        for (size_t i = 0; i < output_vec.size(); i += num_targets) {
            auto predicted_class = std::distance(
                output_vec.begin() + i,
                std::max_element(output_vec.begin() + i, output_vec.begin() + (i + num_targets)));
            auto target_class = std::distance(
                target_vec.begin() + i,
                std::max_element(target_vec.begin() + i, target_vec.begin() + (i + num_targets)));
            num_correct += static_cast<float>(predicted_class == target_class);
            num_samples++;
        }
    }
    model->train();
    return num_correct / num_samples;
};

int main(int argc, char **argv) {
    CLI::App app{"Mnist Example"};
    argv = app.ensure_utf8(argv);

    uint32_t batch_size = 128;
    int logging_interval = 50;
    size_t num_epochs = 10;
    bool is_eval = false;
    int model_save_interval = 500;
    std::string model_path = "/tmp/mnist_mlp.msgpack";

    app.add_option("-b,--batch_size", batch_size, "Batch size")->default_val(batch_size);
    app.add_option("-l,--logging_interval", logging_interval, "Logging interval")->default_val(logging_interval);
    app.add_option("-m,--model_save_interval", model_save_interval, "model save interval")
        ->default_val(model_save_interval);

    app.add_option("-n,--num_epochs", num_epochs, "Number of epochs")->default_val(num_epochs);
    app.add_option("-s,--model_path", model_path, "Model path")->default_val(model_path);
    app.add_option("-e,--eval", is_eval, "eval only mode")->default_val(is_eval);

    CLI11_PARSE(app, argc, argv);
    // Load MNIST data
    const size_t num_targets = 10;
    const size_t num_features = 784;
    mnist::MNIST_dataset<std::vector, std::vector<uint8_t>, uint8_t> dataset =
        mnist::read_dataset<std::vector, std::vector, uint8_t, uint8_t>(MNIST_DATA_LOCATION);
    ttml::datasets::InMemoryDataset<std::vector<uint8_t>, uint8_t> training_dataset(
        dataset.training_images, dataset.training_labels);
    ttml::datasets::InMemoryDataset<std::vector<uint8_t>, uint8_t> test_dataset(
        dataset.test_images, dataset.test_labels);

    auto *device = &ttml::autograd::ctx().get_device();
    std::function<BatchType(std::vector<DatasetSample> && samples)> collate_fn =
        [num_features, num_targets, device](std::vector<DatasetSample> &&samples) {
            const uint32_t batch_size = samples.size();
            std::vector<float> data;
            std::vector<float> targets;
            data.reserve(batch_size * num_features);
            targets.reserve(batch_size * num_targets);
            for (auto &[features, target] : samples) {
                std::copy(features.begin(), features.end(), std::back_inserter(data));

                std::vector<float> one_hot_target(num_targets, 0.0F);
                one_hot_target[target] = 1.0F;
                std::copy(one_hot_target.begin(), one_hot_target.end(), std::back_inserter(targets));
            }

            std::transform(data.begin(), data.end(), data.begin(), [](float pixel) { return pixel / 255.0F - 0.5F; });

            auto data_tensor = ttml::autograd::create_tensor(
                ttml::core::from_vector(data, ttml::core::create_shape({batch_size, 1, 1, num_features}), device));
            auto targets_tensor = ttml::autograd::create_tensor(
                ttml::core::from_vector(targets, ttml::core::create_shape({batch_size, 1, 1, num_targets}), device));
            return std::make_pair(data_tensor, targets_tensor);
        };

    auto train_dataloader = DataLoader(training_dataset, batch_size, /* shuffle */ true, collate_fn);
    auto test_dataloader = DataLoader(test_dataset, batch_size, /* shuffle */ false, collate_fn);

    auto model = create_base_mlp(784, 10);

    const float learning_rate = 0.1F * (static_cast<float>(batch_size) / 128.F);
    const float momentum = 0.9F;
    const float weight_decay = 0.F;
    auto sgd_config =
        ttml::optimizers::SGDConfig{.lr = learning_rate, .momentum = momentum, .weight_decay = weight_decay};

    fmt::print("SGD configuration:\n");
    fmt::print("    Learning rate: {}\n", sgd_config.lr);
    fmt::print("    Momentum: {}\n", sgd_config.momentum);
    fmt::print("    Dampening {}\n", sgd_config.dampening);
    fmt::print("    Weight decay: {}\n", sgd_config.weight_decay);
    fmt::print("    Nesterov: {}\n", sgd_config.nesterov);
    auto optimizer = ttml::optimizers::SGD(model->parameters(), sgd_config);
    if (!model_path.empty() && std::filesystem::exists(model_path)) {
        fmt::print("Loading model from {}\n", model_path);
        load_model_and_optimizer(model_path, model, optimizer, model_name, optimizer_name);
    }

    // evaluate model before training (sanity check to get reasonable accuracy
    // 1/num_targets)
    float accuracy_before_training = evaluate(test_dataloader, model, num_targets);
    fmt::print("Accuracy of the current model training: {}%\n", accuracy_before_training * 100.F);
    if (is_eval) {
        return 0;
    }

    LossAverageMeter loss_meter;
    int training_step = 0;
    for (size_t epoch = 0; epoch < num_epochs; ++epoch) {
        for (const auto &[data, target] : train_dataloader) {
            optimizer.zero_grad();
            auto output = (*model)(data);
            auto loss = ttml::ops::cross_entropy_loss(output, target);
            auto loss_float = ttml::core::to_vector(loss->get_value())[0];
            loss_meter.update(loss_float, batch_size);
            if (training_step % logging_interval == 0) {
                fmt::print("Step: {:5d} | Average Loss: {:.4f}\n", training_step, loss_meter.average());
            }
            if (!model_path.empty() && training_step % model_save_interval == 0) {
                fmt::print("Saving model to {}\n", model_path);
                save_model_and_optimizer(model_path, model, optimizer, model_name, optimizer_name);
            }

            loss->backward();
            optimizer.step();
            ttml::autograd::ctx().reset_graph();
            training_step++;
        }

        const float test_accuracy = evaluate(test_dataloader, model, num_targets);
        fmt::print(
            "Epoch: {:3d} | Average Loss: {:.4f} | Accuracy: {:.4f}%\n",
            epoch + 1,
            loss_meter.average(),
            test_accuracy * 100.F);
        loss_meter.reset();
    }

    if (!model_path.empty()) {
        fmt::print("Saving model to {}\n", model_path);
        save_model_and_optimizer(model_path, model, optimizer, model_name, optimizer_name);
    }

    return 0;
}
