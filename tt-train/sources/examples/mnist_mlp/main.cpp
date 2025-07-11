// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <yaml-cpp/node/node.h>

#include <CLI/CLI.hpp>
#include <chrono>
#include <core/ttnn_all_includes.hpp>
#include <cstdint>
#include <functional>
#include <memory>
#include <mnist/mnist_reader.hpp>
#include <variant>

#include "autograd/auto_context.hpp"
#include "autograd/tensor.hpp"
#include "core/tt_tensor_utils.hpp"
#include "datasets/dataloader.hpp"
#include "datasets/in_memory_dataset.hpp"
#include "model.hpp"
#include "models/mlp.hpp"
#include "ops/losses.hpp"
#include "optimizers/sgd.hpp"
#include "serialization/serializable.hpp"
#include "utils.hpp"

using ttml::autograd::TensorPtr;

using DatasetSample = std::pair<std::vector<uint8_t>, uint8_t>;
using BatchType = std::pair<TensorPtr, TensorPtr>;
using DataLoader = ttml::datasets::DataLoader<
    ttml::datasets::InMemoryDataset<std::vector<uint8_t>, uint8_t>,
    std::function<BatchType(std::vector<DatasetSample> &&samples)>,
    BatchType>;

using Model = std::variant<std::shared_ptr<ttml::modules::MultiLayerPerceptron>, std::shared_ptr<MNISTTensorParallel>>;

const std::string model_name = "mlp";
const std::string optimizer_name = "optimizer";

struct TrainingConfig {
    uint32_t batch_size = 128;
    int logging_interval = 50;
    size_t num_epochs = 10;
    float learning_rate = 0.1;
    float momentum = 0.9F;
    float weight_decay = 0.F;
    int model_save_interval = 500;
    std::string model_path = "/tmp/mnist_mlp.msgpack";
    ttml::modules::MultiLayerPerceptronParameters mlp_config;
};

TrainingConfig parse_config(const YAML::Node &yaml_config) {
    TrainingConfig config;
    auto training_config = yaml_config["training_config"];

    config.batch_size = training_config["batch_size"].as<uint32_t>();
    config.logging_interval = training_config["logging_interval"].as<int>();
    config.num_epochs = training_config["num_epochs"].as<size_t>();
    config.learning_rate = training_config["learning_rate"].as<float>();
    config.momentum = training_config["momentum"].as<float>();
    config.weight_decay = training_config["weight_decay"].as<float>();
    config.model_save_interval = training_config["model_save_interval"].as<int>();
    config.mlp_config = ttml::models::mlp::read_config(training_config["mlp_config"]);
    return config;
}

void initialize_device(bool enable_tp) {
    if (enable_tp) {
        // we support only N300 for now
        ttml::autograd::ctx().open_device(tt::tt_metal::distributed::MeshShape(1, 2));
    } else {
        // use single device defaults
        ttml::autograd::ctx().open_device();
    }
}

void model_to_eval(Model &model) {
    std::visit([](auto &model) { model->eval(); }, model);
}

void model_to_train(Model &model) {
    std::visit([](auto &model) { model->train(); }, model);
}

ttml::autograd::TensorPtr run_model(Model &model, const ttml::autograd::TensorPtr &data) {
    return std::visit([&data](auto &model) { return (*model)(data); }, model);
}

ttml::serialization::NamedParameters get_model_parameters(Model &model) {
    return std::visit([](auto &model) { return model->parameters(); }, model);
}

void load_model(
    Model &model,
    const TrainingConfig &config,
    ttml::optimizers::SGD &optimizer,
    const std::string &model_name,
    const std::string &optimizer_name) {
    std::visit(
        [&config, &optimizer, &model_name, &optimizer_name](auto &model) {
            load_training_state(config.model_path, model, optimizer, model_name, optimizer_name);
        },
        model);
}

void save_model(
    Model &model,
    const TrainingConfig &config,
    ttml::optimizers::SGD &optimizer,
    const std::string &model_name,
    const std::string &optimizer_name) {
    std::visit(
        [&config, &optimizer, &model_name, &optimizer_name](auto &model) {
            save_training_state(config.model_path, model, optimizer, model_name, optimizer_name);
        },
        model);
}

template <typename Model>
float evaluate(DataLoader &test_dataloader, Model &model, size_t num_targets) {
    model_to_eval(model);
    float num_correct = 0;
    float num_samples = 0;
    auto *device = &ttml::autograd::ctx().get_device();
    for (const auto &[data, target] : test_dataloader) {
        auto output = run_model(model, data);
        ttml::core::MeshToXTensorVariant<float> composer = ttml::core::VectorMeshToXTensor<float>(device->shape());
        ttml::core::MeshToXTensorVariant<uint32_t> target_composer =
            ttml::core::VectorMeshToXTensor<uint32_t>(device->shape());
        auto output_xtensor = ttml::core::to_xtensor(output->get_value(), composer)[0];
        auto target_xtensor = ttml::core::to_xtensor<uint32_t>(target->get_value(), target_composer)[0];
        auto output_vec = std::vector<float>(output_xtensor.begin(), output_xtensor.end());
        auto target_vec = std::vector<uint32_t>(target_xtensor.begin(), target_xtensor.end());
        for (size_t i = 0; i < output_vec.size(); i += num_targets) {
            auto predicted_class = std::distance(
                output_vec.begin() + i,
                std::max_element(output_vec.begin() + i, output_vec.begin() + (i + num_targets)));
            auto target_class = target_vec[i / num_targets];
            num_correct += static_cast<float>(predicted_class == target_class);
            num_samples++;
        }
    }
    model_to_train(model);
    return num_correct / num_samples;
};

int main(int argc, char **argv) {
    CLI::App app{"Mnist Example"};
    argv = app.ensure_utf8(argv);

    std::string config_name = std::string(CONFIGS_FOLDER) + "/training_mnist_mlp.yaml";
    bool is_eval = false;
    bool enable_tp = false;
    app.add_option("-c,--config", config_name, "Yaml Config name")->default_val(config_name);
    app.add_option("-e,--eval", is_eval, "Evaluate")->default_val(is_eval);
    app.add_option("-p, --enable_tp", enable_tp, "Enable tensor parallelism")->default_val(enable_tp);

    CLI11_PARSE(app, argc, argv);
    auto yaml_config = YAML::LoadFile(config_name);
    TrainingConfig config = parse_config(yaml_config);

    initialize_device(enable_tp);

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
    device->enable_program_cache();
    std::function<BatchType(std::vector<DatasetSample> && samples)> collate_fn =
        [num_features, num_targets, device](std::vector<DatasetSample> &&samples) {
            const uint32_t batch_size = samples.size();
            std::vector<float> data;
            std::vector<uint32_t> targets;
            data.reserve(batch_size * num_features);
            targets.reserve(batch_size);
            for (auto &[features, target] : samples) {
                std::copy(features.begin(), features.end(), std::back_inserter(data));
                targets.push_back(static_cast<uint32_t>(target));
            }

            std::transform(data.begin(), data.end(), data.begin(), [](float pixel) { return pixel / 255.0F - 0.5F; });

            auto data_tensor = ttml::autograd::create_tensor(
                ttml::core::from_vector(data, ttnn::Shape({batch_size, 1, 1, num_features}), device));

            auto targets_tensor =
                ttml::autograd::create_tensor(ttml::core::from_vector<uint32_t, ttnn::DataType::UINT32>(
                    targets, ttnn::Shape({batch_size, 1U}), device, ttnn::Layout::ROW_MAJOR));
            return std::make_pair(data_tensor, targets_tensor);
        };

    auto train_dataloader = DataLoader(training_dataset, config.batch_size, /* shuffle */ true, collate_fn);
    auto test_dataloader = DataLoader(test_dataset, config.batch_size, /* shuffle */ false, collate_fn);

    Model model;
    if (enable_tp) {
        model = std::make_shared<MNISTTensorParallel>();
    } else {
        model = ttml::models::mlp::create(config.mlp_config);
    }

    const float learning_rate = config.learning_rate * (static_cast<float>(config.batch_size) / 128.F);
    const float momentum = config.momentum;
    const float weight_decay = config.weight_decay;
    auto sgd_config =
        ttml::optimizers::SGDConfig{.lr = learning_rate, .momentum = momentum, .weight_decay = weight_decay};

    fmt::print("SGD configuration:\n");
    fmt::print("    Learning rate: {}\n", sgd_config.lr);
    fmt::print("    Momentum: {}\n", sgd_config.momentum);
    fmt::print("    Dampening {}\n", sgd_config.dampening);
    fmt::print("    Weight decay: {}\n", sgd_config.weight_decay);
    fmt::print("    Nesterov: {}\n", sgd_config.nesterov);
    auto parameters = get_model_parameters(model);
    auto optimizer = ttml::optimizers::SGD(parameters, sgd_config);
    if (!config.model_path.empty() && std::filesystem::exists(config.model_path)) {
        if (enable_tp) {
            fmt::println("Loading model for tensor parallelism is not supported yet. Loading model has been skipped.");
        } else {
            fmt::print("Loading model from {}\n", config.model_path);
            load_model(model, config, optimizer, model_name, optimizer_name);
        }
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

    auto get_loss_value = [device](const TensorPtr &loss) {
        ttml::core::MeshToXTensorVariant<float> composer = ttml::core::VectorMeshToXTensor<float>(device->shape());
        auto loss_xtensors = ttml::core::to_xtensor(loss->get_value(), composer);
        // sum of loss xtensors
        float loss_float =
            std::accumulate(loss_xtensors.begin(), loss_xtensors.end(), 0.0F, [](float acc, auto &xtensor) {
                return acc + xtensor(0);
            });

        return loss_float / static_cast<float>(loss_xtensors.size());
    };

    for (size_t epoch = 0; epoch < config.num_epochs; ++epoch) {
        auto start_time = std::chrono::steady_clock::now();
        uint32_t num_steps_in_epoch = 0;
        for (const auto &[data, target] : train_dataloader) {
            optimizer.zero_grad();
            auto output = run_model(model, data);
            auto loss = ttml::ops::cross_entropy_loss(output, target);
            auto loss_float = get_loss_value(loss);
            loss_meter.update(loss_float, config.batch_size);
            if (training_step % config.logging_interval == 0) {
                fmt::print("Step: {:5d} | Average Loss: {:.4f}\n", training_step, loss_meter.average());
            }
            if (!config.model_path.empty() && training_step % config.model_save_interval == 0) {
                fmt::print("Saving model to {}\n", config.model_path);
                save_model(model, config, optimizer, model_name, optimizer_name);
            }

            loss->backward();
            optimizer.step();
            ttml::autograd::ctx().reset_graph();
            training_step++;
            num_steps_in_epoch++;
        }

        auto end_time = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        fmt::println("per step ms: {} ms", duration.count() / num_steps_in_epoch);

        const float test_accuracy = evaluate(test_dataloader, model, num_targets);
        fmt::print(
            "Epoch: {:3d} | Average Loss: {:.4f} | Accuracy: {:.4f}%\n",
            epoch + 1,
            loss_meter.average(),
            test_accuracy * 100.F);
        loss_meter.reset();
    }

    if (!config.model_path.empty()) {
        fmt::print("Saving model to {}\n", config.model_path);
        save_model(model, config, optimizer, model_name, optimizer_name);
    }

    return 0;
}
