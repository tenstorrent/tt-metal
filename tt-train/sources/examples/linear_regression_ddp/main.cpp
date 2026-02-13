// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <fmt/format.h>

#include <CLI/CLI.hpp>
#include <core/ttnn_all_includes.hpp>
#include <cstdlib>
#include <string>

#include "autograd/auto_context.hpp"
#include "autograd/tensor.hpp"
#include "core/distributed/distributed.hpp"
#include "core/tt_tensor_utils.hpp"
#include "core/xtensor_utils.hpp"
#include "datasets/dataloader.hpp"
#include "datasets/generators.hpp"
#include "models/linear_regression.hpp"
#include "modules/linear_module.hpp"
#include "ops/losses.hpp"
#include "optimizers/sgd.hpp"
#include "ttnn_fixed/distributed/tt_metal.hpp"
#include "utils/memory_utils.hpp"
#include "utils/model_utils.hpp"

using ttml::autograd::TensorPtr;

using DatasetSample = std::pair<std::vector<float>, std::vector<float>>;
using BatchType = std::pair<TensorPtr, TensorPtr>;
using DataLoader = ttml::datasets::DataLoader<
    ttml::datasets::InMemoryFloatVecDataset,
    std::function<BatchType(std::vector<DatasetSample>&& samples)>,
    BatchType>;

namespace {
bool parse_mesh_shape(const std::string& mesh_shape_str, uint32_t& rows, uint32_t& cols) {
    const auto delimiter_pos = mesh_shape_str.find('x');
    if (delimiter_pos == std::string::npos || delimiter_pos == 0 || delimiter_pos + 1 >= mesh_shape_str.size()) {
        return false;
    }

    try {
        rows = static_cast<uint32_t>(std::stoul(mesh_shape_str.substr(0, delimiter_pos)));
        cols = static_cast<uint32_t>(std::stoul(mesh_shape_str.substr(delimiter_pos + 1)));
    } catch (const std::exception&) {
        return false;
    }

    return rows > 0 && cols > 0;
}
}  // namespace

int main(int argc, char** argv) {
    CLI::App app{"Linear Regression DDP Example"};
    argv = app.ensure_utf8(argv);

    uint32_t batch_size = 128;
    const size_t training_samples_count = 1000;
    const size_t num_features = 32;
    const size_t num_targets = 32;
    const float noise = 0.0F;
    const bool bias = true;

    std::string mesh_shape_str = "32x1";
    app.add_option("--mesh_shape", mesh_shape_str, "Logical mesh shape RxC (e.g. 32x1)")->default_val(mesh_shape_str);

    CLI11_PARSE(app, argc, argv);

    uint32_t mesh_rows = 4;
    uint32_t mesh_cols = 8;
    if (!parse_mesh_shape(mesh_shape_str, mesh_rows, mesh_cols)) {
        fmt::print(stderr, "Error: invalid --mesh_shape '{}', expected RxC like 32x1\n", mesh_shape_str);
        return 1;
    }

    TT_FATAL(
        mesh_rows > 0 && mesh_cols > 0 && mesh_rows * mesh_cols == 32,
        "mesh_rows and mesh_cols must be greater than 0 and their product must be 32 (whole galaxy).");

    const auto logical_mesh_shape = tt::tt_metal::distributed::MeshShape(mesh_rows, mesh_cols);
    const auto num_devices = logical_mesh_shape[0] * logical_mesh_shape[1];

    ttml::ttnn_fixed::distributed::enable_fabric(num_devices);
    ttml::autograd::ctx().open_device(logical_mesh_shape);

    // Initialize parallelism context for DDP only
    ttml::autograd::ctx().initialize_parallelism_context({.enable_ddp = true, .enable_tp = false});

    // Get parallelism parameters from context
    const auto& pctx = ttml::autograd::ctx().get_parallelism_context();
    const auto dp_axis = pctx.get_ddp_axis();
    const auto dp_size = pctx.get_ddp_size();

    fmt::print("DDP enabled: {} devices, dp_axis: {}\n", dp_size, dp_axis.value_or(0));

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
        [device](std::vector<DatasetSample>&& samples) -> BatchType {
        const uint32_t batch_size = samples.size();
        std::vector<float> data;
        std::vector<float> targets;
        data.reserve(batch_size * num_features);
        targets.reserve(batch_size * num_targets);
        for (auto& [features, target] : samples) {
            std::move(features.begin(), features.end(), std::back_inserter(data));
            std::move(target.begin(), target.end(), std::back_inserter(targets));
        }

        const auto mapper = ttnn::distributed::shard_tensor_to_mesh_mapper(*device, 0);
        auto data_tensor = ttml::autograd::create_tensor(ttml::core::from_vector(
            data, ttnn::Shape{batch_size, 1, 1, num_features}, device, ttnn::Layout::TILE, mapper.get()));
        auto targets_tensor = ttml::autograd::create_tensor(ttml::core::from_vector(
            targets, ttnn::Shape{batch_size, 1, 1, num_targets}, device, ttnn::Layout::TILE, mapper.get()));

        return {data_tensor, targets_tensor};
    };

    auto train_dataloader = DataLoader(training_dataset, batch_size, /* shuffle */ true, collate_fn);

    auto model = ttml::models::linear_regression::create(num_features, num_targets);

    ttml::utils::MemoryUsageTracker::snapshot("MODEL_CREATION");
    fmt::print("Number of parameters: {}\n", ttml::utils::get_number_of_parameters(model, pctx.is_tp_enabled()));

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
            auto loss_xtensors = ttml::core::to_xtensor(loss->get_value(), ttml::core::IdentityComposer{});
            float mean_loss = 0.0F;
            for (const auto& loss_xtensor : loss_xtensors) {
                mean_loss += loss_xtensor(0);
            }
            mean_loss /= loss_xtensors.size();
            float loss_float_0 = loss_xtensors[0](0);
            float loss_float_1 = loss_xtensors[1](0);
            fmt::print("Step: {} Loss: {} {} {}\n", training_step++, loss_float_0, loss_float_1, mean_loss);
            loss->backward();

            // Synchronize gradients across DDP devices
            ttml::core::distributed::synchronize_gradients(model->parameters());

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
