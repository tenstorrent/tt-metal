// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <fmt/format.h>

#include <CLI/CLI.hpp>
#include <core/ttnn_all_includes.hpp>
#include <string>

#include "autograd/auto_context.hpp"
#include "autograd/tensor.hpp"
#include "core/distributed/distributed.hpp"
#include "core/tt_tensor_utils.hpp"
#include "core/xtensor_utils.hpp"
#include "datasets/dataloader.hpp"
#include "datasets/generators.hpp"
#include "models/linear_regression.hpp"
#include "modules/distributed/linear.hpp"
#include "ops/losses.hpp"
#include "optimizers/sgd.hpp"
#include "ttnn_fixed/distributed/tt_metal.hpp"

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
    CLI::App app{"Linear Regression TP+DP Example"};
    argv = app.ensure_utf8(argv);

    // Training hyperparameters
    const size_t training_samples_count = 100000;
    const size_t num_features = 64;
    const size_t num_targets = 64;
    const float noise = 0.0F;
    const bool bias = true;
    // Parse command line arguments
    // Default to column parallelism, use row if --row flag is passed
    bool use_row_parallel = false;
    uint32_t batch_size = 8192;
    std::string mesh_shape_str = "8x4";

    app.add_flag("--row", use_row_parallel, "Use RowParallelLinear (shards input features), default is ColumnParallelLinear");
    app.add_option("-b,--batch_size", batch_size, "Batch size")->default_val(batch_size);
    app.add_option("--mesh_shape", mesh_shape_str, "Logical mesh shape RxC (e.g. 8x4)")->default_val(mesh_shape_str);

    CLI11_PARSE(app, argc, argv);

    uint32_t mesh_rows = 4;
    uint32_t mesh_cols = 8;
    if (!parse_mesh_shape(mesh_shape_str, mesh_rows, mesh_cols)) {
        fmt::print(stderr, "Error: invalid --mesh_shape '{}', expected RxC like 8x4\n", mesh_shape_str);
        return 1;
    }

    TT_FATAL(
        mesh_rows * mesh_cols == 32,
        "mesh_rows and mesh_cols must be greater than 0 and their product must be 32 (whole galaxy).");

    // - DP groups (data parallelism) along mesh dimension 0
    // - TP devices per group (tensor parallelism) along mesh dimension 1
    // you need a right mgd config file for this (default mesh shape is 1x32, look at enable_fabric function for the mgd
    // config file)
    const auto logical_mesh_shape = tt::tt_metal::distributed::MeshShape(mesh_rows, mesh_cols);
    const uint32_t num_devices = logical_mesh_shape[0] * logical_mesh_shape[1];
    // In these examples, I assume dp is always the first mesh axis and tp is the second one, which will be
    // preserved later on when adding tp+dp llm training support. A user will set if they want to use data
    // parallel or not and which mesh device shape they want to use, the axis will be
    // decided automatically: data parallel will always be the first one if
    // present, the second will be cp (if present, if dp is disabled --- cp will be the first one), then pp,
    // tp and ep.

    ttml::ttnn_fixed::distributed::enable_fabric(num_devices);
    ttml::autograd::ctx().open_device(logical_mesh_shape);
    auto* device = &ttml::autograd::ctx().get_device();

    // Initialize parallelism context for TP+DP
    ttml::autograd::ctx().initialize_parallelism_context({.enable_ddp = true, .enable_tp = true});

    // Get parallelism parameters from context
    const auto& pctx = ttml::autograd::ctx().get_parallelism_context();
    auto tp_axis = pctx.get_tp_axis();
    const uint32_t dp_size = pctx.get_ddp_size();
    const uint32_t tp_size = pctx.get_tp_size();

    TT_FATAL(num_features % tp_size == 0, "num_features must be divisible by tp_size (going to be sharded)");
    TT_FATAL(num_targets % tp_size == 0, "num_targets must be divisible by tp_size (going to be sharded)");
    TT_FATAL(batch_size % dp_size == 0, "batch_size must be divisible by dp_size (going to be sharded)");

    // Validate that batch_size is divisible by dp_size
    if (batch_size == 0) {
        fmt::print(stderr, "Error: batch_size must be > 0\n");
        return 1;
    }
    if (batch_size % dp_size != 0) {
        fmt::print(stderr,
                   "Error: batch_size ({}) must be divisible by dp_size ({})\n",
                   batch_size,
                   dp_size);
        return 1;
    }

    // Generate training dataset
    auto training_params = ttml::datasets::MakeRegressionParams{
        .n_samples = training_samples_count,
        .n_features = num_features,
        .n_targets = num_targets,
        .noise = noise,
        .bias = bias,
    };
    auto training_dataset = ttml::datasets::make_regression(training_params);

    // Collate function: prepare batch data for TP+DP training
    std::function<BatchType(std::vector<DatasetSample> && samples)> collate_fn =
        [device, logical_mesh_shape, use_row_parallel](std::vector<DatasetSample>&& samples) -> BatchType {
        const uint32_t actual_batch_size = samples.size();

        // Flatten samples into contiguous vectors
        std::vector<float> data;
        std::vector<float> targets;
        data.reserve(actual_batch_size * num_features);
        targets.reserve(actual_batch_size * num_targets);
        for (auto& [features, target] : samples) {
            std::move(features.begin(), features.end(), std::back_inserter(data));
            std::move(target.begin(), target.end(), std::back_inserter(targets));
        }

        // Configure data mapper based on parallelism type
        tt::tt_metal::distributed::MeshMapperConfig data_config;
        data_config.placements.push_back(tt::tt_metal::distributed::MeshMapperConfig::Shard{0});  // DP: shard batch
        if (use_row_parallel) {
            // RowParallelLinear: input is sharded, so data should be sharded
            data_config.placements.push_back(tt::tt_metal::distributed::MeshMapperConfig::Shard{3});  // TP: shard
        } else {
            // ColumnParallelLinear: input is broadcast, so data should be broadcast
            data_config.placements.push_back(
                tt::tt_metal::distributed::MeshMapperConfig::Replicate{});  // TP: broadcast
        }
        data_config.mesh_shape_override = logical_mesh_shape;
        const auto data_mapper = ttnn::distributed::create_mesh_mapper(*device, data_config);

        // Create data tensor with proper sharding
        auto data_tensor = ttml::autograd::create_tensor(ttml::core::from_vector(
            data, ttnn::Shape{actual_batch_size, 1, 1, num_features}, device, ttnn::Layout::TILE, data_mapper.get()));

        // Configure targets mapper based on parallelism type
        tt::tt_metal::distributed::MeshMapperConfig targets_config;
        targets_config.placements.push_back(tt::tt_metal::distributed::MeshMapperConfig::Shard{0});  // DP: shard batch
        if (use_row_parallel) {
            // RowParallelLinear: output is always replicated (via all_reduce), so targets should be replicated
            targets_config.placements.push_back(
                tt::tt_metal::distributed::MeshMapperConfig::Replicate{});  // TP: replicate
        } else {
            // ColumnParallelLinear with gather_output=false: output is sharded, so targets should be sharded
            targets_config.placements.push_back(
                tt::tt_metal::distributed::MeshMapperConfig::Shard{3});  // TP: shard output features
        }
        targets_config.mesh_shape_override = logical_mesh_shape;
        const auto targets_mapper = ttnn::distributed::create_mesh_mapper(*device, targets_config);

        // Create targets tensor
        auto targets_tensor = ttml::autograd::create_tensor(ttml::core::from_vector(
            targets,
            ttnn::Shape{actual_batch_size, 1, 1, num_targets},
            device,
            ttnn::Layout::TILE,
            targets_mapper.get()));

        return {data_tensor, targets_tensor};
    };

    auto train_dataloader = DataLoader(training_dataset, batch_size, /* shuffle */ true, collate_fn);

    // Initialize model based on parallelism type
    // shard_dim specifies that weights should be sharded along TP dimension
    std::shared_ptr<ttml::modules::ModuleBase> model;
    if (use_row_parallel) {
        fmt::print("Using RowParallelLinear: shards input features, all_reduces output\n");
        // RowParallelLinear: shards input features, all_reduces output
        model = std::make_shared<ttml::modules::distributed::RowParallelLinear>(
            num_features, num_targets, /* has_bias */ bias, /* input_is_parallel */ true, /* shard_dim */ tp_axis);
    } else {
        fmt::print("Using ColumnParallelLinear: shards output features, sharded output\n");
        // ColumnParallelLinear: shards output features, keeps output sharded
        model = std::make_shared<ttml::modules::distributed::ColumnParallelLinear>(
            num_features, num_targets, /* has_bias */ bias, /* gather_output */ false, /* shard_dim */ tp_axis);
    }
    fmt::print("Batch size: {}, DP groups: {}, TP size: {}\n", batch_size, dp_size, tp_size);

    // Configure optimizer
    float learning_rate = 0.1F * num_targets * (batch_size / 128.F); /* Denys's lr*/
    if (!use_row_parallel) {
        /* loss is calculated for each tp partition, so it is averaged over 1/tp_size times less samples making gradient
         * tp_size times greater*/
        learning_rate /= tp_size;
    }

    auto sgd_config = ttml::optimizers::SGDConfig{.lr = learning_rate, .momentum = 0.0F};
    auto optimizer = ttml::optimizers::SGD(model->parameters(), sgd_config);


    auto get_loss_value = [](const TensorPtr& loss) {
        auto loss_xtensors = ttml::core::to_xtensor(loss->get_value(), ttml::core::IdentityComposer{});
        float loss_float = std::accumulate(
            loss_xtensors.begin(), loss_xtensors.end(), 0.0F, [](float acc, auto& xtensor) {
                return acc + xtensor(0);
            });
        const float mean_loss = loss_float / static_cast<float>(loss_xtensors.size());
        return mean_loss;
    };

    // Training loop
    int training_step = 0;
    const int num_epochs = 10;
    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        for (const auto& [data, targets] : train_dataloader) {
            optimizer.zero_grad();

            // Forward pass
            auto output = (*model)(data);
            auto loss = ttml::ops::mse_loss(output, targets);

            // Log loss
            fmt::print("Step: {} Loss: {}\n", training_step++, get_loss_value(loss));

            // Backward pass
            loss->backward();

            // Synchronize gradients across DP groups (average gradients for data parallelism)
            ttml::core::distributed::synchronize_gradients(model->parameters());

            // Optimizer step
            optimizer.step();
            ttml::autograd::ctx().reset_graph();
        }
    }
}
