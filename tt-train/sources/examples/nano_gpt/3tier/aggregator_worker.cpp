// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <fmt/core.h>

#include <CLI/CLI.hpp>

#include "autograd/module_base.hpp"
#include "common.hpp"
#include "core/distributed/distributed.hpp"
#include "datasets/utils.hpp"
#include "models/distributed/gpt2.hpp"
#include "models/gpt2.hpp"
#include "socket_manager.hpp"
#include "tokenizers/bpe_tokenizer.hpp"
#include "tokenizers/char_tokenizer.hpp"
#include "ttnn_fixed/distributed/ttnn_ops.hpp"

using SortedParameters = std::map<std::string, ttml::autograd::TensorPtr>;
using Rank = ttml::core::distributed::Rank;
using Tag = ttml::core::distributed::Tag;

void send_aggregated_gradients_from_workers_to_optimizer(
    SocketManager &socket_manager,
    const std::shared_ptr<ttml::core::distributed::DistributedContext> &workers_and_aggregator_ctx,
    const std::shared_ptr<ttml::core::distributed::DistributedContext> &aggregator_and_optimizer_ctx,
    const SortedParameters &sorted_model_parameters,
    int workers,
    bool is_ddp = false) {
    Rank optimizer_rank{aggregator_and_optimizer_ctx->rank().get() + 1};
    for (auto &[name, tensor_ptr] : sorted_model_parameters) {
        if (!tensor_ptr->get_requires_grad()) {
            continue;
        }

        // TODO: allow usage of tensor from model parameters (avoids redundant storage of a model)
        auto tensor = ttnn::empty_like(tensor_ptr->get_value());
        socket_manager.recv(tensor, workers_and_aggregator_ctx, ttml::core::distributed::Rank{0});
        for (int worker_id = 1; worker_id < workers; ++worker_id) {
            auto tensor_to_add = ttnn::empty_like(tensor_ptr->get_value());
            socket_manager.recv(tensor_to_add, workers_and_aggregator_ctx, ttml::core::distributed::Rank{worker_id});
            tensor = ttnn::add(tensor, tensor_to_add);
        }
        tensor = ttnn::multiply(tensor, 1.0F / static_cast<float>(workers));
        if (is_ddp) {
            tensor = ttml::ttnn_fixed::distributed::all_reduce(tensor);
        }
        socket_manager.send(tensor, aggregator_and_optimizer_ctx, optimizer_rank);
    }
}

void send_weights_from_optimizer_to_workers(
    SocketManager &socket_manager,
    const std::shared_ptr<ttml::core::distributed::DistributedContext> &workers_and_aggregator_ctx,
    const std::shared_ptr<ttml::core::distributed::DistributedContext> &aggregator_and_optimizer_ctx,
    const SortedParameters &sorted_model_parameters,
    int workers) {
    Rank optimizer_rank{aggregator_and_optimizer_ctx->rank().get() + 1};
    for (auto &[name, tensor_ptr] : sorted_model_parameters) {
        auto tensor = tensor_ptr->get_value();
        socket_manager.recv(tensor, aggregator_and_optimizer_ctx, ttml::core::distributed::Rank{optimizer_rank});

        for (int worker_id = 0; worker_id < workers; ++worker_id) {
            socket_manager.send(tensor, workers_and_aggregator_ctx, ttml::core::distributed::Rank{worker_id});
        }
    }
}

int main(int argc, char **argv) {
    std::cout << "Running aggregator worker" << std::endl;
    auto &ctx = ttml::autograd::ctx();
    ctx.initialize_distributed_context(argc, argv);
    auto distributed_ctx = ctx.get_distributed_context();

    CLI::App app{"Multihost Example"};
    fmt::print("Size {}, Rank {}: Initializing MPI context\n", *distributed_ctx->size(), *distributed_ctx->rank());
    argv = app.ensure_utf8(argv);

    std::string config_name = std::string(CONFIGS_FOLDER) + "/training_shakespeare_nanogpt_3tier.yaml";
    app.add_option("-c,--config", config_name, "Yaml Config name")->default_val(config_name);
    CLI11_PARSE(app, argc, argv);

    auto yaml_config = YAML::LoadFile(config_name);
    three_tier_arch::TrainingConfig config = three_tier_arch::parse_config(yaml_config);
    three_tier_arch::DeviceConfig device_config = three_tier_arch::parse_device_config(yaml_config);

    if (config.socket_type == ttnn::distributed::SocketType::FABRIC) {
        tt::tt_fabric::SetFabricConfig(tt::tt_fabric::FabricConfig::FABRIC_2D_DYNAMIC);
        if (device_config.mesh_shape != tt::tt_metal::distributed::MeshShape(1, 8)) {
            throw std::runtime_error(fmt::format(
                "Fabric config is set to 2D dynamic, but mesh shape is not (1, 8). Mesh shape: {}",
                device_config.mesh_shape));
        }
    }
    three_tier_arch::initialize_device(device_config.mesh_shape, device_config.device_ids);

    auto socket_manager = SocketManager(config.socket_type);

    auto [steps_per_dataset, vocab_size] = three_tier_arch::get_steps_per_dataset_and_vocab_size(config);
    auto *device = &ttml::autograd::ctx().get_device();

    auto num_devices = static_cast<uint32_t>(device->num_devices());
    auto should_be_divisible_by = (device_config.enable_tp ? num_devices : 1U) * 32U;
    vocab_size = three_tier_arch::round_up_to_tile(vocab_size, should_be_divisible_by);
    std::visit(
        [&](auto &&arg) {
            if constexpr (requires { arg.vocab_size; }) {
                arg.vocab_size = vocab_size;
            } else {
                throw std::runtime_error(
                    "Unsupported transformer configuration type: " + std::string(typeid(arg).name()));
            }
        },
        config.transformer_config);

    auto model = std::visit(
        [&device_config](auto &&arg) -> std::shared_ptr<ttml::autograd::ModuleBase> {
            if constexpr (std::is_same_v<std::decay_t<decltype(arg)>, ttml::models::llama::LlamaConfig>) {
                if (device_config.enable_tp) {
                    return ttml::models::distributed::llama::create(arg);
                } else {
                    return ttml::models::llama::create(arg);
                }
            } else if constexpr (std::is_same_v<std::decay_t<decltype(arg)>, ttml::models::gpt2::TransformerConfig>) {
                if (device_config.enable_tp) {
                    return ttml::models::distributed::gpt2::create(arg);
                } else {
                    return ttml::models::gpt2::create(arg);
                }
            } else {
                throw std::runtime_error(
                    "Unsupported transformer configuration type: " + std::string(typeid(arg).name()));
            }
        },
        config.transformer_config);

    auto model_parameters = model->parameters();
    auto sorted_model_parameters = SortedParameters(model_parameters.begin(), model_parameters.end());

    auto workers = config.num_mh_workers;

    auto workers_and_aggregator_ranks =
        three_tier_arch::get_workers_and_aggregator_ranks(static_cast<uint32_t>(distributed_ctx->rank().get()));
    auto workers_and_aggregator_ctx =
        ttml::autograd::ctx().get_distributed_context()->create_sub_context(workers_and_aggregator_ranks);

    auto aggregator_and_optimizer_ranks =
        std::vector<int>{distributed_ctx->rank().get(), distributed_ctx->rank().get() + 1};
    auto aggregator_and_optimizer_ctx =
        ttml::autograd::ctx().get_distributed_context()->create_sub_context(aggregator_and_optimizer_ranks);

    send_weights_from_optimizer_to_workers(
        socket_manager, workers_and_aggregator_ctx, aggregator_and_optimizer_ctx, sorted_model_parameters, workers);

    uint32_t global_step = 0;
    for (uint32_t epoch = 0; epoch < config.num_epochs; ++epoch) {
        for (uint32_t step = 0; step < steps_per_dataset; ++step, ++global_step) {
            send_aggregated_gradients_from_workers_to_optimizer(
                socket_manager,
                workers_and_aggregator_ctx,
                aggregator_and_optimizer_ctx,
                sorted_model_parameters,
                workers,
                device_config.enable_ddp);
            send_weights_from_optimizer_to_workers(
                socket_manager,
                workers_and_aggregator_ctx,
                aggregator_and_optimizer_ctx,
                sorted_model_parameters,
                workers);
            if (global_step >= config.max_steps) {
                break;
            }
        }
        if (global_step >= config.max_steps) {
            break;
        }
    }

    distributed_ctx->barrier();
    fmt::print("Rank {}: Finalized MPI context\n", distributed_ctx->rank().get());
    return 0;
}
