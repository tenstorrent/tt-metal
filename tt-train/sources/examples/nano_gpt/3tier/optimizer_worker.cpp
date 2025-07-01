// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <fmt/core.h>

#include <CLI/CLI.hpp>

#include "autograd/auto_context.hpp"
#include "common.hpp"
#include "core/distributed/distributed.hpp"
#include "datasets/utils.hpp"
#include "models/distributed/gpt2.hpp"
#include "models/gpt2.hpp"
#include "optimizers/adamw.hpp"
#include "tokenizers/bpe_tokenizer.hpp"
#include "tokenizers/char_tokenizer.hpp"

using SortedParameters = std::map<std::string, ttml::autograd::TensorPtr>;

void send_weights_to_aggregator(
    const ttml::autograd::DistributedContext &ctx, const SortedParameters &sorted_model_parameters) {
    auto aggregator_rank = ttml::core::distributed::Rank(*ctx.rank() - 1);
    for (auto &[name, tensor_ptr] : sorted_model_parameters) {
        if (!tensor_ptr->get_requires_grad()) {
            continue;
        }

        auto tensor = tensor_ptr->get_value();
        ttml::core::distributed::send_tensor(ctx, tensor, aggregator_rank);
    }
}

void receive_gradients_from_aggregator(
    const ttml::autograd::DistributedContext &ctx, const SortedParameters &sorted_model_parameters) {
    auto aggregator_rank = ttml::core::distributed::Rank(*ctx.rank() - 1);
    for (auto &[name, tensor_ptr] : sorted_model_parameters) {
        if (!tensor_ptr->get_requires_grad()) {
            continue;
        }

        auto tensor = ttnn::empty_like(tensor_ptr->get_value());
        ttml::core::distributed::recv_tensor(ctx, tensor, aggregator_rank);
        tensor_ptr->set_grad(tensor);
    }
}

int main(int argc, char **argv) {
    auto &ctx = ttml::autograd::ctx();
    ctx.initialize_distributed_context(argc, argv);
    auto &distributed_ctx = ctx.get_distributed_context();

    CLI::App app{"Multihost Example"};
    fmt::print("Size {}, Rank {}: Initializing MPI context\n", *distributed_ctx.size(), *distributed_ctx.rank());
    argv = app.ensure_utf8(argv);

    std::string config_name = std::string(CONFIGS_FOLDER) + "/training_shakespear_nanogpt_3tier.yaml";

    std::vector<int> aggregator_and_optimizer_ranks = {*distributed_ctx.rank() - 1, *distributed_ctx.rank()};

    auto aggregator_and_optimizer_ctx = distributed_ctx.create_sub_context(aggregator_and_optimizer_ranks);

    bool ddp = false;
    bool enable_tp = false;
    app.add_option("-c,--config", config_name, "Yaml Config name")->default_val(config_name);
    app.add_option("-d,--ddp", ddp, "Enable DDP")->default_val(ddp);
    app.add_option("-p,--tp", enable_tp, "Enable TP")->default_val(enable_tp);

    CLI11_PARSE(app, argc, argv);

    // tensor parallel is not supported yet
    three_tier_arch::initialize_device(ddp, enable_tp);

    auto yaml_config = YAML::LoadFile(config_name);
    three_tier_arch::TrainingConfig config = three_tier_arch::parse_config(yaml_config);

    auto [steps_per_dataset, vocab_size] = three_tier_arch::get_steps_per_dataset_and_vocab_size(config);
    fmt::println(
        "[optimizer] Rank {}: Epochs {}: Steps per dataset: {} max steps: {}",
        *distributed_ctx.rank(),
        config.num_epochs,
        steps_per_dataset,
        config.max_steps);

    auto *device = &ctx.get_device();

    auto num_devices = static_cast<uint32_t>(device->num_devices());
    auto should_be_divisible_by = (enable_tp ? num_devices : 1U) * 32U;
    vocab_size = round_up_to_tile(vocab_size, should_be_divisible_by);
    config.transformer_config.vocab_size = vocab_size;

    auto create_model = [enable_tp](const auto &config) -> std::shared_ptr<ttml::autograd::ModuleBase> {
        if (enable_tp) {
            return ttml::models::distributed::gpt2::create(config);
        }
        return ttml::models::gpt2::create(config);
    };
    auto model = create_model(config.transformer_config);

    auto model_parameters = model->parameters();
    auto sorted_model_parameters = SortedParameters(model_parameters.begin(), model_parameters.end());

    auto adamw_params = ttml::optimizers::AdamWConfig();
    adamw_params.lr = config.learning_rate;
    adamw_params.weight_decay = config.weight_decay;
    adamw_params.use_kahan_summation = config.use_kahan_summation;

    auto select_optimizer = [&model_parameters,
                             &adamw_params](bool use_moreh_adamw) -> std::unique_ptr<ttml::optimizers::OptimizerBase> {
        if (use_moreh_adamw) {
            return std::make_unique<ttml::optimizers::MorehAdamW>(model_parameters, adamw_params);
        } else {
            return std::make_unique<ttml::optimizers::AdamW>(model_parameters, adamw_params);
        }
    };

    auto optimizer = select_optimizer(config.use_moreh_adamw);

    send_weights_to_aggregator(*aggregator_and_optimizer_ctx, sorted_model_parameters);

    uint32_t global_step = 0;
    for (uint32_t epoch = 0; epoch < config.num_epochs; ++epoch) {
        for (uint32_t step = 0; step < steps_per_dataset; ++step, ++global_step) {
            receive_gradients_from_aggregator(*aggregator_and_optimizer_ctx, sorted_model_parameters);
            optimizer->step();
            send_weights_to_aggregator(*aggregator_and_optimizer_ctx, sorted_model_parameters);
            if (global_step >= config.max_steps) {
                break;
            }
        }
        if (global_step >= config.max_steps) {
            break;
        }
        fmt::print("[aggregator] Rank {}: Training epoch {} finished\n", *distributed_ctx.rank(), epoch);
    }

    fmt::print("[aggregator] Rank {}: Training finished\n", *distributed_ctx.rank());
    distributed_ctx.barrier();
    fmt::print("Rank {}: Finalized MPI context\n", *distributed_ctx.rank());
    return 0;
}
