// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <fstream>
#include <iostream>
#include <sstream>

#include "autograd/tensor.hpp"
#include "models/gpt2.hpp"
#include "models/llama.hpp"
#include "schedulers/lambda_scheduler.hpp"
#include "schedulers/linear_scheduler.hpp"
#include "schedulers/scheduler_base.hpp"
#include "schedulers/sequential_scheduler.hpp"
#include "serialization/msgpack_file.hpp"
#include "serialization/serialization.hpp"

class LossAverageMeter {
    float m_sum = 0.0F;
    size_t m_count = 0;

public:
    void update(float loss, size_t count = 1);

    [[nodiscard]] float average() const;

    void reset();
};

std::unique_ptr<ttml::schedulers::LRSchedulerBase> create_idendity_scheduler(
    ttml::optimizers::OptimizerBase *optimizer, [[maybe_unused]] size_t total_steps);

std::unique_ptr<ttml::schedulers::LRSchedulerBase> create_warmup_with_linear_scheduler(
    ttml::optimizers::OptimizerBase *optimizer, size_t total_steps);

std::string read_file_to_str(const std::string &file_path);

template <typename Model>
void save_training_state(
    std::string &model_path,
    Model &model,
    const std::unique_ptr<ttml::schedulers::LRSchedulerBase> &scheduler,
    const std::string &model_name,
    const std::string &optimizer_name) {
    ttml::serialization::MsgPackFile serializer;
    ttml::serialization::write_module(serializer, model_name, model.get());
    ttml::serialization::write_optimizer(serializer, optimizer_name, scheduler->get_optimizer().get());
    ttml::serialization::write_state_dict(serializer, "scheduler", scheduler->get_state_dict());
    serializer.serialize(model_path);
}

template <typename Model>
void load_training_state(
    std::string &model_path,
    Model &model,
    const std::unique_ptr<ttml::schedulers::LRSchedulerBase> &scheduler,
    const std::string &model_name,
    const std::string &optimizer_name) {
    ttml::serialization::MsgPackFile deserializer;
    deserializer.deserialize(model_path);
    ttml::serialization::read_module(deserializer, model_name, model.get());
    ttml::serialization::read_optimizer(deserializer, optimizer_name, scheduler->get_optimizer().get());
    auto state_dict = scheduler->get_state_dict();
    ttml::serialization::read_state_dict(deserializer, "scheduler", state_dict);
    scheduler->set_state_dict(state_dict);
}

template <typename Model>
void load_model_parameters(std::string &model_path, Model &model, const std::string &model_name) {
    ttml::serialization::MsgPackFile deserializer;
    deserializer.deserialize(model_path);
    ttml::serialization::read_module(deserializer, model_name, model.get());
}

uint32_t round_up_to_tile(uint32_t value, uint32_t tile_size = 32);

class GradientAccumulator {
public:
    explicit GradientAccumulator(uint32_t accumulation_steps);

    [[nodiscard]] bool should_zero_grad() const;
    [[nodiscard]] bool should_step() const;
    ttml::autograd::TensorPtr scale(ttml::autograd::TensorPtr &tensor_ptr);
    void update(float loss, size_t samples = 1);
    void reset();

    [[nodiscard]] float average_loss() const;

private:
    uint32_t m_accumulation_steps = 1;
    uint32_t m_steps = 0;

    float m_total_loss = 0.0F;
    size_t m_total_samples = 0;
};

template <typename TrainingConfig>
std::string generate_run_name(const std::string &run_name, const TrainingConfig &config, bool add_time_to_run_name) {
    bool use_generated_run_name = run_name.empty();
    std::stringstream ss;

    auto build_run_name = [&]() {
        auto &transformer_config = config.transformer_config;

        auto is_nano_gpt_config = [&transformer_config]() -> bool {
            return std::visit(
                [](auto &&arg) -> bool {
                    constexpr bool is_gpt2_config_type =
                        std::is_same_v<std::decay_t<decltype(arg)>, ttml::models::gpt2::TransformerConfig>;
                    return is_gpt2_config_type && arg.num_heads == 6U && arg.embedding_dim == 384U &&
                           arg.num_blocks == 6U;
                },
                transformer_config);
        };

        auto is_gpt2s_config = [&transformer_config]() -> bool {
            return std::visit(
                [](auto &&arg) -> bool {
                    constexpr bool is_gpt2_config_type =
                        std::is_same_v<std::decay_t<decltype(arg)>, ttml::models::gpt2::TransformerConfig>;
                    return is_gpt2_config_type && arg.num_heads == 12U && arg.embedding_dim == 768U &&
                           arg.num_blocks == 12U;
                },
                transformer_config);
        };

        auto is_llama_config = [&transformer_config]() -> bool {
            return std::visit(
                [](auto &&arg) -> bool {
                    return std::is_same_v<std::decay_t<decltype(arg)>, ttml::models::llama::LlamaConfig>;
                },
                transformer_config);
        };

        auto batch_size = config.batch_size * config.gradient_accumulation_steps;

        if (is_llama_config()) {
            ss << "llama";
        } else if (is_nano_gpt_config()) {
            ss << "nano_gpt";
        } else if (is_gpt2s_config()) {
            ss << "gpt2s";
        } else {
            ss << "transformer";
        }
        ss << "_bs_" << batch_size;
        ss << "_lr_" << config.learning_rate;
        ss << "_wd_" << config.weight_decay;
        if (config.use_kahan_summation) {
            ss << "_kahan";
        }

        if (config.gradient_accumulation_steps > 1) {
            ss << "_grad_acc_" << config.gradient_accumulation_steps;
        }
        ss << "_sched_" << config.scheduler_type;
    };

    if (use_generated_run_name) {
        build_run_name();
    } else {
        ss << run_name;
    }

    if (add_time_to_run_name) {
        auto now = std::chrono::system_clock::now();
        std::time_t current_time = std::chrono::system_clock::to_time_t(now);
        ss << "_date_" << std::put_time(std::localtime(&current_time), "%Y-%m-%d_%H:%M:%S");
    }

    return ss.str();
}

void initialize_device(
    const tt::tt_metal::distributed::MeshShape &mesh_shape,
    const std::vector<int> &device_ids);
