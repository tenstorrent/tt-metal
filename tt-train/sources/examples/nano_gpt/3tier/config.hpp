// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <CLI/CLI.hpp>

// TODO: improve include path
#include "../utils.hpp"
#include "autograd/auto_context.hpp"
#include "core/distributed/distributed.hpp"
#include "datasets/utils.hpp"
#include "models/gpt2.hpp"
#include "tokenizers/bpe_tokenizer.hpp"
#include "tokenizers/char_tokenizer.hpp"

constexpr auto gpt2_tokenizer_file_name = "/gpt2-tokenizer.json";

struct TrainingConfig {
    std::string project_name;
    std::string model_type;  // one of "gpt2", "llama"
    uint32_t seed = 5489U;
    uint32_t model_save_interval = 500;
    uint32_t batch_size = 64;
    uint32_t num_epochs = 1;
    uint32_t max_steps = 5000;
    float learning_rate = 3e-4F;
    float weight_decay = 1e-2F;
    bool use_moreh_adamw = false;
    // works only for AdamW
    bool use_kahan_summation = false;
    // accumulate batches for gradient update
    uint32_t gradient_accumulation_steps = 1;
    std::string model_path;
    std::string data_path;
    std::string tokenizer_type = "char";
    std::string scheduler_type = "identity";
    std::string tokenizer_path = std::string(DATA_FOLDER) + gpt2_tokenizer_file_name;
    bool use_clip_grad_norm = false;
    float clip_grad_norm_max_norm = 1.0F;
    ttml::models::gpt2::TransformerConfig transformer_config;

    bool enable_mpi = false;
    uint32_t num_mpi_workers = 1U;
};

TrainingConfig parse_config(const YAML::Node &yaml_config) {
    TrainingConfig config;
    auto training_config = yaml_config["training_config"];
    config.project_name = training_config["project_name"].as<std::string>("tt_train_nano_gpt");
    config.model_type = training_config["model_type"].as<std::string>();
    config.seed = training_config["seed"].as<uint32_t>();
    config.model_save_interval = training_config["model_save_interval"].as<uint32_t>();
    config.batch_size = training_config["batch_size"].as<uint32_t>();
    config.num_epochs = training_config["num_epochs"].as<uint32_t>();
    config.max_steps = training_config["max_steps"].as<uint32_t>();
    config.learning_rate = training_config["learning_rate"].as<float>();
    config.weight_decay = training_config["weight_decay"].as<float>();
    config.use_moreh_adamw = training_config["use_moreh_adamw"].as<bool>(config.use_moreh_adamw);
    config.use_kahan_summation = training_config["use_kahan_summation"].as<bool>(config.use_kahan_summation);
    config.gradient_accumulation_steps =
        training_config["gradient_accumulation_steps"].as<uint32_t>(config.gradient_accumulation_steps);
    config.model_path = training_config["model_path"].as<std::string>("");
    config.data_path = training_config["data_path"].as<std::string>(std::string(DATA_FOLDER) + "/shakespeare.txt");
    config.tokenizer_type = training_config["tokenizer_type"].as<std::string>(config.tokenizer_type);
    config.scheduler_type = training_config["scheduler_type"].as<std::string>(config.scheduler_type);
    config.tokenizer_path = training_config["tokenizer_path"].as<std::string>(config.tokenizer_path);
    config.use_clip_grad_norm = training_config["use_clip_grad_norm"].as<bool>(config.use_clip_grad_norm);
    config.clip_grad_norm_max_norm =
        training_config["clip_grad_norm_max_norm"].as<float>(config.clip_grad_norm_max_norm);

    // support for gpt2 only
    if (config.model_type != "gpt2") {
        throw std::runtime_error("Unsupported model type: " + config.model_type);
    }
    config.transformer_config = ttml::models::gpt2::read_config(training_config["transformer_config"]);

    auto mpi_config = yaml_config["mpi_config"];
    config.enable_mpi = mpi_config["enabled"].as<bool>(config.enable_mpi);
    config.num_mpi_workers = mpi_config["num_workers"].as<uint32_t>(config.num_mpi_workers);

    return config;
}
