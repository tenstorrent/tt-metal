// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "core/distributed/distributed.hpp"
#include "models/gpt2.hpp"

// namespace name can't start with a digit
namespace three_tier_arch {

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
    uint32_t num_mh_workers = 1U;
};

TrainingConfig parse_config(const YAML::Node &yaml_config);

std::pair<uint32_t, uint32_t> get_steps_per_dataset_and_vocab_size(const TrainingConfig &config);

std::vector<int> get_workers_and_aggregator_ranks(uint32_t workers);

std::string read_file_to_str(const std::string &file_path);

uint32_t round_up_to_tile(uint32_t value, uint32_t tile_size = 32U);

void initialize_device(bool ddp, bool tp);

}  // namespace three_tier_arch
