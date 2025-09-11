// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "common.hpp"

// TODO: improve include path
#include "../utils.hpp"
#include "core/distributed/distributed.hpp"
#include "datasets/utils.hpp"
#include "models/gpt2.hpp"
#include "tokenizers/bpe_tokenizer.hpp"
#include "tokenizers/char_tokenizer.hpp"

// namespace name can't start with a digit
namespace three_tier_arch {

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

    if (config.model_type == "gpt2") {
        config.transformer_config = ttml::models::gpt2::read_config(training_config["transformer_config"]);
    } else if (config.model_type == "llama") {
        config.transformer_config = ttml::models::llama::read_config(training_config["transformer_config"]);
    } else {
        throw std::runtime_error("Unknown model type: " + config.model_type);
    }

    auto multihost_config = yaml_config["multihost_config"];
    config.enable_mpi = multihost_config["enabled"].as<bool>(config.enable_mpi);
    config.num_mh_workers = multihost_config["num_workers"].as<uint32_t>(config.num_mh_workers);
    auto socket_type_str = multihost_config["socket_type"].as<std::string>("mpi");
    if (socket_type_str == "mpi") {
        config.socket_type = ttnn::distributed::SocketType::MPI;
    } else if (socket_type_str == "fabric") {
        config.socket_type = ttnn::distributed::SocketType::FABRIC;
    } else {
        throw std::runtime_error("Unknown socket type: " + socket_type_str);
    }

    return config;
}

std::vector<int> get_workers_and_aggregator_ranks(uint32_t workers) {
    std::vector<int> ranks(workers + 1U);
    std::iota(ranks.begin(), ranks.end(), 0);
    return ranks;
}

std::pair<uint32_t, uint32_t> get_steps_per_dataset_and_vocab_size(const TrainingConfig &config) {
    std::string text;
    std::variant<std::string, std::vector<uint32_t>> text_or_tokens;
    text = read_file_to_str(config.data_path);
    // check file extension:
    if (config.data_path.ends_with(".txt")) {
        text_or_tokens = read_file_to_str(config.data_path);
    } else {
        text_or_tokens = ttml::datasets::load_tokens_from_space_separated_file(config.data_path);
    }
    auto sequence_length = std::visit(
        [&](auto &&arg) {
            if constexpr (requires { arg.max_sequence_length; }) {
                return arg.max_sequence_length;
            } else {
                throw std::runtime_error(
                    "Unsupported transformer configuration type: " + std::string(typeid(arg).name()));
            }
        },
        config.transformer_config);

    auto create_dataset_and_tokenizer =
        [](const auto &text, const auto sequence_length, const auto &tokenizer_path, const auto &tokenizer_type) {
            if (tokenizer_type == "char") {
                return ttml::datasets::create_in_memory_token_dataset<ttml::tokenizers::CharTokenizer>(
                    std::get<0>(text), sequence_length);
            } else if (tokenizer_type == "bpe") {
                return std::visit(
                    [&](const auto &tokens) {
                        return ttml::datasets::create_in_memory_token_dataset<ttml::tokenizers::BPETokenizer>(
                            tokens, sequence_length, tokenizer_path);
                    },
                    text);
            } else {
                throw std::runtime_error("Unknown tokenizer type: " + tokenizer_type);
            }
        };

    auto [dataset, tokenizer] =
        create_dataset_and_tokenizer(text_or_tokens, sequence_length, config.tokenizer_path, config.tokenizer_type);

    auto dataset_size = dataset.get_size();
    auto steps_per_dataset = dataset_size / (config.batch_size * config.gradient_accumulation_steps);
    return {steps_per_dataset, tokenizer->get_vocab_size()};
}

std::string read_file_to_str(const std::string &file_path) {
    std::ifstream file(file_path);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file: " + file_path);
    }

    std::stringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
}

uint32_t round_up_to_tile(uint32_t value, uint32_t tile_size) {
    return (value + tile_size - 1) / tile_size * tile_size;
}

DeviceConfig parse_device_config(const YAML::Node &yaml_config) {
    DeviceConfig config;
    auto device_node = yaml_config["device_config"];
    if (!device_node) {
        return config;
    }

    config.enable_ddp = device_node["enable_ddp"].as<bool>(false);
    config.enable_tp = device_node["enable_tp"].as<bool>(false);

    if (config.enable_ddp && config.enable_tp) {
        throw std::runtime_error("DDP and TP cannot be enabled at the same time. Disable DDP or TP.");
    }

    auto mesh_shape_node = device_node["mesh_shape"];
    bool multidevice = config.enable_ddp || config.enable_tp;
    if (multidevice && !mesh_shape_node) {
        throw std::runtime_error("Mesh shape is required for multidevice training");
    }
    if (mesh_shape_node) {
        assert(mesh_shape_node.size() == 2);
        auto mesh_shape = mesh_shape_node.as<std::vector<int>>();
        config.mesh_shape = tt::tt_metal::distributed::MeshShape(mesh_shape[0], mesh_shape[1]);
    }

    auto device_ids_node = device_node["device_ids"];
    if (device_ids_node) {
        config.device_ids = device_ids_node.as<std::vector<int>>();
    }

    return config;
}

void initialize_device(const tt::tt_metal::distributed::MeshShape &mesh_shape, const std::vector<int> &device_ids) {
    ttml::autograd::ctx().open_device(mesh_shape, device_ids);
}

}  // namespace three_tier_arch
