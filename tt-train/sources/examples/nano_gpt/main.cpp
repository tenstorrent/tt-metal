// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <CLI/CLI.hpp>
#include <chrono>
#include <core/ttnn_all_includes.hpp>
#include <csignal>
#include <cstdint>

#include "autograd/auto_context.hpp"
#include "autograd/tensor.hpp"
#include "core/clip_grad_norm.hpp"
#include "core/distributed/distributed.hpp"
#include "core/distributed/socket_manager.hpp"
#include "core/tt_tensor_utils.hpp"
#include "datasets/dataloader.hpp"
#include "datasets/in_memory_token_dataset.hpp"
#include "datasets/utils.hpp"
#include "models/common/transformer_common.hpp"
#include "models/distributed/gpt2.hpp"
#include "models/distributed/llama.hpp"
#include "models/distributed/pipeline_parallel_llama.hpp"
#include "models/gpt2.hpp"
#include "models/llama.hpp"
#include "ops/binary_ops.hpp"
#include "ops/losses.hpp"
#include "optimizers/adamw.hpp"
#include "optimizers/no_op.hpp"
#include "optimizers/remote_optimizer.hpp"
#include "tokenizers/char_tokenizer.hpp"
#include "ttnn_fixed/distributed/tt_metal.hpp"
#include "ttnn_fixed/trivial_ttnn_ops.hpp"
#include "utils.hpp"

namespace {
constexpr auto gpt2_tokenizer_file_name = "/gpt2-tokenizer.json";
}


using Model = std::shared_ptr<ttml::models::BaseTransformer>;

void model_to_eval(Model &model) {
    model->eval();
}

void model_to_train(Model &model) {
    model->train();
}

ttml::autograd::TensorPtr run_model(
    Model &model, const ttml::autograd::TensorPtr &data, const ttml::autograd::TensorPtr &mask) {
    return (*model)(data, mask);
}

ttml::serialization::NamedParameters get_model_parameters(Model &model) {
    return model->parameters();
}

uint64_t get_number_of_parameters(Model &model, bool tp) {
    auto *device = &ttml::autograd::ctx().get_device();
    auto num_devices = static_cast<uint32_t>(device->num_devices());

    auto contains = [](const std::string &str, const std::string &substr) {
        return str.find(substr) != std::string::npos;
    };

    auto parameters = get_model_parameters(model);
    uint64_t num_params = 0;
    for (const auto &[name, tensor_ptr] : parameters) {
        auto tensor = tensor_ptr->get_value();
        auto params_in_tensor = tensor.logical_volume();
        if (tp && (contains(name, "fc") || contains(name, "linear"))) {
            num_params += params_in_tensor * num_devices;
        } else {
            num_params += params_in_tensor;
        }
    }

    return num_params;
}

using ttml::autograd::TensorPtr;
using SocketManager = ttml::core::distributed::SocketManager;
using SocketType = ttml::core::distributed::SocketType;

using DatasetSample = std::pair<std::span<const uint32_t>, std::span<const uint32_t>>;
// tokens, targets, masks
using BatchType = std::tuple<TensorPtr, TensorPtr, TensorPtr>;
using DataLoader = ttml::datasets::DataLoader<
    ttml::datasets::InMemoryTokenDataset,
    std::function<BatchType(std::vector<DatasetSample> &&samples)>,
    BatchType>;

template <typename Tokenizer>
void generate(
    Model &model,
    const Tokenizer &tokenizer,
    uint32_t max_sequence_length,
    uint32_t num_heads,
    uint32_t tokens_to_generate = 1024U,
    bool enable_tp = false,
    // Additional sampling params:
    float temperature = 1.0F,
    float repetition_penalty = 1.0F,
    int top_k = -1,
    float top_p = 1.0F) {
    model_to_eval(model);

    std::string prompt;
    fmt::print("Enter a prompt: ");
    std::getline(std::cin, prompt);
    if (prompt.empty()) {
        prompt = "\n";
    }

    // Encode the prompt
    auto prompt_tokens = tokenizer.encode(prompt);

    // In case you need a pad token
    auto pad_token_id = 0U;
    auto original_vocab_size = tokenizer.get_vocab_size();
    fmt::println("Original tokenizer vocab size: {}", original_vocab_size);

    auto *device = &ttml::autograd::ctx().get_device();
    auto num_devices = static_cast<uint32_t>(device->num_devices());
    // this is workaround for tensor parallel case, we need to have vocab size divisible by 32 per device
    auto padded_vocab_size = round_up_to_tile(original_vocab_size, (enable_tp ? num_devices : 1U) * 32U);

    // Build mask (causal) for attention
    std::vector<float> mask;
    mask.reserve(static_cast<size_t>(max_sequence_length * max_sequence_length));
    for (uint32_t i = 0; i < max_sequence_length; ++i) {
        for (uint32_t j = 0; j < max_sequence_length; ++j) {
            mask.push_back(i >= j ? 1.0F : 0.0F);
        }
    }

    auto mask_tensor = ttml::autograd::create_tensor(
        ttml::core::from_vector(mask, ttnn::Shape({1, 1, max_sequence_length, max_sequence_length}), device));

    // Prepare a padded buffer for the prompt
    std::vector<uint32_t> prompt_tokens_padded(max_sequence_length, pad_token_id);
    std::vector<float> padded_logits_vector(padded_vocab_size, 0.0F);

    fmt::print("Generated text:\n");
    fmt::print("*******************\n");
    fmt::print("{}", prompt);

    // Sampling setup
    uint32_t prompt_tokens_padded_size = 0U;
    uint32_t next_token_id = 0U;

    auto logits_tensor = ttml::core::from_vector<float, ttnn::DataType::BFLOAT16>(
        std::vector<float>(original_vocab_size, 0.0F),
        ttnn::Shape({1, 1, 1, original_vocab_size}),
        device,
        ttnn::Layout::ROW_MAJOR);

    auto next_token_tensor = ttml::core::zeros(ttnn::Shape({1U, 1U, 1U}), device, tt::tt_metal::DataType::UINT32);

    std::vector<uint32_t> next_token_vector;
    std::vector<float> logits_vector(original_vocab_size, 0.0F);

    // Create a large negative mask for out-of-vocab logits
    bool need_logits_padding = (padded_vocab_size != original_vocab_size);
    std::optional<ttnn::Tensor> logits_padding_mask;
    if (need_logits_padding) {
        auto vocab_mask = std::vector<float>(padded_vocab_size - original_vocab_size, 1e4F);

        auto argmax_zeros =
            ttml::core::zeros(ttnn::Shape({1U, 1U, 1U, original_vocab_size}), device, tt::tt_metal::DataType::BFLOAT16);

        auto argmax_nonzero = ttml::core::from_vector<float, tt::tt_metal::DataType::BFLOAT16>(
            vocab_mask, ttnn::Shape({1U, 1U, 1U, padded_vocab_size - original_vocab_size}), device, ttnn::Layout::TILE);

        auto logits_padding_mask_vector = std::vector<ttnn::Tensor>{argmax_zeros, argmax_nonzero};

        logits_padding_mask = ttnn::concat(logits_padding_mask_vector, 3);
    }
    // Main token generation loop
    for (uint32_t token_idx = 0; token_idx < tokens_to_generate; ++token_idx) {
        // Possibly truncate the prompt if it exceeds max_sequence_length
        uint32_t start_idx = 0;
        if (prompt_tokens.size() > max_sequence_length) {
            start_idx = static_cast<uint32_t>(prompt_tokens.size() - max_sequence_length);
        }
        // Fill padded array
        for (uint32_t i = 0; i < max_sequence_length; ++i) {
            prompt_tokens_padded[i] = pad_token_id;
        }
        for (uint32_t i = start_idx; i < prompt_tokens.size(); ++i) {
            prompt_tokens_padded[i - start_idx] = prompt_tokens[i];
        }
        prompt_tokens_padded_size = static_cast<uint32_t>(prompt_tokens_padded.size());
        auto prompt_tensor = ttml::autograd::create_tensor(ttml::core::from_vector<uint32_t, ttnn::DataType::UINT32>(
            prompt_tokens_padded,
            ttnn::Shape({1U, 1U, 1U, prompt_tokens_padded_size}),
            device,
            ttnn::Layout::ROW_MAJOR));

        // Forward pass
        // 'output' shape is presumably [batch=1, 1, seq_len, padded_vocab_size] or something similar
        auto output = run_model(model, prompt_tensor, mask_tensor);
        next_token_tensor = ttml::ttnn_fixed::sample(
            output->get_value(), temperature, ttml::autograd::ctx().get_generator()(), logits_padding_mask);

        // The index of the last token in the "effective" input
        // (Your indexing may vary depending on how your model outputs are shaped)
        uint32_t predicted_token_idx =
            (prompt_tokens.size() > max_sequence_length) ? (max_sequence_length - 1U) : (prompt_tokens.size() - 1U);

        // ** TTNN Argmax **

        auto next_token_vector = ttml::core::to_vector<uint32_t>(next_token_tensor);
        next_token_id = next_token_vector[predicted_token_idx];

        // Handle out-of-vocabulary token
        if (next_token_id >= original_vocab_size) {
            next_token_id = prompt_tokens.back();
        }

        // Append the new token
        prompt_tokens.push_back(next_token_id);

        // Decode and print
        fmt::print("{}", tokenizer.decode({next_token_id}));
        std::cout.flush();

        // Reset the autograd graph if needed
        ttml::autograd::ctx().reset_graph();
    }

    fmt::print("\n*******************\n");
    model_to_train(model);  // return model to train mode if needed
}

struct EvalConfig {
    float repetition_penalty = 1.0F;
    float temperature = 1.0F;
    int top_k = -1;
    float top_p = 1.0F;
};

EvalConfig parse_eval_config(const YAML::Node &yaml_config) {
    EvalConfig config;
    if (!yaml_config["eval_config"]) {
        return config;
    }
    auto eval_config = yaml_config["eval_config"];
    config.repetition_penalty = eval_config["repetition_penalty"].as<float>(config.repetition_penalty);
    config.temperature = eval_config["temperature"].as<float>(config.temperature);
    config.top_k = eval_config["top_k"].as<int>(config.top_k);
    config.top_p = eval_config["top_p"].as<float>(config.top_p);
    return config;
}

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
    bool use_no_op = false;
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
    std::variant<ttml::models::gpt2::TransformerConfig, ttml::models::llama::LlamaConfig> transformer_config;

    // mpi config
    bool enable_mpi = false;
    uint32_t num_mh_workers = 0U;
    SocketType socket_type = SocketType::MPI;
    std::optional<ttml::models::distributed::pipeline_parallel_llama::PipelineParallelConfig> pipeline_parallel_config;
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
    config.use_no_op = training_config["use_no_op"].as<bool>(config.use_no_op);
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

    if (auto multihost_config = yaml_config["multihost_config"]) {
        config.enable_mpi = multihost_config["enabled"].as<bool>(false);
        config.num_mh_workers = multihost_config["num_workers"].as<uint32_t>(0U);

        auto socket_type_str = multihost_config["socket_type"].as<std::string>("mpi");
        if (socket_type_str == "mpi") {
            config.socket_type = SocketType::MPI;
        } else if (socket_type_str == "fabric") {
            config.socket_type = SocketType::FABRIC;
        } else {
            throw std::runtime_error("Unknown socket type: " + socket_type_str);
        }

        ttml::autograd::ctx().initialize_socket_manager(config.socket_type);

        if (auto pipeline_parallel_config = multihost_config["pipeline_parallel_config"]) {
            config.pipeline_parallel_config =
                ttml::models::distributed::pipeline_parallel_llama::read_config(pipeline_parallel_config);
        }
    }
    return config;
}

struct DeviceConfig {
    // multidevice config: default to single device with default mapping of
    // physical devices onto the mesh shape.
    tt::tt_metal::distributed::MeshShape mesh_shape{1, 1};
    std::vector<int> device_ids{};

    bool enable_ddp = false;
    bool enable_tp = false;
};

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

const std::unordered_map<
    std::string,
    std::function<std::unique_ptr<ttml::schedulers::LRSchedulerBase>(ttml::optimizers::OptimizerBase *, size_t)>>
    schedulers = {{"identity", create_idendity_scheduler}, {"warmup_linear", create_warmup_with_linear_scheduler}};

namespace {

inline bool is_pipeline_parallel_enabled(const TrainingConfig &config) {
    return config.pipeline_parallel_config.has_value();
}

inline int get_mpi_rank_or_zero() {
    auto &ctx = ttml::autograd::ctx();
    auto distributed_ctx = ctx.get_distributed_context();
    return distributed_ctx ? *distributed_ctx->rank() : 0;
}

inline bool is_three_tier_training(const TrainingConfig &config) {
    return config.enable_mpi && !is_pipeline_parallel_enabled(config);
}

inline bool is_last_pipeline_stage(const TrainingConfig &config) {
    if (!is_pipeline_parallel_enabled(config)) {
        return true;
    }
    return static_cast<unsigned>(get_mpi_rank_or_zero()) == (config.num_mh_workers - 1U);
}

inline bool pipeline_needs_to_call_loss(const TrainingConfig &config) {
    return !is_pipeline_parallel_enabled(config) || is_last_pipeline_stage(config);
}

inline void pipeline_transfer_targets_if_needed(const TrainingConfig &config, const TensorPtr &target) {
    if (!is_pipeline_parallel_enabled(config)) {
        return;
    }
    if (config.num_mh_workers <= 1U) {
        return;
    }
    auto &ctx = ttml::autograd::ctx();
    auto distributed_ctx = ctx.get_distributed_context();
    int rank = *distributed_ctx->rank();
    auto &socket_manager = ctx.get_socket_manager();
    if (rank == 0) {
        socket_manager.send(
            target->get_value(), distributed_ctx, ttml::core::distributed::Rank(config.num_mh_workers - 1));
    } else if (static_cast<unsigned>(rank + 1U) == config.num_mh_workers) {
        target->set_value(socket_manager.recv(target->get_value(), distributed_ctx, ttml::core::distributed::Rank(0)));
    }
}

}  // namespace

int main(int argc, char **argv) {
    auto start_timer = std::chrono::high_resolution_clock::now();
    CLI::App app{"NanoGPT Example"};
    argv = app.ensure_utf8(argv);

    std::string config_name = std::string(CONFIGS_FOLDER) + "/training_shakespeare_nanogpt.yaml";

    std::string run_name = "";
    bool is_eval = false;
    bool add_time_to_name = true;
    std::string safetensors_path = "";
    std::string save_and_exit_path = "";
    app.add_option("-c,--config", config_name, "Yaml Config name")->default_val(config_name);
    app.add_option("-e,--eval", is_eval, "Is evaluation")->default_val(is_eval);
    app.add_option("-t,--add_time_to_name", add_time_to_name, "Add time to run name")->default_val(add_time_to_name);
    app.add_option("-n,--name", run_name, "Run name")->default_val(run_name);
    app.add_option("-s,--save_and_exit", save_and_exit_path, "Save and exit (path to dumped msgpack)")
        ->default_val(save_and_exit_path);
    app.add_option("--safetensors", safetensors_path, "Loads safetensors model from the given path")
        ->default_val(safetensors_path);
    CLI11_PARSE(app, argc, argv);

    auto yaml_config = YAML::LoadFile(config_name);
    TrainingConfig config = parse_config(yaml_config);
    EvalConfig eval_config = parse_eval_config(yaml_config);
    DeviceConfig device_config = parse_device_config(yaml_config);

    if (config.enable_mpi) {
        auto &ctx = ttml::autograd::ctx();
        ctx.initialize_distributed_context(argc, argv);

        auto distributed_ctx = ctx.get_distributed_context();
        fmt::print("Size {}, Rank {}: Initializing MPI context\n", *distributed_ctx->size(), *distributed_ctx->rank());

    }

    if (device_config.enable_ddp || device_config.enable_tp) {
        fmt::println("Device config:");
        fmt::println("  Tensor parallel enabled: {}", device_config.enable_tp);
        fmt::println("  Distributed data-parallel enabled: {}", device_config.enable_ddp);
        fmt::println("  Mesh shape: {}", device_config.mesh_shape);
        fmt::println("  Device IDs: {}", device_config.device_ids);
    }

    if (config.enable_mpi) {
        fmt::print("MPI config:\n");
        fmt::print("  enable_mpi: {}\n", config.enable_mpi);
        fmt::print("  num_mh_workers: {}\n", config.num_mh_workers);
        fmt::print("  socket_type: {}\n", config.socket_type == SocketType::MPI ? "MPI" : "FABRIC");
    }

    if (device_config.enable_tp) {
        if (!config.model_path.empty()) {
            throw std::runtime_error("Save and load is not supported with Tensor Parallel model");
        }

        if (is_eval) {
            throw std::runtime_error("Evaluation is not supported with Tensor Parallel model");
        }
    }

    // set seed
    ttml::autograd::ctx().set_seed(config.seed);
    if (config.enable_mpi) {
        int rank = *ttml::autograd::ctx().get_distributed_context()->rank();
        auto seed = config.seed + static_cast<uint32_t>(rank);
        ttml::autograd::ctx().set_seed(seed);
    }
    auto schedule_func = schedulers.at(config.scheduler_type);

    std::string text;
    std::variant<std::string, std::vector<uint32_t>> text_or_tokens;
    try {
        text = read_file_to_str(config.data_path);
        // check file extension:
        if (config.data_path.ends_with(".txt")) {
            text_or_tokens = read_file_to_str(config.data_path);
        } else {
            text_or_tokens = ttml::datasets::load_tokens_from_space_separated_file(config.data_path);
        }
    } catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
        return -1;
    }
    fmt::print("Max steps {}\n", config.max_steps);
    fmt::print("Batch size {}\n", config.batch_size);
    fmt::print("Gradient accumulation steps {}\n", config.gradient_accumulation_steps);
    fmt::print("Total batch size {}\n", config.batch_size * config.gradient_accumulation_steps);
    fmt::print("Scheduler type {}\n", config.scheduler_type);
    fmt::print("Seed {}\n", ttml::autograd::ctx().get_seed());
    auto sequence_length = std::visit([](auto &&arg) { return arg.max_sequence_length; }, config.transformer_config);

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
    fmt::print("Tokenizer path: {}\n", config.tokenizer_path);
    fmt::print("Dataset size: {}\n", dataset.get_size());
    fmt::print("Vocab size: {}\n", tokenizer->get_vocab_size());
    fmt::print("Tokenizer type: {}\n", config.tokenizer_type);

    auto num_devices = device_config.mesh_shape[0] * device_config.mesh_shape[1];
    // enable fabric config for 3-tier architecture, tp, ddp
    if (config.socket_type == SocketType::FABRIC || device_config.enable_tp || device_config.enable_ddp) {
        ttml::ttnn_fixed::distributed::enable_fabric(num_devices);
    }

    initialize_device(device_config.mesh_shape, device_config.device_ids);
    auto *device = &ttml::autograd::ctx().get_device();

    struct CachedHostData {
        std::vector<uint32_t> data;
        std::vector<uint32_t> targets;
        ttml::autograd::TensorPtr masks_tensor;
    };
    CachedHostData cached_data;
    std::vector<float> mask;
    auto num_heads = std::visit([](auto &&arg) { return arg.num_heads; }, config.transformer_config);
    mask.reserve(sequence_length * sequence_length);
    for (int i = 0; i < sequence_length; ++i) {
        for (int j = 0; j < sequence_length; ++j) {
            mask.push_back(i >= j ? 1.0F : 0.0F);
        }
    }
    cached_data.masks_tensor = ttml::autograd::create_tensor(
        ttml::core::from_vector(mask, ttnn::Shape({1U, 1U, sequence_length, sequence_length}), device));

    std::function<BatchType(std::vector<DatasetSample> && samples)> collate_fn =
        [sequence_length, device, &cached_data, &device_config](std::vector<DatasetSample> &&samples) {
            auto start_timer = std::chrono::high_resolution_clock::now();
            const uint32_t batch_size = samples.size();
            std::vector<uint32_t> &data = cached_data.data;
            std::vector<uint32_t> &targets = cached_data.targets;

            data.clear();
            targets.clear();

            data.reserve((size_t)batch_size * sequence_length);
            targets.reserve((size_t)batch_size * sequence_length);
            for (auto &[features, target_span] : samples) {
                std::copy(features.begin(), features.end(), std::back_inserter(data));
                std::copy(target_span.begin(), target_span.end(), std::back_inserter(targets));
            }
            auto end_timer = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_timer - start_timer).count();
            fmt::print("dataloader host only step time {} ms\n", (double)duration / 1000.);

            auto create_data_and_targets = [&]() -> std::tuple<TensorPtr, TensorPtr> {
                if (device_config.enable_ddp) {
                    const auto mapper = ttnn::distributed::shard_tensor_to_mesh_mapper(*device, 0);
                    auto data_tensor =
                        ttml::autograd::create_tensor(ttml::core::from_vector<uint32_t, ttnn::DataType::UINT32>(
                            data,
                            ttnn::Shape({batch_size, 1, 1, sequence_length}),
                            device,
                            ttnn::Layout::ROW_MAJOR,
                            mapper.get()));

                    auto targets_tt_tensor = ttml::core::from_vector<uint32_t, ttnn::DataType::UINT32>(
                        targets,
                        ttnn::Shape({batch_size, sequence_length}),
                        device,
                        ttnn::Layout::ROW_MAJOR,
                        mapper.get());
                    auto targets_tensor = ttml::autograd::create_tensor(targets_tt_tensor);
                    return {data_tensor, targets_tensor};
                }

                auto data_tensor =
                    ttml::autograd::create_tensor(ttml::core::from_vector<uint32_t, ttnn::DataType::UINT32>(
                        data, ttnn::Shape({batch_size, 1, 1, sequence_length}), device, ttnn::Layout::ROW_MAJOR));

                auto targets_tensor =
                    ttml::autograd::create_tensor(ttml::core::from_vector<uint32_t, ttnn::DataType::UINT32>(
                        targets, ttnn::Shape({batch_size, sequence_length}), device, ttnn::Layout::ROW_MAJOR));
                return {data_tensor, targets_tensor};
            };

            auto [data_tensor, targets_tensor] = create_data_and_targets();
            end_timer = std::chrono::high_resolution_clock::now();
            duration = std::chrono::duration_cast<std::chrono::microseconds>(end_timer - start_timer).count();
            fmt::print("dataloader step time {} ms\n", (double)duration / 1000.);
            return std::make_tuple(data_tensor, targets_tensor, cached_data.masks_tensor);
        };

    LossAverageMeter loss_meter;
    auto train_dataloader = DataLoader(dataset, /* batch_size */ config.batch_size, /* shuffle */ true, collate_fn);

    fmt::print("Overriding vocab size to be divisible by 32\n");
    // this is workaround for tensor parallel case, we need to have vocab size divisible by 32 per device
    std::visit(
        [&](auto &&arg) {
            if constexpr (requires { arg.vocab_size; }) {
                arg.vocab_size =
                    round_up_to_tile(tokenizer->get_vocab_size(), (device_config.enable_tp ? num_devices : 1U) * 32U);
            } else {
                throw std::runtime_error(
                    "Unsupported transformer configuration type: " + std::string(typeid(arg).name()));
            }
        },
        config.transformer_config);

    Model model = std::visit(
        [&device_config, &config](auto &&arg) -> Model {
            if constexpr (std::is_same_v<std::decay_t<decltype(arg)>, ttml::models::llama::LlamaConfig>) {
                if (config.pipeline_parallel_config) {
                    return ttml::models::distributed::pipeline_parallel_llama::create(
                        arg, *config.pipeline_parallel_config, device_config.enable_tp);
                } else if (device_config.enable_tp) {
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

    if (!safetensors_path.empty()) {
        fmt::print("Loading model from safetensors path: {}\n", safetensors_path);
        model->load_from_safetensors(safetensors_path);
        fmt::print("Model loaded from safetensors\n");
    }
    if (!save_and_exit_path.empty()) {
        if (std::filesystem::exists(save_and_exit_path)) {
            throw std::runtime_error("Model path already exists: " + save_and_exit_path);
        }
        fmt::println("Saving model and exiting");
        ttml::serialization::MsgPackFile serializer;
        std::string model_prefix = (config.model_type == "llama") ? "llama" : "transformer";
        ttml::serialization::write_module(serializer, model_prefix, model.get());
        serializer.serialize(save_and_exit_path);
        fmt::println("Model saved to {}", save_and_exit_path);
        std::exit(0);
    }

    // Load model parameters if in eval mode and model path exists
    if (is_eval && !config.model_path.empty() && std::filesystem::exists(config.model_path)) {
        fmt::print("Loading model from {}\n", config.model_path);
        std::string model_name = (config.model_type == "llama") ? "llama" : "transformer";
        fmt::print("Loading model parameters\n");
        load_model_parameters(config.model_path, model, model_name);
        fmt::print("Model loaded\n");
    }

    if (is_eval) {
        fmt::print("\nEvaluation started\n");
        for (;;) {
            generate(
                model,
                *tokenizer,
                std::visit([](auto &&arg) { return arg.max_sequence_length; }, config.transformer_config),
                num_heads,
                sequence_length,
                device_config.enable_tp,
                eval_config.temperature,
                eval_config.repetition_penalty,
                eval_config.top_k,
                eval_config.top_p);
        }
        fmt::print("\nEvaluation finished\n");
        return 0;
    }

    auto adamw_params = ttml::optimizers::AdamWConfig();
    adamw_params.lr = config.learning_rate;
    adamw_params.weight_decay = config.weight_decay;
    adamw_params.use_kahan_summation = config.use_kahan_summation;

    if (config.use_no_op) {
        fmt::print("WARNING: Using NoOp optimizer - parameters will NOT be updated.\n");
    } else if (!is_three_tier_training(config)) {
        fmt::print("AdamW configuration:\n");
        fmt::print("    Learning rate: {}\n", adamw_params.lr);
        fmt::print("    Weight decay: {}\n", adamw_params.weight_decay);
        fmt::print("    Use Kahan summation: {}\n", adamw_params.use_kahan_summation);
    } else {
        fmt::println("Remote optimizer configured!");
    }

    fmt::print("Number of parameters: {}\n", get_number_of_parameters(model, device_config.enable_tp));

    auto select_optimizer = [&model, &adamw_params, &config]() -> std::unique_ptr<ttml::optimizers::OptimizerBase> {
        if (is_three_tier_training(config)) {
            return std::make_unique<ttml::optimizers::RemoteOptimizer>(
                get_model_parameters(model), config.num_mh_workers);
        } else if (config.use_no_op) {
            return std::make_unique<ttml::optimizers::NoOp>(get_model_parameters(model));
        } else if (config.use_moreh_adamw) {
            return std::make_unique<ttml::optimizers::MorehAdamW>(get_model_parameters(model), adamw_params);
        } else {
            return std::make_unique<ttml::optimizers::AdamW>(get_model_parameters(model), adamw_params);
        }
    };

    auto optimizer = select_optimizer();
    auto scheduler = schedule_func(optimizer.get(), config.max_steps);

    if (is_three_tier_training(config)) {
        auto *optimizer_ptr = dynamic_cast<ttml::optimizers::RemoteOptimizer *>(optimizer.get());
        if (!optimizer_ptr) {
            throw std::runtime_error("Optimizer is not RemoteOptimizer");
        }
        fmt::println("[worker] Remote optimizer receiving weights from rank {}", config.num_mh_workers);
        optimizer_ptr->receive_weights();
        fmt::println("[worker] Remote optimizer received weights from rank {}", config.num_mh_workers);
    } else if (config.use_no_op) {
        fmt::print("Skipping training state load (NoOp optimizer)\n");
    } else {
        // otherwise proceed with normal loading training state if necessary
        if (!config.model_path.empty() && std::filesystem::exists(config.model_path)) {
            fmt::print("Loading model from {}\n", config.model_path);
            std::string model_name = (config.model_type == "llama") ? "llama" : "transformer";
            fmt::print("Loading training state\n");
            std::string optimizer_name = "adamw";
            load_training_state(config.model_path, model, scheduler, model_name, optimizer_name);
            fmt::print("Model loaded after {} steps\n", optimizer->get_steps());
        }
    }

    if (config.enable_mpi && is_eval) {
        throw std::logic_error("Evaluation is not supported with 3 tier training");
    }

    if (config.enable_mpi && config.use_clip_grad_norm) {
        throw std::logic_error("Clip grad norm is not supported with 3 tier training");
    }

    if (device_config.enable_ddp) {
        auto num_devices = static_cast<uint32_t>(device->num_devices());
        if (config.batch_size % num_devices != 0) {
            throw std::logic_error(fmt::format(
                "Batch size must be divisible by the number of devices. Batch size = {}, devices = {}",
                config.batch_size,
                num_devices));
        }
    }

    auto get_loss_value = [](const TensorPtr &loss) {
        auto loss_xtensors = ttml::core::to_xtensor(loss->get_value(), ttml::core::IdentityComposer{});
        // sum of loss xtensors
        float loss_float =
            std::accumulate(loss_xtensors.begin(), loss_xtensors.end(), 0.0F, [](float acc, auto &xtensor) {
                return acc + xtensor(0);
            });

        return loss_float / static_cast<float>(loss_xtensors.size());
    };

    const uint32_t num_epochs = config.num_epochs;
    auto gradient_accumulator_helper = GradientAccumulator(config.gradient_accumulation_steps);

    bool is_everything_compiled = false;

    const bool needs_to_call_loss = pipeline_needs_to_call_loss(config);

    for (uint32_t epoch = 0; epoch < num_epochs; ++epoch) {
        for (auto [features, target, masks] : train_dataloader) {
            ttml::autograd::ctx().get_profiler().read_results(device, "dataloader_step_done");

            // TODO(rfurko): add mask sending, once mask becomes non-constant
            pipeline_transfer_targets_if_needed(config, target);

            auto start_timer = std::chrono::high_resolution_clock::now();
            if (gradient_accumulator_helper.should_zero_grad()) {
                optimizer->zero_grad();
            }
            auto output = run_model(model, features, masks);
            float loss_float = 0.0F;
            if (needs_to_call_loss) {
                auto loss = ttml::ops::cross_entropy_loss(output, target);
                loss = gradient_accumulator_helper.scale(loss);
                loss_float = get_loss_value(loss);
                ttml::autograd::ctx().get_profiler().read_results(device, "model_forward_done");

                loss->backward();
            } else {
                output->backward();
            }

            ttml::autograd::ctx().reset_graph();

            auto samples = features->get_value().logical_shape()[0];
            gradient_accumulator_helper.update(loss_float, samples);

            if (gradient_accumulator_helper.should_step()) {
                // synchronize gradients for multi-device case, no-op if single device
                auto parameters = get_model_parameters(model);
                if (device_config.enable_ddp && !is_three_tier_training(config)) {
                    ttml::core::distributed::synchronize_parameters(parameters);
                }

                if (config.use_clip_grad_norm) {
                    if (device_config.enable_tp) {
                        throw std::logic_error("Clip grad norm is not supported with TP");
                    }
                    ttml::core::clip_grad_norm(parameters, config.clip_grad_norm_max_norm);
                }
                optimizer->step();
                scheduler->step();
                auto global_step = optimizer->get_steps();
                if (needs_to_call_loss) {
                    if (config.enable_mpi) {
                        fmt::print("[Rank {}] ", *ttml::autograd::ctx().get_distributed_context()->rank());
                    }
                    fmt::print("Step: {}, Loss: {}\n", global_step, gradient_accumulator_helper.average_loss());
                }
                loss_meter.update(gradient_accumulator_helper.average_loss());

                if (!config.enable_mpi) {
                    // save training state if it's not 3 tier training
                    if (!config.model_path.empty() && global_step % config.model_save_interval == 0) {
                        save_training_state(config.model_path, model, scheduler, "transformer", "adamw");
                    }
                }

                ttml::autograd::ctx().get_profiler().read_results(device, fmt::format("iteration_{}", global_step));

                if (global_step >= config.max_steps) {
                    break;
                }

                gradient_accumulator_helper.reset();

                if (!is_everything_compiled) {
                    ttml::autograd::ctx().get_profiler().read_results(device, "compilation_finished");
                    is_everything_compiled = true;
                }
            }
            auto end_timer = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_timer - start_timer).count();
            if (needs_to_call_loss) {
                fmt::print(
                    "Full step time {} ms, cache entries: {}\n",
                    (double)duration / 1000,
                    device->num_program_cache_entries());
            }
        }
        if (optimizer->get_steps() >= config.max_steps) {
            break;
        }
    }

    if (!config.enable_mpi) {
        // save training state if it's not 3 tier training
        if (!config.model_path.empty()) {
            save_training_state(config.model_path, model, scheduler, "transformer", "adamw");
        }
    }

    auto end_timer = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_timer - start_timer).count();
    fmt::print(
        "{} Steps training time: {} s, cache entries: {}\n",
        config.max_steps,
        (double)duration / 1000000.,
        device->num_program_cache_entries());

    if (config.enable_mpi) {
        auto &ctx = ttml::autograd::ctx();
        auto distributed_ctx = ctx.get_distributed_context();
        distributed_ctx->barrier();
        fmt::print("Rank {}: Finalizing MPI context\n", distributed_ctx->rank());
    }

    ttml::autograd::ctx().get_profiler().read_results(device, "before close device", 0);
    ttml::autograd::ctx().close_profiler();
    return 0;
}
