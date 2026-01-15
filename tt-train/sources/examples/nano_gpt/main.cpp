// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <CLI/CLI.hpp>
#include <chrono>
#include <core/ttnn_all_includes.hpp>
#include <csignal>
#include <cstdint>
#include <filesystem>

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
#include "utils/memory_utils.hpp"

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

struct TrainingConfig {
    std::string project_name;
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
    std::string model_config;
    std::string data_path;
    std::string scheduler_type = "identity";
    std::string tokenizer_type = "char";
    bool use_clip_grad_norm = false;
    float clip_grad_norm_max_norm = 1.0F;
};

TrainingConfig parse_config(const YAML::Node &yaml_config) {
    TrainingConfig config;
    auto training_config = yaml_config["training_config"];
    config.project_name = training_config["project_name"].as<std::string>("tt_train_nano_gpt");
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
    config.model_config = training_config["model_config"].as<std::string>("");
    config.data_path = training_config["data_path"].as<std::string>(std::string(DATA_FOLDER) + "/shakespeare.txt");
    config.scheduler_type = training_config["scheduler_type"].as<std::string>(config.scheduler_type);
    config.use_clip_grad_norm = training_config["use_clip_grad_norm"].as<bool>(config.use_clip_grad_norm);
    config.clip_grad_norm_max_norm =
        training_config["clip_grad_norm_max_norm"].as<float>(config.clip_grad_norm_max_norm);
    config.tokenizer_type = training_config["tokenizer_type"].as<std::string>(config.tokenizer_type);

    return config;
}

struct MultihostConfig {
    bool enable_mpi = false;
    uint32_t num_mh_workers = 0U;
    SocketType socket_type = SocketType::MPI;
    std::optional<ttml::models::distributed::pipeline_parallel_llama::PipelineParallelConfig> pipeline_parallel_config;
};

MultihostConfig parse_multihost_config(const YAML::Node &yaml_config) {
    MultihostConfig config;
    auto multihost_config = yaml_config["multihost_config"];
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

struct ModelConfig {
    std::string model_type = "gpt2";
    std::string model_path = "";
    std::variant<ttml::models::gpt2::TransformerConfig, ttml::models::llama::LlamaConfig> transformer_config;
};

ModelConfig parse_model_config(const YAML::Node &yaml_config) {
    ModelConfig config;
    auto model_config = yaml_config["transformer_config"];
    config.model_type = model_config["model_type"].as<std::string>();
    config.model_path = model_config["model_path"].as<std::string>("");

    if (config.model_type == "gpt2") {
        config.transformer_config = ttml::models::gpt2::read_config(model_config);
    } else if (config.model_type == "llama") {
        config.transformer_config = ttml::models::llama::read_config(model_config);
    } else {
        throw std::runtime_error("Unknown model type: " + config.model_type);
    }

    return config;
}

const std::unordered_map<
    std::string,
    std::function<std::unique_ptr<ttml::schedulers::LRSchedulerBase>(ttml::optimizers::OptimizerBase *, size_t)>>
    schedulers = {{"identity", create_idendity_scheduler}, {"warmup_linear", create_warmup_with_linear_scheduler}};

namespace {

inline bool is_pipeline_parallel_enabled(const MultihostConfig &config) {
    return config.pipeline_parallel_config.has_value();
}

inline int get_mpi_rank_or_zero() {
    auto &ctx = ttml::autograd::ctx();
    auto distributed_ctx = ctx.get_distributed_context();
    return distributed_ctx ? *distributed_ctx->rank() : 0;
}

inline bool is_three_tier_training(const MultihostConfig &config) {
    return config.enable_mpi && !is_pipeline_parallel_enabled(config);
}

inline bool is_last_pipeline_stage(const MultihostConfig &config) {
    if (!is_pipeline_parallel_enabled(config)) {
        return true;
    }
    return static_cast<unsigned>(get_mpi_rank_or_zero()) == (config.num_mh_workers - 1U);
}

inline bool pipeline_needs_to_call_loss(const MultihostConfig &config) {
    return !is_pipeline_parallel_enabled(config) || is_last_pipeline_stage(config);
}

inline void pipeline_transfer_targets_if_needed(const MultihostConfig &config, const TensorPtr &target) {
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

    std::string training_config_name =
        std::filesystem::current_path().string() + "/configs/training_configs/training_shakespeare_nanogpt.yaml";
    std::string multihost_config_name = "";

    std::string run_name = "";
    bool add_time_to_name = true;
    std::string safetensors_path = "";
    std::string save_and_exit_path = "";

    app.add_option("-c,--config", training_config_name, "Training Config name")->default_val(training_config_name);
    app.add_option("--multihost", multihost_config_name, "Multihost Config name")->default_val(multihost_config_name);

    app.add_option("-t,--add_time_to_name", add_time_to_name, "Add time to run name")->default_val(add_time_to_name);
    app.add_option("-n,--name", run_name, "Run name")->default_val(run_name);
    app.add_option(
           "-s,--save_and_exit", save_and_exit_path, "Save and exit (path to directory for model serialization)")
        ->default_val(save_and_exit_path);
    app.add_option("--safetensors", safetensors_path, "Loads safetensors model from the given path")
        ->default_val(safetensors_path);
    CLI11_PARSE(app, argc, argv);

    auto yaml_config = YAML::LoadFile(training_config_name);

    TrainingConfig training_config = parse_config(yaml_config);
    DeviceConfig device_config = parse_device_config(yaml_config);
    ModelConfig model_config = parse_model_config(YAML::LoadFile(training_config.model_config));

    // Pass tt::tt_metal::IGraphProcessor::RunMode::NO_DISPATCH to measure memory usage
    // of model that doesn't fit in the memory of the device.
    ttnn::ScopeGuard memory_usage_guard = ttml::utils::MemoryUsageTracker::begin_capture();

    MultihostConfig multihost_config;
    if (!multihost_config_name.empty()) {
        multihost_config = parse_multihost_config(YAML::LoadFile(multihost_config_name));
    }

    if (multihost_config.enable_mpi) {
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

    if (multihost_config.enable_mpi) {
        fmt::print("MPI config:\n");
        fmt::print("  enable_mpi: {}\n", multihost_config.enable_mpi);
        fmt::print("  num_mh_workers: {}\n", multihost_config.num_mh_workers);
        fmt::print("  socket_type: {}\n", multihost_config.socket_type == SocketType::MPI ? "MPI" : "FABRIC");
    }

    if (device_config.enable_tp) {
        if (!model_config.model_path.empty()) {
            throw std::runtime_error("Save and load is not supported with Tensor Parallel model");
        }
    }

    // set seed
    ttml::autograd::ctx().set_seed(training_config.seed);
    if (multihost_config.enable_mpi) {
        int rank = *ttml::autograd::ctx().get_distributed_context()->rank();
        auto seed = training_config.seed + static_cast<uint32_t>(rank);
        ttml::autograd::ctx().set_seed(seed);
    }
    auto schedule_func = schedulers.at(training_config.scheduler_type);

    fmt::print("Max steps {}\n", training_config.max_steps);
    fmt::print("Batch size {}\n", training_config.batch_size);
    fmt::print("Gradient accumulation steps {}\n", training_config.gradient_accumulation_steps);
    fmt::print("Total batch size {}\n", training_config.batch_size * training_config.gradient_accumulation_steps);
    fmt::print("Scheduler type {}\n", training_config.scheduler_type);
    fmt::print("Seed {}\n", ttml::autograd::ctx().get_seed());
    auto sequence_length =
        std::visit([](auto &&arg) { return arg.max_sequence_length; }, model_config.transformer_config);

    std::variant<std::string, YAML::Node> text_or_tokens;

    try {
        // check file extension:
        if (training_config.data_path.ends_with(".txt")) {
            text_or_tokens = read_file_to_str(training_config.data_path);
        } else {
            auto yaml_data = YAML::LoadFile(training_config.data_path);
            yaml_data["sequence_length"] = sequence_length;
            text_or_tokens = yaml_data;
        }
    } catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
        return -1;
    }

    auto create_dataset =
        [](const auto &data_source, const auto sequence_length, const auto &train_config, auto &model_config) {
            std::string tokenizer_type = train_config.tokenizer_type;

            if (tokenizer_type == "char") {
                auto [dataset, tokenizer] =
                    ttml::datasets::create_in_memory_token_dataset<ttml::tokenizers::CharTokenizer>(
                        std::get<std::string>(data_source), sequence_length);

                std::visit(
                    [&](auto &&arg) { arg.vocab_size = tokenizer->get_vocab_size(); }, model_config.transformer_config);

                return dataset;
            } else if (tokenizer_type == "bpe") {
                auto &yaml_node = std::get<YAML::Node>(data_source);

                auto dataset = ttml::datasets::create_token_dataset_from_yaml(yaml_node);

                std::visit(
                    [&](auto &&arg) { arg.vocab_size = yaml_node["tokenizer_vocab_size"].template as<uint32_t>(); },
                    model_config.transformer_config);

                return dataset;
            } else {
                throw std::runtime_error("Unknown tokenizer type: " + tokenizer_type);
            }
        };

    auto dataset = create_dataset(text_or_tokens, sequence_length, training_config, model_config);

    fmt::print("Dataset size: {}\n", dataset.get_size());

    auto num_devices = device_config.mesh_shape[0] * device_config.mesh_shape[1];
    // enable fabric config for 3-tier architecture, tp, ddp
    if (multihost_config.socket_type == SocketType::FABRIC || device_config.enable_tp || device_config.enable_ddp) {
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
    auto train_dataloader =
        DataLoader(dataset, /* batch_size */ training_config.batch_size, /* shuffle */ true, collate_fn);

    fmt::print("Overriding vocab size to be divisible by 32\n");
    // this is workaround for tensor parallel case, we need to have vocab size divisible by 32 per device
    std::visit(
        [&](auto &&arg) {
            if constexpr (requires { arg.vocab_size; }) {
                arg.vocab_size = round_up_to_tile(arg.vocab_size, (device_config.enable_tp ? num_devices : 1U) * 32U);
            } else {
                throw std::runtime_error(
                    "Unsupported transformer configuration type: " + std::string(typeid(arg).name()));
            }
        },
        model_config.transformer_config);

    Model model = std::visit(
        [&device_config, &multihost_config](auto &&arg) -> Model {
            if constexpr (std::is_same_v<std::decay_t<decltype(arg)>, ttml::models::llama::LlamaConfig>) {
                if (multihost_config.pipeline_parallel_config) {
                    return ttml::models::distributed::pipeline_parallel_llama::create(
                        arg, *multihost_config.pipeline_parallel_config, device_config.enable_tp);
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
        model_config.transformer_config);

    ttml::utils::MemoryUsageTracker::snapshot("MODEL_CREATION");

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
        ttml::serialization::FlatBufferFile serializer;
        ttml::serialization::write_module(serializer, model_config.model_type, model.get());
        serializer.serialize(save_and_exit_path);
        fmt::println("Model saved to {}", save_and_exit_path);
        std::exit(0);
    }

    // Load model parameters if in eval mode and model path exists
    if (!safetensors_path.empty() && !model_config.model_path.empty() &&
        std::filesystem::exists(model_config.model_path)) {
        fmt::print("Loading model parameters\n");
        load_model_parameters(model_config.model_path, model, model_config.model_type);
        fmt::print("Model loaded\n");
    }

    auto adamw_params = ttml::optimizers::AdamWConfig();
    adamw_params.lr = training_config.learning_rate;
    adamw_params.weight_decay = training_config.weight_decay;
    adamw_params.use_kahan_summation = training_config.use_kahan_summation;

    if (training_config.use_no_op) {
        fmt::print("WARNING: Using NoOp optimizer - parameters will NOT be updated.\n");
    } else if (!is_three_tier_training(multihost_config)) {
        fmt::print("AdamW configuration:\n");
        fmt::print("    Learning rate: {}\n", adamw_params.lr);
        fmt::print("    Weight decay: {}\n", adamw_params.weight_decay);
        fmt::print("    Use Kahan summation: {}\n", adamw_params.use_kahan_summation);
    } else {
        fmt::println("Remote optimizer configured!");
    }

    fmt::print("Number of parameters: {}\n", get_number_of_parameters(model, device_config.enable_tp));

    auto select_optimizer = [&model,
                             &adamw_params,
                             &training_config,
                             &multihost_config]() -> std::unique_ptr<ttml::optimizers::OptimizerBase> {
        if (is_three_tier_training(multihost_config)) {
            return std::make_unique<ttml::optimizers::RemoteOptimizer>(
                get_model_parameters(model), multihost_config.num_mh_workers);
        } else if (training_config.use_no_op) {
            return std::make_unique<ttml::optimizers::NoOp>(get_model_parameters(model));
        } else if (training_config.use_moreh_adamw) {
            return std::make_unique<ttml::optimizers::MorehAdamW>(get_model_parameters(model), adamw_params);
        } else {
            return std::make_unique<ttml::optimizers::AdamW>(get_model_parameters(model), adamw_params);
        }
    };

    auto optimizer = select_optimizer();
    auto scheduler = schedule_func(optimizer.get(), training_config.max_steps);

    if (is_three_tier_training(multihost_config)) {
        auto *optimizer_ptr = dynamic_cast<ttml::optimizers::RemoteOptimizer *>(optimizer.get());
        if (!optimizer_ptr) {
            throw std::runtime_error("Optimizer is not RemoteOptimizer");
        }
        fmt::println("[worker] Remote optimizer receiving weights from rank {}", multihost_config.num_mh_workers);
        optimizer_ptr->receive_weights();
        fmt::println("[worker] Remote optimizer received weights from rank {}", multihost_config.num_mh_workers);
    } else if (training_config.use_no_op) {
        fmt::print("Skipping training state load (NoOp optimizer)\n");
    } else {
        // otherwise proceed with normal loading training state if necessary
        if (!model_config.model_path.empty() && std::filesystem::exists(model_config.model_path)) {
            fmt::print("Loading model from {}\n", model_config.model_path);
            fmt::print("Loading training state\n");
            load_training_state(
                model_config.model_path, model, scheduler, model_config.model_type, optimizer->get_name());
            fmt::print("Model loaded after {} steps\n", optimizer->get_steps());
        }
    }

    ttml::utils::MemoryUsageTracker::snapshot("OPTIMIZER_CREATION");

    if (multihost_config.enable_mpi && training_config.use_clip_grad_norm) {
        throw std::logic_error("Clip grad norm is not supported with 3 tier training");
    }

    if (device_config.enable_ddp) {
        auto num_devices = static_cast<uint32_t>(device->num_devices());
        if (training_config.batch_size % num_devices != 0) {
            throw std::logic_error(fmt::format(
                "Batch size must be divisible by the number of devices. Batch size = {}, devices = {}",
                training_config.batch_size,
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

    const uint32_t num_epochs = training_config.num_epochs;
    auto gradient_accumulator_helper = GradientAccumulator(training_config.gradient_accumulation_steps);

    bool is_everything_compiled = false;
    auto memory_snapshot = [&is_everything_compiled](const std::string &name) {
        if (!is_everything_compiled) {
            ttml::utils::MemoryUsageTracker::snapshot(name);
        }
    };

    const bool needs_to_call_loss = pipeline_needs_to_call_loss(multihost_config);

    // Training loop
    for (uint32_t epoch = 0; epoch < num_epochs; ++epoch) {
        for (auto [features, target, masks] : train_dataloader) {
            ttml::autograd::ctx().get_profiler().read_results(device, "dataloader_step_done");

            // TODO(rfurko): add mask sending, once mask becomes non-constant
            pipeline_transfer_targets_if_needed(multihost_config, target);

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

                memory_snapshot("FORWARD_PASS");
                loss->backward();
                memory_snapshot("BACKWARD_PASS");
            } else {
                memory_snapshot("FORWARD_PASS");
                output->backward();
                memory_snapshot("BACKWARD_PASS");
            }

            ttml::autograd::ctx().reset_graph();

            auto samples = features->get_value().logical_shape()[0];
            gradient_accumulator_helper.update(loss_float, samples);

            if (gradient_accumulator_helper.should_step()) {
                // synchronize gradients for multi-device case, no-op if single device
                auto parameters = get_model_parameters(model);
                if (device_config.enable_ddp && !is_three_tier_training(multihost_config)) {
                    ttml::core::distributed::synchronize_gradients(parameters);
                }

                if (training_config.use_clip_grad_norm) {
                    if (device_config.enable_tp) {
                        throw std::logic_error("Clip grad norm is not supported with TP");
                    }
                    ttml::core::clip_grad_norm(parameters, training_config.clip_grad_norm_max_norm);
                }
                optimizer->step();
                scheduler->step();
                auto global_step = optimizer->get_steps();
                if (needs_to_call_loss) {
                    if (multihost_config.enable_mpi) {
                        fmt::print("[Rank {}] ", *ttml::autograd::ctx().get_distributed_context()->rank());
                    }
                    fmt::print("Step: {}, Loss: {}\n", global_step, gradient_accumulator_helper.average_loss());
                }
                loss_meter.update(gradient_accumulator_helper.average_loss());

                if (!multihost_config.enable_mpi) {
                    // save training state if it's not 3 tier training
                    if (!model_config.model_path.empty() && global_step % training_config.model_save_interval == 0) {
                        save_training_state(
                            model_config.model_path, model, scheduler, model_config.model_type, optimizer->get_name());
                    }
                }

                ttml::autograd::ctx().get_profiler().read_results(device, fmt::format("iteration_{}", global_step));

                if (global_step >= training_config.max_steps) {
                    break;
                }

                gradient_accumulator_helper.reset();

                if (!is_everything_compiled) {
                    ttml::autograd::ctx().get_profiler().read_results(device, "compilation_finished");
                    is_everything_compiled = true;
                    ttml::utils::MemoryUsageTracker::end_capture("FIRST_ITERATION_COMPLETE");
                    ttml::utils::MemoryUsageTracker::print_memory_usage();
                    ttml::utils::MemoryUsageTracker::clear();
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
        if (optimizer->get_steps() >= training_config.max_steps) {
            break;
        }
    }

    if (!multihost_config.enable_mpi) {
        // save training state if it's not 3 tier training
        if (!model_config.model_path.empty()) {
            save_training_state(
                model_config.model_path, model, scheduler, model_config.model_type, optimizer->get_name());
        }
    }

    auto end_timer = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_timer - start_timer).count();
    fmt::print(
        "{} Steps training time: {} s, cache entries: {}\n",
        training_config.max_steps,
        (double)duration / 1000000.,
        device->num_program_cache_entries());

    if (multihost_config.enable_mpi) {
        auto &ctx = ttml::autograd::ctx();
        auto distributed_ctx = ctx.get_distributed_context();
        distributed_ctx->barrier();
        fmt::print("Rank {}: Finalizing MPI context\n", distributed_ctx->rank());
    }

    ttml::autograd::ctx().get_profiler().read_results(device, "before close device", 0);
    ttml::autograd::ctx().close_device();
    ttml::autograd::ctx().close_profiler();
    return 0;
}
