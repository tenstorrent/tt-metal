// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <fmt/core.h>

#include <CLI/CLI.hpp>
#include <array>
#include <cstdio>  // for popen, pclose
#include <iostream>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>

#include "autograd/auto_context.hpp"
#include "core/distributed/distributed.hpp"
#include "core/distributed/mpi_context.hpp"
#include "core/tt_tensor_utils.hpp"
#include "datasets/dataloader.hpp"
#include "datasets/generators.hpp"
#include "optimizers/adamw.hpp"
#include "roles/optimizer.hpp"
#include "roles/worker.hpp"

constexpr int WORKER_RANK = 1;
constexpr int AGGREGATOR_RANK = 0;

struct board_entry {
    std::string pci_dev_id;
    std::string board_type;
    std::string device_series;
    std::string board_number;
};

std::string replace_all(const std::string& s, const std::string& old_val, const std::string& new_val) {
    std::string result = s;
    size_t pos = 0;
    while ((pos = result.find(old_val, pos)) != std::string::npos) {
        result.replace(pos, old_val.size(), new_val);
        pos += new_val.size();
    }
    return result;
}

std::vector<board_entry> parse_tt_smi_output(const std::string& command_output) {
    std::vector<board_entry> board_list;

    // Regex for ASCII pipes:
    static const std::regex row_pattern(R"(^\|?\s*(.*?)\s*\|\s*(.*?)\s*\|\s*(.*?)\s*\|\s*(.*?)\s*\|?$)");

    std::stringstream stream(command_output);
    std::string line;
    while (std::getline(stream, line)) {
        line = replace_all(line, "\xE2\x94\x82", "|");

        // Now we have a line with normal ASCII pipes, e.g.:
        // "| 0          | wormhole   | n150 L        | 0100018511732049 |"

        std::smatch match;
        if (std::regex_match(line, match, row_pattern)) {
            // Skip header lines containing "Pci Dev ID"
            if (match[1].str().find("Pci Dev ID") != std::string::npos) {
                continue;
            }

            board_entry entry;
            entry.pci_dev_id = match[1];
            entry.board_type = match[2];
            entry.device_series = match[3];
            entry.board_number = match[4];

            board_list.push_back(std::move(entry));
        }
    }

    return board_list;
}

// Runs "tt-smi -ls" via popen, parses the result, and returns a list of boards.
std::vector<board_entry> get_tt_smi_boards() {
    std::array<char, 256> buffer{};
    std::string command_output;

    FILE* pipe_ptr = popen("tt-smi -ls", "r");
    if (!pipe_ptr) {
        throw std::runtime_error("Failed to run tt-smi -ls via popen()");
    }

    while (fgets(buffer.data(), static_cast<int>(buffer.size()), pipe_ptr)) {
        fmt::println("Buffer: {}", buffer.data());
        command_output += buffer.data();
    }

    int return_code = pclose(pipe_ptr);
    if (return_code != 0) {
        std::cerr << "Warning: tt-smi -ls exited with code " << return_code << std::endl;
    }

    return parse_tt_smi_output(command_output);
}

void print_tt_smi() {
    auto& ctx = ttml::autograd::ctx();
    auto& mpi_ctx = ctx.get_mpi_context();
    auto rank = mpi_ctx.get_rank();
    try {
        std::vector<board_entry> boards = get_tt_smi_boards();
        fmt::print("Rank {}: Parsed boards:\n", rank);
        for (const auto& b : boards) {
            fmt::print(
                "PCI Dev ID:      {}\n"
                "Board Type:      {}\n"
                "Device Series:   {}\n"
                "Board Number:    {}\n"
                "-----------------------------\n",
                b.pci_dev_id,
                b.board_type,
                b.device_series,
                b.board_number);
        }
    } catch (const std::exception& ex) {
        fmt::print(stderr, "Error: {}\n", ex.what());
    }
}

void test_send_recv_tensor() {
    fmt::print("test_send_recv_tensor started\n");
    auto& ctx = ttml::autograd::ctx();
    auto& mpi_ctx = ctx.get_mpi_context();
    auto rank = mpi_ctx.get_rank();
    auto size = mpi_ctx.get_size();
    fmt::print("Rank {}:, Testing send/recv tensor\n", rank);
    auto& device = ctx.get_device();
    if (size < 2) {
        fmt::print("This example requires at least 2 processes.\n");
        return;
    }
    auto shape = ttml::core::create_shape({1, 1, 2, 3});
    if (rank == 0) {
        auto tensor = ttml::core::ones(shape, &device);
        auto vec_ones = ttml::core::to_vector(tensor);
        fmt::print("Rank {}, vector size: {}\n", rank, vec_ones.size());
        fmt::print("Rank {}: sending tensor: [{}]\n", rank, vec_ones);
        ttml::core::distributed::send_tensor(tensor, 1);
        fmt::print("Rank {}: sent tensor\n", rank);
    } else if (rank == 1) {
        auto tensor = ttml::core::zeros(shape, &device);
        auto vec_zeros = ttml::core::to_vector(tensor);
        fmt::print("Rank {}: original tensor: [{}]\n", rank, vec_zeros);
        ttml::core::distributed::recv_tensor(tensor, 0);
        auto vec = ttml::core::to_vector(tensor);
        fmt::print("Rank {}: received tensor: [{}]\n", rank, vec);
    }
}

struct LinearRegressionParameters {
    const size_t training_samples_count = 128000;
    const uint32_t num_features = 64;
    const uint32_t num_targets = 32;
    const float noise = 0.0F;
    const bool bias = true;
    const uint32_t batch_size = 128;
    const int num_epochs = 10;
};

void regression_training() {
    auto& ctx = ttml::autograd::ctx();
    auto& device = ctx.get_device();
    auto& mpi_ctx = ctx.get_mpi_context();
    int rank = mpi_ctx.get_rank();
    LinearRegressionParameters params{};
    auto training_params = ttml::datasets::MakeRegressionParams{
        .n_samples = params.training_samples_count,
        .n_features = params.num_features,
        .n_targets = params.num_targets,
        .noise = params.noise,
        .bias = params.bias,
    };

    auto model = std::make_shared<ttml::modules::LinearLayer>(params.num_features, params.num_targets);

    if (rank == WORKER_RANK) {
        fmt::print("Worker rank {} initializing worker\n", rank);
        auto train_dataset = ttml::datasets::make_regression(training_params);

        std::function<roles::BatchType(std::vector<roles::DatasetSample> && samples)> collate_fn =
            [&params, &device](std::vector<roles::DatasetSample>&& samples) {
                const uint32_t batch_size = samples.size();
                std::vector<float> data;
                std::vector<float> targets;
                data.reserve(params.batch_size * params.num_features);
                targets.reserve(params.batch_size * params.num_targets);
                for (auto& [features, target] : samples) {
                    std::move(features.begin(), features.end(), std::back_inserter(data));
                    std::move(target.begin(), target.end(), std::back_inserter(targets));
                }
                auto feature_shape = ttml::core::create_shape({params.batch_size, 1, 1, params.num_features});
                auto target_shape = ttml::core::create_shape({params.batch_size, 1, 1, params.num_targets});
                auto data_tensor =
                    ttml::autograd::create_tensor(ttml::core::from_vector<float>(data, feature_shape, &device));
                auto targets_tensor =
                    ttml::autograd::create_tensor(ttml::core::from_vector(targets, target_shape, &device));
                return std::make_pair(data_tensor, targets_tensor);
            };

        auto train_dataloader = roles::DataLoader(train_dataset, params.batch_size, /* shuffle */ true, collate_fn);
        roles::Worker worker(train_dataloader, model, AGGREGATOR_RANK);
        for (int i = 0; i < params.num_epochs; i++) {
            fmt::print("Rank {}: Training epoch {}\n", rank, i);
            worker.training_step();
        }
        fmt::print("Rank {}: Training finished\n", rank);
    }
    if (rank == AGGREGATOR_RANK) {
        // use optimizer now
        roles::Optimizer optimizer(model, rank, WORKER_RANK);

        for (int i = 0; i < params.num_epochs; i++) {
            fmt::print("Rank {}: Optimizer epoch {}\n", rank, i);
            for (int i = 0; i < params.training_samples_count / params.batch_size; i++) {
                optimizer.optimization_step();
                optimizer.send_weights();
            }
            fmt::print("Rank {}: Optimizer epoch {} finished\n", rank, i);
        }

        fmt::print("Rank {}: Optimizer finished\n", rank);
    }
}

int main(int argc, char** argv) {
    auto& ctx = ttml::autograd::ctx();
    ctx.init_mpi_context(argc, argv);
    auto& mpi_ctx = ctx.get_mpi_context();
    CLI::App app{"Multihost Example"};
    fmt::print("Size {}, Rank {}: Initializing MPI context\n", mpi_ctx.get_size(), mpi_ctx.get_rank());
    argv = app.ensure_utf8(argv);

    bool print_tt_smi_output = false;
    bool run_test_send_recv_tensor = false;
    bool run_regression_training = true;

    // app.add_option("-c,--config", config_name, "Yaml Config name")->default_val(config_name);
    app.add_option("-t,--tt_smi", print_tt_smi_output, "print tt-smi on all hosts")->default_val(print_tt_smi_output);
    app.add_option("--run_test_send_recv_tensor", run_test_send_recv_tensor, "run simple send recv tensor test")
        ->default_val(run_test_send_recv_tensor);
    app.add_option("--run_regression_training", run_regression_training, "runs regression training")
        ->default_val(run_regression_training);

    CLI11_PARSE(app, argc, argv);

    if (print_tt_smi_output) {
        print_tt_smi();
    }
    if (run_test_send_recv_tensor) {
        test_send_recv_tensor();
    }
    if (run_regression_training) {
        regression_training();
    }

    mpi_ctx.barrier();
    mpi_ctx.finalize();
    fmt::print("Rank {}: Finalized MPI context\n", mpi_ctx.get_rank());
    return 0;
}
