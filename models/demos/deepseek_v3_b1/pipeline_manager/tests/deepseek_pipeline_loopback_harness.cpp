// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdlib>
#include <optional>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "models/demos/deepseek_v3_b1/pipeline_manager/pipeline_manager.hpp"

namespace {

std::string get_arg_value(const std::vector<std::string>& arguments, const std::string& key) {
    for (size_t idx = 0; idx + 1 < arguments.size(); ++idx) {
        if (arguments[idx] == key) {
            return arguments[idx + 1];
        }
    }
    throw std::runtime_error("Missing required argument: " + key);
}

std::vector<std::string> split_tab_separated(const std::string& line) {
    std::vector<std::string> parts;
    std::stringstream stream(line);
    std::string part;
    while (std::getline(stream, part, '\t')) {
        parts.push_back(part);
    }
    return parts;
}

}  // namespace

int main(int argc, char* argv[]) {
    using models::demos::deepseek_v3_b1::pipeline_manager::PipelineManager;
    using models::demos::deepseek_v3_b1::pipeline_manager::PipelineManagerRequest;

    try {
        std::vector<std::string> arguments;
        arguments.reserve(static_cast<size_t>(argc > 1 ? argc - 1 : 0));
        for (int idx = 1; idx < argc; ++idx) {
            arguments.emplace_back(argv[idx]);
        }

        const std::string h2d_socket_id = get_arg_value(arguments, "--h2d-socket-id");
        const std::string d2h_socket_id = get_arg_value(arguments, "--d2h-socket-id");
        const uint32_t page_size_bytes =
            static_cast<uint32_t>(std::stoul(get_arg_value(arguments, "--page-size-bytes")));
        const uint32_t iterations = static_cast<uint32_t>(std::stoul(get_arg_value(arguments, "--iterations")));
        const uint32_t initial_token = static_cast<uint32_t>(std::stoul(get_arg_value(arguments, "--initial-token")));
        const uint32_t connect_timeout_ms =
            static_cast<uint32_t>(std::stoul(get_arg_value(arguments, "--connect-timeout-ms")));

        PipelineManager pipeline_manager(h2d_socket_id, d2h_socket_id, page_size_bytes, connect_timeout_ms);
        PipelineManagerRequest request{
            .request_id = "loopback_harness",
            .prompt_token_ids = {initial_token},
            .max_new_tokens = iterations,
            .eos_token_id = std::nullopt,
        };
        pipeline_manager.run_one_shot(request, std::cout);
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << error.what() << std::endl;
        return EXIT_FAILURE;
    }
}
