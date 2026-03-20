// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdlib>
#include <iostream>
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

}  // namespace

int main(int argc, char* argv[]) {
    using models::demos::deepseek_v3_b1::pipeline_manager::PipelineManager;

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
        if (iterations == 0) {
            throw std::runtime_error("iterations must be greater than zero");
        }

        pipeline_manager.start();
        pipeline_manager.write_token(initial_token);

        std::vector<uint32_t> observed_tokens;
        observed_tokens.reserve(iterations);

        for (uint32_t step = 0; step < iterations; ++step) {
            const uint32_t token_id = pipeline_manager.read_token();
            observed_tokens.push_back(token_id);
            std::cout << token_id << std::endl;

            if (step + 1 < iterations) {
                pipeline_manager.write_token(token_id);
            }
        }

        pipeline_manager.stop();
        std::cout << "COMPLETE\t" << observed_tokens.size();
        for (uint32_t token_id : observed_tokens) {
            std::cout << "\t" << token_id;
        }
        std::cout << std::endl;
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << error.what() << std::endl;
        return EXIT_FAILURE;
    }
}
