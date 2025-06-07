// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <fmt/base.h>
#include <stdint.h>
#include <cstddef>
#include <string>
#include <vector>

#include <tt-metalium/assert.hpp>
#include <tt-metalium/fabric_edm_packet_header.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-logger/tt-logger.hpp>
#include "tests/ttnn/unit_tests/gtests/ccl/test_fabric_edm_common.hpp"

// Global state for daemon mode
static bool daemon_running = true;
// NOTE: This path need to be same as the one written in test_fabric_edm_bandwidth.py
static std::string daemon_pipe_path = "/tmp/tt_metal_fabric_edm_daemon";

// Signal handler for graceful shutdown
void signal_handler(int signum) { daemon_running = false; }

struct TestParams {
    WriteThroughputStabilityTestWithPersistentFabricParams params;
    std::string message_noc_type;
    size_t num_messages;
    size_t packet_payload_size_bytes;
    bool fabric_unicast;
};

// noc_send_type, flush
static std::tuple<tt::tt_fabric::NocSendType, bool> get_noc_send_type(const std::string& message_noc_type) {
    tt::tt_fabric::NocSendType noc_send_type;
    bool flush = true;
    if (message_noc_type == "noc_unicast_write") {
        noc_send_type = tt::tt_fabric::NocSendType::NOC_UNICAST_WRITE;
    } else if (message_noc_type == "noc_multicast_write") {
        noc_send_type = tt::tt_fabric::NocSendType::NOC_MULTICAST_WRITE;
    } else if (message_noc_type == "noc_unicast_flush_atomic_inc") {
        noc_send_type = tt::tt_fabric::NocSendType::NOC_UNICAST_ATOMIC_INC;
        flush = true;
    } else if (message_noc_type == "noc_unicast_no_flush_atomic_inc") {
        noc_send_type = tt::tt_fabric::NocSendType::NOC_UNICAST_ATOMIC_INC;
        flush = false;
    } else if (message_noc_type == "noc_fused_unicast_write_flush_atomic_inc") {
        noc_send_type = tt::tt_fabric::NocSendType::NOC_FUSED_UNICAST_ATOMIC_INC;
        flush = true;
    } else if (message_noc_type == "noc_fused_unicast_write_no_flush_atomic_inc") {
        noc_send_type = tt::tt_fabric::NocSendType::NOC_FUSED_UNICAST_ATOMIC_INC;
        flush = false;
    } else {
        TT_THROW("Invalid message type: {}", message_noc_type.c_str());
    }

    return std::make_tuple(noc_send_type, flush);
}

static int baseline_validate_test_environment(const WriteThroughputStabilityTestWithPersistentFabricParams& params) {
    uint32_t min_test_num_devices = 8;
    if (tt::tt_metal::GetNumAvailableDevices() < min_test_num_devices) {
        log_warning(tt::LogTest, "This test can only be run on T3000 or TG devices");
        return 1;
    }

    uint32_t galaxy_num_devices = 32;
    if (params.num_links > 2 && tt::tt_metal::GetNumAvailableDevices() < galaxy_num_devices) {
        log_warning(tt::LogTest, "This test with {} links can only be run on Galaxy systems", params.num_links);
        return 1;
    }

    if (tt::tt_metal::GetNumAvailableDevices() == min_test_num_devices && params.num_links > 1 && params.line_size > 4) {
        log_warning(tt::LogTest, "T3000 cannot run multi-link with more than 4 devices");
        return 1;
    }

    return 0;
}

static int run_single_test(TestParams& test_params, const std::string& test_mode) {
    auto chip_send_type = test_params.fabric_unicast ? tt::tt_fabric::CHIP_UNICAST : tt::tt_fabric::CHIP_MULTICAST;
    auto [noc_send_type, flush] = get_noc_send_type(test_params.message_noc_type);

    std::vector<Fabric1DPacketSendTestSpec> test_specs{
        {.chip_send_type = chip_send_type,
         .noc_send_type = noc_send_type,
         .num_messages = test_params.num_messages,
         .packet_payload_size_bytes = test_params.packet_payload_size_bytes,
         .flush = flush}};

    try {
        if (test_mode == "1_fabric_instance") {
            Run1DFabricPacketSendTest(test_specs, test_params.params);
        } else if (test_mode == "1D_fabric_on_mesh") {
            if (test_params.params.fabric_mode == FabricTestMode::Linear) {
                Run1DFabricPacketSendTest<Fabric1DLineDeviceInitFixture>(test_specs, test_params.params);
            } else if (test_params.params.fabric_mode == FabricTestMode::FullRing) {
                Run1DFabricPacketSendTest<Fabric1DRingDeviceInitFixture>(test_specs, test_params.params);
            } else {
                TT_THROW(
                    "Invalid fabric mode when using device init fabric in 1D fabric on mesh BW test: {}",
                    test_params.params.fabric_mode);
            }
        } else {
            TT_THROW("Invalid test mode: {}", test_mode.c_str());
        }
        return 0;
    } catch (const std::exception& e) {
        log_error(tt::LogTest, "Test failed: {}", e.what());
        return 1;
    }
}

// Unified parameter parser that works with both command line args and tokenized strings
template <typename StringProvider>
static TestParams parse_parameters(StringProvider& provider, bool has_mesh_params) {
    TestParams test_params;

    test_params.fabric_unicast = std::stoi(provider.next());
    test_params.message_noc_type = provider.next();
    test_params.num_messages = std::stoi(provider.next());
    test_params.params.num_links = std::stoi(provider.next());
    test_params.params.num_op_invocations = std::stoi(provider.next());
    test_params.params.line_sync = std::stoi(provider.next());
    test_params.params.line_size = std::stoi(provider.next());
    test_params.packet_payload_size_bytes = std::stoi(provider.next());
    test_params.params.fabric_mode = static_cast<FabricTestMode>(std::stoi(provider.next()));
    test_params.params.disable_sends_for_interior_workers = std::stoi(provider.next());
    test_params.params.disable_end_workers_in_backward_direction = std::stoi(provider.next());
    test_params.params.senders_are_unidirectional = std::stoi(provider.next());

    // Handle mesh parameters if present
    if (has_mesh_params) {
        test_params.params.num_fabric_rows = std::stoi(provider.next());
        test_params.params.num_fabric_cols = std::stoi(provider.next());
        test_params.params.first_link_offset = std::stoi(provider.next());
    } else {
        test_params.params.num_fabric_rows = 0;
        test_params.params.num_fabric_cols = 0;
        test_params.params.first_link_offset = 0;
    }

    return test_params;
}

// String provider for command line arguments
class ArgvProvider {
    char** argv;
    size_t& idx;

public:
    ArgvProvider(char** argv, size_t& idx) : argv(argv), idx(idx) {}
    std::string next() { return std::string(argv[idx++]); }
};

// String provider for tokenized pipe-separated strings
class TokenProvider {
    const std::vector<std::string>& tokens;
    size_t& idx;

public:
    TokenProvider(const std::vector<std::string>& tokens, size_t& idx) : tokens(tokens), idx(idx) {}
    std::string next() { return tokens[idx++]; }
};

// Helper function to parse command line arguments
static TestParams parse_command_line_args(char** argv, size_t& arg_idx, const std::string& test_mode, int argc) {
    ArgvProvider provider(argv, arg_idx);
    bool has_mesh_params = (test_mode == "1D_fabric_on_mesh" && arg_idx < argc);
    return parse_parameters(provider, has_mesh_params);
}

// Helper function to parse pipe-separated string
static TestParams parse_pipe_separated_params(const std::string& params_str, const std::string& test_mode) {
    std::istringstream iss(params_str);
    std::vector<std::string> tokens;
    std::string token;

    while (std::getline(iss, token, '|')) {
        tokens.push_back(token);
    }

    if (tokens.size() < 12) {
        TT_THROW("Invalid parameter string format");
    }

    size_t idx = 0;
    TokenProvider provider(tokens, idx);
    bool has_mesh_params = (test_mode == "1D_fabric_on_mesh" && tokens.size() > 12);
    return parse_parameters(provider, has_mesh_params);
}

static void run_daemon_mode() {
    log_info(tt::LogTest, "Starting fabric EDM daemon mode...");

    // Setup signal handlers
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);

    // Create named pipe
    unlink(daemon_pipe_path.c_str());
    if (mkfifo(daemon_pipe_path.c_str(), 0666) == -1) {
        TT_THROW("Failed to create named pipe: {}", daemon_pipe_path);
    }

    log_info(tt::LogTest, "Daemon listening on pipe: {}", daemon_pipe_path);

    while (daemon_running) {
        std::ifstream pipe(daemon_pipe_path);
        if (!pipe.is_open()) {
            log_warning(tt::LogTest, "Failed to open pipe, retrying...");
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            continue;
        }

        std::string command;
        if (std::getline(pipe, command)) {
            if (command == "SHUTDOWN") {
                log_info(tt::LogTest, "Received shutdown command");
                break;
            } else if (command.starts_with("TEST:")) {
                std::string test_params_str = command.substr(5);
                size_t separator_pos = test_params_str.find(':');
                if (separator_pos == std::string::npos) {
                    log_error(tt::LogTest, "Invalid test command format");
                    continue;
                }

                std::string test_mode = test_params_str.substr(0, separator_pos);
                std::string params_str = test_params_str.substr(separator_pos + 1);

                // Lambda to write result to pipe
                auto write_result_to_pipe = [](int result) {
                    std::string result_pipe_path = daemon_pipe_path + "_result";
                    std::ofstream result_pipe(result_pipe_path);
                    if (result_pipe.is_open()) {
                        result_pipe << result << std::endl;
                        result_pipe.close();
                    }
                };

                try {
                    TestParams test_params = parse_pipe_separated_params(params_str, test_mode);

                    auto rc = baseline_validate_test_environment(test_params.params);
                    int result;
                    if (rc != 0) {
                        log_warning(tt::LogTest, "Test environment validation failed");
                        result = 1;  // Return 1 for environment validation failure
                    } else {
                        result = run_single_test(test_params, test_mode);
                    }

                    write_result_to_pipe(result);
                    log_info(tt::LogTest, "Test completed with result: {}", result);

                } catch (const std::exception& e) {
                    log_error(tt::LogTest, "Error running test: {}", e.what());
                    write_result_to_pipe(1);  // Write error result
                }
            }
        }

        pipe.close();
        // Small delay to prevent busy waiting
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    // Cleanup
    unlink(daemon_pipe_path.c_str());
    unlink((daemon_pipe_path + "_result").c_str());
    log_info(tt::LogTest, "Daemon shutdown complete");
}

int main(int argc, char** argv) {
    if (argc > 1 && std::string(argv[1]) == "daemon_mode") {
        run_daemon_mode();
        return 0;
    }

    // Original single-run mode
    std::size_t arg_idx = 1;
    std::string test_mode = argv[arg_idx++];

    // Parse command line arguments directly into TestParams
    TestParams test_params = parse_command_line_args(argv, arg_idx, test_mode, argc);

    auto rc = baseline_validate_test_environment(test_params.params);
    if (rc != 0) {
        return rc;
    }

    TT_FATAL(test_params.packet_payload_size_bytes > 0, "packet_payload_size_bytes must be greater than 0");
    TT_FATAL(test_params.params.num_links > 0, "num_links must be greater than 0");
    TT_FATAL(test_params.params.num_op_invocations > 0, "num_op_invocations must be greater than 0");
    TT_FATAL(test_params.params.line_size > 0, "line_size must be greater than 0");

    return run_single_test(test_params, test_mode);
}
