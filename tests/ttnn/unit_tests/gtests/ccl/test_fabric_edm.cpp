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
#include <tt-metalium/logger.hpp>
#include "tests/ttnn/unit_tests/gtests/ccl/test_fabric_edm_common.hpp"

// Global state for daemon mode
static bool daemon_mode = false;
static bool daemon_running = true;
static std::string daemon_pipe_path = "/tmp/tt_metal_fabric_edm_daemon";
static FILE* debug_log = nullptr;

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
        tt::log_warning("This test can only be run on T3000 or TG devices");
        return 1;
    }

    uint32_t galaxy_num_devices = 32;
    if (params.num_links > 2 && tt::tt_metal::GetNumAvailableDevices() < galaxy_num_devices) {
        tt::log_warning("This test with {} links can only be run on Galaxy systems", params.num_links);
        return 1;
    }

    if (tt::tt_metal::GetNumAvailableDevices() == min_test_num_devices && params.num_links > 1 && params.line_size > 4) {
        tt::log_warning("T3000 cannot run multi-link with more than 4 devices");
        return 1;
    }

    return 0;
}

static void dispatch_1d_fabric_on_mesh(
    const std::vector<Fabric1DPacketSendTestSpec>& test_specs,
    TestParams& test_params,
    int argc,
    char** argv,
    size_t arg_idx) {
    size_t num_rows = std::stoi(argv[arg_idx++]);
    size_t num_cols = std::stoi(argv[arg_idx++]);
    size_t first_link_offset = std::stoi(argv[arg_idx++]);
    TT_FATAL(arg_idx == argc, "Missing args");

    test_params.params.num_fabric_rows = num_rows;
    test_params.params.num_fabric_cols = num_cols;
    test_params.params.first_link_offset = first_link_offset;
    TT_FATAL(first_link_offset == 0, "first_link_offset must be 0. Higher offset not tested yet");

    TT_FATAL(
        num_rows > 0 ^ num_cols > 0,
        "Either num_rows or num_cols (but not both) must be greater than 0 when running 1D fabric on mesh BW test");

    if (test_params.params.fabric_mode == FabricTestMode::Linear) {
        Run1DFabricPacketSendTest<Fabric1DLineDeviceInitFixture>(test_specs, test_params.params);
    } else if (test_params.params.fabric_mode == FabricTestMode::FullRing) {
        Run1DFabricPacketSendTest<Fabric1DRingDeviceInitFixture>(test_specs, test_params.params);
    } else {
        TT_THROW(
            "Invalid fabric mode when using device init fabric in 1D fabric on mesh BW test: {}",
            test_params.params.fabric_mode);
    }
}

static void dispatch_single_line_bw_test(
    const std::vector<Fabric1DPacketSendTestSpec>& test_specs,
    TestParams& test_params,
    int argc,
    char** argv,
    size_t arg_idx) {
    TT_FATAL(arg_idx == argc, "Missing args");
    Run1DFabricPacketSendTest(test_specs, test_params.params);
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
        tt::log_error("Test failed: {}", e.what());
        return 1;
    }
}

static TestParams parse_test_params_from_string(const std::string& params_str) {
    std::istringstream iss(params_str);
    std::vector<std::string> tokens;
    std::string token;

    while (std::getline(iss, token, '|')) {
        tokens.push_back(token);
    }

    if (tokens.size() < 12) {
        TT_THROW("Invalid parameter string format");
    }

    TestParams test_params;
    size_t idx = 0;

    test_params.fabric_unicast = std::stoi(tokens[idx++]);
    test_params.message_noc_type = tokens[idx++];
    test_params.num_messages = std::stoi(tokens[idx++]);
    test_params.params.num_links = std::stoi(tokens[idx++]);
    test_params.params.num_op_invocations = std::stoi(tokens[idx++]);
    test_params.params.line_sync = std::stoi(tokens[idx++]);
    test_params.params.line_size = std::stoi(tokens[idx++]);
    test_params.packet_payload_size_bytes = std::stoi(tokens[idx++]);
    test_params.params.fabric_mode = static_cast<FabricTestMode>(std::stoi(tokens[idx++]));
    test_params.params.disable_sends_for_interior_workers = std::stoi(tokens[idx++]);
    test_params.params.disable_end_workers_in_backward_direction = std::stoi(tokens[idx++]);
    test_params.params.senders_are_unidirectional = std::stoi(tokens[idx++]);

    // Optional mesh parameters
    if (tokens.size() > 12) {
        test_params.params.num_fabric_rows = std::stoi(tokens[idx++]);
        test_params.params.num_fabric_cols = std::stoi(tokens[idx++]);
        test_params.params.first_link_offset = std::stoi(tokens[idx++]);
    }

    return test_params;
}

static void run_daemon_mode() {
    tt::log_info("Starting fabric EDM daemon mode...");

    // Setup signal handlers
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);

    // Create named pipe
    unlink(daemon_pipe_path.c_str());
    if (mkfifo(daemon_pipe_path.c_str(), 0666) == -1) {
        TT_THROW("Failed to create named pipe: {}", daemon_pipe_path);
    }

    tt::log_info("Daemon listening on pipe: {}", daemon_pipe_path);

    while (daemon_running) {
        std::ifstream pipe(daemon_pipe_path);
        if (!pipe.is_open()) {
            tt::log_warning("Failed to open pipe, retrying...");
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            continue;
        }

        std::string command;
        if (std::getline(pipe, command)) {
            if (command == "SHUTDOWN") {
                tt::log_info("Received shutdown command");
                break;
            } else if (command.starts_with("TEST:")) {
                std::string test_params_str = command.substr(5);
                size_t separator_pos = test_params_str.find(':');
                if (separator_pos == std::string::npos) {
                    tt::log_error("Invalid test command format");
                    continue;
                }

                std::string test_mode = test_params_str.substr(0, separator_pos);
                std::string params_str = test_params_str.substr(separator_pos + 1);

                try {
                    TestParams test_params = parse_test_params_from_string(params_str);

                    auto rc = baseline_validate_test_environment(test_params.params);
                    int result;
                    if (rc != 0) {
                        tt::log_warning("Test environment validation failed");
                        result = 1;  // Return 1 for environment validation failure
                    } else {
                        result = run_single_test(test_params, test_mode);
                    }

                    // Write result back to a result pipe
                    std::string result_pipe_path = daemon_pipe_path + "_result";
                    std::ofstream result_pipe(result_pipe_path);
                    if (result_pipe.is_open()) {
                        result_pipe << result << std::endl;
                        result_pipe.close();
                    }

                    tt::log_info("Test completed with result: {}", result);

                } catch (const std::exception& e) {
                    tt::log_error("Error running test: {}", e.what());

                    // Write error result
                    std::string result_pipe_path = daemon_pipe_path + "_result";
                    std::ofstream result_pipe(result_pipe_path);
                    if (result_pipe.is_open()) {
                        result_pipe << 1 << std::endl;
                        result_pipe.close();
                    }
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
    if (debug_log) {
        fclose(debug_log);
        debug_log = nullptr;
    }
    tt::log_info("Daemon shutdown complete");
}

int main(int argc, char** argv) {
    if (argc > 1 && std::string(argv[1]) == "daemon_mode") {
        daemon_mode = true;
        run_daemon_mode();
        return 0;
    }

    // Original single-run mode
    std::size_t arg_idx = 1;
    std::string test_mode = argv[arg_idx++];

    bool fabric_unicast = std::stoi(argv[arg_idx++]);
    const std::string& message_noc_type = std::string(argv[arg_idx++]);
    std::size_t num_messages = std::stoi(argv[arg_idx++]);
    std::size_t num_links = std::stoi(argv[arg_idx++]);
    std::size_t num_op_invocations = std::stoi(argv[arg_idx++]);
    bool line_sync = std::stoi(argv[arg_idx++]);
    std::size_t line_size = std::stoi(argv[arg_idx++]);
    std::size_t packet_payload_size_bytes = std::stoi(argv[arg_idx++]);
    uint32_t fabric_mode = std::stoi(argv[arg_idx++]);
    bool disable_sends_for_interior_workers = std::stoi(argv[arg_idx++]);
    bool unidirectional_test = std::stoi(argv[arg_idx++]);
    bool senders_are_unidirectional = std::stoi(argv[arg_idx++]);

    // WriteThroughputStabilityTestWithPersistentFabricParams params;
    TestParams test_params;
    test_params.fabric_unicast = fabric_unicast;
    test_params.params.line_sync = line_sync;
    test_params.params.line_size = line_size;
    test_params.params.num_links = num_links;
    test_params.num_messages = num_messages;
    test_params.params.num_op_invocations = num_op_invocations;
    test_params.packet_payload_size_bytes = packet_payload_size_bytes;
    test_params.params.fabric_mode = static_cast<FabricTestMode>(fabric_mode);
    test_params.params.disable_sends_for_interior_workers = disable_sends_for_interior_workers;
    test_params.params.disable_end_workers_in_backward_direction = unidirectional_test;
    test_params.params.senders_are_unidirectional = senders_are_unidirectional;
    test_params.message_noc_type = message_noc_type;

    auto rc = baseline_validate_test_environment(test_params.params);
    if (rc != 0) {
        return rc;
    }

    TT_FATAL(test_params.packet_payload_size_bytes > 0, "packet_payload_size_bytes must be greater than 0");
    TT_FATAL(test_params.params.num_links > 0, "num_links must be greater than 0");
    TT_FATAL(test_params.params.num_op_invocations > 0, "num_op_invocations must be greater than 0");
    TT_FATAL(test_params.params.line_size > 0, "line_size must be greater than 0");

    // Handle mesh parameters if present
    if (test_mode == "1D_fabric_on_mesh" && arg_idx < argc) {
        test_params.params.num_fabric_rows = std::stoi(argv[arg_idx++]);
        test_params.params.num_fabric_cols = std::stoi(argv[arg_idx++]);
        test_params.params.first_link_offset = std::stoi(argv[arg_idx++]);
    }

    return run_single_test(test_params, test_mode);
}
