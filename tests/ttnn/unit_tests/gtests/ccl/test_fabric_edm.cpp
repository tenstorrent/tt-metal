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


int main(int argc, char** argv) {
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

    auto chip_send_type = test_params.fabric_unicast ? tt::tt_fabric::CHIP_UNICAST : tt::tt_fabric::CHIP_MULTICAST;

    auto [noc_send_type, flush] = get_noc_send_type(test_params.message_noc_type);

    std::vector<Fabric1DPacketSendTestSpec> test_specs{
        {.chip_send_type = chip_send_type,
         .noc_send_type = noc_send_type,
         .num_messages = test_params.num_messages,
         .packet_payload_size_bytes = test_params.packet_payload_size_bytes,
         .flush = flush}};

    if (test_mode == "1_fabric_instance") {
        dispatch_single_line_bw_test(test_specs, test_params, argc, argv, arg_idx);
    } else if (test_mode == "1D_fabric_on_mesh") {
        dispatch_1d_fabric_on_mesh(test_specs, test_params, argc, argv, arg_idx);
    } else {
        TT_THROW("Invalid test mode: {}", test_mode.c_str());
    }

    return 0;
}
