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

int main(int argc, char** argv) {
    std::size_t arg_idx = 1;
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
    TT_FATAL(arg_idx == argc, "Missing args");

    uint32_t min_test_num_devices = 8;
    if (tt::tt_metal::GetNumAvailableDevices() < min_test_num_devices) {
        tt::log_warning("This test can only be run on T3000 or TG devices");
        return 1;
    }

    uint32_t tg_num_devices = 32;
    if (num_links > 2 && tt::tt_metal::GetNumAvailableDevices() < tg_num_devices) {
        tt::log_warning("This test with {} links can only be run on TG devices", num_links);
        return 1;
    }
    TT_FATAL(num_messages > 0, "num_messages must be greater than 0");
    TT_FATAL(packet_payload_size_bytes > 0, "packet_payload_size_bytes must be greater than 0");
    TT_FATAL(num_links > 0, "num_links must be greater than 0");
    TT_FATAL(num_op_invocations > 0, "num_op_invocations must be greater than 0");
    TT_FATAL(line_size > 0, "line_size must be greater than 0");

    WriteThroughputStabilityTestWithPersistentFabricParams params;
    params.line_sync = line_sync;
    params.line_size = line_size;
    params.fabric_mode = static_cast<FabricTestMode>(fabric_mode);
    params.disable_sends_for_interior_workers = disable_sends_for_interior_workers;
    params.disable_end_workers_in_backward_direction = unidirectional_test;
    params.senders_are_unidirectional = senders_are_unidirectional;

    auto chip_send_type = fabric_unicast ? tt::tt_fabric::CHIP_UNICAST : tt::tt_fabric::CHIP_MULTICAST;

    bool flush = true;
    tt::tt_fabric::NocSendType noc_send_type;
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

    std::vector<Fabric1DPacketSendTestSpec> test_specs{
        {.chip_send_type = chip_send_type,
         .noc_send_type = noc_send_type,
         .num_messages = num_messages,
         .packet_payload_size_bytes = packet_payload_size_bytes,
         .flush = flush}};

    Run1DFabricPacketSendTest(num_links, num_op_invocations, test_specs, params);
}
