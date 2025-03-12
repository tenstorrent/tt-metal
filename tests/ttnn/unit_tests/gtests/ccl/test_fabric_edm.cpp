// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tests/ttnn/unit_tests/gtests/ccl/test_fabric_edm_common.hpp"

int main(int argc, char** argv) {
    std::size_t arg_idx = 1;
    std::size_t num_mcasts = std::stoi(argv[arg_idx++]);
    std::size_t num_unicasts = std::stoi(argv[arg_idx++]);
    std::size_t num_links = std::stoi(argv[arg_idx++]);
    std::size_t num_op_invocations = std::stoi(argv[arg_idx++]);
    bool line_sync = std::stoi(argv[arg_idx++]);
    std::size_t line_size = std::stoi(argv[arg_idx++]);
    std::size_t packet_payload_size_bytes = std::stoi(argv[arg_idx++]);

    uint32_t min_test_num_devices = 8;
    if (tt::tt_metal::GetNumAvailableDevices() < min_test_num_devices) {
        tt::log_warning("This test can only be run on T3000 or TG devices");
        return 1;
    }

    WriteThroughputStabilityTestWithPersistentFabricParams params;
    params.line_sync = line_sync;
    params.line_size = line_size;
    RunWriteThroughputStabilityTestWithPersistentFabric(
        num_mcasts, num_unicasts, num_links, num_op_invocations, params, packet_payload_size_bytes);
}
