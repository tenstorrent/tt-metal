// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tests/ttnn/unit_tests/gtests/ccl/test_fabric_edm_common.hpp"
#include <cstdint>
#include <cstddef>

int main(int argc, char** argv) {
    std::size_t arg_idx = 1;
    // the line length, not total length of fabric after including loopback
    std::size_t line_size = std::stoi(argv[arg_idx++]);
    std::size_t latency_measurement_worker_line_index = std::stoi(argv[arg_idx++]);
    std::size_t latency_ping_message_size_bytes = std::stoi(argv[arg_idx++]);
    std::size_t latency_ping_burst_size = std::stoi(argv[arg_idx++]);
    std::size_t latency_ping_burst_count = std::stoi(argv[arg_idx++]);
    TT_FATAL(
        latency_ping_burst_size == 1,
        "Latency ping burst size must be 1. Support for accurately measuring latency with burst size > 1 is not "
        "implemented");

    bool add_upstream_fabric_congestion_writers = std::stoi(argv[arg_idx++]) != 0;
    std::size_t num_downstream_fabric_congestion_writers = std::stoi(argv[arg_idx++]);
    std::size_t congestion_writers_message_size = std::stoi(argv[arg_idx++]);
    bool congestion_writers_use_mcast = std::stoi(argv[arg_idx++]) != 0;
    TT_FATAL(arg_idx == argc, "Read past end of args or didn't read all args");

    uint32_t test_expected_num_devices = 8;
    if (tt::tt_metal::GetNumAvailableDevices() < test_expected_num_devices) {
        tt::log_warning("This test can only be run on T3000 devices");
        return 1;
    }

    auto compute_loopback_distance_to_start_of_line = [line_size](std::size_t line_index) {
        return ((line_size - 1) * 2) - line_index;
    };

    LatencyTestWriterSpecs writer_specs(line_size - 1, std::nullopt);
    writer_specs.at(latency_measurement_worker_line_index) = WriterSpec{
        .spec =
            LatencyPacketTestWriterSpec{
                .num_bursts = latency_ping_burst_count,
                .burst_size_num_messages = latency_ping_burst_size,
            },
        .worker_core_logical = CoreCoord(0, 0),
        .message_size_bytes = latency_ping_message_size_bytes};

    if (add_upstream_fabric_congestion_writers) {
        TT_FATAL(
            latency_measurement_worker_line_index != 0,
            "Tried adding upstream congestion writer but the latency measurement packet router was added to line index "
            "0. If there is an upstream congestion writer, then the latency test writer cannot be at line index 0.");
        TT_FATAL(congestion_writers_message_size != 0, "upstream congestion writer message size must be non-zero");
        size_t upstream_worker_line_index = latency_measurement_worker_line_index - 1;
        writer_specs.at(upstream_worker_line_index) = WriterSpec{
            .spec =
                DatapathBusyDataWriterSpec{
                    .message_size_bytes = congestion_writers_message_size,
                    .mcast = congestion_writers_use_mcast,
                    .write_distance = compute_loopback_distance_to_start_of_line(upstream_worker_line_index),
                },
            .worker_core_logical = CoreCoord(0, 0),
            .message_size_bytes = congestion_writers_message_size};
    }

    TT_FATAL(
        num_downstream_fabric_congestion_writers + latency_measurement_worker_line_index < line_size,
        "Tried adding {} downstream congestion writers but there is not enough space left in the line."
        "the latency packet writer is at index {} and line_size is {}. Therefore, the largest number of downstream "
        "writers for this configuration is {}.",
        num_downstream_fabric_congestion_writers,
        latency_measurement_worker_line_index,
        line_size,
        line_size - 1 - latency_measurement_worker_line_index);
    for (size_t i = 0; i < num_downstream_fabric_congestion_writers; i++) {
        TT_FATAL(congestion_writers_message_size != 0, "downstream congestion writer message size must be non-zero");
        size_t downstream_worker_line_index = latency_measurement_worker_line_index + 1 + i;
        size_t distance = downstream_worker_line_index - latency_measurement_worker_line_index;
        writer_specs.at(downstream_worker_line_index) = WriterSpec{
            .spec =
                DatapathBusyDataWriterSpec{
                    .message_size_bytes = congestion_writers_message_size,
                    .mcast = congestion_writers_use_mcast,
                    .write_distance = distance,
                },
            .worker_core_logical = CoreCoord(0, 0),
            .message_size_bytes = congestion_writers_message_size};
    }

    RunPersistent1dFabricLatencyTest(writer_specs, line_size);
}
