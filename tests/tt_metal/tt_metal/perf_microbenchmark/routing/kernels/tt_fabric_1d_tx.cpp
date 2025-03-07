// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

// clang-format off
#include "dataflow_api.h"
#include "debug/dprint.h"
#include "tests/tt_metal/tt_metal/perf_microbenchmark/common/kernel_utils.hpp"
#include "tt_metal/fabric/hw/inc/tt_fabric_status.h"
#include "tt_metal/fabric/hw/inc/tt_fabric.h"
#include "tests/tt_metal/tt_metal/perf_microbenchmark/routing/kernels/tt_fabric_traffic_gen.hpp"
#include "tt_metal/api/tt-metalium/fabric_edm_packet_header.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_connection_manager.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_edm_packet_transmission.hpp"

// clang-format on

constexpr uint32_t test_results_addr_arg = get_compile_time_arg_val(0);
constexpr uint32_t test_results_size_bytes = get_compile_time_arg_val(1);
tt_l1_ptr uint32_t* const test_results = reinterpret_cast<tt_l1_ptr uint32_t*>(test_results_addr_arg);

constexpr uint32_t target_address = get_compile_time_arg_val(2);

void kernel_main() {
    using namespace tt::tt_fabric;

    size_t rt_args_idx = 0;
    uint32_t packet_header_buffer_address = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t source_l1_buffer_address = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t packet_payload_size_bytes = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t num_packets = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t unicast_hops = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t rx_noc_encoding = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t time_seed = get_arg_val<uint32_t>(rt_args_idx++);

    uint64_t noc_dest_addr = get_noc_addr_helper(rx_noc_encoding, target_address);

    auto fabric_connection =
        tt::tt_fabric::WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(rt_args_idx);

    zero_l1_buf(test_results, test_results_size_bytes);
    test_results[TT_FABRIC_STATUS_INDEX] = TT_FABRIC_STATUS_STARTED;

    // connect to edm
    fabric_connection.open();

    // construct packet header
    volatile auto* unicast_packet_header =
        reinterpret_cast<volatile tt_l1_ptr LowLatencyPacketHeader*>(packet_header_buffer_address);
    unicast_packet_header->to_chip_unicast(static_cast<uint8_t>(unicast_hops));
    unicast_packet_header->to_noc_unicast_write(NocUnicastCommandHeader{noc_dest_addr}, packet_payload_size_bytes);

    uint64_t start_timestamp = get_timestamp();

    // loop over for num packets
    for (uint32_t i = 0; i < num_packets; i++) {
#ifndef BENCHMARK_MODE
        unicast_packet_header->to_noc_unicast_write(NocUnicastCommandHeader{noc_dest_addr}, packet_payload_size_bytes);
        // fill packet data for sanity testing
        tt_l1_ptr uint32_t* start_addr = reinterpret_cast<tt_l1_ptr uint32_t*>(source_l1_buffer_address);
        time_seed = prng_next(time_seed);
        fill_packet_data(start_addr, packet_payload_size_bytes / 16, time_seed);
        tt_l1_ptr uint32_t* last_word_addr =
            reinterpret_cast<tt_l1_ptr uint32_t*>(source_l1_buffer_address + packet_payload_size_bytes - 4);
        noc_dest_addr += packet_payload_size_bytes;
#endif
        fabric_connection.wait_for_empty_write_slot();
        fabric_connection.send_payload_without_header_non_blocking_from_address(
            source_l1_buffer_address, packet_payload_size_bytes);
        fabric_connection.send_payload_blocking_from_address(
            (uint32_t)unicast_packet_header, sizeof(tt::tt_fabric::PacketHeader));
    }

    uint64_t cycles_elapsed = get_timestamp() - start_timestamp;

    fabric_connection.close();
    noc_async_write_barrier();

    uint64_t bytes_sent = packet_payload_size_bytes * num_packets;

    // write out results
    test_results[TT_FABRIC_STATUS_INDEX] = TT_FABRIC_STATUS_PASS;
    test_results[TT_FABRIC_CYCLES_INDEX] = (uint32_t)cycles_elapsed;
    test_results[TT_FABRIC_CYCLES_INDEX + 1] = cycles_elapsed >> 32;
    test_results[TT_FABRIC_WORD_CNT_INDEX] = (uint32_t)bytes_sent;
    test_results[TT_FABRIC_WORD_CNT_INDEX + 1] = bytes_sent >> 32;
}
