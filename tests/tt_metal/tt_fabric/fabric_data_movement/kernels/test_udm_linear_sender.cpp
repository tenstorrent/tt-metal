// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/fabric/hw/inc/tt_fabric_status.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"
#include "tt_metal/fabric/hw/inc/udm/tt_fabric_udm.hpp"
#include "tests/tt_metal/tt_metal/perf_microbenchmark/routing/kernels/tt_fabric_traffic_gen.hpp"
#include "dataflow_api.h"

constexpr uint32_t test_results_addr_arg = get_compile_time_arg_val(0);
constexpr uint32_t test_results_size_bytes = get_compile_time_arg_val(1);
tt_l1_ptr uint32_t* const test_results = reinterpret_cast<tt_l1_ptr uint32_t*>(test_results_addr_arg);
constexpr uint32_t notification_mailbox_address = get_compile_time_arg_val(2);
uint32_t target_address = get_compile_time_arg_val(3);
constexpr NocSendType noc_send_type = static_cast<NocSendType>(get_compile_time_arg_val(4));
constexpr uint32_t source_l1_buffer_address = get_compile_time_arg_val(5);
constexpr uint16_t packet_payload_size_bytes = static_cast<uint16_t>(get_compile_time_arg_val(6));
constexpr uint32_t num_packets = get_compile_time_arg_val(7);
constexpr uint32_t time_seed_init = get_compile_time_arg_val(8);
constexpr uint32_t noc_x_start = get_compile_time_arg_val(9);
constexpr uint32_t noc_y_start = get_compile_time_arg_val(10);
constexpr uint32_t dst_dev_id = get_compile_time_arg_val(11);
constexpr uint32_t dst_mesh_id = get_compile_time_arg_val(12);

void kernel_main() {
    // Runtime args are set up by append_fabric_connection_rt_args and used internally by fabric_fast_write
    // We don't need to read them explicitly here
    uint32_t time_seed = time_seed_init;

    zero_l1_buf(test_results, test_results_size_bytes);
    test_results[TT_FABRIC_STATUS_INDEX] = TT_FABRIC_STATUS_STARTED;

    uint64_t start_timestamp = get_timestamp();

    for (uint32_t i = 0; i < num_packets; i++) {
        time_seed = prng_next(time_seed);
        tt_l1_ptr uint32_t* start_addr = reinterpret_cast<tt_l1_ptr uint32_t*>(source_l1_buffer_address);
        fill_packet_data(start_addr, packet_payload_size_bytes / 16, time_seed);

        switch (noc_send_type) {
            case NOC_UNICAST_WRITE: {
                tt::tt_fabric::udm::fabric_fast_write(
                    dst_dev_id,
                    dst_mesh_id,
                    source_l1_buffer_address,
                    get_noc_addr(noc_x_start, noc_y_start, target_address),
                    packet_payload_size_bytes);
            } break;
            default: {
                ASSERT(false);
            } break;
        }
        noc_async_writes_flushed();
        target_address += packet_payload_size_bytes;
    }

    uint64_t cycles_elapsed = get_timestamp() - start_timestamp;

    noc_async_write_barrier();

    uint64_t bytes_sent = packet_payload_size_bytes * num_packets;

    test_results[TT_FABRIC_STATUS_INDEX] = TT_FABRIC_STATUS_PASS;
    test_results[TT_FABRIC_CYCLES_INDEX] = (uint32_t)cycles_elapsed;
    test_results[TT_FABRIC_CYCLES_INDEX + 1] = cycles_elapsed >> 32;
    test_results[TT_FABRIC_WORD_CNT_INDEX] = (uint32_t)bytes_sent;
    test_results[TT_FABRIC_WORD_CNT_INDEX + 1] = bytes_sent >> 32;
}
