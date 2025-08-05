// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"
#include "tt_metal/fabric/hw/inc/linear/api.h"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_status.h"
#include "tt_metal/fabric/hw/inc/tt_fabric.h"
#include "tests/tt_metal/tt_metal/perf_microbenchmark/routing/kernels/tt_fabric_traffic_gen.hpp"

using namespace tt::tt_fabric::linear::experimental;
constexpr uint32_t test_results_addr_arg = get_compile_time_arg_val(0);
constexpr uint32_t test_results_size_bytes = get_compile_time_arg_val(1);
tt_l1_ptr uint32_t* const test_results = reinterpret_cast<tt_l1_ptr uint32_t*>(test_results_addr_arg);
constexpr uint32_t notification_mailbox_address = get_compile_time_arg_val(2);
uint32_t target_address = get_compile_time_arg_val(3);
constexpr NocSendType noc_send_type = static_cast<NocSendType>(get_compile_time_arg_val(4));
constexpr bool is_chip_multicast = get_compile_time_arg_val(5) == 1;

void kernel_main() {
    size_t rt_arg_idx = 0;
    uint32_t source_l1_buffer_address = get_arg_val<uint32_t>(rt_arg_idx++);
    uint32_t packet_payload_size_bytes = get_arg_val<uint32_t>(rt_arg_idx++);
    uint32_t num_packets = get_arg_val<uint32_t>(rt_arg_idx++);
    uint32_t time_seed = get_arg_val<uint32_t>(rt_arg_idx++);
    uint32_t noc_x_start = get_arg_val<uint32_t>(rt_arg_idx++);
    uint32_t noc_y_start = get_arg_val<uint32_t>(rt_arg_idx++);
    // mcast only
    uint32_t start_distance = get_arg_val<uint32_t>(rt_arg_idx++);
    uint32_t range = get_arg_val<uint32_t>(rt_arg_idx++);

    auto connection = tt::tt_fabric::WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(rt_arg_idx);
    auto packet_header = PacketHeaderPool::allocate_header();
    zero_l1_buf((uint32_t*)packet_header, sizeof(PACKET_HEADER_TYPE));

    connection.open();
    zero_l1_buf(test_results, test_results_size_bytes);
    test_results[TT_FABRIC_STATUS_INDEX] = TT_FABRIC_STATUS_STARTED;

    uint64_t start_timestamp = get_timestamp();

    for (uint32_t i = 0; i < num_packets; i++) {
        if constexpr (is_chip_multicast) {
            switch (noc_send_type) {
                case NOC_UNICAST_ATOMIC_INC: {
                    fabric_multicast_noc_unicast_atomic_inc(
                        &connection,
                        packet_header,
                        tt::tt_fabric::NocUnicastAtomicIncCommandHeader{
                            get_noc_addr(noc_x_start, noc_y_start, notification_mailbox_address),
                            1,
                            std::numeric_limits<uint16_t>::max(),
                            true},
                        start_distance,
                        range);
                } break;
                case NOC_FUSED_UNICAST_ATOMIC_INC: {
                    fabric_multicast_noc_fused_unicast_with_atomic_inc(
                        &connection,
                        packet_header,
                        source_l1_buffer_address,
                        packet_payload_size_bytes,
                        tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader{
                            get_noc_addr(noc_x_start, noc_y_start, target_address),
                            get_noc_addr(noc_x_start, noc_y_start, notification_mailbox_address),
                            1,
                            std::numeric_limits<uint16_t>::max(),
                            true},
                        start_distance,
                        range);
                } break;
                default: {
                    ASSERT(false);
                } break;
            }
        } else {
            switch (noc_send_type) {
                case NOC_UNICAST_ATOMIC_INC: {
                    fabric_unicast_noc_unicast_atomic_inc(
                        &connection,
                        packet_header,
                        tt::tt_fabric::NocUnicastAtomicIncCommandHeader{
                            get_noc_addr(noc_x_start, noc_y_start, notification_mailbox_address),
                            1,
                            std::numeric_limits<uint16_t>::max(),
                            true},
                        1);
                } break;
                case NOC_FUSED_UNICAST_ATOMIC_INC: {
                    fabric_unicast_noc_fused_unicast_with_atomic_inc(
                        &connection,
                        packet_header,
                        source_l1_buffer_address,
                        packet_payload_size_bytes,
                        tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader{
                            get_noc_addr(noc_x_start, noc_y_start, target_address),
                            get_noc_addr(noc_x_start, noc_y_start, notification_mailbox_address),
                            1,
                            std::numeric_limits<uint16_t>::max(),
                            true},
                        1);
                } break;
                default: {
                    ASSERT(false);
                } break;
            }
        }
        noc_async_writes_flushed();
    }

    uint64_t cycles_elapsed = get_timestamp() - start_timestamp;
    connection.close();

    noc_async_write_barrier();

    uint64_t total_operations = num_packets;

    test_results[TT_FABRIC_STATUS_INDEX] = TT_FABRIC_STATUS_PASS;
    test_results[TT_FABRIC_CYCLES_INDEX] = (uint32_t)cycles_elapsed;
    test_results[TT_FABRIC_CYCLES_INDEX + 1] = cycles_elapsed >> 32;
    test_results[TT_FABRIC_WORD_CNT_INDEX] = (uint32_t)total_operations;
    test_results[TT_FABRIC_WORD_CNT_INDEX + 1] = total_operations >> 32;
}
