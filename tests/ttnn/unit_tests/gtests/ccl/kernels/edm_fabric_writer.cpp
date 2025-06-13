// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/api/tt-metalium/fabric_edm_packet_header.hpp"
#include "dataflow_api.h"
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_connection_manager.hpp"
#include "tt_metal/fabric/hw/inc/noc_addr.h"

#include <cstdint>
#include <cstddef>

static constexpr bool enable_start_synchronization = get_compile_time_arg_val(0) != 0;
static constexpr bool enable_finish_synchronization = get_compile_time_arg_val(1) != 0;
static constexpr bool enable_any_synchronization = enable_start_synchronization || enable_finish_synchronization;

void line_sync(
    FabricConnectionManager& fabric_connection,
    bool sync_forward,
    bool sync_backward,
    volatile PACKET_HEADER_TYPE* fwd_packet_header,
    volatile PACKET_HEADER_TYPE* bwd_packet_header,
    size_t sync_bank_addr,
    size_t sync_noc_x,
    size_t sync_noc_y,
    size_t sync_val) {
    using namespace tt::tt_fabric;

    auto dest_noc_addr =
        safe_get_noc_addr(static_cast<uint8_t>(sync_noc_x), static_cast<uint8_t>(sync_noc_y), sync_bank_addr, 0);
    if (sync_forward) {
        fwd_packet_header->to_noc_unicast_atomic_inc(NocUnicastAtomicIncCommandHeader{dest_noc_addr, 1, 128});
        fabric_connection.get_forward_connection().wait_for_empty_write_slot();
        fabric_connection.get_forward_connection().send_payload_flush_non_blocking_from_address(
            (uint32_t)fwd_packet_header, sizeof(PACKET_HEADER_TYPE));
    }

    if (sync_backward) {
        bwd_packet_header->to_noc_unicast_atomic_inc(NocUnicastAtomicIncCommandHeader{dest_noc_addr, 1, 128});
        fabric_connection.get_backward_connection().wait_for_empty_write_slot();
        fabric_connection.get_backward_connection().send_payload_flush_non_blocking_from_address(
            (uint32_t)bwd_packet_header, sizeof(PACKET_HEADER_TYPE));
    }
    noc_semaphore_inc(get_noc_addr(sync_noc_x, sync_noc_y, sync_bank_addr), 1);
    if (sync_noc_x == my_x[0] && sync_noc_y == my_y[0]) {
        noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sync_bank_addr), sync_val);
    }
}

struct TestParams {
    size_t send_count = 0;
    size_t num_fwd_hops = 0;
    size_t num_bwd_hops = 0;
    size_t dest_noc_x_fwd = 0;
    size_t dest_noc_y_fwd = 0;
    size_t dest_bank_addr_fwd = 0;
    size_t dest_noc_x_bwd = 0;
    size_t dest_noc_y_bwd = 0;
    size_t dest_bank_addr_bwd = 0;
    size_t payload_size_bytes = 0;
    tt::tt_fabric::ChipSendType chip_send_type = tt::tt_fabric::CHIP_UNICAST;
    bool flush = true;
};

static FORCE_INLINE void setup_packet_header(
    volatile PACKET_HEADER_TYPE* pkt_hdr, size_t num_hops, tt::tt_fabric::ChipSendType chip_send_type) {
    if (num_hops > 0) {
        if (chip_send_type == tt::tt_fabric::CHIP_UNICAST) {
            pkt_hdr->to_chip_unicast(num_hops);
        } else {
            pkt_hdr->to_chip_multicast(tt::tt_fabric::MulticastRoutingCommandHeader{1, static_cast<uint8_t>(num_hops)});
        }
    }
}

template <tt::tt_fabric::NocSendType T>
static void send_packets(
    FabricConnectionManager& fabric_connection,
    volatile PACKET_HEADER_TYPE* pkt_hdr_fwd,
    volatile PACKET_HEADER_TYPE* pkt_hdr_bwd,
    const TestParams& params,
    size_t source_buffer_address) {
    ASSERT(false);

    // hang because we enterring here means the test is malformed (not implemented)
    while (1);
}

template <>
FORCE_INLINE void send_packets<tt::tt_fabric::NocSendType::NOC_UNICAST_WRITE>(
    FabricConnectionManager& fabric_connection,
    volatile PACKET_HEADER_TYPE* pkt_hdr_fwd,
    volatile PACKET_HEADER_TYPE* pkt_hdr_bwd,
    const TestParams& params,
    size_t source_buffer_address) {
    using namespace tt::tt_fabric;

    // Setup packet headers for both directions
    setup_packet_header(pkt_hdr_fwd, params.num_fwd_hops, params.chip_send_type);
    setup_packet_header(pkt_hdr_bwd, params.num_bwd_hops, params.chip_send_type);

    auto noc0_dest_addr_fwd = safe_get_noc_addr(
        static_cast<uint8_t>(params.dest_noc_x_fwd),
        static_cast<uint8_t>(params.dest_noc_y_fwd),
        params.dest_bank_addr_fwd,
        0);

    pkt_hdr_fwd->to_noc_unicast_write(NocUnicastCommandHeader{noc0_dest_addr_fwd}, params.payload_size_bytes);
    auto noc0_dest_addr_bwd = safe_get_noc_addr(
        static_cast<uint8_t>(params.dest_noc_x_bwd),
        static_cast<uint8_t>(params.dest_noc_y_bwd),
        params.dest_bank_addr_bwd,
        0);

    pkt_hdr_bwd->to_noc_unicast_write(NocUnicastCommandHeader{noc0_dest_addr_bwd}, params.payload_size_bytes);
    for (size_t i = 0; i < params.send_count; i++) {
        // Forward direction
        if (params.num_fwd_hops > 0) {
            fabric_connection.get_forward_connection().wait_for_empty_write_slot();
            // fabric_connection.get_forward_connection().send_payload_without_header_non_blocking_from_address(
            //     source_buffer_address, params.payload_size_bytes);
            fabric_connection.get_forward_connection().send_payload_flush_non_blocking_from_address(
                (uint32_t)pkt_hdr_fwd, sizeof(PACKET_HEADER_TYPE));
        }

        // Backward direction
        if (params.num_bwd_hops > 0) {
            fabric_connection.get_backward_connection().wait_for_empty_write_slot();
            // fabric_connection.get_backward_connection().send_payload_without_header_non_blocking_from_address(
            //     source_buffer_address, params.payload_size_bytes);
            fabric_connection.get_backward_connection().send_payload_flush_non_blocking_from_address(
                (uint32_t)pkt_hdr_bwd, sizeof(PACKET_HEADER_TYPE));
        }

        noc_async_writes_flushed();
    }
}

template <>
void send_packets<tt::tt_fabric::NocSendType::NOC_UNICAST_ATOMIC_INC>(
    FabricConnectionManager& fabric_connection,
    volatile PACKET_HEADER_TYPE* pkt_hdr_fwd,
    volatile PACKET_HEADER_TYPE* pkt_hdr_bwd,
    const TestParams& params,
    size_t source_buffer_address) {
    using namespace tt::tt_fabric;

    // Setup packet headers for both directions
    setup_packet_header(pkt_hdr_fwd, params.num_fwd_hops, params.chip_send_type);
    setup_packet_header(pkt_hdr_bwd, params.num_bwd_hops, params.chip_send_type);

    auto noc0_dest_addr_fwd = safe_get_noc_addr(
        static_cast<uint8_t>(params.dest_noc_x_fwd),
        static_cast<uint8_t>(params.dest_noc_y_fwd),
        params.dest_bank_addr_fwd,
        0);

    pkt_hdr_fwd->to_noc_unicast_atomic_inc(NocUnicastAtomicIncCommandHeader{noc0_dest_addr_fwd, 1, 128, params.flush});
    auto noc0_dest_addr_bwd = safe_get_noc_addr(
        static_cast<uint8_t>(params.dest_noc_x_bwd),
        static_cast<uint8_t>(params.dest_noc_y_bwd),
        params.dest_bank_addr_bwd,
        0);

    pkt_hdr_bwd->to_noc_unicast_atomic_inc(NocUnicastAtomicIncCommandHeader{noc0_dest_addr_bwd, 1, 128, params.flush});

    if (params.num_fwd_hops > 0 && params.num_bwd_hops > 0) {
        for (size_t i = 0; i < params.send_count; i++) {
            fabric_connection.get_forward_connection().wait_for_empty_write_slot();
            fabric_connection.get_forward_connection().send_payload_flush_non_blocking_from_address(
                (uint32_t)pkt_hdr_fwd, sizeof(PACKET_HEADER_TYPE));

            fabric_connection.get_backward_connection().wait_for_empty_write_slot();
            fabric_connection.get_backward_connection().send_payload_flush_non_blocking_from_address(
                (uint32_t)pkt_hdr_bwd, sizeof(PACKET_HEADER_TYPE));

            noc_async_writes_flushed();
        }

    } else if (params.num_fwd_hops > 0) {
        for (size_t i = 0; i < params.send_count; i++) {
            fabric_connection.get_forward_connection().wait_for_empty_write_slot();
            fabric_connection.get_forward_connection().send_payload_flush_non_blocking_from_address(
                (uint32_t)pkt_hdr_fwd, sizeof(PACKET_HEADER_TYPE));

            noc_async_writes_flushed();
        }

    } else if (params.num_bwd_hops > 0) {
        for (size_t i = 0; i < params.send_count; i++) {
            fabric_connection.get_backward_connection().wait_for_empty_write_slot();
            fabric_connection.get_backward_connection().send_payload_flush_non_blocking_from_address(
                (uint32_t)pkt_hdr_bwd, sizeof(PACKET_HEADER_TYPE));
            noc_async_writes_flushed();
        }

    } else {
        ASSERT(false);  // Invalid path with no hops in either direction. The test was misconfigured or args were passed
                        // incorrectly.

        // In case we are not running with watcher, we want to indicate to the runner that there is a problem
        // rather than spit out numbers that are not meaningful/are incorrect.
        while (1);
    }

    for (size_t i = 0; i < params.send_count; i++) {
        // Forward direction
        if (params.num_fwd_hops > 0) {
            fabric_connection.get_forward_connection().wait_for_empty_write_slot();
            fabric_connection.get_forward_connection().send_payload_flush_non_blocking_from_address(
                (uint32_t)pkt_hdr_fwd, sizeof(PACKET_HEADER_TYPE));
        }

        // Backward direction
        if (params.num_bwd_hops > 0) {
            fabric_connection.get_backward_connection().wait_for_empty_write_slot();
            fabric_connection.get_backward_connection().send_payload_flush_non_blocking_from_address(
                (uint32_t)pkt_hdr_bwd, sizeof(PACKET_HEADER_TYPE));
        }

        noc_async_writes_flushed();
    }
}

template <>
void send_packets<tt::tt_fabric::NocSendType::NOC_FUSED_UNICAST_ATOMIC_INC>(
    FabricConnectionManager& fabric_connection,
    volatile PACKET_HEADER_TYPE* pkt_hdr_fwd,
    volatile PACKET_HEADER_TYPE* pkt_hdr_bwd,
    const TestParams& params,
    size_t source_buffer_address) {
    using namespace tt::tt_fabric;

    // Setup packet headers for both directions
    setup_packet_header(pkt_hdr_fwd, params.num_fwd_hops, params.chip_send_type);
    setup_packet_header(pkt_hdr_bwd, params.num_bwd_hops, params.chip_send_type);

    auto noc0_dest_addr_fwd = safe_get_noc_addr(
        static_cast<uint8_t>(params.dest_noc_x_fwd),
        static_cast<uint8_t>(params.dest_noc_y_fwd),
        params.dest_bank_addr_fwd,
        0);

    auto sem_noc0_dest_addr_fwd = safe_get_noc_addr(
        static_cast<uint8_t>(params.dest_noc_x_fwd),
        static_cast<uint8_t>(params.dest_noc_y_fwd),
        params.dest_bank_addr_fwd + 4,
        0);
    pkt_hdr_fwd->to_noc_fused_unicast_write_atomic_inc(
        NocUnicastAtomicIncFusedCommandHeader{noc0_dest_addr_fwd, sem_noc0_dest_addr_fwd, 1, 128, params.flush},
        params.payload_size_bytes);
    auto noc0_dest_addr_bwd = safe_get_noc_addr(
        static_cast<uint8_t>(params.dest_noc_x_bwd),
        static_cast<uint8_t>(params.dest_noc_y_bwd),
        params.dest_bank_addr_bwd,
        0);

    auto sem_noc0_dest_addr_bwd = safe_get_noc_addr(
        static_cast<uint8_t>(params.dest_noc_x_bwd),
        static_cast<uint8_t>(params.dest_noc_y_bwd),
        params.dest_bank_addr_bwd + 4,
        0);

    pkt_hdr_bwd->to_noc_fused_unicast_write_atomic_inc(
        NocUnicastAtomicIncFusedCommandHeader{noc0_dest_addr_bwd, sem_noc0_dest_addr_bwd, 1, 128, params.flush},
        params.payload_size_bytes);
    for (size_t i = 0; i < params.send_count; i++) {
        // Forward direction
        if (params.num_fwd_hops > 0) {
            fabric_connection.get_forward_connection().wait_for_empty_write_slot();
            // Don't send payload from worker since we only want to test fabric, not worker
            fabric_connection.get_forward_connection().send_payload_flush_non_blocking_from_address(
                (uint32_t)pkt_hdr_fwd, sizeof(PACKET_HEADER_TYPE));
        }

        // Backward direction
        if (params.num_bwd_hops > 0) {
            fabric_connection.get_backward_connection().wait_for_empty_write_slot();
            fabric_connection.get_backward_connection().send_payload_flush_non_blocking_from_address(
                (uint32_t)pkt_hdr_bwd, sizeof(PACKET_HEADER_TYPE));
        }

        noc_async_writes_flushed();
    }
}

void kernel_main() {
    using namespace tt::tt_fabric;
    size_t arg_idx = 0;

    const size_t dest_bank_addr = get_arg_val<uint32_t>(arg_idx++);
    // const size_t packet_payload_size_bytes = get_arg_val<uint32_t>(arg_idx++);
    const size_t dest_noc_x_fwd = get_arg_val<uint32_t>(arg_idx++);
    const size_t dest_noc_y_fwd = get_arg_val<uint32_t>(arg_idx++);
    const size_t dest_noc_x_bwd = get_arg_val<uint32_t>(arg_idx++);
    const size_t dest_noc_y_bwd = get_arg_val<uint32_t>(arg_idx++);

    const size_t num_send_types = get_arg_val<uint32_t>(arg_idx++);
    size_t* send_types_int = reinterpret_cast<size_t*>(get_arg_addr(arg_idx));
    arg_idx += num_send_types;
    size_t* chip_send_types_int = reinterpret_cast<size_t*>(get_arg_addr(arg_idx));
    arg_idx += num_send_types;
    size_t* send_counts_per_type = reinterpret_cast<size_t*>(get_arg_addr(arg_idx));
    arg_idx += num_send_types;
    size_t* num_fwd_hops_per_type = reinterpret_cast<size_t*>(get_arg_addr(arg_idx));
    arg_idx += num_send_types;
    size_t* num_bwd_hops_per_type = reinterpret_cast<size_t*>(get_arg_addr(arg_idx));
    arg_idx += num_send_types;
    size_t* send_type_payload_sizes = reinterpret_cast<size_t*>(get_arg_addr(arg_idx));
    arg_idx += num_send_types;
    size_t* flush_send = reinterpret_cast<size_t*>(get_arg_addr(arg_idx));
    arg_idx += num_send_types;

    const size_t source_l1_cb_index = get_arg_val<uint32_t>(arg_idx++);
    const size_t packet_header_cb = get_arg_val<uint32_t>(arg_idx++);
    const size_t packet_header_size_in_headers = get_arg_val<uint32_t>(arg_idx++);

    auto fabric_connection = FabricConnectionManager::build_from_args(arg_idx);

    ASSERT(fabric_connection.is_logically_connected());

    if (!fabric_connection.is_logically_connected()) {
        return;
    }

    size_t sync_noc_x = 0;
    size_t sync_noc_y = 0;
    size_t sync_bank_addr = 0;
    size_t total_workers_per_sync = 0;
    size_t sync_mcast_fwd_hops = 0;
    size_t sync_mcast_bwd_hops = 0;
    if (enable_any_synchronization) {
        sync_noc_x = get_arg_val<uint32_t>(arg_idx++);
        sync_noc_y = get_arg_val<uint32_t>(arg_idx++);
        sync_bank_addr = get_arg_val<uint32_t>(arg_idx++);
        total_workers_per_sync = get_arg_val<uint32_t>(arg_idx++);
        sync_mcast_fwd_hops = get_arg_val<uint32_t>(arg_idx++);
        sync_mcast_bwd_hops = get_arg_val<uint32_t>(arg_idx++);
    }
    bool sync_fwd = sync_mcast_fwd_hops > 0;
    bool sync_bwd = sync_mcast_bwd_hops > 0;

    const size_t start_sync_val = total_workers_per_sync;
    const size_t finish_sync_val = 3 * total_workers_per_sync;
    const size_t second_finish_sync_val = 4 * total_workers_per_sync;

    fabric_connection.open();

    cb_reserve_back(source_l1_cb_index, 1);
    cb_reserve_back(packet_header_cb, 1);
    const auto source_l1_buffer_address = get_write_ptr(source_l1_cb_index);
    const auto packet_header_buffer_address = get_write_ptr(packet_header_cb);

    auto* fwd_packet_header = reinterpret_cast<volatile PACKET_HEADER_TYPE*>(packet_header_buffer_address);
    auto* bwd_packet_header =
        reinterpret_cast<volatile PACKET_HEADER_TYPE*>(packet_header_buffer_address + sizeof(PACKET_HEADER_TYPE));
    auto* sync_fwd_packet_header =
        reinterpret_cast<volatile PACKET_HEADER_TYPE*>(packet_header_buffer_address + sizeof(PACKET_HEADER_TYPE) * 2);
    auto* sync_bwd_packet_header =
        reinterpret_cast<volatile PACKET_HEADER_TYPE*>(packet_header_buffer_address + sizeof(PACKET_HEADER_TYPE) * 3);

    if (enable_any_synchronization) {
        sync_fwd_packet_header->to_chip_multicast(
            MulticastRoutingCommandHeader{1, static_cast<uint8_t>(sync_mcast_fwd_hops)});
        sync_bwd_packet_header->to_chip_multicast(
            MulticastRoutingCommandHeader{1, static_cast<uint8_t>(sync_mcast_bwd_hops)});
    }

    if (enable_start_synchronization) {
        line_sync(
            fabric_connection,
            sync_fwd,
            sync_bwd,
            sync_fwd_packet_header,
            sync_bwd_packet_header,
            sync_bank_addr,
            sync_noc_x,
            sync_noc_y,
            start_sync_val);
        noc_async_writes_flushed();
        line_sync(
            fabric_connection,
            sync_fwd,
            sync_bwd,
            sync_fwd_packet_header,
            sync_bwd_packet_header,
            sync_bank_addr,
            sync_noc_x,
            sync_noc_y,
            2 * start_sync_val);
    }

    {
        DeviceZoneScopedN("MAIN-TEST-BODY");
        {
            DeviceZoneScopedN("MAIN-TEST-BODY-INNER");
            for (size_t i = 0; i < num_send_types; i++) {
                auto send_type = static_cast<NocSendType>(send_types_int[i]);
                auto chip_send_type = static_cast<ChipSendType>(chip_send_types_int[i]);

                TestParams params;
                params.send_count = send_counts_per_type[i];
                params.num_fwd_hops = num_fwd_hops_per_type[i];
                params.num_bwd_hops = num_bwd_hops_per_type[i];
                params.dest_noc_x_fwd = dest_noc_x_fwd;
                params.dest_noc_y_fwd = dest_noc_y_fwd;
                params.dest_noc_x_bwd = dest_noc_x_bwd;
                params.dest_noc_y_bwd = dest_noc_y_bwd;
                params.dest_bank_addr_fwd = dest_bank_addr;
                params.dest_bank_addr_bwd = dest_bank_addr;
                params.payload_size_bytes = send_type_payload_sizes[i];
                params.chip_send_type = chip_send_type;
                params.flush = flush_send[i];

                switch (send_type) {
                    case NocSendType::NOC_UNICAST_WRITE:
                        send_packets<NocSendType::NOC_UNICAST_WRITE>(
                            fabric_connection, fwd_packet_header, bwd_packet_header, params, source_l1_buffer_address);
                        break;
                    case NocSendType::NOC_UNICAST_ATOMIC_INC:
                        send_packets<NocSendType::NOC_UNICAST_ATOMIC_INC>(
                            fabric_connection, fwd_packet_header, bwd_packet_header, params, source_l1_buffer_address);
                        break;
                    case NocSendType::NOC_FUSED_UNICAST_ATOMIC_INC:
                        send_packets<NocSendType::NOC_FUSED_UNICAST_ATOMIC_INC>(
                            fabric_connection, fwd_packet_header, bwd_packet_header, params, source_l1_buffer_address);
                        break;
                    default: ASSERT(false); break;
                }
            }
        }

        if (enable_finish_synchronization) {
            // Send a completion message
            line_sync(
                fabric_connection,
                sync_fwd,
                sync_bwd,
                sync_fwd_packet_header,
                sync_bwd_packet_header,
                sync_bank_addr,
                sync_noc_x,
                sync_noc_y,
                finish_sync_val);
            // Ack the completion and wait for everyone to do the same. This guarantees
            // all other workers have received all messages.
            line_sync(
                fabric_connection,
                sync_fwd,
                sync_bwd,
                sync_fwd_packet_header,
                sync_bwd_packet_header,
                sync_bank_addr,
                sync_noc_x,
                sync_noc_y,
                second_finish_sync_val);

            if (sync_noc_x == my_x[0] && sync_noc_y == my_y[0]) {
                // Sanity check to ensure we don't receive more acks than expected
                ASSERT(*reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sync_bank_addr) == second_finish_sync_val);
                // reset the global semaphore in case it is used in a op/kernel invocation
                *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sync_bank_addr) = 0;
            }
        }
    }

    {
        DeviceZoneScopedN("WR-CLOSE");
        fabric_connection.close();
    }
    noc_async_write_barrier();
}
