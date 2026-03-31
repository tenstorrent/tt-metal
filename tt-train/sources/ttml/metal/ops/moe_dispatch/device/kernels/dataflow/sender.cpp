// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// MoE Dispatch — Sender (DRAM dispatch_buf + streaming semaphore)
//
// For each expert (EP-serialized via go_sem):
//   1. Read tile-row from sorted_hidden DRAM
//   2. Write to dispatch_buf DRAM at remapped offset (local NoC or fabric)
//   3. Increment tiles_ready_sem on receiver
//
// Receiver reads from dispatch_buf as tiles become ready (no full barrier).

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "api/debug/dprint.h"
#include "fabric/fabric_edm_packet_header.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_connection_manager.hpp"

constexpr uint32_t sender_cb_id = get_compile_time_arg_val(0);
constexpr uint32_t pkt_hdr_cb_id = get_compile_time_arg_val(1);
constexpr uint32_t tile_bytes = get_compile_time_arg_val(2);
constexpr uint32_t D_t = get_compile_time_arg_val(3);
constexpr uint32_t num_experts = get_compile_time_arg_val(4);
constexpr uint32_t E_local = get_compile_time_arg_val(5);
constexpr uint32_t my_device_index = get_compile_time_arg_val(6);
constexpr uint32_t num_devices = get_compile_time_arg_val(7);

constexpr auto input_ta = TensorAccessorArgs<8>();
constexpr auto dispatch_ta = TensorAccessorArgs<input_ta.next_compile_time_args_offset()>();

void kernel_main() {
    size_t ra = 0;
    uint32_t input_addr = get_arg_val<uint32_t>(ra++);
    uint32_t dispatch_buf_addr = get_arg_val<uint32_t>(ra++);
    uint32_t receiver_noc_x = get_arg_val<uint32_t>(ra++);
    uint32_t receiver_noc_y = get_arg_val<uint32_t>(ra++);
    uint32_t tiles_ready_sem_addr = get_semaphore(get_arg_val<uint32_t>(ra++));
    uint32_t go_sem_addr = get_semaphore(get_arg_val<uint32_t>(ra++));
    uint32_t next_sender_noc_x = get_arg_val<uint32_t>(ra++);
    uint32_t next_sender_noc_y = get_arg_val<uint32_t>(ra++);
    uint32_t is_last_ep_device = get_arg_val<uint32_t>(ra++);

    uint32_t expert_n_rows[num_experts];
    uint32_t expert_start_row[num_experts];
    for (uint32_t e = 0; e < num_experts; e++) expert_n_rows[e] = get_arg_val<uint32_t>(ra++);
    for (uint32_t e = 0; e < num_experts; e++) expert_start_row[e] = get_arg_val<uint32_t>(ra++);
    uint32_t expert_dst_row[num_experts];
    for (uint32_t e = 0; e < num_experts; e++) expert_dst_row[e] = get_arg_val<uint32_t>(ra++);

    auto fabric_connection =
        FabricConnectionManager::build_from_args<FabricConnectionManager::BUILD_AND_OPEN_CONNECTION_START_ONLY>(ra);

    const auto input_acc = TensorAccessor(input_ta, input_addr, tile_bytes);
    const auto dispatch_acc = TensorAccessor(dispatch_ta, dispatch_buf_addr, tile_bytes);

    cb_reserve_back(pkt_hdr_cb_id, 1);
    auto pkt_hdr_l1 = get_write_ptr(pkt_hdr_cb_id);
    cb_push_back(pkt_hdr_cb_id, 1);
    volatile PACKET_HEADER_TYPE* pkt_hdr = reinterpret_cast<volatile PACKET_HEADER_TYPE*>(pkt_hdr_l1);

    DPRINT << "SENDER[" << my_device_index << "]: opening fabric" << ENDL();
    fabric_connection.open_finish();
    DPRINT << "SENDER[" << my_device_index << "]: fabric open, go_sem wait" << ENDL();

    volatile tt_l1_ptr uint32_t* go_sem_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(go_sem_addr);

    const uint32_t row_bytes = D_t * tile_bytes;
    uint32_t turn = 0;

    for (uint32_t e = 0; e < num_experts; e++) {
        uint32_t owner = e / E_local;
        uint32_t n_rows = expert_n_rows[e];
        uint32_t src_start = expert_start_row[e];
        uint32_t dst_start = expert_dst_row[e];
        bool is_local = (owner == my_device_index);

        // Wait for EP turn
        noc_semaphore_wait_min(go_sem_ptr, turn + 1);

        if (n_rows > 0) {
            uint64_t owner_sem_noc = get_noc_addr(receiver_noc_x, receiver_noc_y, tiles_ready_sem_addr);

            for (uint32_t r = 0; r < n_rows; r++) {
                uint32_t src_tile_idx = (src_start + r) * D_t;
                uint32_t dst_tile_idx = (dst_start + r) * D_t;

                // Read from sorted_hidden
                cb_reserve_back(sender_cb_id, 1);
                uint32_t l1 = get_write_ptr(sender_cb_id);
                for (uint32_t k = 0; k < D_t; k++) {
                    noc_async_read(input_acc.get_noc_addr(src_tile_idx + k), l1 + k * tile_bytes, tile_bytes);
                }
                noc_async_read_barrier();

                if (is_local) {
                    // Write to local dispatch_buf DRAM
                    for (uint32_t k = 0; k < D_t; k++) {
                        noc_async_write(l1 + k * tile_bytes, dispatch_acc.get_noc_addr(dst_tile_idx + k), tile_bytes);
                    }
                    noc_async_write_barrier();
                    // Signal receiver: one more tile-row ready
                    noc_semaphore_inc(owner_sem_noc, 1);
                } else {
                    // Fabric: write to remote dispatch_buf + inc sem
                    bool use_forward = (owner > my_device_index);
                    uint8_t num_hops =
                        use_forward ? (uint8_t)(owner - my_device_index) : (uint8_t)(my_device_index - owner);

                    uint64_t dest_noc = dispatch_acc.get_noc_addr(dst_tile_idx);

                    pkt_hdr->to_chip_unicast(num_hops);
                    pkt_hdr->to_noc_fused_unicast_write_atomic_inc(
                        tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader{dest_noc, owner_sem_noc, 1, false},
                        row_bytes);

                    if (use_forward && fabric_connection.has_forward_connection()) {
                        fabric_connection.get_forward_connection().wait_for_empty_write_slot();
                        fabric_connection.get_forward_connection()
                            .send_payload_without_header_non_blocking_from_address(l1, row_bytes);
                        fabric_connection.get_forward_connection().send_payload_flush_blocking_from_address(
                            (uint32_t)pkt_hdr, sizeof(PACKET_HEADER_TYPE));
                    } else if (!use_forward && fabric_connection.has_backward_connection()) {
                        fabric_connection.get_backward_connection().wait_for_empty_write_slot();
                        fabric_connection.get_backward_connection()
                            .send_payload_without_header_non_blocking_from_address(l1, row_bytes);
                        fabric_connection.get_backward_connection().send_payload_flush_blocking_from_address(
                            (uint32_t)pkt_hdr, sizeof(PACKET_HEADER_TYPE));
                    }
                }

                cb_push_back(sender_cb_id, 1);
                cb_pop_front(sender_cb_id, 1);
            }
        }

        turn++;

        // Signal next EP device
        if (!is_last_ep_device) {
            uint64_t next_go = get_noc_addr(next_sender_noc_x, next_sender_noc_y, go_sem_addr);
            if (fabric_connection.has_forward_connection()) {
                pkt_hdr->to_chip_unicast(1);
                pkt_hdr->to_noc_unicast_atomic_inc(tt::tt_fabric::NocUnicastAtomicIncCommandHeader{next_go, 1, false});
                fabric_connection.get_forward_connection().wait_for_empty_write_slot();
                fabric_connection.get_forward_connection().send_payload_flush_blocking_from_address(
                    (uint32_t)pkt_hdr, sizeof(PACKET_HEADER_TYPE));
            }
        }
    }

    noc_async_write_barrier();
    noc_async_atomic_barrier();

    if (fabric_connection.is_logically_connected()) {
        fabric_connection.close();
    }

    DPRINT << "SENDER[" << my_device_index << "]: ALL DONE" << ENDL();
}
