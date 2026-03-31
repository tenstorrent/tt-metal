// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// MoE Dispatch — Sender (DRAM dispatch_buf + streaming semaphore)
//
// EP serialization via go_sem ring:
//   - All devices share one go_sem per device
//   - Device 0: go_sem init=1 (first expert pre-granted); waits go_sem >= turn+1 before each expert
//   - Middle devices: go_sem init=0; wait go_sem >= turn+1; signal next device (+1 hop forward)
//   - Last device: waits go_sem >= turn+1; signals device 0's go_sem (num_devices-1 hops backward)
//
// Only one EP device sends bulk tile data at a time.

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "api/debug/dprint.h"
#include "fabric/fabric_edm_packet_header.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_connection_manager.hpp"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"

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
    uint32_t first_sender_noc_x = get_arg_val<uint32_t>(ra++);
    uint32_t first_sender_noc_y = get_arg_val<uint32_t>(ra++);
    uint32_t is_first_ep_device = get_arg_val<uint32_t>(ra++);
    uint32_t is_last_ep_device = get_arg_val<uint32_t>(ra++);
    uint32_t my_mesh_id = get_arg_val<uint32_t>(ra++);
    uint32_t next_device_id = get_arg_val<uint32_t>(ra++);
    uint32_t first_device_id = get_arg_val<uint32_t>(ra++);

    uint32_t expert_n_rows[num_experts];
    uint32_t expert_start_row[num_experts];
    for (uint32_t e = 0; e < num_experts; e++) expert_n_rows[e] = get_arg_val<uint32_t>(ra++);
    for (uint32_t e = 0; e < num_experts; e++) expert_start_row[e] = get_arg_val<uint32_t>(ra++);
    uint32_t expert_dst_row[num_experts];
    for (uint32_t e = 0; e < num_experts; e++) expert_dst_row[e] = get_arg_val<uint32_t>(ra++);
    uint32_t expert_owner_device_id[num_experts];
    for (uint32_t e = 0; e < num_experts; e++) expert_owner_device_id[e] = get_arg_val<uint32_t>(ra++);

    auto fabric_connection =
        FabricConnectionManager::build_from_args<FabricConnectionManager::BUILD_AND_OPEN_CONNECTION_START_ONLY>(ra);

    const auto input_acc = TensorAccessor(input_ta, input_addr, tile_bytes);
    const auto dispatch_acc = TensorAccessor(dispatch_ta, dispatch_buf_addr, tile_bytes);

    cb_reserve_back(pkt_hdr_cb_id, 1);
    auto pkt_hdr_l1 = get_write_ptr(pkt_hdr_cb_id);
    cb_push_back(pkt_hdr_cb_id, 1);
    volatile PACKET_HEADER_TYPE* pkt_hdr = reinterpret_cast<volatile PACKET_HEADER_TYPE*>(pkt_hdr_l1);

    volatile tt_l1_ptr uint32_t* go_sem_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(go_sem_addr);

    // Pre-compute the forward direction so we can select the right connection at runtime.
    // This is only valid when !is_last_ep_device (we have a forward neighbor).
    uint32_t fwd_direction = 0;
    if (!is_last_ep_device) {
        fwd_direction = static_cast<uint32_t>(tt::tt_fabric::get_next_hop_router_direction(my_mesh_id, next_device_id));
    }

    DPRINT << "SENDER[" << my_device_index << "]: next=(" << next_sender_noc_x << "," << next_sender_noc_y
           << ") first=(" << first_sender_noc_x << "," << first_sender_noc_y << ") opening fabric" << ENDL();
    DPRINT << "SENDER[" << my_device_index << "]: go_sem_addr=0x" << HEX() << go_sem_addr << " tiles_ready_sem_addr=0x"
           << tiles_ready_sem_addr << DEC() << " receiver=(" << receiver_noc_x << "," << receiver_noc_y << ")"
           << " has_fwd=" << fabric_connection.has_forward_connection()
           << " has_bwd=" << fabric_connection.has_backward_connection() << ENDL();
    fabric_connection.open_finish();
    DPRINT << "SENDER[" << my_device_index << "]: fabric open, go_sem=" << *go_sem_ptr
           << " is_first=" << is_first_ep_device << " is_last=" << is_last_ep_device << ENDL();
    DPRINT << "SENDER[" << my_device_index << "]: entering loop, my_mesh_id=" << my_mesh_id
           << " next_dev=" << next_device_id << " first_dev=" << first_device_id << ENDL();

    const uint32_t row_bytes = D_t * tile_bytes;
    uint32_t turn = 0;

    // Outer loop: num_devices turns, one go_sem grant per turn.
    // Each turn, this device sends for E_local experts, strided across device groups:
    //   expert = turn + i * num_devices   (i = 0..E_local-1)
    // All devices hold the token simultaneously within a turn, so all compute in parallel.
    for (uint32_t t = 0; t < num_experts / E_local; t++) {
        // Wait for our turn
        DPRINT << "SENDER[" << my_device_index << "]: turn=" << t << " waiting go_sem=" << (turn + 1)
               << " cur=" << *go_sem_ptr << ENDL();
        noc_semaphore_wait_min(go_sem_ptr, turn + 1);
        DPRINT << "SENDER[" << my_device_index << "]: turn=" << t << " go_sem ok" << ENDL();

        for (uint32_t i = 0; i < num_devices; i++) {
            uint32_t e = i * E_local + t;
            uint32_t owner = e / E_local;
            uint32_t n_rows = expert_n_rows[e];
            uint32_t src_start = expert_start_row[e];
            uint32_t dst_start = expert_dst_row[e];
            uint32_t owner_dev_id = expert_owner_device_id[e];
            bool is_local = (owner == my_device_index);

            DPRINT << "SENDER[" << my_device_index << "]: turn=" << t << " i=" << i << " e=" << e
                   << " n_rows=" << n_rows << ENDL();

            if (n_rows > 0) {
                uint64_t owner_sem_noc = get_noc_addr(receiver_noc_x, receiver_noc_y, tiles_ready_sem_addr);

                for (uint32_t r = 0; r < n_rows; r++) {
                    uint32_t src_tile_idx = (src_start + r) * D_t;
                    uint32_t dst_tile_idx = (dst_start + r) * D_t;

                    cb_reserve_back(sender_cb_id, 1);
                    uint32_t l1 = get_write_ptr(sender_cb_id);
                    for (uint32_t k = 0; k < D_t; k++) {
                        noc_async_read(input_acc.get_noc_addr(src_tile_idx + k), l1 + k * tile_bytes, tile_bytes);
                    }
                    noc_async_read_barrier();

                    if (is_local) {
                        for (uint32_t k = 0; k < D_t; k++) {
                            noc_async_write(
                                l1 + k * tile_bytes, dispatch_acc.get_noc_addr(dst_tile_idx + k), tile_bytes);
                        }
                        noc_async_write_barrier();
                        noc_semaphore_inc(owner_sem_noc, 1);
                    } else {
                        uint64_t dest_noc = dispatch_acc.get_noc_addr(dst_tile_idx);

                        pkt_hdr->to_noc_fused_unicast_write_atomic_inc(
                            tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader{dest_noc, owner_sem_noc, 1, false},
                            row_bytes);
                        tt::tt_fabric::fabric_set_unicast_route(
                            (volatile tt::tt_fabric::HybridMeshPacketHeader*)pkt_hdr, owner_dev_id, my_mesh_id);

                        uint32_t route = static_cast<uint32_t>(
                            tt::tt_fabric::get_next_hop_router_direction(my_mesh_id, owner_dev_id));
                        if (route == fwd_direction) {
                            fabric_connection.get_forward_connection().wait_for_empty_write_slot();
                            fabric_connection.get_forward_connection()
                                .send_payload_without_header_non_blocking_from_address(l1, row_bytes);
                            fabric_connection.get_forward_connection().send_payload_flush_blocking_from_address(
                                (uint32_t)pkt_hdr, sizeof(PACKET_HEADER_TYPE));
                        } else {
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
        }

        turn++;

        if (!is_last_ep_device) {
            uint64_t next_go = get_noc_addr(next_sender_noc_x, next_sender_noc_y, go_sem_addr);
            pkt_hdr->to_noc_unicast_atomic_inc(tt::tt_fabric::NocUnicastAtomicIncCommandHeader{next_go, 1, false});
            tt::tt_fabric::fabric_set_unicast_route(
                (volatile tt::tt_fabric::HybridMeshPacketHeader*)pkt_hdr, next_device_id, my_mesh_id);
            DPRINT << "SENDER[" << my_device_index << "]: turn=" << t << " signaling next dev=" << next_device_id
                   << " noc=0x" << HEX() << next_go << DEC() << ENDL();
            fabric_connection.get_forward_connection().wait_for_empty_write_slot();
            fabric_connection.get_forward_connection().send_payload_flush_blocking_from_address(
                (uint32_t)pkt_hdr, sizeof(PACKET_HEADER_TYPE));
            DPRINT << "SENDER[" << my_device_index << "]: turn=" << t << " fwd signal sent" << ENDL();
        } else {
            uint64_t first_go = get_noc_addr(first_sender_noc_x, first_sender_noc_y, go_sem_addr);
            pkt_hdr->to_noc_unicast_atomic_inc(tt::tt_fabric::NocUnicastAtomicIncCommandHeader{first_go, 1, false});
            tt::tt_fabric::fabric_set_unicast_route(
                (volatile tt::tt_fabric::HybridMeshPacketHeader*)pkt_hdr, first_device_id, my_mesh_id);
            DPRINT << "SENDER[" << my_device_index << "]: turn=" << t << " signaling first dev=" << first_device_id
                   << " noc=0x" << HEX() << first_go << DEC() << ENDL();
            fabric_connection.get_backward_connection().wait_for_empty_write_slot();
            fabric_connection.get_backward_connection().send_payload_flush_blocking_from_address(
                (uint32_t)pkt_hdr, sizeof(PACKET_HEADER_TYPE));
            DPRINT << "SENDER[" << my_device_index << "]: turn=" << t << " bwd signal sent" << ENDL();
        }
    }

    noc_async_write_barrier();
    noc_async_atomic_barrier();

    if (fabric_connection.is_logically_connected()) {
        fabric_connection.close();
    }

    DPRINT << "SENDER[" << my_device_index << "]: ALL DONE" << ENDL();
}
