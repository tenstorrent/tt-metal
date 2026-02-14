// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

#include "api/debug/dprint.h"
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_connection_manager.hpp"
#include "cpp/ttnn/operations/ccl/common/kernels/minimal_ccl_common.hpp"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"
#include "tt_metal/fabric/hw/inc/linear/api.h"

void kernel_main() {
    // same arg indices as in reader_binary_diff_lengths for compat
    size_t arg_idx = 0;
    uint32_t device_id = get_arg_val<uint32_t>(arg_idx++);
    uint32_t global_semaphore_addr = get_arg_val<uint32_t>(arg_idx++);
    uint32_t src0_addr = get_arg_val<uint32_t>(arg_idx++);
    uint32_t src1_addr = get_arg_val<uint32_t>(arg_idx++);
    uint32_t Mt = get_arg_val<uint32_t>(arg_idx++);
    uint32_t Kt = get_arg_val<uint32_t>(arg_idx++);
    uint32_t Nt = get_arg_val<uint32_t>(arg_idx++);

    auto fabric_connection =
        FabricConnectionManager::build_from_args<FabricConnectionManager::BuildFromArgsMode::BUILD_AND_OPEN_CONNECTION>(
            arg_idx);

    volatile tt_l1_ptr uint32_t* global_semaphore_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(global_semaphore_addr);

    uint32_t tile_size = get_tile_size(0);

    // 18,18 corresponds to the virtual coord of Worker Core 0,0
    uint64_t semaphore_noc_addr = safe_get_noc_addr(18, 18, global_semaphore_addr, 0);

    constexpr uint32_t cb_id_in0 = 0;
    constexpr uint32_t cb_id_in1 = 1;

    tt::tt_fabric::WorkerToFabricEdmSender cur_connection;

    if (device_id == 0) {
        // Device 0 has a forward connection to Device 1
        cur_connection = fabric_connection.get_forward_connection();
    } else {
        // Device 1 has a backward connection to Device 0
        cur_connection = fabric_connection.get_backward_connection();
    }

    constexpr auto s0_args = TensorAccessorArgs<0>();
    const auto s0 = TensorAccessor(s0_args, src0_addr, get_tile_size(cb_id_in0));

    constexpr auto s1_args = TensorAccessorArgs<s0_args.next_compile_time_args_offset()>();
    const auto s1 = TensorAccessor(s1_args, src1_addr, get_tile_size(cb_id_in1));

    // Packet Header for semaphore increment
    auto* pkt_semaphore_hdr = PacketHeaderPool::allocate_header();

    // Packet Header for sending payload with fused semaphore increment
    auto* pkt_payload_sem_hdr = PacketHeaderPool::allocate_header();
    pkt_semaphore_hdr->to_noc_unicast_atomic_inc(
        tt::tt_fabric::NocUnicastAtomicIncCommandHeader{semaphore_noc_addr, static_cast<uint32_t>(1)});  // increment 1

    // Num Hops is 1 as we have a direct connection between the two devices.
    fabric_set_unicast_route<false>(pkt_payload_sem_hdr, 1);
    fabric_set_unicast_route<false>(pkt_semaphore_hdr, 1);

    // Loop through the dimensions of the matrices. Read them and push to the circular buffers.
    // Dimension names are called M, N and K. `t` in `mt` means tile.
    uint32_t pkt_write_index = 1;
    for (uint32_t mt = 0; mt < Mt; mt++) {
        uint32_t itileB = 0;
        // Nt is for one device, but because it's sharded across two devices, we loop 2*Nt
        for (uint32_t nt = 0; nt < 2 * Nt; nt++) {
            for (uint32_t kt = 0; kt < Kt; kt++) {
                // Load Input A from DRAM to CB
                uint32_t a_tile_index = mt * Kt + kt;  // A is MK, so we stride by Kt
                cb_reserve_back(cb_id_in0, 1);
                uint32_t l1_write_addr_in0 = get_write_ptr(cb_id_in0);
                noc_async_read_tile(a_tile_index, s0, l1_write_addr_in0);
                noc_async_read_barrier();
                cb_push_back(cb_id_in0, 1);

                cb_reserve_back(cb_id_in1, 1);
                uint32_t l1_write_addr_in1 = get_write_ptr(cb_id_in1);
                if (device_id == nt / Nt) {
                    // If this tile belongs to this device, read from DRAM to CB
                    uint32_t b_tile_index = kt * Nt + (nt % Nt);
                    noc_async_read_tile(b_tile_index, s1, l1_write_addr_in1);
                    noc_async_read_barrier();
                    uint64_t cb_noc_addr = safe_get_noc_addr(18, 18, l1_write_addr_in1);
                    pkt_payload_sem_hdr->to_noc_fused_unicast_write_atomic_inc(
                        NocUnicastAtomicIncFusedCommandHeader(cb_noc_addr, semaphore_noc_addr, 1, true), tile_size);

                    // Send this tile to the other device via fabric with a fused semaphore increment
                    cur_connection.wait_for_empty_write_slot();
                    cur_connection.send_payload_without_header_non_blocking_from_address(l1_write_addr_in1, tile_size);
                    cur_connection.send_payload_flush_blocking_from_address(
                        (uint32_t)pkt_payload_sem_hdr, sizeof(PACKET_HEADER_TYPE));

                    // Wait for the other device to ack that it has received the tile.
                    noc_semaphore_wait_min(global_semaphore_ptr, pkt_write_index);

                } else {
                    // If this tile belongs to the other device, wait for the other device to send it via fabric
                    noc_semaphore_wait_min(global_semaphore_ptr, pkt_write_index);

                    // Send an ack back to the other device that we have received the tile by performing a semaphore
                    // increment
                    cur_connection.wait_for_empty_write_slot();
                    cur_connection.send_payload_flush_blocking_from_address(
                        (uint32_t)pkt_semaphore_hdr, sizeof(PACKET_HEADER_TYPE));
                    noc_async_writes_flushed();
                    noc_async_write_barrier();
                }
                pkt_write_index++;
                cb_push_back(cb_id_in1, 1);
            }  // Kt loop
        }  // Nt loop
    }  // Mt loop

    if (fabric_connection.is_logically_connected()) {
        fabric_connection.close();
    }
}
