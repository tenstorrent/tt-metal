// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Pipeline shift at the start of a prefill tick (BRISC, one core per chip): the stage input x_in of
// chip p > 0 becomes chip p - 1's last output x_last (chip 0 keeps the embedding rows the tick wrote into
// x_in before). Both are DRAM interleaved tensors of the same shape, so every bank holds the same byte
// range of each, and a bank's range moves as whole fabric packets (one NoC read, one packet per 4 KB).
// Chip p + 1 first signals chip p that its x_in may be overwritten (everything before this program on
// chip p + 1 has finished: its previous tick and its own embedding copy); chip p then writes, fused with
// increments of chip p + 1's receive semaphore.
//
// Runtime args: 0 chip, 1 chips, 2 x_last, 3 x_in, 4 unused, 5 bytes per bank, 6 ready semaphore,
// 7 receive semaphore (global semaphores), then FabricConnectionManager args (fwd, bwd).

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_connection_manager.hpp"
#include "tt_metal/fabric/hw/inc/noc_addr.h"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"
#include "cpp/ttnn/operations/ccl/common/kernels/minimal_ccl_common.hpp"

constexpr uint32_t cb_buf = 0;
constexpr uint32_t kPacket = 4096;
constexpr uint32_t kSlots = 4;

void kernel_main() {
    size_t arg_idx = 0;
    const uint32_t chip = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t chips = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t x_last = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t x_in = get_arg_val<uint32_t>(arg_idx++);
    arg_idx++;
    const uint32_t bank_bytes = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t ready_addr = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t recv_addr = get_arg_val<uint32_t>(arg_idx++);
    auto fabric =
        FabricConnectionManager::build_from_args<FabricConnectionManager::BUILD_AND_OPEN_CONNECTION_START_ONLY>(
            arg_idx);
    PacketHeaderPool::reset();
    volatile PACKET_HEADER_TYPE* hdr = PacketHeaderPool::allocate_header();
    if (fabric.is_logically_connected()) {
        fabric.open_finish();
    }
    const uint32_t buf = get_write_ptr(cb_buf);
    const uint64_t ready_noc = get_noc_addr(my_x[0], my_y[0], ready_addr);
    const uint64_t recv_noc = get_noc_addr(my_x[0], my_y[0], recv_addr);
    volatile tt_l1_ptr uint32_t* ready = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ready_addr);
    volatile tt_l1_ptr uint32_t* recv = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(recv_addr);
    const uint32_t packets_per_bank = (bank_bytes + kPacket - 1) / kPacket;

    // x_in free: tell the previous chip
    if (chip > 0) {
        hdr->to_chip_unicast(1);
        hdr->to_noc_unicast_atomic_inc(tt::tt_fabric::NocUnicastAtomicIncCommandHeader{ready_noc, 1, true});
        auto& conn = fabric.get_backward_connection();
        conn.wait_for_empty_write_slot();
        conn.send_payload_flush_blocking_from_address((uint32_t)hdr, sizeof(PACKET_HEADER_TYPE));
    }

    // x_last to the next chip
    if (chip + 1 < chips) {
        noc_semaphore_wait_min(ready, 1);
        noc_semaphore_set(ready, 0);
        auto& conn = fabric.get_forward_connection();
        hdr->to_chip_unicast(1);
        for (uint32_t b = 0; b < NUM_DRAM_BANKS; b++) {
            for (uint32_t o0 = 0; o0 < bank_bytes; o0 += kSlots * kPacket) {
                const uint32_t span = bank_bytes - o0 < kSlots * kPacket ? bank_bytes - o0 : kSlots * kPacket;
                noc_async_read(get_noc_addr_from_bank_id<true>(b, x_last + o0), buf, span);
                noc_async_read_barrier();
                for (uint32_t o = 0; o < span; o += kPacket) {
                    const uint32_t len = span - o < kPacket ? span - o : kPacket;
                    hdr->to_noc_fused_unicast_write_atomic_inc(
                        tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader{
                            get_noc_addr_from_bank_id<true>(b, x_in + o0 + o), recv_noc, 1, true},
                        len);
                    perform_payload_send<true, true>(conn, buf + o, len, hdr);
                }
            }
        }
    }
    if (chip > 0) {
        noc_semaphore_wait_min(recv, NUM_DRAM_BANKS * packets_per_bank);
        noc_semaphore_set(recv, 0);
    }
    if (fabric.is_logically_connected()) {
        fabric.close();
    }
}
