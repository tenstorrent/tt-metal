// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// In-kernel one-shot all-reduce transport on a 1D line of chips (resident decode prototype).
//
// Every chip line-multicasts its local vector in both directions straight into slot[my_index] of the
// receive buffer on every other chip (same L1 address on all chips), fusing an atomic increment of
// the receivers' global semaphore into each packet. A chip is done with round r once the semaphore
// reaches (num_chips - 1) * packets_per_vector * (r + 1); the sum of the num_chips slots is then the
// all-reduced vector. Receive slots ping-pong on round parity: a peer can only send round r + 2 after
// it has received this chip's round r + 1 data, which this chip sends after consuming round r.
//
// Runs `rounds` back-to-back all-reduces inside one launch (models the serial chain of a decode
// step), so launch cost cancels out of the per-round slope.
//
// Compile-time args: 0 num_chips, 1 my_index, 2 vector_bytes, 3 packet_bytes, 4 rounds
// Runtime args: 0 src_l1, 1 recv_l1, 2 sem_l1, then FabricConnectionManager args (fwd, bwd)

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_connection_manager.hpp"
#include "tt_metal/fabric/hw/inc/noc_addr.h"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"
#include "cpp/ttnn/operations/ccl/common/kernels/minimal_ccl_common.hpp"

constexpr uint32_t num_chips = get_compile_time_arg_val(0);
constexpr uint32_t my_index = get_compile_time_arg_val(1);
constexpr uint32_t vector_bytes = get_compile_time_arg_val(2);
constexpr uint32_t packet_bytes = get_compile_time_arg_val(3);
constexpr uint32_t rounds = get_compile_time_arg_val(4);
constexpr uint32_t packets_per_vector = (vector_bytes + packet_bytes - 1) / packet_bytes;
constexpr uint32_t num_forward = num_chips - 1 - my_index;
constexpr uint32_t num_backward = my_index;

void kernel_main() {
    size_t arg_idx = 0;
    const uint32_t src_l1 = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t recv_l1 = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t sem_l1 = get_arg_val<uint32_t>(arg_idx++);
    auto fabric =
        FabricConnectionManager::build_from_args<FabricConnectionManager::BUILD_AND_OPEN_CONNECTION_START_ONLY>(
            arg_idx);

    PacketHeaderPool::reset();
    volatile PACKET_HEADER_TYPE* hdr_fwd = PacketHeaderPool::allocate_header();
    volatile PACKET_HEADER_TYPE* hdr_bwd = PacketHeaderPool::allocate_header();
    hdr_fwd->to_chip_multicast(tt::tt_fabric::MulticastRoutingCommandHeader{1, static_cast<uint8_t>(num_forward)});
    hdr_bwd->to_chip_multicast(tt::tt_fabric::MulticastRoutingCommandHeader{1, static_cast<uint8_t>(num_backward)});
    if (fabric.is_logically_connected()) {
        fabric.open_finish();
    }

    volatile tt_l1_ptr uint32_t* sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem_l1);
    const uint64_t sem_noc = get_noc_addr(my_x[0], my_y[0], sem_l1);
    uint32_t expected = 0;
    for (uint32_t r = 0; r < rounds; r++) {
        const uint32_t slot = recv_l1 + ((r & 1) * num_chips + my_index) * vector_bytes;
        size_t rd = src_l1;
        for (uint32_t p = 0; p < packets_per_vector; p++) {
            const uint32_t off = p * packet_bytes;
            const uint32_t bytes = off + packet_bytes <= vector_bytes ? packet_bytes : vector_bytes - off;
            fused_write_atomic_and_advance_local_read_address_for_fabric_write(
                get_noc_addr(my_x[0], my_y[0], slot + off), hdr_fwd, hdr_bwd, fabric, rd, bytes, sem_noc, 1, true);
        }
        expected += (num_chips - 1) * packets_per_vector;
        noc_semaphore_wait_min(sem, expected);
    }
    noc_async_write_barrier();
    // Every peer's increments for the last round have landed, so the semaphore can be rearmed.
    noc_semaphore_set(sem, 0);
    if (fabric.is_logically_connected()) {
        fabric.close();
    }
}
