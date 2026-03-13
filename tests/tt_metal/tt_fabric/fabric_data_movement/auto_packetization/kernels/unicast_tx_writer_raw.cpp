// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Unified unicast TX kernel for auto-packetization integration tests.
// Dispatches to the correct fabric unicast API based on the send_op RT arg.
//
// RT args:
//   0: src_l1_addr   (u32)
//   1: total_size    (u32)
//   2: dst_base_addr (u32)
//   [FABRIC_2D] 3: dst_mesh_id, 4: dst_dev_id
//   next: rx_noc_x, rx_noc_y, sem_l1_addr
//   [!FABRIC_2D] next: num_hops
//   next: scatter_offset (u32, 0 for non-scatter ops)
//   next: send_op       (u32, one of TX_OP_*)
//   ... fabric connection args

#include "tx_kernel_common.h"

void kernel_main() {
    size_t idx = 0;
    TX_KERNEL_PARSE_UNICAST_ARGS(idx)
    const uint32_t scatter_offset = get_arg_val<uint32_t>(idx++);
    const uint32_t send_op        = get_arg_val<uint32_t>(idx++);
    TX_KERNEL_SETUP(idx)

    const uint64_t dst_noc_addr  = safe_get_noc_addr(rx_noc_x, rx_noc_y, dst_base_addr, 0);
    const uint64_t dst_noc_addr1 = safe_get_noc_addr(rx_noc_x, rx_noc_y, dst_base_addr + scatter_offset, 0);
    const uint64_t sem_noc_addr  = safe_get_noc_addr(rx_noc_x, rx_noc_y, sem_l1_addr, 0);
    const uint16_t scatter_chunk_size = static_cast<uint16_t>(total_size / 2);

    if (send_op == TX_OP_WRITE) {
#ifdef FABRIC_2D
        fabric_unicast_noc_unicast_write(
            &sender, packet_header, dst_dev_id, dst_mesh_id,
            src_l1_addr, total_size, tt::tt_fabric::NocUnicastCommandHeader{dst_noc_addr});
#else
        fabric_unicast_noc_unicast_write(
            &sender, packet_header,
            src_l1_addr, total_size, tt::tt_fabric::NocUnicastCommandHeader{dst_noc_addr}, num_hops);
#endif
        noc_async_writes_flushed();
#ifdef FABRIC_2D
        fabric_unicast_noc_unicast_atomic_inc(
            &sender, packet_header, dst_dev_id, dst_mesh_id,
            tt::tt_fabric::NocUnicastAtomicIncCommandHeader(sem_noc_addr, 1, 32));
#else
        fabric_unicast_noc_unicast_atomic_inc(
            &sender, packet_header,
            tt::tt_fabric::NocUnicastAtomicIncCommandHeader(sem_noc_addr, 1, 32), num_hops);
#endif
    } else if (send_op == TX_OP_SCATTER_WRITE) {
#ifdef FABRIC_2D
        fabric_unicast_noc_scatter_write(
            &sender, packet_header, dst_dev_id, dst_mesh_id,
            src_l1_addr, total_size,
            tt::tt_fabric::NocUnicastScatterCommandHeader{{dst_noc_addr, dst_noc_addr1}, {scatter_chunk_size}});
#else
        fabric_unicast_noc_scatter_write(
            &sender, packet_header,
            src_l1_addr, total_size,
            tt::tt_fabric::NocUnicastScatterCommandHeader{{dst_noc_addr, dst_noc_addr1}, {scatter_chunk_size}},
            num_hops);
#endif
        noc_async_writes_flushed();
#ifdef FABRIC_2D
        fabric_unicast_noc_unicast_atomic_inc(
            &sender, packet_header, dst_dev_id, dst_mesh_id,
            tt::tt_fabric::NocUnicastAtomicIncCommandHeader(sem_noc_addr, 1, 32));
#else
        fabric_unicast_noc_unicast_atomic_inc(
            &sender, packet_header,
            tt::tt_fabric::NocUnicastAtomicIncCommandHeader(sem_noc_addr, 1, 32), num_hops);
#endif
    } else if (send_op == TX_OP_FUSED_ATOMIC_INC) {
#ifdef FABRIC_2D
        fabric_unicast_noc_fused_unicast_with_atomic_inc(
            &sender, packet_header, dst_dev_id, dst_mesh_id,
            src_l1_addr, total_size,
            tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader{dst_noc_addr, sem_noc_addr, 1, true});
#else
        fabric_unicast_noc_fused_unicast_with_atomic_inc(
            &sender, packet_header,
            src_l1_addr, total_size,
            tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader{dst_noc_addr, sem_noc_addr, 1, true},
            num_hops);
#endif
    } else {  // TX_OP_FUSED_SCATTER_ATOMIC_INC
#ifdef FABRIC_2D
        fabric_unicast_noc_fused_scatter_write_atomic_inc(
            &sender, packet_header, dst_dev_id, dst_mesh_id,
            src_l1_addr, total_size,
            tt::tt_fabric::NocUnicastScatterAtomicIncFusedCommandHeader{
                {dst_noc_addr, dst_noc_addr1}, sem_noc_addr, {scatter_chunk_size}, 1, true});
#else
        fabric_unicast_noc_fused_scatter_write_atomic_inc(
            &sender, packet_header,
            src_l1_addr, total_size,
            tt::tt_fabric::NocUnicastScatterAtomicIncFusedCommandHeader{
                {dst_noc_addr, dst_noc_addr1}, sem_noc_addr, {scatter_chunk_size}, 1, true},
            num_hops);
#endif
    }

    TX_KERNEL_TEARDOWN()
}
