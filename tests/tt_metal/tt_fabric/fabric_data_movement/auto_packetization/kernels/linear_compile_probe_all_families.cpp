// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Compile-only probe kernel for ALL 7 missing linear wrapper families.
// Only includes linear/api.h -- does NOT require FABRIC_2D.
// This kernel instantiates template wrappers to force device toolchain compilation.
// It should NOT be expected to produce correct results at runtime.

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "fabric/fabric_edm_packet_header.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_worker_adapters.hpp"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"
#include "tt_metal/fabric/hw/inc/noc_addr.h"
#include "tt_metal/fabric/hw/inc/linear/api.h"

using namespace tt::tt_fabric;
using namespace tt::tt_fabric::linear::experimental;

void kernel_main() {
    size_t idx = 0;
    const uint32_t src_l1_addr   = get_arg_val<uint32_t>(idx++);
    const uint32_t total_size    = get_arg_val<uint32_t>(idx++);
    const uint32_t dst_base_addr = get_arg_val<uint32_t>(idx++);
    const uint32_t rx_noc_x      = get_arg_val<uint32_t>(idx++);
    const uint32_t rx_noc_y      = get_arg_val<uint32_t>(idx++);
    const uint32_t sem_l1_addr   = get_arg_val<uint32_t>(idx++);
    const uint8_t  num_hops      = static_cast<uint8_t>(get_arg_val<uint32_t>(idx++));

    auto sender = WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(idx);
    volatile tt_l1_ptr PACKET_HEADER_TYPE* packet_header = PacketHeaderPool::allocate_header();
    sender.open<true>();

    const uint64_t dst_noc_addr = safe_get_noc_addr(rx_noc_x, rx_noc_y, dst_base_addr, 0);
    const uint64_t sem_noc_addr = safe_get_noc_addr(rx_noc_x, rx_noc_y, sem_l1_addr, 0);

    // --- Unicast families ---

    // 1. fabric_unicast_noc_scatter_write
    fabric_unicast_noc_scatter_write(
        &sender, packet_header,
        src_l1_addr, total_size,
        NocUnicastScatterCommandHeader({dst_noc_addr, dst_noc_addr}, {static_cast<uint16_t>(total_size / 2)}),
        num_hops);

    // 2. fabric_unicast_noc_fused_scatter_write_atomic_inc
    fabric_unicast_noc_fused_scatter_write_atomic_inc(
        &sender, packet_header,
        src_l1_addr, total_size,
        NocUnicastScatterAtomicIncFusedCommandHeader(
            {dst_noc_addr, dst_noc_addr}, sem_noc_addr,
            {static_cast<uint16_t>(total_size / 2)}, 1, true),
        num_hops);

    // 3. fabric_unicast_noc_fused_unicast_with_atomic_inc
    fabric_unicast_noc_fused_unicast_with_atomic_inc(
        &sender, packet_header,
        src_l1_addr, total_size,
        NocUnicastAtomicIncFusedCommandHeader{dst_noc_addr, sem_noc_addr, 1, true},
        num_hops);

    // --- Multicast families ---

    // 4. fabric_multicast_noc_scatter_write
    fabric_multicast_noc_scatter_write(
        &sender, packet_header,
        src_l1_addr, total_size,
        NocUnicastScatterCommandHeader({dst_noc_addr, dst_noc_addr}, {static_cast<uint16_t>(total_size / 2)}),
        1, 1);

    // 5. fabric_multicast_noc_fused_unicast_with_atomic_inc
    fabric_multicast_noc_fused_unicast_with_atomic_inc(
        &sender, packet_header,
        src_l1_addr, total_size,
        NocUnicastAtomicIncFusedCommandHeader{dst_noc_addr, sem_noc_addr, 1, true},
        1, 1);

    // 6. fabric_multicast_noc_fused_scatter_write_atomic_inc
    fabric_multicast_noc_fused_scatter_write_atomic_inc(
        &sender, packet_header,
        src_l1_addr, total_size,
        NocUnicastScatterAtomicIncFusedCommandHeader(
            {dst_noc_addr, dst_noc_addr}, sem_noc_addr,
            {static_cast<uint16_t>(total_size / 2)}, 1, true),
        1, 1);

    // 7. fabric_sparse_multicast_noc_unicast_write (linear-only)
    fabric_sparse_multicast_noc_unicast_write(
        &sender, packet_header,
        src_l1_addr, total_size,
        NocUnicastCommandHeader{dst_noc_addr},
        static_cast<uint16_t>(0x3));

    sender.close();
}
