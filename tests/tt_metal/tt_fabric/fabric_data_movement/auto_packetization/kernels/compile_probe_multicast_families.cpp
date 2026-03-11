// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Compile-only probe kernel for multicast scatter, multicast fused_unicast_atomic_inc,
// and multicast fused_scatter_atomic_inc wrapper families (mesh + linear).
// This kernel instantiates template wrappers to force device toolchain compilation.
// It should NOT be expected to produce correct results at runtime.
//
// Requires FABRIC_2D=1 define (includes mesh/api.h).

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "fabric/fabric_edm_packet_header.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_worker_adapters.hpp"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"
#include "tt_metal/fabric/hw/inc/noc_addr.h"
#include "tt_metal/fabric/hw/inc/mesh/api.h"
#include "tt_metal/fabric/hw/inc/linear/api.h"

using namespace tt::tt_fabric;
using namespace tt::tt_fabric::mesh::experimental;

void kernel_main() {
    size_t idx = 0;
    const uint32_t src_l1_addr   = get_arg_val<uint32_t>(idx++);
    const uint32_t total_size    = get_arg_val<uint32_t>(idx++);
    const uint32_t dst_base_addr = get_arg_val<uint32_t>(idx++);
    const uint16_t dst_mesh_id   = static_cast<uint16_t>(get_arg_val<uint32_t>(idx++));
    const uint8_t  dst_dev_id    = static_cast<uint8_t>(get_arg_val<uint32_t>(idx++));
    const uint32_t rx_noc_x      = get_arg_val<uint32_t>(idx++);
    const uint32_t rx_noc_y      = get_arg_val<uint32_t>(idx++);
    const uint32_t sem_l1_addr   = get_arg_val<uint32_t>(idx++);

    auto sender = WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(idx);
    volatile tt_l1_ptr PACKET_HEADER_TYPE* packet_header = PacketHeaderPool::allocate_header();
    sender.open<true>();

    const uint64_t dst_noc_addr = safe_get_noc_addr(rx_noc_x, rx_noc_y, dst_base_addr, 0);
    const uint64_t sem_noc_addr = safe_get_noc_addr(rx_noc_x, rx_noc_y, sem_l1_addr, 0);

    // Dummy multicast range (1-hop east)
    MeshMcastRange ranges{1, 0, 0, 0};

    // --- Mesh multicast families ---

    // 1. fabric_multicast_noc_scatter_write (mesh)
    fabric_multicast_noc_scatter_write(
        &sender, packet_header, dst_dev_id, dst_mesh_id, ranges,
        src_l1_addr, total_size,
        NocUnicastScatterCommandHeader({dst_noc_addr, dst_noc_addr}, {static_cast<uint16_t>(total_size / 2)}));

    // 2. fabric_multicast_noc_fused_unicast_with_atomic_inc (mesh)
    fabric_multicast_noc_fused_unicast_with_atomic_inc(
        &sender, packet_header, dst_dev_id, dst_mesh_id, ranges,
        src_l1_addr, total_size,
        NocUnicastAtomicIncFusedCommandHeader{dst_noc_addr, sem_noc_addr, 1, true});

    // 3. fabric_multicast_noc_fused_scatter_write_atomic_inc (mesh)
    fabric_multicast_noc_fused_scatter_write_atomic_inc(
        &sender, packet_header, dst_dev_id, dst_mesh_id, ranges,
        src_l1_addr, total_size,
        NocUnicastScatterAtomicIncFusedCommandHeader(
            {dst_noc_addr, dst_noc_addr}, sem_noc_addr,
            {static_cast<uint16_t>(total_size / 2)}, 1, true));

    // --- Linear multicast families ---
    uint8_t start_distance = 1;
    uint8_t range = 1;

    // 4. fabric_multicast_noc_scatter_write (linear)
    tt::tt_fabric::linear::experimental::fabric_multicast_noc_scatter_write(
        &sender, packet_header,
        src_l1_addr, total_size,
        NocUnicastScatterCommandHeader({dst_noc_addr, dst_noc_addr}, {static_cast<uint16_t>(total_size / 2)}),
        start_distance, range);

    // 5. fabric_multicast_noc_fused_unicast_with_atomic_inc (linear)
    tt::tt_fabric::linear::experimental::fabric_multicast_noc_fused_unicast_with_atomic_inc(
        &sender, packet_header,
        src_l1_addr, total_size,
        NocUnicastAtomicIncFusedCommandHeader{dst_noc_addr, sem_noc_addr, 1, true},
        start_distance, range);

    // 6. fabric_multicast_noc_fused_scatter_write_atomic_inc (linear)
    tt::tt_fabric::linear::experimental::fabric_multicast_noc_fused_scatter_write_atomic_inc(
        &sender, packet_header,
        src_l1_addr, total_size,
        NocUnicastScatterAtomicIncFusedCommandHeader(
            {dst_noc_addr, dst_noc_addr}, sem_noc_addr,
            {static_cast<uint16_t>(total_size / 2)}, 1, true),
        start_distance, range);

    sender.close();
}
