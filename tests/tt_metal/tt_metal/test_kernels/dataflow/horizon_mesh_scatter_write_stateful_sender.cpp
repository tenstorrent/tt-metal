// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Stateful scatter write sender.
// Calls set_state once then with_state twice:
//   - First with_state: send two chunks to (addr_a, addr_b) on dst_dev_id.
//   - Second with_state: update DstAddrs to (addr_c, addr_d) and send again.

#include "api/dataflow/dataflow_api.h"
#include "experimental/kernel_args.h"
#include "tt_metal/fabric/hw/inc/mesh/api.h"
#include "tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_worker_adapters.hpp"
#include "hostdev/fabric_edm_packet_header.hpp"

void kernel_main() {
    const uint32_t src_l1_addr = get_arg(args::src_l1_addr);
    const uint32_t dst_noc_x = get_arg(args::dst_noc_x);
    const uint32_t dst_noc_y = get_arg(args::dst_noc_y);
    const uint32_t dst_dev_id = get_arg(args::dst_dev_id);
    const uint32_t dst_l1_addr_a = get_arg(args::dst_l1_addr_a);
    const uint32_t dst_l1_addr_b = get_arg(args::dst_l1_addr_b);
    const uint32_t dst_l1_addr_c = get_arg(args::dst_l1_addr_c);
    const uint32_t dst_l1_addr_d = get_arg(args::dst_l1_addr_d);
    const uint32_t chunk_size_0 = get_arg(args::chunk_size_0);
    const uint32_t total_size = get_arg(args::total_size);

    const uint64_t noc_addr_a =
        get_noc_addr(static_cast<uint8_t>(dst_noc_x), static_cast<uint8_t>(dst_noc_y), dst_l1_addr_a);
    const uint64_t noc_addr_b =
        get_noc_addr(static_cast<uint8_t>(dst_noc_x), static_cast<uint8_t>(dst_noc_y), dst_l1_addr_b);
    const uint64_t noc_addr_c =
        get_noc_addr(static_cast<uint8_t>(dst_noc_x), static_cast<uint8_t>(dst_noc_y), dst_l1_addr_c);
    const uint64_t noc_addr_d =
        get_noc_addr(static_cast<uint8_t>(dst_noc_x), static_cast<uint8_t>(dst_noc_y), dst_l1_addr_d);

    // Build the initial scatter header (2 chunks, addresses a+b).
    tt::tt_fabric::NocUnicastScatterCommandHeader init_hdr{
        {noc_addr_a, noc_addr_b}, {static_cast<uint16_t>(chunk_size_0)}};

    alignas(16) uint8_t hdr_storage[sizeof(PACKET_HEADER_TYPE)]{};
    volatile PACKET_HEADER_TYPE* ph = reinterpret_cast<volatile PACKET_HEADER_TYPE*>(hdr_storage);

    using namespace tt::tt_fabric;
    using namespace tt::tt_fabric::mesh::experimental;

    // set_state: store routing + addresses (a,b) + chunk sizes + payload size.
    fabric_unicast_noc_scatter_write_set_state<UnicastScatterWriteUpdateMask::All>(
        ph, static_cast<uint8_t>(dst_dev_id), 0u, init_hdr, static_cast<uint16_t>(total_size));

    // with_state (no updates): send chunks to addr_a and addr_b.
    fabric_unicast_noc_scatter_write_with_state<UnicastScatterWriteUpdateMask::None>(
        (tt_l1_ptr WorkerToFabricEdmSender*)nullptr, ph, src_l1_addr, init_hdr, 0u);

    // Build updated scatter header (same chunk sizes, new addresses c+d).
    tt::tt_fabric::NocUnicastScatterCommandHeader update_hdr{{noc_addr_c, noc_addr_d}};

    // with_state (DstAddrs update): send chunks to addr_c and addr_d.
    fabric_unicast_noc_scatter_write_with_state<UnicastScatterWriteUpdateMask::DstAddrs>(
        (tt_l1_ptr WorkerToFabricEdmSender*)nullptr, ph, src_l1_addr, update_hdr, 0u);

    noc_async_write_barrier();
}
