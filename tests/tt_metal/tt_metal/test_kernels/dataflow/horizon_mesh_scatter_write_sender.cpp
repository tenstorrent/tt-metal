// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Stateless scatter write sender.
// Writes the first chunk_size_a bytes of src_l1_addr to dst_l1_addr_a on the destination
// device, and the remaining bytes to dst_l1_addr_b, using fabric_unicast_noc_scatter_write.

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
    const uint32_t chunk_size_a = get_arg(args::chunk_size_a);
    const uint32_t total_size = get_arg(args::total_size);

    const uint64_t noc_addr_a =
        get_noc_addr(static_cast<uint8_t>(dst_noc_x), static_cast<uint8_t>(dst_noc_y), dst_l1_addr_a);
    const uint64_t noc_addr_b =
        get_noc_addr(static_cast<uint8_t>(dst_noc_x), static_cast<uint8_t>(dst_noc_y), dst_l1_addr_b);

    tt::tt_fabric::NocUnicastScatterCommandHeader scatter_hdr(
        {noc_addr_a, noc_addr_b}, {static_cast<uint16_t>(chunk_size_a)});

    tt::tt_fabric::mesh::experimental::fabric_unicast_noc_scatter_write<tt::tt_fabric::WorkerToFabricEdmSender, false>(
        (tt_l1_ptr tt::tt_fabric::WorkerToFabricEdmSender*)nullptr,
        (volatile PACKET_HEADER_TYPE*)nullptr,
        static_cast<uint8_t>(dst_dev_id),
        0u,
        src_l1_addr,
        total_size,
        scatter_hdr);

    noc_async_write_barrier();
}
