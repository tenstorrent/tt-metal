// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// Horizon fabric unicast write — sender side.
// Calls fabric_unicast_noc_unicast_write() to push size bytes of this core's L1 at src_l1_addr
// to (dst_noc_x, dst_noc_y, dst_l1_addr) on device dst_dev_id, using the routing table
// written to ROUTING_TABLE_BASE by SetFabricConfig(FABRIC_1D) during device open.
// The host pre-fills src_l1_addr with the test pattern before dispatching this kernel.

#include "api/dataflow/dataflow_api.h"
#include "experimental/kernel_args.h"
#include "tt_metal/fabric/hw/inc/mesh/api.h"
#include "tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_worker_adapters.hpp"
#include "hostdev/fabric_edm_packet_header.hpp"

void kernel_main() {
    const uint32_t src_l1_addr = get_arg(args::src_l1_addr);
    const uint32_t dst_l1_addr = get_arg(args::dst_l1_addr);
    const uint32_t dst_noc_x = get_arg(args::dst_noc_x);
    const uint32_t dst_noc_y = get_arg(args::dst_noc_y);
    const uint32_t dst_dev_id = get_arg(args::dst_dev_id);
    const uint32_t size = get_arg(args::size);

    const uint64_t noc_addr =
        get_noc_addr(static_cast<uint8_t>(dst_noc_x), static_cast<uint8_t>(dst_noc_y), dst_l1_addr);
    tt::tt_fabric::NocUnicastCommandHeader cmd{noc_addr};

    tt::tt_fabric::mesh::experimental::fabric_unicast_noc_unicast_write(
        (tt_l1_ptr tt::tt_fabric::WorkerToFabricEdmSender*)nullptr,
        (volatile PACKET_HEADER_TYPE*)nullptr,
        static_cast<uint8_t>(dst_dev_id),
        0u,
        src_l1_addr,
        size,
        cmd);

    noc_async_write_barrier();
}
