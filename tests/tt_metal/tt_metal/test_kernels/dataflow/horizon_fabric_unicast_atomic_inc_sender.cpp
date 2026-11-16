// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Horizon fabric unicast atomic increment: adds value to the 32-bit word at (dst_noc_x, dst_noc_y, dst_l1_addr)
// on device dst_dev_id, count times, through fabric_unicast_noc_unicast_atomic_inc().

#include "api/dataflow/dataflow_api.h"
#include "experimental/kernel_args.h"
#include "tt_metal/fabric/hw/inc/mesh/api.h"
#include "tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_worker_adapters.hpp"
#include "hostdev/fabric_edm_packet_header.hpp"

void kernel_main() {
    const uint32_t dst_l1_addr = get_arg(args::dst_l1_addr);
    const uint32_t dst_noc_x = get_arg(args::dst_noc_x);
    const uint32_t dst_noc_y = get_arg(args::dst_noc_y);
    const uint32_t dst_dev_id = get_arg(args::dst_dev_id);
    const uint32_t count = get_arg(args::count);
    const uint32_t value = get_arg(args::value);

    const uint64_t noc_addr =
        get_noc_addr(static_cast<uint8_t>(dst_noc_x), static_cast<uint8_t>(dst_noc_y), dst_l1_addr);
    for (uint32_t i = 0; i < count; i++) {
        tt::tt_fabric::mesh::experimental::fabric_unicast_noc_unicast_atomic_inc(
            (tt_l1_ptr tt::tt_fabric::WorkerToFabricEdmSender*)nullptr,
            (volatile PACKET_HEADER_TYPE*)nullptr,
            static_cast<uint8_t>(dst_dev_id),
            0u,
            tt::tt_fabric::NocUnicastAtomicIncCommandHeader{noc_addr, value});
    }
    noc_async_write_barrier();
    noc_async_atomic_barrier();
}
