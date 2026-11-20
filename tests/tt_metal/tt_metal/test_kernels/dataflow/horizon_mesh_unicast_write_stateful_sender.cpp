// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Stateful unicast write sender.
// Calls set_state once then with_state twice to push the same payload to two different
// L1 addresses on the destination device.

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
    const uint32_t size = get_arg(args::size);

    const uint64_t noc_addr_a =
        get_noc_addr(static_cast<uint8_t>(dst_noc_x), static_cast<uint8_t>(dst_noc_y), dst_l1_addr_a);
    const uint64_t noc_addr_b =
        get_noc_addr(static_cast<uint8_t>(dst_noc_x), static_cast<uint8_t>(dst_noc_y), dst_l1_addr_b);

    // Stack-allocate the packet header used to carry stateful routing info.
    alignas(16) uint8_t hdr_storage[sizeof(PACKET_HEADER_TYPE)]{};
    volatile PACKET_HEADER_TYPE* ph = reinterpret_cast<volatile PACKET_HEADER_TYPE*>(hdr_storage);

    using namespace tt::tt_fabric;
    using namespace tt::tt_fabric::mesh::experimental;

    // set_state: store routing + first destination address + payload size.
    fabric_unicast_noc_unicast_write_set_state<UnicastWriteUpdateMask::All>(
        ph, static_cast<uint8_t>(dst_dev_id), 0u, NocUnicastCommandHeader{noc_addr_a}, static_cast<uint16_t>(size));

    // with_state (no mask updates): send to addr_a using the stored state.
    fabric_unicast_noc_unicast_write_with_state<UnicastWriteUpdateMask::None>(
        (tt_l1_ptr WorkerToFabricEdmSender*)nullptr, ph, src_l1_addr, NocUnicastCommandHeader{}, 0u);

    // with_state (DstAddr update): change destination to addr_b and send.
    fabric_unicast_noc_unicast_write_with_state<UnicastWriteUpdateMask::DstAddr>(
        (tt_l1_ptr WorkerToFabricEdmSender*)nullptr, ph, src_l1_addr, NocUnicastCommandHeader{noc_addr_b}, 0u);

    noc_async_write_barrier();
}
