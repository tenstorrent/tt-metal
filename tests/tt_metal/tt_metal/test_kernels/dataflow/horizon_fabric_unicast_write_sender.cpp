// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Horizon fabric unicast write — sender side (runs on chip 0).
//
// 1. Fills the local source buffer at src_l1_addr with base_value+i for i in [0, num_words).
// 2. Calls mesh::experimental::fabric_unicast_noc_unicast_write() to push num_words*4 bytes to chip dst_dev_id,
//    NOC (dst_noc_x, dst_noc_y), L1 offset dst_l1_addr.
// 3. Issues noc_async_write_barrier() to ensure the write has left chip 0.
// 4. Exits; host calls Finish(chip0) which waits for this point.
//
// client_interface and packet_header are passed as nullptr — the Horizon crossbar path
// does not dereference them (it uses the flat-address routing table instead).

#include "api/dataflow/dataflow_api.h"
#include "experimental/kernel_args.h"
#include "tt_metal/fabric/hw/inc/mesh/api.h"
#include "tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_worker_adapters.hpp"
#include "hostdev/fabric_edm_packet_header.hpp"

void kernel_main() {
    const uint32_t num_words = get_arg(args::num_words);
    const uint32_t src_l1_addr = get_arg(args::src_l1_addr);
    const uint32_t dst_l1_addr = get_arg(args::dst_l1_addr);
    const uint32_t dst_noc_x = get_arg(args::dst_noc_x);
    const uint32_t dst_noc_y = get_arg(args::dst_noc_y);
    const uint32_t dst_dev_id = get_arg(args::dst_dev_id);
    const uint32_t base_value = get_arg(args::base_value);

    // Fill source buffer with the test pattern.
    volatile uint32_t* src = reinterpret_cast<volatile uint32_t*>(src_l1_addr);
    for (uint32_t i = 0; i < num_words; i++) {
        src[i] = base_value + i;
    }

    // get_noc_addr() returns a flat ATT address when NOC_ATT_ENABLED is defined
    // (i.e. whenever TT_METAL_NOC_ATT=horizon_2x3 is set at runtime), or standard XY otherwise.
    // Format under ATT: REMOTE_WINDOW.compare | (selector<<HORIZON_ATT_ENDPOINT_SHIFT) | offset.
    // api_impl.h decodes the selector back to (dev_id, local_y) for the routing table lookup.
    const uint64_t noc_address =
        get_noc_addr(static_cast<uint8_t>(dst_noc_x), static_cast<uint8_t>(dst_noc_y), dst_l1_addr);
    tt::tt_fabric::NocUnicastCommandHeader cmd{noc_address};

    // Issue the fabric write.  client_interface and packet_header are unused on the
    // Horizon crossbar path; nullptr satisfies the template constraint.
    tt::tt_fabric::mesh::experimental::fabric_unicast_noc_unicast_write(
        (tt_l1_ptr tt::tt_fabric::WorkerToFabricEdmSender*)nullptr,
        (volatile PACKET_HEADER_TYPE*)nullptr,
        static_cast<uint8_t>(dst_dev_id),
        0u,  // dst_mesh_id
        src_l1_addr,
        num_words * sizeof(uint32_t),
        cmd);

    noc_async_write_barrier();
}
