// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// A Tensix worker that sends fabric packets, knowing NOTHING about the bridge -- cable or host is
// decided inside the router. If this file needs a bridge-specific line, the design has leaked.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "fabric/fabric_edm_packet_header.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_worker_adapters.hpp"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"
#include "tt_metal/fabric/hw/inc/noc_addr.h"

using namespace tt;
using namespace tt::tt_fabric;

void kernel_main() {
    constexpr uint32_t PAYLOAD_BYTES = get_compile_time_arg_val(0);
    constexpr uint32_t ITERATIONS = get_compile_time_arg_val(1);
    constexpr uint32_t SRC_L1_ADDR = get_compile_time_arg_val(2);
    constexpr uint32_t STATUS_ADDR = get_compile_time_arg_val(3);  // {stage, frames sent}, read on a stall

    size_t idx = 0;
    const uint32_t dst_noc_x = get_arg_val<uint32_t>(idx++);
    const uint32_t dst_noc_y = get_arg_val<uint32_t>(idx++);
    const uint32_t dst_l1_addr = get_arg_val<uint32_t>(idx++);
    // USED UNDER 2D, ignored under 1D. The 1D overload routes by device id alone, intra-mesh;
    // the 2D header (HybridMeshPacketHeader) needs the mesh too. The host packs it either way.
    const uint16_t dst_mesh_id = static_cast<uint16_t>(get_arg_val<uint32_t>(idx++));
    const uint16_t dst_dev_id = static_cast<uint16_t>(get_arg_val<uint32_t>(idx++));

    // Binds this core to a routing plane and a link. The host packed the args; which link it
    // picked is what decides whether the bridge is in the path.
    auto sender = WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(idx);

    volatile tt_l1_ptr PACKET_HEADER_TYPE* header = PacketHeaderPool::allocate_header();
    // Route setters differ: 2 args under 1D, 3 under 2D. Switch on ROUTING_MODE, since tt-metal
    // never defines FABRIC_2D for a kernel.

// An undefined macro is 0 in #if, so losing this include would read `162 & 0` and silently route
// every packet as 1D. Fail the build instead.
#ifndef ROUTING_MODE_2D
#error "ROUTING_MODE_2D not in scope (fabric_routing_mode.h) -- the branch below would silently take 1D"
#endif
#if defined(ROUTING_MODE) && ((ROUTING_MODE & ROUTING_MODE_2D) != 0)
    (void)fabric_set_unicast_route(header, dst_dev_id, dst_mesh_id);
#else
    (void)dst_mesh_id;
    (void)fabric_set_unicast_route(header, dst_dev_id);
#endif

    // Pattern written once, outside the loop, or the memset lands in the measured window. Words
    // [0] counter, [1..2] this core's rdcycle -- a delta against itself only, not host time.
    volatile tt_l1_ptr uint32_t* const payload = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(SRC_L1_ADDR);
    for (uint32_t k = 3; k < PAYLOAD_BYTES / sizeof(uint32_t); ++k) {
        payload[k] = 0xC0DE0000u | (k & 0xFFFFu);
    }

    const uint64_t dst_noc_addr = safe_get_noc_addr(dst_noc_x, dst_noc_y, dst_l1_addr, /*noc=*/0);

    volatile tt_l1_ptr uint32_t* const status = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(STATUS_ADDR);
    status[0] = 1;  // started: stuck here means open() never completed
    sender.open<true>();
    status[0] = 2;  // connected

    for (uint32_t i = 0; i < ITERATIONS; ++i) {
        payload[0] = i;
        const uint64_t ts = get_timestamp();
        payload[1] = static_cast<uint32_t>(ts);
        payload[2] = static_cast<uint32_t>(ts >> 32);

        // Pace against the router's sender channel. With the bridge engaged this is where the
        // host ring's backpressure surfaces -- the router declines, so this waits longer.
        sender.wait_for_empty_write_slot();

        header->to_noc_unicast_write(NocUnicastCommandHeader{dst_noc_addr}, PAYLOAD_BYTES);

        // Payload then header: the header is what completes the packet, so it goes last for
        // the same reason the bridge's guard does.
        sender.send_payload_without_header_non_blocking_from_address(SRC_L1_ADDR, PAYLOAD_BYTES);
        sender.send_payload_blocking_from_address(reinterpret_cast<uint32_t>(header), sizeof(PACKET_HEADER_TYPE));
        status[1] = i + 1;
    }
    status[0] = 3;  // all sent

    noc_async_writes_flushed();
    sender.close();
}
