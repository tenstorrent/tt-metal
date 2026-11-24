// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// A2A sender (one core a fabric connection: a direction and link of this chip): ROWS rows to each of its NDST
// destination chips, interleaved over the destinations, each row one fabric unicast of PB bytes from this core's L1
// into a landing core of the destination (row r of a destination on landing core r % NLAND; the slot is overwritten:
// traffic only), the last row to each destination fused with an atomic inc of the destination's counter.
// CT: 0 PB, 1 NDST, 2 ROWS, 3 NLAND, 4 the L1 buffer (source here, slot on the landing cores), 5 the counter address
// RT: 0 mesh id, 1 the counter core's xy, 2 .. 1 + NLAND the landing cores' xy (x | y << 8), then NDST chip ids, then
//     the fabric connection's args
#include "api/dataflow/dataflow_api.h"
#include "tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_worker_adapters.hpp"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"
#include "tt_metal/fabric/hw/inc/mesh/api.h"
using namespace tt::tt_fabric::mesh::experimental;

void kernel_main() {
    constexpr uint32_t PB = get_compile_time_arg_val(0), NDST = get_compile_time_arg_val(1);
    constexpr uint32_t ROWS = get_compile_time_arg_val(2), NLAND = get_compile_time_arg_val(3);
    constexpr uint32_t buf = get_compile_time_arg_val(4), cnt = get_compile_time_arg_val(5);
    const uint16_t mesh = get_arg_val<uint32_t>(0);
    const uint32_t cxy = get_arg_val<uint32_t>(1);
    uint32_t lxy[NLAND];
    for (uint32_t i = 0; i < NLAND; ++i) {
        lxy[i] = get_arg_val<uint32_t>(2 + i);
    }
    size_t a = 2 + NLAND + NDST;
    auto conn = tt::tt_fabric::WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(a);
    conn.open();
    PacketHeaderPool::reset();
    volatile tt_l1_ptr PACKET_HEADER_TYPE* h[NDST];
    for (uint32_t d = 0; d < NDST; ++d) {
        h[d] = PacketHeaderPool::allocate_header();
        (void)fabric_set_unicast_route(h[d], static_cast<uint8_t>(get_arg_val<uint32_t>(2 + NLAND + d)), mesh);
    }
    const uint64_t c_noc = get_noc_addr(cxy & 0xFF, cxy >> 8, cnt);
    for (uint32_t r = 0; r < ROWS; ++r) {
        const uint32_t l = lxy[r % NLAND];
        const uint64_t dst = get_noc_addr(l & 0xFF, l >> 8, buf);
        for (uint32_t d = 0; d < NDST; ++d) {
            if (r + 1 == ROWS) {
                h[d]->to_noc_fused_unicast_write_atomic_inc(
                    tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader{dst, c_noc, 1, false}, PB);
            } else {
                h[d]->to_noc_unicast_write(tt::tt_fabric::NocUnicastCommandHeader{dst}, PB);
            }
            conn.wait_for_empty_write_slot();
            conn.send_current_slot_non_blocking(buf, PB, reinterpret_cast<uint32_t>(h[d]));
        }
    }
    noc_async_full_barrier();
    conn.close();
}
