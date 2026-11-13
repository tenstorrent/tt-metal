// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// ws_expert fabric weight landing, sender chip, fabric sender (NCRISC): one fabric connection (one link toward the
// receiving chip). For every expert in the global order and each of this connection's target cores of the expert's
// group: once the core has freed the slot (credit word c >= n - 1, written over the fabric by the receiver's relay),
// sends the slice from CB 0 one CHT-tile packet at a time into the core's 2-slot rings (slot n & 1: gate/up part to
// wgu_ring, down part to wd_ring); the last packet is fused with an atomic inc of the core's warr word.
// RT: 0 dst mesh id, 1 dst chip id, 2 ntargets, 3.. targets (c | x << 8 | y << 16), then the fabric connection args
// CT: 0 NEXP, 1 KT, 2 KTD, 3 DCOLS, 4 CHT, 5 G, 6 GC, 7 wgu_ring, 8 wd_ring, 9 warr, 10 icred
#include "api/dataflow/dataflow_api.h"
#include "tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_worker_adapters.hpp"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"
#include "tt_metal/fabric/hw/inc/mesh/api.h"
using namespace tt::tt_fabric::mesh::experimental;

void kernel_main() {
    constexpr uint32_t NEXP = get_compile_time_arg_val(0), KT = get_compile_time_arg_val(1);
    constexpr uint32_t KTD = get_compile_time_arg_val(2), DCOLS = get_compile_time_arg_val(3);
    constexpr uint32_t CHT = get_compile_time_arg_val(4), G = get_compile_time_arg_val(5);
    constexpr uint32_t GC = get_compile_time_arg_val(6);
    constexpr uint32_t wgu_ring = get_compile_time_arg_val(7), wd_ring = get_compile_time_arg_val(8);
    constexpr uint32_t warr = get_compile_time_arg_val(9), icred = get_compile_time_arg_val(10);
    constexpr uint32_t WGU = KT * 2, WDN = KTD * DCOLS, WTOT = WGU + WDN;
    const uint32_t tb = get_tile_size(0);
    const uint32_t CHB = CHT * tb;

    const uint16_t dst_mesh = get_arg_val<uint32_t>(0);
    const uint8_t dst_chip = get_arg_val<uint32_t>(1);
    const uint32_t nt = get_arg_val<uint32_t>(2);
    size_t arg_idx = 3 + nt;
    auto conn = tt::tt_fabric::WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(arg_idx);
    conn.open();
    PacketHeaderPool::reset();
    volatile tt_l1_ptr PACKET_HEADER_TYPE* hdr[2] = {
        PacketHeaderPool::allocate_header(), PacketHeaderPool::allocate_header()};
    uint32_t hi = 0;

    for (uint32_t e = 0; e < NEXP; ++e) {
        const uint32_t g = e % G, n = e / G;
        for (uint32_t t = 0; t < nt; ++t) {
            const uint32_t a = get_arg_val<uint32_t>(3 + t);
            const uint32_t c = a & 0xFF, tx = (a >> 8) & 0xFF, ty = (a >> 16) & 0xFF;
            if (c / GC != g) {
                continue;
            }
            if (n >= 2) {  // the core freed expert n - 2's slot
                volatile tt_l1_ptr uint32_t* cr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(icred + c * 16);
                do {
                    invalidate_l1_cache();
                } while (*cr < n - 1);
            }
            for (uint32_t u0 = 0; u0 < WTOT; u0 += CHT) {
                const uint32_t dst =
                    u0 < WGU ? wgu_ring + ((n & 1) * WGU + u0) * tb : wd_ring + ((n & 1) * WDN + u0 - WGU) * tb;
                cb_wait_front(0, CHT);
                volatile tt_l1_ptr PACKET_HEADER_TYPE* h = hdr[hi];
                hi ^= 1;
                if (u0 + CHT < WTOT) {
                    fabric_unicast_noc_unicast_write(
                        &conn,
                        h,
                        dst_chip,
                        dst_mesh,
                        get_read_ptr(0),
                        CHB,
                        tt::tt_fabric::NocUnicastCommandHeader{get_noc_addr(tx, ty, dst)});
                } else {
                    fabric_unicast_noc_fused_unicast_with_atomic_inc(
                        &conn,
                        h,
                        dst_chip,
                        dst_mesh,
                        get_read_ptr(0),
                        CHB,
                        tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader{
                            get_noc_addr(tx, ty, dst), get_noc_addr(tx, ty, warr), 1, true});
                }
                noc_async_writes_flushed();  // (payload read out of CB 0 and the header out of L1)
                cb_pop_front(0, CHT);
            }
        }
    }
    conn.close();
    noc_async_full_barrier();
}
