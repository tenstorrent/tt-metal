// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// ws_expert all-chip weight landing, credit relay (BRISC). This chip's group cores write their freed-expert counts into
// this core's words c (c = g * GC + j); each change is forwarded as word me * NC + c to every other chip's owner-side
// sender that covers this chip (fabric inline write; first hop = the connection in the routing direction) and to this
// chip's local lander (NoC inline write). Ends once every core's final count has been forwarded.
// RT: 0 my chip id, 1 nconn, per connection: the fabric connection args (first = its eth channel); then NCHIP words
//     (dst chip | x << 8 | y << 16: the sender core on that chip that covers this one; for this chip the local lander)
// CT: 0 NC, 1 GC, 2 G, 3 NEXP, 4 icred, 5 NCHIP, 6 mesh id
#include "api/dataflow/dataflow_api.h"
#include "tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_worker_adapters.hpp"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"
#include "tt_metal/fabric/hw/inc/mesh/api.h"
using namespace tt::tt_fabric::mesh::experimental;

constexpr uint32_t MAXC = 4;

void kernel_main() {
    constexpr uint32_t NC = get_compile_time_arg_val(0), GC = get_compile_time_arg_val(1);
    constexpr uint32_t G = get_compile_time_arg_val(2), NEXP = get_compile_time_arg_val(3);
    constexpr uint32_t icred = get_compile_time_arg_val(4), NCHIP = get_compile_time_arg_val(5);
    constexpr uint32_t mesh = get_compile_time_arg_val(6);
    size_t arg_idx = 0;
    const uint32_t me = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t nconn = get_arg_val<uint32_t>(arg_idx++);
    tt::tt_fabric::WorkerToFabricEdmSender conn[MAXC];
    uint32_t dir[MAXC];
    for (uint32_t k = 0; k < nconn; ++k) {
        dir[k] = get_router_direction(get_arg_val<uint32_t>(arg_idx));
        conn[k] = tt::tt_fabric::WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(arg_idx);
    }
    const size_t map0 = arg_idx;
    uint32_t via[NCHIP];  // connection to each chip's first hop
    for (uint32_t o = 0; o < NCHIP; ++o) {
        via[o] = 0;
        if (o != me) {
            const uint32_t d = get_next_hop_router_direction(mesh, o);
            for (uint32_t k = 0; k < nconn; ++k) {
                if (dir[k] == d) {
                    via[o] = k;
                }
            }
        }
    }
    for (uint32_t k = 0; k < nconn; ++k) {
        conn[k].open();
    }
    PacketHeaderPool::reset();
    volatile tt_l1_ptr PACKET_HEADER_TYPE* hdr = PacketHeaderPool::allocate_header();

    uint32_t fwd[NC];
    for (uint32_t c = 0; c < NC; ++c) {
        fwd[c] = 0;
    }
    uint32_t done = 0;
    while (done < NC) {
        invalidate_l1_cache();
        for (uint32_t c = 0; c < NC; ++c) {
            const uint32_t v = *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(icred + c * 16);
            if (v <= fwd[c]) {
                continue;
            }
            const uint32_t word = icred + (me * NC + c) * 16;
            for (uint32_t o = 0; o < NCHIP; ++o) {
                const uint32_t a = get_arg_val<uint32_t>(map0 + o);
                const uint32_t sx = (a >> 8) & 0xFF, sy = (a >> 16) & 0xFF;
                if (o == me) {
                    noc_inline_dw_write(get_noc_addr(sx, sy, word), v);
                    continue;
                }
                fabric_unicast_noc_unicast_inline_write(
                    &conn[via[o]],
                    hdr,
                    o,
                    mesh,
                    tt::tt_fabric::NocUnicastInlineWriteCommandHeader{get_noc_addr(sx, sy, word), v});
                noc_async_writes_flushed();  // (header out of L1 before it is rewritten)
            }
            fwd[c] = v;
            const uint32_t g = c / GC;
            if (v >= (NEXP - g + G - 1) / G) {  // this core's last expert
                ++done;
            }
        }
    }
    for (uint32_t k = 0; k < nconn; ++k) {
        conn[k].close();
    }
    noc_async_full_barrier();
}
