// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// ws_expert fabric weight landing, receiving chip, credit relay (BRISC): the group cores write their freed-expert
// counts into this core's icred words (word c for core c = g * GC + j, by NoC inline write); each change is forwarded
// by fabric inline write to the same word of the sender core that lands core c's slices (one connection per sender
// chip). Ends once every core's final count (its group's experts) has been forwarded.
// RT: 0 nconn, then per connection: dst mesh id, dst chip id, the fabric connection args; then NC words
//     (sender connection index | x << 8 | y << 16)
// CT: 0 NC, 1 GC, 2 G, 3 NEXP, 4 icred
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
    constexpr uint32_t icred = get_compile_time_arg_val(4);
    size_t arg_idx = 0;
    const uint32_t nconn = get_arg_val<uint32_t>(arg_idx++);
    tt::tt_fabric::WorkerToFabricEdmSender conn[MAXC];
    uint16_t mesh[MAXC];
    uint8_t chip[MAXC];
    for (uint32_t k = 0; k < nconn; ++k) {
        mesh[k] = get_arg_val<uint32_t>(arg_idx++);
        chip[k] = get_arg_val<uint32_t>(arg_idx++);
        conn[k] = tt::tt_fabric::WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(arg_idx);
    }
    const size_t map0 = arg_idx;
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
            const uint32_t a = get_arg_val<uint32_t>(map0 + c);
            const uint32_t k = a & 0xFF, sx = (a >> 8) & 0xFF, sy = (a >> 16) & 0xFF;
            fabric_unicast_noc_unicast_inline_write(
                &conn[k],
                hdr,
                chip[k],
                mesh[k],
                tt::tt_fabric::NocUnicastInlineWriteCommandHeader{get_noc_addr(sx, sy, icred + c * 16), v});
            noc_async_writes_flushed();  // (header out of L1 before it is rewritten)
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
