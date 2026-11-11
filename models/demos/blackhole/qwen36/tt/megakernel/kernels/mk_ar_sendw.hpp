// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Megakernel AR, AR_SEND writer phase (BRISC, link-sender cores): mkar::SendWriter. See mk_ar_send.hpp for the engine
// contract and the config macros. init() opens this core's fabric connection ONCE per pass (open_start ... open_finish);
// iter(it) sends one all-reduce's reduce-scatter unicasts and all-gather multicasts; fini() closes the connection.
// The object (connection state + packet-header pointers) must live for the whole pass: make it file-scope static.
#pragma once
#include "mk_ar_send.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_connection_manager.hpp"
#include "tt_metal/fabric/hw/inc/noc_addr.h"
#include "cpp/ttnn/operations/ccl/common/kernels/minimal_ccl_common.hpp"

namespace mkar {

#if defined(GDN_GTEXT2) && GDN_GTEXT2 && !defined(MKAR_GT2_SP_INLINE)
// GTEXT2 (GDN-hosted sender on the AR corners only): one out-of-line copy of the packet send (it was inlined at every RS / AG
// fwd / bwd site, ~1.5 KB of BRISC text); same header writes, same payload send, same order
__attribute__((noinline)) void send_pkt(tt::tt_fabric::WorkerToFabricEdmSender& connection,
#else
FORCE_INLINE void send_pkt(tt::tt_fabric::WorkerToFabricEdmSender& connection,
#endif
                           volatile PACKET_HEADER_TYPE* pkt_hdr,
                           uint32_t l1_read_addr, uint64_t dest_noc_addr, uint64_t sem_noc_addr, uint32_t payload_bytes,
                           bool last) {
    if (last) {
        pkt_hdr->to_noc_fused_unicast_write_atomic_inc(
            tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader{dest_noc_addr, sem_noc_addr, 1, false}, payload_bytes);
    } else {
        pkt_hdr->to_noc_unicast_write(tt::tt_fabric::NocUnicastCommandHeader{dest_noc_addr}, payload_bytes);
    }
    perform_payload_send(connection, l1_read_addr, payload_bytes, pkt_hdr);
}

struct SendWriter {
    uint32_t cbase = 0, rbase = 0, num_rs = 0, num_ag = 0, fwd_range = 0, bwd_range = 0;
    uint32_t rs_arg_base = 0, ag_arg_base = 0, sem_addr = 0, out_addr = 0, slot_addr = 0, red_base = 0;
    FabricConnectionManager fab;
    volatile PACKET_HEADER_TYPE* hf = nullptr;
    volatile PACKET_HEADER_TYPE* hb = nullptr;

    FORCE_INLINE uint32_t carg(uint32_t i) const {
        if constexpr (MKAR_ALL_RT) {
            return get_arg_val<uint32_t>(cbase + i);
        } else {
            return get_common_arg_val<uint32_t>(i);
        }
    }

    FORCE_INLINE void init(uint32_t common_base, uint32_t role_base) {
        cbase = common_base;
        rbase = role_base;
        size_t a = rbase;
        num_rs = get_arg_val<uint32_t>(a++);
        num_ag = get_arg_val<uint32_t>(a++);
        fwd_range = get_arg_val<uint32_t>(a++);
        bwd_range = get_arg_val<uint32_t>(a++);
        rs_arg_base = a;
        a += 2 * num_rs;
        ag_arg_base = a;
        a += num_ag;
        fab = FabricConnectionManager::build_from_args<FabricConnectionManager::BUILD_AND_OPEN_CONNECTION_START_ONLY>(a);
        sem_addr = carg(ttnn::ccl::AllReduceSemaphoreCommonArgs::semaphore);
        out_addr = carg(1);
#if defined(MKAR_S_PUB)   // hosted (GDN24): this core does not span CB_RED; the owners publish its address at init
        {
            volatile tt_l1_ptr uint32_t* pw = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(MKAR_S_PUB);
            while (*pw == 0) {}
            red_base = *pw;
        }
#else
        red_base = get_write_ptr(MKAR_S_CB_RED);
#endif
        slot_addr = red_base + MKAR_S_MY_CHIP * MKAR_S_CHUNK_TILES * MKAR_S_IN_PAGE;
        CircularBuffer cb_hdr(MKAR_S_CB_HDR);
        cb_hdr.reserve_back(1);
        hf = reinterpret_cast<volatile PACKET_HEADER_TYPE*>(cb_hdr.get_write_ptr());
        cb_hdr.push_back(1);
        cb_hdr.reserve_back(1);
        hb = reinterpret_cast<volatile PACKET_HEADER_TYPE*>(cb_hdr.get_write_ptr());
        cb_hdr.push_back(1);
        if (fab.is_logically_connected()) {
            fab.open_finish();
        }
    }

#if defined(GDN_GTEXT2) && GDN_GTEXT2 && !defined(MKAR_GT2_SP_INLINE)
    __attribute__((noinline)) void send_ag(uint32_t j, uint32_t l1_read_addr) {  // GTEXT2: 2 call sites in iter()
#else
    FORCE_INLINE void send_ag(uint32_t j, uint32_t l1_read_addr) {
#endif
        if (fwd_range > 0) {
            hf->to_chip_multicast(tt::tt_fabric::MulticastRoutingCommandHeader{1, static_cast<uint8_t>(fwd_range)});
        }
        if (bwd_range > 0) {
            hb->to_chip_multicast(tt::tt_fabric::MulticastRoutingCommandHeader{1, static_cast<uint8_t>(bwd_range)});
        }
        const uint32_t xy = get_arg_val<uint32_t>(ag_arg_base + j);
        const uint8_t noc_x = xy >> 16, noc_y = xy & 0xFFFF;
        const uint64_t sem_noc_addr = safe_get_noc_addr(noc_x, noc_y, sem_addr, 0);
        for (uint32_t t = 0; t < MKAR_S_CHUNK_TILES; t += MKAR_S_OUT_PPP) {
            const uint32_t n = std::min<uint32_t>(MKAR_S_OUT_PPP, MKAR_S_CHUNK_TILES - t);
            const uint64_t dest = safe_get_noc_addr(noc_x, noc_y, out_addr + t * MKAR_S_OUT_PAGE, 0);
            const bool last = t + n >= MKAR_S_CHUNK_TILES;
            if (fwd_range > 0) {
                send_pkt(fab.get_forward_connection(), hf, l1_read_addr + t * MKAR_S_OUT_PAGE, dest, sem_noc_addr,
                         n * MKAR_S_OUT_PAGE, last);
            }
            if (bwd_range > 0) {
                send_pkt(fab.get_backward_connection(), hb, l1_read_addr + t * MKAR_S_OUT_PAGE, dest, sem_noc_addr,
                         n * MKAR_S_OUT_PAGE, last);
            }
            noc_async_writes_flushed();
        }
    }

    FORCE_INLINE void iter(uint32_t it) {
#if defined(MKAR_SZONE)
        DeviceZoneScopedN("S-WIT");
#endif
        CircularBuffer cb_rs(MKAR_S_CB_RS), cb_ag(MKAR_S_CB_AG);
        uint32_t ag_sent = 0;
        for (uint32_t i = 0; i < num_rs; i++) {
            if constexpr (MKAR_S_PROG) {
                while (!cb_rs.pages_available_at_front(MKAR_S_CHUNK_TILES)) {
                    if (ag_sent < num_ag && cb_ag.pages_available_at_front(MKAR_S_CHUNK_TILES)) {
                        send_ag(ag_sent++, cb_ag.get_read_ptr());
                        cb_ag.pop_front(MKAR_S_CHUNK_TILES);
                    }
                }
            }
            const uint32_t xy = get_arg_val<uint32_t>(rs_arg_base + 2 * i);
            const uint32_t route = get_arg_val<uint32_t>(rs_arg_base + 2 * i + 1);
            const uint8_t noc_x = xy >> 16, noc_y = xy & 0xFFFF;
            const bool forward = ((route >> 8) & 0xFF) == 0;
#if defined(MKAR_NOOWN)   // the destination owner has no slot for itself: slot index (route >> 16) from the host
            const uint32_t slot_addr = red_base + (route >> 16) * MKAR_S_CHUNK_TILES * MKAR_S_IN_PAGE;
#endif
            auto& connection = forward ? fab.get_forward_connection() : fab.get_backward_connection();
            volatile PACKET_HEADER_TYPE* h = forward ? hf : hb;
            h->to_chip_unicast(static_cast<uint8_t>(route & 0xFF));
            const uint64_t sem_noc_addr = safe_get_noc_addr(noc_x, noc_y, sem_addr, 0);
            cb_rs.wait_front(MKAR_S_CHUNK_TILES);
            const uint32_t src = cb_rs.get_read_ptr();
            for (uint32_t t = 0; t < MKAR_S_CHUNK_TILES; t += MKAR_S_IN_PPP) {
                const uint32_t n = std::min<uint32_t>(MKAR_S_IN_PPP, MKAR_S_CHUNK_TILES - t);
                send_pkt(connection, h, src + t * MKAR_S_IN_PAGE,
                         safe_get_noc_addr(noc_x, noc_y, slot_addr + t * MKAR_S_IN_PAGE, 0), sem_noc_addr,
                         n * MKAR_S_IN_PAGE, t + n >= MKAR_S_CHUNK_TILES);
                noc_async_writes_flushed();
            }
            cb_rs.pop_front(MKAR_S_CHUNK_TILES);
        }
        for (; ag_sent < num_ag; ag_sent++) {
            cb_ag.wait_front(MKAR_S_CHUNK_TILES);
            send_ag(ag_sent, cb_ag.get_read_ptr());
            cb_ag.pop_front(MKAR_S_CHUNK_TILES);
        }
    }

    FORCE_INLINE void fini() {
        if (fab.is_logically_connected()) {
            fab.close_start();
        }
        if (fab.is_logically_connected()) {
            fab.close_finish();
        }
        noc_async_write_barrier();
    }
};

}  // namespace mkar
