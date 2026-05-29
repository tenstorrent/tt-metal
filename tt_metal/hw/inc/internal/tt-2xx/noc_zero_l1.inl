// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Single iDMA zero-device transaction with the full cmdbuf sequence
// spelled out, plus the matching write_zeros_l1_barrier (iDMA ack spin) and
// (DM only) the L2 invalidate-after-write for cache coherence with
// program-cache-hit dirty lines from prior dispatches.

#include "internal/tt-2xx/quasar/overlay/cmdbuff_api.hpp"

#if defined(COMPILE_FOR_DM)
#include "risc_common.h"  // invalidate_l2_cache_line, L2_CACHE_LINE_SIZE
#endif

template <typename Dst>
inline void Noc::async_write_zeros(const Dst& dst, uint32_t size_bytes, const dst_args_t<Dst>& args) const {
    static_assert(
        std::is_same_v<Dst, CircularBuffer> || std::is_same_v<Dst, DataflowBuffer>,
        "noc.async_write_zeros local-L1 overload accepts CircularBuffer or DataflowBuffer only. "
        "Use the TensorAccessor overload for DRAM.");
    uint32_t local_addr = get_dst_ptr<AddressType::LOCAL_L1>(dst, args);

    // Engage the Quasar iDMA zero device (Overlay Spec §4.12). The zero mode is
    // a HW overlay on top of the iDMA copy path: same MISC.idma_en + MISC.write_trans
    // setup as iDMA copy, but with AXI_OPT_1.src_protocol = 4 and decouple_aw = 1. The
    // payload bytes coming out the back are forced to zero. Source address is
    // ignored. We reuse idma_setup_as_copy_cmdbuf_0 for the MISC setup, then flip
    // src_protocol / decouple_aw via set_axi_opt_1_cmdbuf_0.
    //
    // The cmdbuf splits the transaction into packets and round-robins them across
    // the 8 backend engines via per-packet VC autoincrement wrapping the 8 write-VCs
    // (CMDBUF_WR_REQ_VC..+7). idma_acked_cmdbuf_0 returns true only after every
    // backend packet has acked.
    overlay::reset_cmdbuf_0();
    overlay::idma_setup_as_copy_cmdbuf_0(/*wrapping_en=*/false);              // MISC.idma_en + MISC.write_trans
    overlay::set_axi_opt_1_cmdbuf_0(/*src_protocol=*/4, /*decouple_aw=*/1);   // flip to zero mode
    overlay::setup_ongoing_cmdbuf_0(
        /*src_addr_inc_en=*/false,
        /*dest_addr_inc_en=*/false,
        /*trid_inc_en=*/false,
        /*req_vc_inc_en=*/true,                                              // per-packet VC autoincrement
        /*resp_vc_inc_en=*/false);
    overlay::setup_wrapping_vcs_cmdbuf_0(
        /*wr=*/true,
        /*req_start_vc=*/overlay::CMDBUF_WR_REQ_VC,
        /*req_end_vc=*/overlay::CMDBUF_WR_REQ_VC + 7);                       // wrap across all 8 write-VCs
    overlay::setup_trids_cmdbuf_0(overlay::CMDBUF_DEF_TRID);
    overlay::set_dest_cmdbuf_0(local_addr);
    overlay::set_len_cmdbuf_0(size_bytes);
    overlay::issue_cmdbuf_0();

#if defined(COMPILE_FOR_DM)
    // The iDMA zero device writes directly to TL1 (node memory), bypassing this DM
    // core's L1 D$ and L2. If the caller had previously CPU-written the same region,
    // those dirty cache entries would shadow the zeros on subsequent reads. Discard
    // L2 lines covering the destination range; L1 D$ and L2 are coherent on hardware.
    constexpr uintptr_t kLineMask = static_cast<uintptr_t>(L2_CACHE_LINE_SIZE) - 1;
    const uintptr_t lo = static_cast<uintptr_t>(local_addr) & ~kLineMask;
    const uintptr_t hi = static_cast<uintptr_t>(local_addr) + size_bytes;
    for (uintptr_t a = lo; a < hi; a += L2_CACHE_LINE_SIZE) {
        invalidate_l2_cache_line(a);
    }
#endif  // COMPILE_FOR_DM
}

inline void Noc::write_zeros_l1_barrier() const {
    while (!overlay::idma_acked_cmdbuf_0()) {
        // Spin until all per-backend split packets ack.
    }
}
