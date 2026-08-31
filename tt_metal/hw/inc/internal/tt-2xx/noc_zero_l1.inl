// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Single iDMA zero-device transaction with the full cmdbuf sequence
// spelled out, plus the matching write_zeros_l1_barrier (iDMA ack spin).
// No cache invalidate is needed here: zeroed buffers are assumed not resident
// in the DM core's cache. Access to a buffer should be bracketed by lock/unlock
// (unlock will be responsible for cache eviction). Zeroing a locked buffer
// should be flagged by the NOC transaction debug tool -- see TODO below.

#include "internal/tt-2xx/quasar/overlay/cmdbuff_api.hpp"
#include "internal/tt-2xx/quasar/noc_nonblocking_api.h"
#include "internal/debug/noc_zero_guard.h"

template <typename Dst>
inline void Noc::async_write_zeros(const Dst& dst, uint32_t size_bytes, const dst_args_t<Dst>& args) const {
    static_assert(
        noc_zero_l1_endpoint_v<Dst>,
        "noc.async_write_zeros: unsupported local-L1 destination. Supported: CircularBuffer, "
        "DataflowBuffer, CoreLocalMem, Scratchpad, LocalTensorAccessor. Use the TensorAccessor overload for DRAM.");

    if constexpr (is_scratchpad_v<Dst>) {
        ASSERT(static_cast<uint64_t>(args.offset_bytes) + size_bytes <= dst.size_in_bytes());
    }
    const uint32_t local_addr = static_cast<uint32_t>(get_dst_ptr<AddressType::LOCAL_L1>(dst, args));
    DEBUG_SANITIZE_L1_ADDR(local_addr, size_bytes);

    // Engage the Quasar iDMA zero device (Overlay Spec §4.12). The zero mode is
    // a HW overlay on top of the iDMA copy path: same MISC.idma_en + MISC.write_trans
    // setup as iDMA copy, but with AXI_OPT_1.src_protocol = 4 and decouple_aw = 1. The
    // payload bytes coming out the back are forced to zero. Source address is
    // ignored. We reuse idma_setup_as_copy_cmdbuf_0 for the MISC setup, then flip
    // src_protocol / decouple_aw via set_axi_opt_1_cmdbuf_0.
    //
    // The cmdbuf splits the transaction into packets and round-robins them across
    // the 8 backend engines via per-packet VC autoincrement wrapping the 8 iDMA
    // backend VCs (CMDBUF_FIRST_IDMA_VC..+7). idma_acked_cmdbuf_0 returns true only
    // after every backend packet has acked. The range is pinned to the backend VCs
    // rather than derived from CMDBUF_WR_REQ_VC so it cannot walk into the multicast
    // request VCs when the unicast write VC moves.
    //
    // No reset_cmdbuf_0() here: resetting per-call would be unsafe when callers batch
    // several async_write_zeros before a single barrier — a CMDBUF_RESET on the next
    // call may disturb a previous zero whose iDMA ack is still pending.
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
        /*req_start_vc=*/overlay::CMDBUF_FIRST_IDMA_VC,
        /*req_end_vc=*/overlay::CMDBUF_FIRST_IDMA_VC + overlay::CMDBUF_NUM_IDMA_VCS - 1);  // all 8 iDMA VCs
    overlay::setup_trids_cmdbuf_0(overlay::CMDBUF_DEF_TRID);
    overlay::set_dest_cmdbuf_0(local_addr);
    overlay::set_len_cmdbuf_0(size_bytes);
    overlay::issue_cmdbuf_0();

    // cmdbuf 0 is now in the zero-borrowed configuration (AXI_OPT_1 in zero mode, AUTOINC
    // enabling per-packet VC autoincrement). It is reset and reprogrammed to its write-ready
    // default by write_zeros_l1_barrier() below, after the ack, so the reset there cannot
    // disturb this transaction's pending iDMA ack. Do not issue other cmdbuf-0 NoC ops
    // between this call and write_zeros_l1_barrier(): barrier first, then reuse cmdbuf 0.

    // TODO: this zero should record a NOC-debug write event (dst = local_addr, size_bytes) so
    // the NOC transaction debug tool flags zeroing a locked buffer (WRITE_TO_LOCKED_*). Not
    // wired up here yet: the RECORD_NOC_EVENT_WITH_ADDR machinery is currently only enabled for
    // COMPILE_FOR_NCRISC/BRISC, not Quasar's COMPILE_FOR_DM.

    // WATCHER: mark cmd buffer 0 borrowed for zeroing; write_zeros_l1_barrier() clears it. Lets
    // watcher builds catch any NoC write issued before the barrier (the zero->barrier->reuse rule).
    NOC_ZERO_MODE_ENTER();
}

inline void Noc::write_zeros_l1_barrier() const {
    // TODO: this barrier should record a NOC-debug event so the tool can flag a missing
    // write_zeros_l1_barrier (use-before-flush), the way read/write barriers do.
    while (!overlay::idma_acked_cmdbuf_0()) {
        // Spin until all per-backend split packets ack.
    }
    // The zero borrowed cmd buffer 0 in a non-write configuration (iDMA zero mode + VC
    // autoincrement). Now that every packet has acked, reset it. V2 uses the overlay command
    // buffer for normal writes, so restore its write-ready configuration. V1 performs normal
    // writes through memory-mapped NoC registers and only needs the borrowed overlay buffer
    // reset. Resetting before the ack could disturb the pending iDMA ack we just waited on.
#ifdef NOC_API_V2
    init_wr_cmd_buf(noc_local_xy());
#else
    overlay::reset_cmdbuf_0();
#endif
    NOC_ZERO_MODE_EXIT();  // cmd buffer 0 restored; NoC writes are safe again
}
