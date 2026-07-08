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
        /*req_start_vc=*/overlay::CMDBUF_WR_REQ_VC,
        /*req_end_vc=*/overlay::CMDBUF_WR_REQ_VC + 7);                       // wrap across all 8 write-VCs
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
    // autoincrement). Now that every packet has acked, reset it and reprogram the standard
    // write-ready config so the next noc_async_write on cmd buffer 0 behaves normally. This
    // is done here, after the ack, rather than in async_write_zeros: init_wr_cmd_buf() resets
    // cmd buffer 0, and resetting before the ack could disturb the pending iDMA ack we just
    // waited on.
    //
    // TODO: Quasar has architecture different enough that we may want to get out of using
    // noc_nonblocking_api. Refactor this when we move away from it.
    init_wr_cmd_buf(noc_local_xy());
    NOC_ZERO_MODE_EXIT();  // cmd buffer 0 restored; NoC writes are safe again
}
