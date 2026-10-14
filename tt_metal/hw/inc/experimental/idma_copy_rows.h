// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#if defined(ARCH_QUASAR) && defined(COMPILE_FOR_DM)

#include "api/debug/assert.h"
#include "internal/tt-2xx/quasar/noc_nonblocking_api.h"    // init_wr_cmd_buf, noc_local_xy
#include "internal/tt-2xx/quasar/overlay/addrgen_api.hpp"  // addrgen src/dest loops
#include "internal/tt-2xx/quasar/overlay/cmdbuff_api.hpp"  // overlay:: cmdbuf API

namespace experimental {

/*
    Quasar iDMA row copy for DM kernels, local L1 -> local L1.

    Copies num_rows rows of row_bytes each. Source rows are src_stride_bytes apart; destination rows are
    packed back to back. This is the compaction step that gives Quasar narrow-row pack-untilize: pack whole
    tiles, then copy the valid prefix of each padded row into a dense buffer. iDMA addresses L1 by the byte,
    so row_bytes and the addresses need no alignment.

    One iDMA transaction per row, spread over num_vcs iDMA VCs (default: all of them). Blocks until every
    row has landed.

    Borrows cmdbuf 0, which is also the NOC write cmdbuf, and addrgen 0. Call it with no NOC writes
    outstanding; it hands cmdbuf 0 back to the NOC write path before returning.
*/
inline void idma_copy_rows(
    std::uint32_t src_addr,
    std::uint32_t src_stride_bytes,
    std::uint32_t dst_addr,
    std::uint32_t row_bytes,
    std::uint32_t num_rows,
    std::uint32_t num_vcs = overlay::CMDBUF_NUM_IDMA_VCS) {
    ASSERT(num_vcs >= 1 && num_vcs <= overlay::CMDBUF_NUM_IDMA_VCS);
    if (num_rows == 0) {
        return;
    }
    // DM-side DFB pointers carry the uncached L1 alias; the cmdbuf registers take the physical address.
    auto l1_phys = [](std::uint32_t addr) {
        return (addr >= MEM_L1_UNCACHED_BASE && addr < MEM_L1_UNCACHED_BASE + MEM_L1_SIZE) ? addr - MEM_L1_UNCACHED_BASE
                                                                                           : addr;
    };

    overlay::reset_cmdbuf_0();
    overlay::idma_setup_as_copy_cmdbuf_0(/*wrapping_en=*/false);
    overlay::setup_ongoing_cmdbuf_0({.req_vc = true});
    // resp must be explicit: its default VC 0 sits inside the request window (CMDBUF_FIRST_IDMA_VC is 0).
    overlay::setup_wrapping_vcs_cmdbuf_0(
        /*req=*/{.start = overlay::CMDBUF_FIRST_IDMA_VC, .end = overlay::CMDBUF_FIRST_IDMA_VC + num_vcs - 1},
        /*resp=*/{.start = overlay::CMDBUF_WR_RESP_VC, .end = overlay::CMDBUF_WR_RESP_VC});
    // One packet per row. CMDBUF_RESET leaves this at 0 ("never split"), which would put every row on one VC.
    overlay::setup_max_bytes_in_packet_cmdbuf_0(row_bytes);
    overlay::setup_trids_cmdbuf_0(overlay::CMDBUF_DEF_TRID);
    overlay::set_len_cmdbuf_0(row_bytes);

    // Full reset: other addrgen 0 users may have left outer-loop, face-size or banking state behind.
    overlay::reset_addrgen_0();
    overlay::setup_src_base_start_addrgen_0(l1_phys(src_addr));
    overlay::setup_src_inner_loop_addrgen_0(src_stride_bytes, (std::uint64_t)num_rows * src_stride_bytes);
    overlay::setup_dest_base_start_addrgen_0(l1_phys(dst_addr));
    overlay::setup_dest_inner_loop_addrgen_0(row_bytes, (std::uint64_t)num_rows * row_bytes);

    for (std::uint32_t r = 0; r < num_rows; r++) {
        overlay::push_both_addrgen_0();
        overlay::issue_cmdbuf_0();
    }
    while (!overlay::idma_acked_cmdbuf_0()) {
    }
    init_wr_cmd_buf(noc_local_xy());
}

}  // namespace experimental

#endif  // ARCH_QUASAR && COMPILE_FOR_DM
