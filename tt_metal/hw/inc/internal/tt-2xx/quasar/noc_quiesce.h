// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#if !defined(COMPILE_FOR_TRISC)

#include <cstdint>
#include "internal/risc_attribs.h"
#include "internal/tt-2xx/quasar/noc_nonblocking_api.h"
#include "internal/tt-2xx/quasar/overlay/rocc_instructions.hpp"

constexpr uint32_t NOC_QUIESCE_USER_TRID_MAX = 7U;
constexpr uint32_t NOC_QUIESCE_HW_TRID_MAX = 31U;

#ifdef USER_TXN_ID_MAX
static_assert(NOC_QUIESCE_USER_TRID_MAX == USER_TXN_ID_MAX);
#endif
#ifdef HW_TXN_ID_MAX
static_assert(NOC_QUIESCE_HW_TRID_MAX == HW_TXN_ID_MAX);
#endif

// Extra idle cycles after all NIU/RoCC TRIDs have drained before publishing QUIESCE_STATUS_QUIESCED,
// allowing last-hop posted write flits to clear NOC routers and DRAM controllers.
#ifndef NOC_QUIESCE_ROUTER_DRAIN_CYCLES
#define NOC_QUIESCE_ROUTER_DRAIN_CYCLES 512U
#endif

// Waits until every transaction ID in [first_trid, last_trid] issued by this DM hart has zero
// un-sent writes, zero un-acked reads/non-posted writes/atomics, and zero un-acked iDMA transfers
// across all three overlay command buffers (WR=0, RD=1, simple/AT=2).
FORCE_INLINE void noc_drain_trid_range(uint32_t first_trid, uint32_t last_trid) {
#if defined(NOC_API_V1)
    while (!ncrisc_noc_reads_flushed(0));
    while (!ncrisc_noc_nonposted_writes_sent(0));
    while (!ncrisc_noc_nonposted_writes_flushed(0));
    while (!ncrisc_noc_nonposted_atomics_flushed(0));
    while (!ncrisc_noc_posted_writes_sent(0));
    for (uint32_t trid = first_trid; trid <= last_trid; ++trid) {
        while (NOC_STATUS_READ_REG(0, NIU_MST_REQS_OUTSTANDING_ID(trid)) != 0U) {
        }
    }
#else
    for (uint32_t trid = first_trid; trid <= last_trid; ++trid) {
        while (__builtin_riscv_ttrocc_cmdbuf_wr_sent_trid(OVERLAY_WR_CMD_BUF, trid) != 0 ||
               __builtin_riscv_ttrocc_cmdbuf_tr_ack_trid(OVERLAY_WR_CMD_BUF, trid) != 0 ||
               __builtin_riscv_ttrocc_cmdbuf_idma_tr_ack_trid(OVERLAY_WR_CMD_BUF, trid) != 0 ||
               __builtin_riscv_ttrocc_cmdbuf_wr_sent_trid(OVERLAY_RD_CMD_BUF, trid) != 0 ||
               __builtin_riscv_ttrocc_cmdbuf_tr_ack_trid(OVERLAY_RD_CMD_BUF, trid) != 0 ||
               __builtin_riscv_ttrocc_cmdbuf_idma_tr_ack_trid(OVERLAY_RD_CMD_BUF, trid) != 0 ||
               __builtin_riscv_ttrocc_scmdbuf_wr_sent_trid(trid) != 0 ||
               __builtin_riscv_ttrocc_scmdbuf_tr_ack_trid(trid) != 0) {
        }
    }
#endif
}

// Full per-hart NOC quiesce barrier: drains all 32 hardware TRIDs [0..31] on this DM hart and
// commits any inbound flits already in the local NIU to L1 via NOC_FENCE().
FORCE_INLINE void noc_quiesce_hart_barrier() {
    noc_drain_trid_range(0U, NOC_QUIESCE_HW_TRID_MAX);
    asm volatile("fence" ::: "memory");
    NOC_FENCE();
    asm volatile("fence" ::: "memory");
}

FORCE_INLINE void noc_quiesce_router_drain_delay() {
    for (volatile uint32_t i = 0; i < NOC_QUIESCE_ROUTER_DRAIN_CYCLES; ++i) {
        asm volatile("nop");
    }
}

#endif  // !COMPILE_FOR_TRISC
