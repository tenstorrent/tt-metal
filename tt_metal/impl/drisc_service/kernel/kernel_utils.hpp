// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once
// Kernel-side NoC and GDDR DMA helpers for the DRISC service kernel, taking abi::NocEndpoint.
//
// NocEndpoint.local_addr goes to NOC_*_ADDR_LO/MID unmasked and noc_xy_addr to NOC_*_ADDR_COORDINATE, so host
// IOVAs wider than 36 bits pass through. A host endpoint's local_addr is the raw IOVA routed to the PCIe tile.

#include <cstdint>

#include "hostdev/drisc_service/abi.hpp"  // shared host<->kernel contract (abi::NocEndpoint)

// clang-format off
#include "risc_common.h"  // brings in noc_nonblocking_api.h in the order metal expects
#include "internal/dataflow/dataflow_api_common.h"
#include "internal/dataflow/dataflow_cmd_bufs.h"
// clang-format on
#include "experimental/gddr_dma.h"
#include "l1_layout.hpp"

namespace abi = tt::tt_metal::drisc_service::abi;

// Metal's NoC API declares these extern and expects firmware to define them; this kernel runs without firmware.
inline uint32_t noc_reads_num_issued[NUM_NOCS];
inline uint32_t noc_nonposted_writes_num_issued[NUM_NOCS];
inline uint32_t noc_nonposted_writes_acked[NUM_NOCS];
inline uint32_t noc_nonposted_atomics_acked[NUM_NOCS];
inline uint32_t noc_posted_writes_num_issued[NUM_NOCS];

// Readies the NoC for this kernel. The kernel never writes NIU_CFG_0 (experimental/drisc_mode.h, "Who sets the
// mode"): the host put the core in reset and set both NIU modes to tt-metal's DRISC firmware layout before starting it
// (DriscCore::force_stop), and the seat's NIU is one that layout leaves in stream mode (bank_seats).
inline void noc_boot() {
    // Requests of a kernel stopped mid-flight survive the relaunch; let them complete before reprogramming the NoC.
    for (uint32_t trid = 0; trid <= NOC_MAX_TRANSACTION_ID; ++trid) {
        while (!ncrisc_noc_read_with_transaction_id_flushed(noc_index, trid));
    }

    // Off: the D-cache is not coherent with NoC-landed control words.
    set_l1_data_cache<false>();

    // The argument is the atomic-return address; the kernel issues no atomics.
    noc_init(l1::SCRATCH);

    // Non-posted: done_count is published only after the data writes are acked.
    noc_write_init_state<write_cmd_buf, CQ_NOC_mkp>(noc_index, NOC_UNICAST_WRITE_VC);

    noc_local_state_init(noc_index);
}

static_assert(
    NOC_ADDR_NODE_ID_BITS == abi::NOC_COORD_BITS,
    "tt-metal's NOC coordinate width no longer matches the host<->kernel contract");

// Transaction-ID tagged transfers, chunked into packets of at most NOC_MAX_BURST_SIZE. The request counts against
// NIU_MST_REQS_OUTSTANDING_ID(trid). IDs are shared across cmd bufs and directions, so streams complete in any order
// only if their IDs differ. The separate-coordinate with_state calls (rather than metal's uint64_t NoC-address APIs)
// let host IOVAs wider than NOC_ADDR_LOCAL_BITS through.
inline void noc_async_read_with_trid(uint32_t dst_l1, abi::NocEndpoint src, uint32_t len_bytes, uint32_t trid) {
    while (len_bytes > 0) {
        const uint32_t n = len_bytes > NOC_MAX_BURST_SIZE ? uint32_t(NOC_MAX_BURST_SIZE) : len_bytes;
        ncrisc_noc_set_transaction_id(noc_index, read_cmd_buf, trid);
        noc_read_with_state<DM_DEDICATED_NOC, read_cmd_buf, CQ_NOC_SNDL>(
            noc_index, src.noc_xy_addr, src.local_addr, dst_l1, n);
        dst_l1 += n;
        src = src.at_offset(n);
        len_bytes -= n;
    }
}

// Issues one packet of at most len_bytes and returns the bytes issued, or 0 if the write cmd buf is busy.
inline uint32_t noc_try_write_packet_with_trid(
    uint32_t src_l1, abi::NocEndpoint dst, uint32_t len_bytes, uint32_t trid) {
    if (!noc_cmd_buf_ready(noc_index, write_cmd_buf)) {
        return 0;
    }
    const uint32_t n = len_bytes > NOC_MAX_BURST_SIZE ? uint32_t(NOC_MAX_BURST_SIZE) : len_bytes;
    ncrisc_noc_set_transaction_id(noc_index, write_cmd_buf, trid);
    noc_wwrite_with_state<DM_DEDICATED_NOC, write_cmd_buf, CQ_NOC_SNDL>(
        noc_index, src_l1, dst.noc_xy_addr, dst.local_addr, n);
    return n;
}

// GDDR DMA: reads go on one TX stream and writes on the other, and a stream completes its requests in issue order.
// Issue skips the engine's ready poll, so callers keep fewer than experimental::kMaxOutstandingWrites requests in
// flight per stream. Lengths must be multiples of 16.
constexpr uint8_t dma_read_stream = 0;
constexpr uint8_t dma_write_stream = 1;

// The engine's attributes and in-flight requests survive a kernel relaunch.
inline void dma_boot() {
    experimental::dma_restore_default_transfer_attrs();
    for (uint8_t stream = 0; stream < 2; ++stream) {
        experimental::dma_async_read_barrier(stream);
        experimental::dma_async_write_barrier(stream);
    }
}
