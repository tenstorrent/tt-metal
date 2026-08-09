// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "internal/cross_node_dfb_interface.h"
#include "internal/circular_buffer_interface.h"
#include "api/alignment.h"
#include "api/debug/waypoint.h"
#include "internal/risc_attribs.h"

#if defined(KERNEL_BUILD) && !defined(COMPILE_FOR_TRISC)
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/endpoints.h"
#include "api/core_local_mem.h"
#include "api/remote_circular_buffer.h"

// Credit helpers take Remote*CBInterface; CrossNode prefix layout matches for reinterpret_cast.
static_assert(sizeof(CrossNodeSenderDFBInterface) == sizeof(RemoteSenderCBInterface));
static_assert(
    offsetof(CrossNodeSenderDFBInterface, aligned_pages_sent_ptr) ==
    offsetof(RemoteSenderCBInterface, aligned_pages_sent_ptr));
static_assert(
    offsetof(CrossNodeReceiverDFBInterface, aligned_pages_acked_ptr) ==
    offsetof(RemoteReceiverCBInterface, aligned_pages_acked_ptr));
static_assert(
    offsetof(CrossNodeReceiverDFBInterface, remote_pages_acked_ptr) ==
    offsetof(RemoteReceiverCBInterface, remote_pages_acked_ptr));
#endif

namespace experimental {

// CrossNodeDFB: device-side kernel class for a globally-allocated ring FIFO shared across
// kernels within a single program (WH/BH only; Quasar support planned for a future PR).
// Not persistent across programs: firmware resets fifo ptrs and credit counters on every
// program init. Cross-program persistence is GlobalDFB.
//
// entry_size is fixed at Create for the life of a CrossNodeDFB. Mid-flight entry_size
// reconfiguration (remote CB / prefetcher style) is intentionally omitted here — CrossNode
// is reset-on-init and same-program only. Add set_*_entry_size later if a same-program
// multi-phase op needs it; prefer that path on GlobalDFB where the ring outlives programs.
//
// Sync counters (pages_sent / pages_acked) are in L1_ALIGNMENT-byte units.
//
// ═══════════════════════════════════════════════════════════════════════
//  SENDER FLOWS
// ═══════════════════════════════════════════════════════════════════════
//
//  Flow A — Broadcast (same data to all receivers):
//    reserve_back(n);                         // wait for space on all receivers
//    write_broadcast(src, n);                 // post NOC writes to all receivers
//    flush_writes();                          // flush posted writes before credit
//    push_back(n);                            // advance wr_ptr + signal all receivers
//
//  Flow B — Receiver-contiguous / unique-per-receiver:
//    reserve_back(n);
//    write_to_receiver(0, src_a, n);          // receiver 0 gets tensor shard A
//    write_to_receiver(1, src_b, n);          // receiver 1 gets tensor shard B
//    ...
//    flush_writes();
//    push_back(n);                            // one collective credit to all receivers
//
//  Flow C — Per-receiver credit (round-robin, uneven shards):
//    Use reserve_back_for_receiver(r, n) to check only receiver r's space; reserve_back(n)
//    would poll ALL receivers and block on the slowest even when receiver r is ready.
//    for r in 0..num_recv:
//      reserve_back_for_receiver(r, n);         // polls only receiver r, no head-of-line block
//      write_to_receiver(r, src, n);            // NOC write only to receiver r
//      flush_writes();
//      push_back_to_receiver(r, n);             // advance wr_ptr + credit only receiver r
//
//  Flow D — Interleaved scatter (prefetcher / write_strided):
//    write_strided is a single call that handles ALL receivers simultaneously.
//    Staging buffer layout: [recv0_chunk][recv1_chunk]...[recvN_chunk]
//    Each chunk is written to the corresponding receiver's FIFO in one loop.
//    reserve_back(n);
//    write_strided(src, num_rows, pages_per_row, page_size);  // all receivers, one call
//    flush_writes();
//    push_back(n);                              // advance wr_ptr + credit all receivers
//
// ═══════════════════════════════════════════════════════════════════════
//  RECEIVER FLOW
// ═══════════════════════════════════════════════════════════════════════
//
//  Standard receiver (NCRISC/BRISC consumes data):
//    wait_front(n);
//    rd_ptr = get_read_ptr();
//    // process data at rd_ptr ...
//    pop_front(n);                            // advance rd_ptr + NOC-ack sender
//
// ═══════════════════════════════════════════════════════════════════════
//  RELAY DFB FLOW — bridging CrossNodeDFB to Compute
// ═══════════════════════════════════════════════════════════════════════
//
//  Compute cannot issue NOC atomics. Data is bridged via a local
//  DataflowBuffer (relay DFB): DMs owns the CrossNodeDFB and issues all acks
//  via pop_front; Compute reads from the relay DFB's local interface.
//
//  The relay DFB shares the same L1 FIFO buffer as the CrossNodeDFB.  Its rd_ptr is
//  managed independently by Compute (via pop_front) and starts aligned to the
//  CrossNodeDFB rd_ptr at registration time.
//
//  DM (receiver kernel):
//    DataflowBuffer<...> relay_dfb(local_handle);     // local DFB backed by same FIFO
//    cn_dfb.register_relay_dfb(relay_dfb);            // init: wr_ptr = rd_ptr = start
//    while (has_more) {
//        relay_dfb.reserve_back(n);                   // wait until TRISC consumed previous data
//        cn_dfb.wait_front(n);                        // wait for sender's data (pages_sent)
//        cn_dfb.push_relay_front(n);                  // advance relay wr_ptr → TRISC unblocks
//        // optional: wait for TRISC done signal (sync_cb pattern)
//        cn_dfb.pop_front(n);                         // advance DM rd_ptr + NOC-ack sender
//    }
//
//  Compute kernel (reads relay DFB, no CrossNodeDFB or NOC knowledge):
//    cb_wait_front(relay_dfb_id, n);                  // polls relay fifo_wr_ptr (set by DM)
//    // read from cb_read_ptr(relay_dfb_id) ...
//    cb_pop_front(relay_dfb_id, n);                   // advance Compute's fifo_rd_ptr only
//                                                     // DM's pop_front already acked sender
class CrossNodeDFB {
public:
    FORCE_INLINE explicit CrossNodeDFB(uint8_t remote_dfb_id) : id_(remote_dfb_id) {}

    // -----------------------------------------------------------------------
    // Sender-side API
    // -----------------------------------------------------------------------

    // Spin until ALL receivers have space for num_entries entries of the current entry_size.
    // Use this for Flows A, B, D (collective credit patterns).
    // For Flow C (per-receiver credit), use reserve_back_for_receiver(r, n) instead.
    FORCE_INLINE void reserve_back(uint32_t num_entries) {
        WAYPOINT("GSRW");
        CrossNodeSenderDFBInterface& iface = get_cross_node_sender_dfb_interface(id_);
        const uint32_t entry_size = iface.fifo_page_size;
        const uint32_t fifo_size = get_config_word(iface.config_ptr, 3);
        const uint32_t num_units = fifo_size / L1_ALIGNMENT;

        // Adjust for wrap: if wr_ptr + len_bytes crosses the limit, add the tail region.
        uint32_t len_bytes = num_entries * entry_size;
        if (iface.fifo_wr_ptr + len_bytes >= iface.fifo_limit_page_aligned) {
            len_bytes += iface.fifo_start_addr + fifo_size - iface.fifo_limit_page_aligned;
        }
        const uint32_t total_units_needed = len_bytes / L1_ALIGNMENT;

        const uint32_t num_recv = cross_node_dfb_num_receivers(iface.num_receivers_and_remote_pages_sent_ptr);

        volatile tt_l1_ptr uint32_t* base_sent_ptr =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(iface.aligned_pages_sent_ptr);

        for (uint32_t i = 0; i < num_recv; ++i) {
            volatile tt_l1_ptr uint32_t* sent_ptr = base_sent_ptr + (2 * i * L1_ALIGNMENT / sizeof(uint32_t));
            volatile tt_l1_ptr uint32_t* acked_ptr = sent_ptr + (L1_ALIGNMENT / sizeof(uint32_t));
            do {
                invalidate_l1_cache();
            } while ((num_units - (*sent_ptr - *acked_ptr)) < total_units_needed);
        }
        WAYPOINT("GSRD");
    }

    // Spin until a SINGLE receiver (receiver_idx) has space for num_entries entries.
    // Use this for Flow C (per-receiver credit) to avoid blocking on unrelated receivers.
    FORCE_INLINE void reserve_back_for_receiver(uint32_t receiver_idx, uint32_t num_entries) {
        WAYPOINT("GSRW");
        CrossNodeSenderDFBInterface& iface = get_cross_node_sender_dfb_interface(id_);
        const uint32_t entry_size = iface.fifo_page_size;
        const uint32_t fifo_size = get_config_word(iface.config_ptr, 3);
        const uint32_t num_units = fifo_size / L1_ALIGNMENT;

        uint32_t len_bytes = num_entries * entry_size;
        if (iface.fifo_wr_ptr + len_bytes >= iface.fifo_limit_page_aligned) {
            len_bytes += iface.fifo_start_addr + fifo_size - iface.fifo_limit_page_aligned;
        }
        const uint32_t total_units_needed = len_bytes / L1_ALIGNMENT;

        volatile tt_l1_ptr uint32_t* base_sent_ptr =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(iface.aligned_pages_sent_ptr);

        volatile tt_l1_ptr uint32_t* sent_ptr = base_sent_ptr + (2 * receiver_idx * L1_ALIGNMENT / sizeof(uint32_t));
        volatile tt_l1_ptr uint32_t* acked_ptr = sent_ptr + (L1_ALIGNMENT / sizeof(uint32_t));
        do {
            invalidate_l1_cache();
        } while ((num_units - (*sent_ptr - *acked_ptr)) < total_units_needed);
        WAYPOINT("GSRD");
    }

#if defined(KERNEL_BUILD) && !defined(COMPILE_FOR_TRISC)

    // ------------------------------------------------------------------
    // Write primitives — kick off NoC writes at fifo_wr_ptr.
    // These do NOT advance fifo_wr_ptr and do NOT increment credits.
    // Call push_back() or push_back_to_receiver() after all writes.
    // ------------------------------------------------------------------

    FORCE_INLINE void noc_unicast_write_l1(
        uint32_t src_l1_addr,
        uint32_t dest_l1_addr,
        uint32_t len_bytes,
        uint32_t noc_x,
        uint32_t noc_y,
        const Noc& noc) {
        UnicastEndpoint dst;
        noc.async_write<NocOptions::POSTED>(
            CoreLocalMem<uint32_t>(src_l1_addr),
            dst,
            len_bytes,
            {},
            {.noc_x = noc_x, .noc_y = noc_y, .addr = dest_l1_addr});
    }

    // Interleaved scatter
    // Writes rows from src_l1_addr interleaved across num_receivers destinations.
    // Each receiver i gets rows at src_l1_addr + i * (num_rows * coalesced_page_size),
    // written to fifo_wr_ptr of that receiver.
    FORCE_INLINE void write_strided(
        uint32_t src_l1_addr,
        uint32_t num_rows,
        uint32_t coalesced_num_pages_per_row,
        uint32_t coalesced_page_size,
        const Noc& noc = Noc{}) {
        CrossNodeSenderDFBInterface& iface = get_cross_node_sender_dfb_interface(id_);
        const uint32_t dest_l1_base = iface.fifo_wr_ptr;
        const uint32_t num_recv = cross_node_dfb_num_receivers(iface.num_receivers_and_remote_pages_sent_ptr);
        volatile tt_l1_ptr uint32_t* xy_base =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(iface.receiver_noc_xy_ptr);

        const uint32_t row_bytes_per_recv = coalesced_num_pages_per_row * coalesced_page_size;
        const uint32_t row_stride_in_stage = row_bytes_per_recv * num_recv;

        UnicastEndpoint dst;
        uint32_t recv_src_offset = 0;
        for (uint32_t i = 0; i < num_recv; ++i) {
            const uint32_t noc_x = xy_base[2 * i];
            const uint32_t noc_y = xy_base[2 * i + 1];

            uint32_t dest_addr = dest_l1_base;
            uint32_t src_addr = src_l1_addr + recv_src_offset;
            noc.set_async_write_state<NocOptions::POSTED>(
                dst, coalesced_page_size, {.noc_x = noc_x, .noc_y = noc_y, .addr = dest_addr});
            for (uint32_t h = 0; h < num_rows; ++h) {
                const uint32_t row_src_start = src_addr;
                for (uint32_t w = 0; w < coalesced_num_pages_per_row; ++w) {
                    noc.async_write_with_state<NocOptions::POSTED>(
                        CoreLocalMem<uint32_t>(src_addr), dst, coalesced_page_size, {}, {.addr = dest_addr});
                    src_addr += coalesced_page_size;
                    dest_addr += coalesced_page_size;
                }
                src_addr = row_src_start + row_stride_in_stage;
            }
            recv_src_offset += row_bytes_per_recv;
        }
    }

    // Broadcast: write n entries of identical data from src_l1_addr to all receivers
    // at their current fifo_wr_ptr. Uses loop-unicast (hardware NOC multicast requires
    // a rectangular destination grid). For different bytes per receiver, use
    // write_to_receiver / write_strided instead.
    FORCE_INLINE void write_broadcast(uint32_t src_l1_addr, uint32_t num_entries, const Noc& noc = Noc{}) {
        CrossNodeSenderDFBInterface& iface = get_cross_node_sender_dfb_interface(id_);
        const uint32_t entry_size = iface.fifo_page_size;
        const uint32_t len_bytes = num_entries * entry_size;
        const uint32_t dest_l1_base = iface.fifo_wr_ptr;
        const uint32_t num_recv = cross_node_dfb_num_receivers(iface.num_receivers_and_remote_pages_sent_ptr);
        volatile tt_l1_ptr uint32_t* xy_base =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(iface.receiver_noc_xy_ptr);

        DPRINT("src_l1_addr: {}\n", src_l1_addr);
        DPRINT("dest_l1_base: {}\n", dest_l1_base);

        for (uint32_t i = 0; i < num_recv; ++i) {
            const uint32_t noc_x = xy_base[2 * i];
            const uint32_t noc_y = xy_base[2 * i + 1];
            DPRINT("noc_x: {} noc_y: {}\n", noc_x, noc_y);
            noc_unicast_write_l1(src_l1_addr, dest_l1_base, len_bytes, noc_x, noc_y, noc);
        }
    }

    // Write n entries from src_l1_addr to a single receiver (receiver_idx) at its
    // current fifo_wr_ptr.  Does NOT advance wr_ptr or increment credits.
    // Pair with push_back() (collective credit after all per-receiver writes) or
    // push_back_to_receiver() (per-receiver credit) as appropriate.
    FORCE_INLINE void write_to_receiver(
        uint32_t receiver_idx, uint32_t src_l1_addr, uint32_t num_entries, const Noc& noc = Noc{}) {
        CrossNodeSenderDFBInterface& iface = get_cross_node_sender_dfb_interface(id_);
        const uint32_t entry_size = iface.fifo_page_size;
        const uint32_t len_bytes = num_entries * entry_size;
        const uint32_t dest_l1_base = iface.fifo_wr_ptr;
        volatile tt_l1_ptr uint32_t* xy_base =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(iface.receiver_noc_xy_ptr);

        const uint32_t noc_x = xy_base[2 * receiver_idx];
        const uint32_t noc_y = xy_base[2 * receiver_idx + 1];
        noc_unicast_write_l1(src_l1_addr, dest_l1_base, len_bytes, noc_x, noc_y, noc);
    }

    // Flush all posted payload writes from this core before publishing pages_sent.
    // Posted writes do not return completion acknowledgements; receiver visibility of
    // the subsequently posted credit is the end-to-end synchronization point.
    FORCE_INLINE void flush_writes(const Noc& noc = Noc{}) {
        noc.async_writes_flushed<NocOptions::POSTED>();
    }

    // Credit-only: advance fifo_wr_ptr by num_entries and NOC-inc pages_sent on ALL
    // receivers.  Call after all write_* for this slot.
    FORCE_INLINE void push_back(uint32_t num_entries, const Noc& noc = Noc{}) {
        CrossNodeSenderDFBInterface& iface = get_cross_node_sender_dfb_interface(id_);
        const uint32_t entry_size = iface.fifo_page_size;
        const uint32_t fifo_size = get_config_word(iface.config_ptr, 3);

        uint32_t len_bytes = num_entries * entry_size;
        uint32_t new_wr = iface.fifo_wr_ptr + len_bytes;
        if (new_wr >= iface.fifo_limit_page_aligned) {
            len_bytes += iface.fifo_start_addr + fifo_size - iface.fifo_limit_page_aligned;
            new_wr = iface.fifo_start_addr + (new_wr - iface.fifo_limit_page_aligned);
        }
        iface.fifo_wr_ptr = new_wr;

        const uint32_t num_units = len_bytes / L1_ALIGNMENT;
        const uint8_t noc_id = noc.get_noc_id();
        detail::update_pages_sent(
            reinterpret_cast<const RemoteSenderCBInterface&>(iface), num_units, noc_id, true, write_at_cmd_buf);
    }

    // Credit-only for one receiver: advance fifo_wr_ptr by num_entries and NOC-inc
    // pages_sent on receiver_idx only.  Used for round-robin / uneven per-receiver
    // credit distribution (caller manages receiver index).
    FORCE_INLINE void push_back_to_receiver(uint32_t receiver_idx, uint32_t num_entries, const Noc& noc = Noc{}) {
        CrossNodeSenderDFBInterface& iface = get_cross_node_sender_dfb_interface(id_);
        const uint32_t entry_size = iface.fifo_page_size;
        const uint32_t fifo_size = get_config_word(iface.config_ptr, 3);

        uint32_t len_bytes = num_entries * entry_size;
        uint32_t new_wr = iface.fifo_wr_ptr + len_bytes;
        if (new_wr >= iface.fifo_limit_page_aligned) {
            len_bytes += iface.fifo_start_addr + fifo_size - iface.fifo_limit_page_aligned;
            new_wr = iface.fifo_start_addr + (new_wr - iface.fifo_limit_page_aligned);
        }
        iface.fifo_wr_ptr = new_wr;

        const uint32_t num_units = len_bytes / L1_ALIGNMENT;
        volatile tt_l1_ptr uint32_t* sent_base =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(iface.aligned_pages_sent_ptr);
        volatile tt_l1_ptr uint32_t* xy_base =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(iface.receiver_noc_xy_ptr);
        volatile tt_l1_ptr uint32_t* local_sent = sent_base + (2 * receiver_idx * L1_ALIGNMENT / sizeof(uint32_t));
        const uint8_t noc_id = noc.get_noc_id();
        const uint32_t noc_x = xy_base[2 * receiver_idx];
        const uint32_t noc_y = xy_base[2 * receiver_idx + 1];
        const uint32_t noc_xy = uint32_t(NOC_XY_ENCODING(DYNAMIC_NOC_X(noc_id, noc_x), DYNAMIC_NOC_Y(noc_id, noc_y)));
        *local_sent += num_units;
        const uint64_t remote_addr = get_noc_addr_helper(noc_xy, (uint32_t)local_sent);
        noc_semaphore_inc<true>(remote_addr, num_units, noc_id);
    }

#endif  // KERNEL_BUILD && !COMPILE_FOR_TRISC

    // Wait until all receivers have acked all pages_sent (drains the pipeline).
    FORCE_INLINE void barrier() {
        WAYPOINT("CNBW");
        CrossNodeSenderDFBInterface& iface = get_cross_node_sender_dfb_interface(id_);
        const uint32_t num_recv = cross_node_dfb_num_receivers(iface.num_receivers_and_remote_pages_sent_ptr);
        volatile tt_l1_ptr uint32_t* base =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(iface.aligned_pages_sent_ptr);
        for (uint32_t i = 0; i < num_recv; ++i) {
            volatile tt_l1_ptr uint32_t* sent_ptr = base + (2 * i * L1_ALIGNMENT / sizeof(uint32_t));
            volatile tt_l1_ptr uint32_t* acked_ptr = sent_ptr + (L1_ALIGNMENT / sizeof(uint32_t));
            while (true) {
                invalidate_l1_cache();
                if (*acked_ptr == *sent_ptr) {
                    break;
                }
            }
        }
        WAYPOINT("CNBD");
    }

    // -----------------------------------------------------------------------
    // Receiver-side API
    // -----------------------------------------------------------------------

    // Spin until pages_sent - pages_acked >= num_entries (in L1_ALIGNMENT units).
    FORCE_INLINE void wait_front(uint32_t num_entries) {
        WAYPOINT("CNWF");
        CrossNodeReceiverDFBInterface& iface = get_cross_node_receiver_dfb_interface(id_);
        const uint32_t entry_size = iface.fifo_page_size;
        const uint32_t fifo_size = get_config_word(iface.config_ptr, 3);

        uint32_t len_bytes = num_entries * entry_size;
        if (iface.fifo_rd_ptr + len_bytes >= iface.fifo_limit_page_aligned) {
            len_bytes += iface.fifo_start_addr + fifo_size - iface.fifo_limit_page_aligned;
        }
        const uint32_t units_needed = len_bytes / L1_ALIGNMENT;

        // pages_sent is at aligned_pages_acked_ptr - L1_ALIGNMENT (same as GlobalCB).
        volatile tt_l1_ptr uint32_t* acked_ptr =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(iface.aligned_pages_acked_ptr);
        volatile tt_l1_ptr uint32_t* sent_ptr =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(iface.aligned_pages_acked_ptr - L1_ALIGNMENT);
        do {
            invalidate_l1_cache();
        } while ((*sent_ptr - *acked_ptr) < units_needed);
        WAYPOINT("CNWD");
    }

#if defined(KERNEL_BUILD) && !defined(COMPILE_FOR_TRISC)
    // Advance read pointer and NOC-inc pages_acked on sender.
    FORCE_INLINE void pop_front(uint32_t num_entries, const Noc& noc = Noc{}) {
        CrossNodeReceiverDFBInterface& iface = get_cross_node_receiver_dfb_interface(id_);
        const uint32_t entry_size = iface.fifo_page_size;
        const uint32_t fifo_size = get_config_word(iface.config_ptr, 3);

        uint32_t len_bytes = num_entries * entry_size;
        if (iface.fifo_rd_ptr + len_bytes >= iface.fifo_limit_page_aligned) {
            iface.fifo_rd_ptr = iface.fifo_start_addr + (iface.fifo_rd_ptr + len_bytes - iface.fifo_limit_page_aligned);
            len_bytes += iface.fifo_start_addr + fifo_size - iface.fifo_limit_page_aligned;
        } else {
            iface.fifo_rd_ptr += len_bytes;
        }
        const uint32_t num_units = len_bytes / L1_ALIGNMENT;

        const uint8_t noc_id = noc.get_noc_id();
        detail::update_pages_acked(
            reinterpret_cast<const RemoteReceiverCBInterface&>(iface), num_units, noc_id, false, write_at_cmd_buf);
    }
#endif  // KERNEL_BUILD && !COMPILE_FOR_TRISC

    // -----------------------------------------------------------------------
    // Relay DFB registration and alignment
    // -----------------------------------------------------------------------

    // Register one local DataflowBuffer as the relay for this CrossNodeDFB (receiver side only).
    //
    // The relay shares the same L1 FIFO as the CrossNodeDFB and exposes it to TRISC via the
    // standard DataflowBuffer / CB API. Host must co-allocate them at the same L1 address.
    //
    // Intra-core sync (same as GlobalCB receiver):
    //   - relay.fifo_wr_ptr is managed by the receiver DM via push_relay_front() after wait_front()
    //   - relay.fifo_rd_ptr is managed by TRISC via pop_front()
    //   - The receiver DM must never write relay.fifo_rd_ptr after init
    //
    // Typical receiver-DM loop:
    //   register_relay_dfb(relay);            // init: wr_ptr = rd_ptr = start
    //   while (has_more) {
    //       relay.reserve_back(n);
    //       gdfb.wait_front(n);
    //       gdfb.push_relay_front(n);
    //       gdfb.pop_front(n);
    //   }
    template <typename DFB>
    FORCE_INLINE void register_relay_dfb(DFB& local_dfb) {
        CrossNodeReceiverDFBInterface& iface = get_cross_node_receiver_dfb_interface(id_);
        iface.relay_id = local_dfb.get_logical_handle();
        align_relay_init();
    }

    // Advance the relay DFB's fifo_wr_ptr by num_entries.
    // Call after wait_front(n) to unblock TRISC's cb_wait_front(relay_id, n).
    // Does NOT touch fifo_rd_ptr.
    FORCE_INLINE void push_relay_front(uint32_t num_entries) {
        CrossNodeReceiverDFBInterface& iface = get_cross_node_receiver_dfb_interface(id_);
        if (iface.relay_id == RELAY_DFB_INVALID) {
            return;
        }
        LocalCBInterface& relay = get_local_cb_interface(iface.relay_id);
        const uint32_t len_bytes = num_entries * iface.fifo_page_size;
        uint32_t new_wr = relay.fifo_wr_ptr + len_bytes;
        if (new_wr >= relay.fifo_limit) {
            new_wr = iface.fifo_start_addr + (new_wr - relay.fifo_limit);
        }
        relay.fifo_wr_ptr = new_wr;
    }

    // Manual one-shot alignment into a local DataflowBuffer (when not using register_relay_dfb).
    template <typename DFB>
    FORCE_INLINE void align_local_dfb(DFB& local_dfb) {
        align_one_relay_init(local_dfb);
    }

    // -----------------------------------------------------------------------
    // Accessors
    // -----------------------------------------------------------------------

    // Number of receivers connected to this CrossNodeDFB (sender-attached cores only).
    FORCE_INLINE uint32_t num_receivers() {
        const CrossNodeSenderDFBInterface& iface = get_cross_node_sender_dfb_interface(id_);
        return cross_node_dfb_num_receivers(iface.num_receivers_and_remote_pages_sent_ptr);
    }

    FORCE_INLINE uint32_t get_write_ptr() { return get_cross_node_sender_dfb_interface(id_).fifo_wr_ptr; }

    FORCE_INLINE uint32_t get_read_ptr() { return get_cross_node_receiver_dfb_interface(id_).fifo_rd_ptr; }

    FORCE_INLINE uint32_t get_entry_size() { return get_cross_node_sender_dfb_interface(id_).fifo_page_size; }

private:
    uint8_t id_;

    // Read a word from the config page.
    FORCE_INLINE uint32_t get_config_word(uint32_t config_ptr, uint32_t word_idx) {
        return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(config_ptr)[word_idx];
    }

    // Init: set relay fifo_wr_ptr = fifo_rd_ptr = CrossNodeDFB start, plus limit/page_size.
    // Called once from register_relay_dfb(). After this, the receiver DM owns fifo_wr_ptr
    // (advances via push_relay_front) and TRISC owns fifo_rd_ptr (advances via pop_front).
    FORCE_INLINE void align_relay_init() {
        CrossNodeReceiverDFBInterface& iface = get_cross_node_receiver_dfb_interface(id_);
        if (iface.relay_id == RELAY_DFB_INVALID) {
            return;
        }
        LocalCBInterface& relay = get_local_cb_interface(iface.relay_id);
        relay.fifo_wr_ptr = iface.fifo_rd_ptr;
        relay.fifo_rd_ptr = iface.fifo_rd_ptr;
        relay.fifo_limit = iface.fifo_limit_page_aligned;
        relay.fifo_page_size = iface.fifo_page_size;
    }

    template <typename DFB>
    FORCE_INLINE void align_one_relay_init(DFB& dfb) {
        const CrossNodeReceiverDFBInterface& iface = get_cross_node_receiver_dfb_interface(id_);
        LocalCBInterface& relay = get_local_cb_interface(dfb.get_logical_handle());
        relay.fifo_wr_ptr = iface.fifo_rd_ptr;
        relay.fifo_rd_ptr = iface.fifo_rd_ptr;
        relay.fifo_limit = iface.fifo_limit_page_aligned;
        relay.fifo_page_size = iface.fifo_page_size;
    }
};

}  // namespace experimental
