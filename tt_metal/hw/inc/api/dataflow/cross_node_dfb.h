// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// WH/BH only for now; Quasar CrossNode device API is a follow-up.
#ifndef ARCH_QUASAR

#include <cstdint>
#include "internal/cross_node_dfb_init.h"
#include "internal/circular_buffer_interface.h"
#include "api/alignment.h"
#include "api/debug/assert.h"
#include "api/debug/waypoint.h"
#include "internal/risc_attribs.h"

#if defined(KERNEL_BUILD) && !defined(COMPILE_FOR_TRISC)
#include <new>
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
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
// kernels within a single program (WH/BH only).
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
// Each receiver owns a private ring, so the sender needs an independent write position per
// receiver. No cursor state is stored anywhere: a receiver's write offset is derived from
// its local entries_sent counter (sent % ring_entries), which dispatch resets to zero on
// every launch.
//
// Writes are contiguous: a reserve/write/push of n entries must fit from the current
// write position to fifo_limit without straddling the wrap (same rule as local CBs). The
// derived position wraps only when the credited entries reach a multiple of the ring
// depth. If fewer than n slots remain before the limit, issue a smaller batch (or drain
// to wrap), then continue.
//
// ═══════════════════════════════════════════════════════════════════════
//  SENDER FLOWS
// ═══════════════════════════════════════════════════════════════════════
//
//  Flow A — Broadcast (same data to all receivers):
//    reserve_back(n);                         // wait for space on all receivers
//    write_broadcast(src, n);                 // post NOC writes to all receivers
//    flush_writes();                          // flush posted writes before credit
//    push_back(n);                            // credit all receivers (advances positions)
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
//      push_back_to_receiver(r, n);             // credit receiver r (advances its position)
//
//  Flow D — Interleaved scatter (prefetcher / write_strided):
//    write_strided is a single call that handles ALL receivers simultaneously.
//    Staging buffer layout: [recv0_chunk][recv1_chunk]...[recvN_chunk]
//    Each chunk is written to the corresponding receiver's FIFO in one loop.
//    reserve_back(n);
//    write_strided(src, num_rows, pages_per_row, page_size);  // all receivers, one call
//    flush_writes();
//    push_back(n);                              // credit all receivers (advances positions)
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
//  Compute cannot issue NOC atomics. Data is bridged via a host-declared local
//  DataflowBuffer that aliases the CrossNode ring. DM owns CrossNode credits;
//  TRISC consumes through the normal local DFB API.
//
//  Host writes the relay local DFB's device slot into the receiver interface.
//  DM deliberately receives no relay binding token and must use bind_relay().
//
//  DM (receiver kernel):
//    auto relay = cn_dfb.bind_relay();                 // constructs the real local DFB
//    while (has_more) {
//        relay.reserve_back(n);                       // wait for local free space
//        cn_dfb.wait_front(n);                        // wait for sender's data (pages_sent)
//        relay.push_back(n);                          // publish via CB credits
//        cn_dfb.pop_front(n);                         // wait for TRISC if relay-bound, then NOC-ack sender
//    }
//
//  Compute kernel (reads relay DFB, no CrossNodeDFB or NOC knowledge):
//    DataflowBuffer relay(dfb::relay_name);           // normal generated binding token
//    relay.wait_front(n);
//    // consume ...
//    relay.pop_front(n);
class CrossNodeDFB {
public:
    FORCE_INLINE explicit CrossNodeDFB(uint8_t remote_dfb_id) {
        const uint32_t launch_index = *GET_MAILBOX_ADDRESS_DEV(launch_msg_rd_ptr);
        const auto* launch_msg = GET_MAILBOX_ADDRESS_DEV(launch[launch_index]);
        const auto& kernel_config = launch_msg->kernel_config;
        ASSERT(kernel_config.cross_node_dfb_offset != REMOTE_DFB_OFFSET_NONE);

        const uint32_t kernel_config_base = kernel_config.kernel_config_base[PROGRAMMABLE_CORE_TYPE];
        volatile tt_l1_ptr uint32_t* region =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(kernel_config_base + kernel_config.cross_node_dfb_offset);
        ASSERT(remote_dfb_id < region[0]);

        volatile tt_l1_ptr uint32_t* slot =
            region + REMOTE_DFB_REGION_HEADER_WORDS + remote_dfb_id * UINT32_WORDS_PER_REMOTE_DFB_CONFIG;
        setup_cross_node_dfb_interface(
            interface_, /*config_page_addr=*/slot[0], /*entry_size=*/slot[1], /*relay_dfb_id=*/slot[2]);
    }

    // -----------------------------------------------------------------------
    // Sender-side API
    // -----------------------------------------------------------------------

    // Spin until ALL receivers have space for num_entries entries of the current entry_size.
    // Use this for Flows A, B, D (collective credit patterns).
    // For Flow C (per-receiver credit), use reserve_back_for_receiver(r, n) instead.
    FORCE_INLINE void reserve_back(uint32_t num_entries) {
        WAYPOINT("GSRW");
        CrossNodeSenderDFBInterface& iface = interface_.sender;
        const uint32_t fifo_size = get_config_word(iface.config_ptr, 3);
        const uint32_t num_units = fifo_size / L1_ALIGNMENT;
        const uint32_t num_recv = cross_node_dfb_num_receivers(iface.num_receivers_and_remote_pages_sent_ptr);
        const uint32_t total_units_needed = units_for_write(iface, num_entries);

        for (uint32_t i = 0; i < num_recv; ++i) {
            volatile tt_l1_ptr uint32_t* sent_ptr = local_sent_ptr(iface, i);
            volatile tt_l1_ptr uint32_t* acked_ptr = sent_ptr + (L1_ALIGNMENT / sizeof(uint32_t));
            assert_contiguous_write(iface, wr_offset_from_sent(iface, *sent_ptr), num_entries);
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
        CrossNodeSenderDFBInterface& iface = interface_.sender;
        const uint32_t fifo_size = get_config_word(iface.config_ptr, 3);
        const uint32_t num_units = fifo_size / L1_ALIGNMENT;
        const uint32_t total_units_needed = units_for_write(iface, num_entries);

        volatile tt_l1_ptr uint32_t* sent_ptr = local_sent_ptr(iface, receiver_idx);
        volatile tt_l1_ptr uint32_t* acked_ptr = sent_ptr + (L1_ALIGNMENT / sizeof(uint32_t));
        assert_contiguous_write(iface, wr_offset_from_sent(iface, *sent_ptr), num_entries);
        do {
            invalidate_l1_cache();
        } while ((num_units - (*sent_ptr - *acked_ptr)) < total_units_needed);
        WAYPOINT("GSRD");
    }

#if defined(KERNEL_BUILD) && !defined(COMPILE_FOR_TRISC)

    // ------------------------------------------------------------------
    // Write primitives — kick off NoC writes at each receiver's write position,
    // derived from that receiver's entries_sent credits. They do NOT increment
    // credits, so repeating a write before crediting overwrites the same slots.
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
    // written at that receiver's write position.
    FORCE_INLINE void write_strided(
        uint32_t src_l1_addr,
        uint32_t num_rows,
        uint32_t coalesced_num_pages_per_row,
        uint32_t coalesced_page_size,
        const Noc& noc = Noc{}) {
        CrossNodeSenderDFBInterface& iface = interface_.sender;
        const uint32_t num_recv = cross_node_dfb_num_receivers(iface.num_receivers_and_remote_pages_sent_ptr);
        volatile tt_l1_ptr uint32_t* xy_base =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(iface.receiver_noc_xy_ptr);

        const uint32_t row_bytes_per_recv = coalesced_num_pages_per_row * coalesced_page_size;
        const uint32_t bytes_per_recv = num_rows * row_bytes_per_recv;
        const uint32_t row_stride_in_stage = row_bytes_per_recv * num_recv;

        UnicastEndpoint dst;
        uint32_t recv_src_offset = 0;
        for (uint32_t i = 0; i < num_recv; ++i) {
            const uint32_t wr_offset = derived_wr_offset(iface, i);
            assert_contiguous_bytes(iface, wr_offset, bytes_per_recv);
            const uint32_t noc_x = xy_base[2 * i];
            const uint32_t noc_y = xy_base[2 * i + 1];

            uint32_t dest_addr = iface.fifo_start_addr + wr_offset;
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
    // at their current write position. Uses loop-unicast (hardware NOC multicast requires
    // a rectangular destination grid). For different bytes per receiver, use
    // write_to_receiver / write_strided instead.
    FORCE_INLINE void write_broadcast(uint32_t src_l1_addr, uint32_t num_entries, const Noc& noc = Noc{}) {
        CrossNodeSenderDFBInterface& iface = interface_.sender;
        const uint32_t entry_size = iface.fifo_page_size;
        const uint32_t len_bytes = num_entries * entry_size;
        const uint32_t num_recv = cross_node_dfb_num_receivers(iface.num_receivers_and_remote_pages_sent_ptr);
        volatile tt_l1_ptr uint32_t* xy_base =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(iface.receiver_noc_xy_ptr);

        DPRINT("src_l1_addr: {}\n", src_l1_addr);

        for (uint32_t i = 0; i < num_recv; ++i) {
            const uint32_t wr_offset = derived_wr_offset(iface, i);
            assert_contiguous_write(iface, wr_offset, num_entries);
            const uint32_t noc_x = xy_base[2 * i];
            const uint32_t noc_y = xy_base[2 * i + 1];
            const uint32_t dest_l1_addr = iface.fifo_start_addr + wr_offset;
            DPRINT("noc_x: {} noc_y: {} dest: {}\n", noc_x, noc_y, dest_l1_addr);
            noc_unicast_write_l1(src_l1_addr, dest_l1_addr, len_bytes, noc_x, noc_y, noc);
        }
    }

    // Write n entries from src_l1_addr to a single receiver (receiver_idx) at that
    // receiver's write position.  Does NOT increment credits.  Pair with push_back()
    // (collective credit after all per-receiver writes) or push_back_to_receiver()
    // (per-receiver credit) as appropriate.
    FORCE_INLINE void write_to_receiver(
        uint32_t receiver_idx, uint32_t src_l1_addr, uint32_t num_entries, const Noc& noc = Noc{}) {
        CrossNodeSenderDFBInterface& iface = interface_.sender;
        const uint32_t entry_size = iface.fifo_page_size;
        const uint32_t len_bytes = num_entries * entry_size;
        const uint32_t wr_offset = derived_wr_offset(iface, receiver_idx);
        assert_contiguous_write(iface, wr_offset, num_entries);
        const uint32_t dest_l1_addr = iface.fifo_start_addr + wr_offset;
        volatile tt_l1_ptr uint32_t* xy_base =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(iface.receiver_noc_xy_ptr);

        const uint32_t noc_x = xy_base[2 * receiver_idx];
        const uint32_t noc_y = xy_base[2 * receiver_idx + 1];
        noc_unicast_write_l1(src_l1_addr, dest_l1_addr, len_bytes, noc_x, noc_y, noc);
    }

    // Flush all posted payload writes from this core before publishing pages_sent.
    // Posted writes do not return completion acknowledgements; receiver visibility of
    // the subsequently posted credit is the end-to-end synchronization point.
    FORCE_INLINE void flush_writes(const Noc& noc = Noc{}) { noc.async_writes_flushed<NocOptions::POSTED>(); }

    // Credit-only: NOC-inc pages_sent on ALL receivers by num_entries. Advancing each
    // receiver's entries_sent is what moves its derived write position forward.
    // Call after all write_* for this slot.
    FORCE_INLINE void push_back(uint32_t num_entries, const Noc& noc = Noc{}) {
        CrossNodeSenderDFBInterface& iface = interface_.sender;
        const uint32_t num_recv = cross_node_dfb_num_receivers(iface.num_receivers_and_remote_pages_sent_ptr);
        const uint32_t num_units = units_for_write(iface, num_entries);

        // The batch being credited starts at each receiver's current derived position and
        // must not straddle the ring wrap (contiguous-write contract).
        for (uint32_t i = 0; i < num_recv; ++i) {
            assert_contiguous_write(iface, derived_wr_offset(iface, i), num_entries);
        }

        const uint8_t noc_id = noc.get_noc_id();
        detail::update_pages_sent(
            reinterpret_cast<const RemoteSenderCBInterface&>(iface), num_units, noc_id, true, write_at_cmd_buf);
    }

    // Credit-only for one receiver: NOC-inc pages_sent on receiver_idx by num_entries,
    // which also advances that receiver's derived write position. Used for round-robin /
    // uneven per-receiver credit distribution (caller manages receiver index).
    FORCE_INLINE void push_back_to_receiver(uint32_t receiver_idx, uint32_t num_entries, const Noc& noc = Noc{}) {
        CrossNodeSenderDFBInterface& iface = interface_.sender;

        const uint32_t num_units = units_for_write(iface, num_entries);
        assert_contiguous_write(iface, derived_wr_offset(iface, receiver_idx), num_entries);

        volatile tt_l1_ptr uint32_t* xy_base =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(iface.receiver_noc_xy_ptr);
        volatile tt_l1_ptr uint32_t* local_sent = local_sent_ptr(iface, receiver_idx);
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
        CrossNodeSenderDFBInterface& iface = interface_.sender;
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
        CrossNodeReceiverDFBInterface& iface = interface_.receiver;
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
    // If bind_relay() was called, waits until compute has consumed num_entries first.
    FORCE_INLINE void pop_front(uint32_t num_entries, const Noc& noc = Noc{}) {
        if (interface_.receiver.relay_id != RELAY_DFB_INVALID) {
            ASSERT(relay_dfb_ != nullptr);
            wait_relay_consumed(num_entries);
        }
        pop_front_impl(num_entries, noc);
    }
#endif  // KERNEL_BUILD && !COMPILE_FOR_TRISC

#if defined(KERNEL_BUILD) && !defined(COMPILE_FOR_TRISC)
    // -----------------------------------------------------------------------
    // Host-declared relay DFB
    // -----------------------------------------------------------------------

    // Producer-only view over the DFB owned by this CrossNode object. Reserve/push
    // stay here; pop_front waits on that same DFB so WH/BH and Quasar share one object.
    class RelayView {
    public:
        FORCE_INLINE void reserve_back(uint16_t num_entries) { dfb_.reserve_back(num_entries); }
        FORCE_INLINE void push_back(uint16_t num_entries) { dfb_.push_back(num_entries); }

    private:
        friend class CrossNodeDFB;
        FORCE_INLINE explicit RelayView(DataflowBuffer& dfb) : dfb_(dfb) {}
        DataflowBuffer& dfb_;
    };

    // Open the relay declared by CreateCrossNodeRelayDataflowBuffer on the host.
    // No token is accepted here: receiver DM kernels cannot select an arbitrary
    // local DFB.
    FORCE_INLINE RelayView bind_relay() {
        const CrossNodeReceiverDFBInterface& iface = interface_.receiver;
        ASSERT(iface.relay_id != RELAY_DFB_INVALID);
        ASSERT(relay_dfb_ == nullptr);
        relay_dfb_ = new (relay_dfb_storage_) DataflowBuffer(RelayDFBBindingToken{iface.relay_id});
        const uintptr_t entries_acked_ptr = reinterpret_cast<uintptr_t>(get_cb_tiles_acked_ptr(iface.relay_id));
        relay_entries_acked_checkpoint_ = static_cast<uint16_t>(reg_read(entries_acked_ptr));
        return RelayView(*relay_dfb_);
    }
#endif

    // -----------------------------------------------------------------------
    // Accessors
    // -----------------------------------------------------------------------

    // Number of receivers connected to this CrossNodeDFB (sender participant cores only).
    FORCE_INLINE uint32_t num_receivers() {
        const CrossNodeSenderDFBInterface& iface = interface_.sender;
        return cross_node_dfb_num_receivers(iface.num_receivers_and_remote_pages_sent_ptr);
    }

    // Next write address in one receiver's ring, derived from that receiver's credits.
    // Every receiver has its own position, so callers that fan out unevenly must ask
    // per receiver.
    FORCE_INLINE uint32_t get_write_ptr(uint32_t receiver_idx = 0) {
        CrossNodeSenderDFBInterface& iface = interface_.sender;
        return iface.fifo_start_addr + derived_wr_offset(iface, receiver_idx);
    }

    FORCE_INLINE uint32_t get_read_ptr() { return interface_.receiver.fifo_rd_ptr; }

    FORCE_INLINE uint32_t get_entry_size() { return interface_.sender.fifo_page_size; }

private:
    CrossNodeDFBInterface interface_;

#if defined(KERNEL_BUILD) && !defined(COMPILE_FOR_TRISC)
    DataflowBuffer* relay_dfb_ = nullptr;
    alignas(DataflowBuffer) unsigned char relay_dfb_storage_[sizeof(DataflowBuffer)];
    uint16_t relay_entries_acked_checkpoint_ = 0;

    FORCE_INLINE void wait_relay_consumed(uint32_t num_entries) {
        ASSERT(num_entries <= relay_dfb_->get_local_num_entries());
        WAYPOINT("CNCW");
        const uint16_t relay_dfb_id = relay_dfb_->get_id();
        const uintptr_t entries_acked_ptr = reinterpret_cast<uintptr_t>(get_cb_tiles_acked_ptr(relay_dfb_id));
        uint16_t entries_acked;
        do {
            invalidate_l1_cache();
            entries_acked = static_cast<uint16_t>(reg_read(entries_acked_ptr));
        } while (static_cast<uint16_t>(entries_acked - relay_entries_acked_checkpoint_) < num_entries);
        relay_entries_acked_checkpoint_ = static_cast<uint16_t>(relay_entries_acked_checkpoint_ + num_entries);
        WAYPOINT("CNCD");
    }

    FORCE_INLINE void pop_front_impl(uint32_t num_entries, const Noc& noc) {
        CrossNodeReceiverDFBInterface& iface = interface_.receiver;
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
#endif

    // Read a word from the config page.
    FORCE_INLINE uint32_t get_config_word(uint32_t config_ptr, uint32_t word_idx) {
        return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(config_ptr)[word_idx];
    }

    // Local entries_sent counter for one receiver (L1_ALIGNMENT units; written only by
    // this core, remotely mirrored on the receiver).
    FORCE_INLINE static volatile tt_l1_ptr uint32_t* local_sent_ptr(
        const CrossNodeSenderDFBInterface& iface, uint32_t receiver_idx) {
        return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(iface.aligned_pages_sent_ptr) +
               (2 * receiver_idx * L1_ALIGNMENT / sizeof(uint32_t));
    }

    // Distance from fifo_start_addr at which a write wraps.
    FORCE_INLINE static uint32_t wrap_offset(const CrossNodeSenderDFBInterface& iface) {
        return iface.fifo_limit_page_aligned - iface.fifo_start_addr;
    }

    // Byte offset of a receiver's next free slot given its entries_sent counter value.
    // The counter is monotonic and resets to zero on every launch, so the credited byte
    // count modulo the ring size is exactly the write position — no stored cursor needed.
    FORCE_INLINE static uint32_t wr_offset_from_sent(const CrossNodeSenderDFBInterface& iface, uint32_t sent_units) {
        const uint32_t ring_units = wrap_offset(iface) / L1_ALIGNMENT;
        return (sent_units % ring_units) * L1_ALIGNMENT;
    }

    FORCE_INLINE static uint32_t derived_wr_offset(const CrossNodeSenderDFBInterface& iface, uint32_t receiver_idx) {
        return wr_offset_from_sent(iface, *local_sent_ptr(iface, receiver_idx));
    }

    // Producer writes must be contiguous (same rule as local CBs): wr_offset + len must
    // land at or before the limit. Crossing the wrap in one call is illegal.
    FORCE_INLINE static void assert_contiguous_bytes(
        const CrossNodeSenderDFBInterface& iface, uint32_t wr_offset, uint32_t len_bytes) {
        ASSERT(wr_offset + len_bytes <= wrap_offset(iface));
    }

    FORCE_INLINE static void assert_contiguous_write(
        const CrossNodeSenderDFBInterface& iface, uint32_t wr_offset, uint32_t num_entries) {
        assert_contiguous_bytes(iface, wr_offset, num_entries * iface.fifo_page_size);
    }

    // Credit units a contiguous write of num_entries consumes.
    FORCE_INLINE static uint32_t units_for_write(const CrossNodeSenderDFBInterface& iface, uint32_t num_entries) {
        return (num_entries * iface.fifo_page_size) / L1_ALIGNMENT;
    }
};

}  // namespace experimental

#endif  // !ARCH_QUASAR
