// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// WH/BH only for now; Quasar CrossNode device API is a follow-up.
#ifndef ARCH_QUASAR

#include <cstdint>
#include "internal/persistent_dfb_init.h"
#include "internal/circular_buffer_interface.h"
#include "api/alignment.h"
#include "api/debug/assert.h"
#include "api/debug/waypoint.h"
#include "hostdev/remote_dfb_config_layout.h"
#include "hostdev/remote_dfb_constants.h"
#include "internal/risc_attribs.h"

#if defined(KERNEL_BUILD) && !defined(COMPILE_FOR_TRISC)
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

// PersistentDFB: device-side kernel class for a cross-program durable remote DFB (WH/BH).
// Config pages + credits persist across programs; ctor loads word[4]
// (PERSISTENT_DFB_CFG_FIFO_PTR_CHECKPOINT — durable sender wr / receiver rd cursor)
// and may resize to this launch's dense entry_size. commit() / dtor store ptr back
// when the epoch (word[2] fifo_start, word[5] applied_entry_size) matches the iface.
//
// Same push/pop/write API as CrossNodeDFB. Mid-flight page-size changes use
// set_entry_size / set_receiver_entry_size at author-defined
// safe points; unilateral Attach(E2) while a peer is still live on E1 is illegal.
//
// Sync counters (pages_sent / pages_acked) are in L1_ALIGNMENT-byte units.
//
// Each receiver owns a private ring, so the sender needs an independent write position
// per receiver. No cursor state is stored for that: a receiver's write offset is derived
// from its local entries_sent counter (sent % ring), which persists across programs
// (unlike CrossNode, which zeros credits every launch).
//
// Writes are contiguous: a reserve/write/push of n entries must fit from the current
// write position to fifo_limit without straddling the wrap (same rule as local CBs).
//
// ═══════════════════════════════════════════════════════════════════════
//  SENDER FLOWS
// ═══════════════════════════════════════════════════════════════════════
//
//  Flow A — Broadcast (same data to all receivers):
//    reserve_back(n);
//    write_broadcast(src, n);
//    flush_writes();
//    push_back(n);
//
//  Flow B — Receiver-contiguous / unique-per-receiver:
//    reserve_back(n);
//    write_to_receiver(0, src_a, n);
//    write_to_receiver(1, src_b, n);
//    flush_writes();
//    push_back(n);
//
//  Flow C — Per-receiver credit:
//    for r in 0..num_recv:
//      reserve_back_for_receiver(r, n);
//      write_to_receiver(r, src, n);
//      flush_writes();
//      push_back_to_receiver(r, n);
//
//  Flow D — Interleaved scatter (write_strided):
//    reserve_back(n);
//    write_strided(src, num_rows, pages_per_row, page_size);
//    flush_writes();
//    push_back(n);
//
//  Mid-flight resize (sender, coordinated with peers):
//    // only at an author-defined safe point
//    set_entry_size(E2);           // default: NOC credit fixup + barrier
//    // then continue with E2-sized pushes, or signal host to Attach a new consumer
//
// ═══════════════════════════════════════════════════════════════════════
//  RECEIVER FLOW
// ═══════════════════════════════════════════════════════════════════════
//
//  Standard receiver (DM consumes data):
//    wait_front(n);
//    auto rd = get_read_ptr();  // CoreLocalMem at fifo front
//    // process data at rd.get_address() / rd.get_unsafe_ptr() ...
//    pop_front(n);
//
//  Mid-flight resize (receiver):
//    set_receiver_entry_size(E2);  // use on receiver cores only
//
// ═══════════════════════════════════════════════════════════════════════
//  RELAY DFB FLOW — bridging PersistentDFB to Compute
// ═══════════════════════════════════════════════════════════════════════
//
//  Compute cannot issue NOC atomics. Data is bridged via a host-declared local
//  DataflowBuffer that aliases the Persistent ring. DM owns Persistent credits;
//  TRISC consumes through the normal local DFB API.
//
//  Host: AttachPersistentDFB(..., receivers) then CreatePersistentRelayDataflowBuffer.
//  DM deliberately receives no relay binding token and must use bind_relay().
//
//  DM (receiver kernel):
//    PersistentDFB pdfb(id);
//    auto relay = pdfb.bind_relay();  // aligns DM's local iface to post-resize cursor
//    while (has_more) {
//        relay.reserve_back(n);       // wait for local free space
//        pdfb.wait_front(n);          // wait for sender's data (pages_sent)
//        relay.push_back(n);          // publish via CB credits
//        relay.wait_consumed(n);      // wait for TRISC to free the local slots
//        pdfb.pop_front(n);           // advance DM rd_ptr + NOC-ack sender
//    }
//
//  Compute kernel (reads relay DFB, no PersistentDFB or NOC knowledge):
//    DataflowBuffer relay(RelayDFBBindingToken{relay_id, persistent_dfb_id});
//    // or dfb::relay from kernel_bindings_generated.h — construction snaps the
//    // borrowed iface to the durable checkpoint (O(1) launch-msg slot lookup)
//    relay.wait_front(n);
//    // consume ...
//    relay.pop_front(n);
//
class PersistentDFB {
public:
    FORCE_INLINE explicit PersistentDFB(uint8_t persistent_dfb_id) : persistent_dfb_id_(persistent_dfb_id) {
        const uint32_t launch_index = *GET_MAILBOX_ADDRESS_DEV(launch_msg_rd_ptr);
        const auto* launch_msg = GET_MAILBOX_ADDRESS_DEV(launch[launch_index]);
        const auto& kernel_config = launch_msg->kernel_config;
        ASSERT(kernel_config.persistent_dfb_offset != PERSISTENT_DFB_OFFSET_NONE);

        const uint32_t kernel_config_base = kernel_config.kernel_config_base[PROGRAMMABLE_CORE_TYPE];
        volatile tt_l1_ptr uint32_t* region =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(kernel_config_base + kernel_config.persistent_dfb_offset);
        ASSERT(persistent_dfb_id < region[0]);

        volatile tt_l1_ptr uint32_t* slot =
            region + PERSISTENT_DFB_REGION_HEADER_WORDS + persistent_dfb_id * UINT32_WORDS_PER_PERSISTENT_DFB_CONFIG;
        setup_persistent_dfb_interface(
            interface_, /*config_page_addr=*/slot[0], /*entry_size=*/slot[1], /*relay_dfb_id=*/slot[2]);

#if defined(KERNEL_BUILD) && !defined(COMPILE_FOR_TRISC)

        volatile tt_l1_ptr uint32_t* l1_config =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(interface_.sender.config_ptr);
        const bool is_sender = static_cast<bool>(l1_config[REMOTE_DFB_CFG_IS_SENDER]);
        const uint32_t dense_entry_size = slot[1];
        const uint8_t noc_id = noc_index;
        if (is_sender) {
            sync_sender_wr_ptr_from_credits();
            resize_sender_interface<true>(dense_entry_size, noc_id);
            l1_config[PERSISTENT_DFB_CFG_APPLIED_ENTRY_SIZE] = interface_.sender.fifo_page_size;
            barrier_sender_credits();
        } else {
            resize_receiver_interface<true>(dense_entry_size, noc_id);
            l1_config[PERSISTENT_DFB_CFG_APPLIED_ENTRY_SIZE] = interface_.receiver.fifo_page_size;
        }
#endif
    }

    FORCE_INLINE ~PersistentDFB() { commit(); }

    FORCE_INLINE void commit() {
        volatile tt_l1_ptr uint32_t* l1_config =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(interface_.sender.config_ptr);
        const bool is_sender = static_cast<bool>(l1_config[REMOTE_DFB_CFG_IS_SENDER]);
        const uint32_t epoch_fifo_start = l1_config[REMOTE_DFB_CFG_FIFO_START];
        const uint32_t epoch_entry_size = l1_config[PERSISTENT_DFB_CFG_APPLIED_ENTRY_SIZE];
        if (is_sender) {
            CrossNodeSenderDFBInterface& iface = interface_.sender;
            if (iface.fifo_start_addr == epoch_fifo_start && iface.fifo_page_size == epoch_entry_size) {
                l1_config[PERSISTENT_DFB_CFG_FIFO_PTR_CHECKPOINT] = iface.fifo_start_addr + derived_wr_offset(iface, 0);
            }
        } else {
            CrossNodeReceiverDFBInterface& iface = interface_.receiver;
            if (iface.fifo_start_addr == epoch_fifo_start && iface.fifo_page_size == epoch_entry_size) {
                l1_config[PERSISTENT_DFB_CFG_FIFO_PTR_CHECKPOINT] = iface.fifo_rd_ptr;
            }
        }
    }

#if defined(KERNEL_BUILD) && !defined(COMPILE_FOR_TRISC)
    // Coordinated sender resize. Credit fixup is mandatory: exposing a local-only
    // option would let the local cursor diverge silently from live receivers.
    FORCE_INLINE void set_entry_size(uint32_t entry_size) {
        volatile tt_l1_ptr uint32_t* l1_config =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(interface_.sender.config_ptr);
        ASSERT(static_cast<bool>(l1_config[REMOTE_DFB_CFG_IS_SENDER]));
        const uint8_t noc_id = noc_index;
        sync_sender_wr_ptr_from_credits();
        resize_sender_interface<true>(entry_size, noc_id);
        barrier_sender_credits();
        l1_config[PERSISTENT_DFB_CFG_APPLIED_ENTRY_SIZE] = interface_.sender.fifo_page_size;
    }

    // Coordinated receiver resize. Credit fixup is mandatory for the public API.
    FORCE_INLINE void set_receiver_entry_size(uint32_t entry_size) {
        volatile tt_l1_ptr uint32_t* l1_config =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(interface_.receiver.config_ptr);
        ASSERT(!static_cast<bool>(l1_config[REMOTE_DFB_CFG_IS_SENDER]));
        const uint8_t noc_id = noc_index;
        resize_receiver_interface<true>(entry_size, noc_id);
        l1_config[PERSISTENT_DFB_CFG_APPLIED_ENTRY_SIZE] = interface_.receiver.fifo_page_size;
    }
#endif

    // -----------------------------------------------------------------------
    // Sender-side API (same as CrossNodeDFB)
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
    FORCE_INLINE void pop_front(uint32_t num_entries, const Noc& noc = Noc{}) {
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
#endif  // KERNEL_BUILD && !COMPILE_FOR_TRISC

    // -----------------------------------------------------------------------
    // Accessors
    // -----------------------------------------------------------------------

    // Number of receivers connected to this PersistentDFB (sender participant cores only).
    FORCE_INLINE uint32_t num_receivers() {
        const CrossNodeSenderDFBInterface& iface = interface_.sender;
        return cross_node_dfb_num_receivers(iface.num_receivers_and_remote_pages_sent_ptr);
    }

    // Front of the receiver ring as a typed L1 handle (raw address via .get_address()).
    FORCE_INLINE CoreLocalMem<uint32_t> get_read_ptr() {
        return CoreLocalMem<uint32_t>(interface_.receiver.fifo_rd_ptr);
    }

    FORCE_INLINE uint32_t get_entry_size() { return interface_.sender.fifo_page_size; }

#if defined(KERNEL_BUILD) && !defined(COMPILE_FOR_TRISC)
    // -----------------------------------------------------------------------
    // Host-declared relay DFB (DM → compute)
    // -----------------------------------------------------------------------

    // Producer-only view. It intentionally exposes only the local operations the
    // receiver DM owns; TRISC constructs the same local DFB from its normal token.
    class RelayView {
    public:
        FORCE_INLINE explicit RelayView(uint16_t relay_dfb_id) : dfb_(RelayDFBBindingToken{relay_dfb_id}) {}

        FORCE_INLINE void reserve_back(uint16_t num_entries) { dfb_.reserve_back(num_entries); }
        FORCE_INLINE void push_back(uint16_t num_entries) { dfb_.push_back(num_entries); }

        FORCE_INLINE void wait_consumed(uint16_t num_entries) {
            ASSERT(num_entries <= dfb_.get_local_num_entries());
            WAYPOINT("PDCW");
            const uint16_t relay_dfb_id = dfb_.get_id();
            volatile tt_reg_ptr uint32_t* entries_received_ptr = get_cb_tiles_received_ptr(relay_dfb_id);
            const uintptr_t entries_acked_ptr = reinterpret_cast<uintptr_t>(get_cb_tiles_acked_ptr(relay_dfb_id));
            uint16_t entries_received;
            uint16_t entries_acked;
            do {
                invalidate_l1_cache();
                entries_received = static_cast<uint16_t>(entries_received_ptr[0]);
                entries_acked = static_cast<uint16_t>(reg_read(entries_acked_ptr));
            } while (entries_received != entries_acked);
            WAYPOINT("PDCD");
        }

    private:
        DataflowBuffer dfb_;
    };

    // Open the relay declared by CreatePersistentRelayDataflowBuffer. Aligns the local
    // DFB iface to the post-resize Persistent receiver cursor (TRISC is JIT-aligned separately).
    FORCE_INLINE RelayView bind_relay() {
        const CrossNodeReceiverDFBInterface& iface = interface_.receiver;
        ASSERT(iface.relay_id != RELAY_DFB_INVALID);
        align_local_dfb_to_persistent_receiver_iface(iface.relay_id, iface);
        return RelayView(iface.relay_id);
    }
#endif

private:
    CrossNodeDFBInterface interface_;
    uint8_t persistent_dfb_id_ = 0;

#ifdef PERSISTENT_DFB_TEST_HELPERS
    friend void test_stale_commit_after_resize(
        PersistentDFB&, uint32_t new_entry_size, uint32_t stale_entry_size, uint32_t poison_wr_ptr);
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

    // Sender resize has one cursor/checkpoint but a sender may otherwise advance receivers
    // independently. Resizing is valid only at a coordinated point where all receiver
    // credit-derived cursors agree.
    FORCE_INLINE void sync_sender_wr_ptr_from_credits() {
        CrossNodeSenderDFBInterface& iface = interface_.sender;
        const uint32_t num_recv = cross_node_dfb_num_receivers(iface.num_receivers_and_remote_pages_sent_ptr);
        ASSERT(num_recv > 0);
        const uint32_t wr_offset = derived_wr_offset(iface, 0);
        for (uint32_t i = 1; i < num_recv; ++i) {
            ASSERT(derived_wr_offset(iface, i) == wr_offset);
        }
        iface.fifo_wr_ptr = iface.fifo_start_addr + wr_offset;
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

#if defined(KERNEL_BUILD) && !defined(COMPILE_FOR_TRISC)
    // Snap sender wr_ptr / page size to `page_size`. When update_remote_over_noc, also NOC-adjust
    // remote pages_sent for any pad/wrap introduced by the snap.
    template <bool update_remote_over_noc = false>
    FORCE_INLINE void resize_sender_interface(
        uint32_t page_size,
        uint8_t noc,
        uint8_t nm = detail::default_noc_mode,
        bool posted = true,
        uint8_t cmd_buf = detail::default_cmd_buf) {
        CrossNodeSenderDFBInterface& sender_cb_interface = interface_.sender;
        ASSERT(static_cast<bool>(
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sender_cb_interface.config_ptr)[REMOTE_DFB_CFG_IS_SENDER]));
        ASSERT(page_size % REMOTE_CIRCULAR_BUFFER_ALIGNED_PAGE_SIZE == 0);
        uint32_t fifo_size = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sender_cb_interface.config_ptr)[3];
        uint32_t fifo_start_addr = sender_cb_interface.fifo_start_addr;
        uint32_t fifo_wr_ptr = sender_cb_interface.fifo_wr_ptr;
        uint32_t cb_size_page_aligned = fifo_size - fifo_size % page_size;
        uint32_t fifo_limit_page_aligned = fifo_start_addr + cb_size_page_aligned;

        uint32_t next_fifo_wr_ptr = fifo_start_addr + align(fifo_wr_ptr - fifo_start_addr, page_size);
        if constexpr (update_remote_over_noc) {
            uint32_t aligned_page_adjustment = 0;
            if (next_fifo_wr_ptr >= fifo_limit_page_aligned) {
                aligned_page_adjustment =
                    (fifo_start_addr + fifo_size - fifo_wr_ptr) / REMOTE_CIRCULAR_BUFFER_ALIGNED_PAGE_SIZE;
                next_fifo_wr_ptr = fifo_start_addr;
            } else if (next_fifo_wr_ptr != fifo_wr_ptr) {
                aligned_page_adjustment = (next_fifo_wr_ptr - fifo_wr_ptr) / REMOTE_CIRCULAR_BUFFER_ALIGNED_PAGE_SIZE;
            }
            if (aligned_page_adjustment != 0) {
                if (nm == DM_DYNAMIC_NOC) {
                    detail::update_pages_sent<DM_DYNAMIC_NOC>(
                        reinterpret_cast<const RemoteSenderCBInterface&>(sender_cb_interface),
                        aligned_page_adjustment,
                        noc,
                        posted,
                        cmd_buf);
                } else {
                    detail::update_pages_sent<DM_DEDICATED_NOC>(
                        reinterpret_cast<const RemoteSenderCBInterface&>(sender_cb_interface),
                        aligned_page_adjustment,
                        noc,
                        posted,
                        cmd_buf);
                }
            }
        } else if (next_fifo_wr_ptr >= fifo_limit_page_aligned) {
            next_fifo_wr_ptr = fifo_start_addr;
        }
        sender_cb_interface.fifo_wr_ptr = next_fifo_wr_ptr;
        sender_cb_interface.fifo_limit_page_aligned = fifo_limit_page_aligned;
        sender_cb_interface.fifo_page_size = page_size;
    }

    // Snap receiver rd_ptr / page size to `page_size`. When update_remote_over_noc, also NOC-adjust
    // remote pages_acked for any pad/wrap introduced by the snap.
    template <bool update_remote_over_noc = false>
    FORCE_INLINE void resize_receiver_interface(
        uint32_t page_size,
        uint8_t noc,
        uint8_t nm = detail::default_noc_mode,
        bool posted = true,
        uint8_t cmd_buf = detail::default_cmd_buf) {
        CrossNodeReceiverDFBInterface& receiver_cb_interface = interface_.receiver;
        ASSERT(!static_cast<bool>(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
            receiver_cb_interface.config_ptr)[REMOTE_DFB_CFG_IS_SENDER]));
        ASSERT(page_size % REMOTE_CIRCULAR_BUFFER_ALIGNED_PAGE_SIZE == 0);
        uint32_t fifo_size = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(receiver_cb_interface.config_ptr)[3];
        uint32_t fifo_start_addr = receiver_cb_interface.fifo_start_addr;
        uint32_t fifo_rd_ptr = receiver_cb_interface.fifo_rd_ptr;
        uint32_t cb_size_page_aligned = fifo_size - fifo_size % page_size;
        uint32_t fifo_limit_page_aligned = fifo_start_addr + cb_size_page_aligned;

        uint32_t next_fifo_rd_ptr = fifo_start_addr + align(fifo_rd_ptr - fifo_start_addr, page_size);
        if constexpr (update_remote_over_noc) {
            uint32_t aligned_page_adjustment = 0;
            if (next_fifo_rd_ptr >= fifo_limit_page_aligned) {
                aligned_page_adjustment =
                    (fifo_start_addr + fifo_size - fifo_rd_ptr) / REMOTE_CIRCULAR_BUFFER_ALIGNED_PAGE_SIZE;
                next_fifo_rd_ptr = fifo_start_addr;
            } else if (next_fifo_rd_ptr != fifo_rd_ptr) {
                aligned_page_adjustment = (next_fifo_rd_ptr - fifo_rd_ptr) / REMOTE_CIRCULAR_BUFFER_ALIGNED_PAGE_SIZE;
            }
            if (aligned_page_adjustment != 0) {
                uint32_t pages_acked = 0;
                uint32_t pages_sent = 0;
                uint32_t num_pages_recv = 0;
                volatile tt_l1_ptr uint32_t* pages_acked_ptr =
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(receiver_cb_interface.aligned_pages_acked_ptr);
                volatile tt_l1_ptr uint32_t* pages_sent_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                    receiver_cb_interface.aligned_pages_acked_ptr - L1_ALIGNMENT);
                do {
                    invalidate_l1_cache();
                    pages_acked = *pages_acked_ptr;
                    pages_sent = *pages_sent_ptr;
                    num_pages_recv = pages_sent - pages_acked;
                } while (num_pages_recv < aligned_page_adjustment);

                if (nm == DM_DYNAMIC_NOC) {
                    detail::update_pages_acked<DM_DYNAMIC_NOC>(
                        reinterpret_cast<const RemoteReceiverCBInterface&>(receiver_cb_interface),
                        aligned_page_adjustment,
                        noc,
                        posted,
                        cmd_buf);
                } else {
                    detail::update_pages_acked<DM_DEDICATED_NOC>(
                        reinterpret_cast<const RemoteReceiverCBInterface&>(receiver_cb_interface),
                        aligned_page_adjustment,
                        noc,
                        posted,
                        cmd_buf);
                }
            }
        } else if (next_fifo_rd_ptr >= fifo_limit_page_aligned) {
            next_fifo_rd_ptr = fifo_start_addr;
        }
        receiver_cb_interface.fifo_rd_ptr = next_fifo_rd_ptr;
        receiver_cb_interface.fifo_limit_page_aligned = fifo_limit_page_aligned;
        receiver_cb_interface.fifo_page_size = page_size;
    }

    // Wait until every receiver has acked all locally recorded pages_sent (post-resize barrier).
    FORCE_INLINE void barrier_sender_credits() {
        const CrossNodeSenderDFBInterface& iface = interface_.sender;
        ASSERT(static_cast<bool>(
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(iface.config_ptr)[REMOTE_DFB_CFG_IS_SENDER]));
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
    }
#endif  // KERNEL_BUILD && !COMPILE_FOR_TRISC
};

}  // namespace experimental

#endif  // !ARCH_QUASAR
