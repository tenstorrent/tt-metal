// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "internal/prefetcher_pipe_init.h"
#include "internal/circular_buffer_interface.h"
#include "api/alignment.h"
#include "api/debug/assert.h"
#include "api/debug/waypoint.h"
#include "api/kernel_thread_globals.h"
#include "hostdev/remote_dfb_config_layout.h"
#include "hostdev/remote_dfb_constants.h"
#include "internal/risc_attribs.h"
#include "api/dataflow/dfb_binding_token.h"
#include "api/dataflow/prefetcher_pipe_binding_token.h"

#if !defined(COMPILE_FOR_TRISC)
#include <optional>
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/core_local_mem.h"
#include "api/remote_circular_buffer.h"

// Credit helpers take Remote*CBInterface; CrossNode prefix layout matches for reinterpret_cast
// (CrossNode structs carry trailing PrefetcherPipe-only fields).
static_assert(sizeof(CrossNodeSenderDFBInterface) >= sizeof(RemoteSenderCBInterface));
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
class PrefetcherPipe;
}

#if !defined(COMPILE_FOR_TRISC)
// dst_args_type must be visible before PrefetcherPipe::write_* method bodies.
// RISC-V g++ parses those templates at class definition (-Wtemplate-body).
template <>
struct noc_traits_t<experimental::PrefetcherPipe> {
    struct src_args_type {};
    struct dst_args_type {
        uint32_t receiver_idx{};
    };

    template <Noc::AddressType address_type>
    static uint32_t src_addr(const experimental::PrefetcherPipe& src, const Noc& noc, const src_args_type& args);

    template <Noc::AddressType address_type>
    static uint64_t dst_addr(const experimental::PrefetcherPipe& dst, const Noc& noc, const dst_args_type& args);
};
#endif

namespace experimental {

// PrefetcherPipe: device-side kernel class for a cross-program durable remote DFB.
// Config pages + credits persist across programs; ctor loads word[4]
// (PREFETCHER_PIPE_CFG_FIFO_PTR_CHECKPOINT — durable sender wr / receiver rd cursor).
// If this program's dense entry_size differs from word[5] (applied_entry_size), ctor
// resizes with NOC pad credits. Same-epoch relaunch skips that so
// a producer can keep filling free space while outstanding credits wait for a consumer.
// commit() / dtor store ptr back when the epoch (word[2] fifo_start, word[5]
// applied_entry_size) matches the iface.
//
// Same push/pop/write API as CrossNodeDFB. Mid-flight page-size changes use
// set_entry_size / set_receiver_entry_size after that endpoint's E1 operations.
// The sender may switch to E2 and prefetch while a live receiver still consumes E1.
//
// Sync counters (pages_sent / pages_acked) are in L1_ALIGNMENT-byte units.
//
// Each receiver owns a private ring, so the sender needs an independent write position
// per receiver. Each one is stored beside that receiver's credit counters as a byte offset
// from the ring base, and is advanced whenever that receiver is credited. It is not derived
// from entries_sent: that counter wraps at 2^32, which only preserves a (sent % ring_units)
// derivation when ring_units is a power of two. Cursors persist across programs for the same
// reason the counters do (unlike CrossNode, which zeros credits every launch).
//
// Writes are contiguous: a reserve/write/push of n entries must fit from the current
// write position to fifo_limit without straddling the wrap (same rule as local CBs).
//
// ═══════════════════════════════════════════════════════════════════════
//  SENDER FLOWS
// ═══════════════════════════════════════════════════════════════════════
//
//  Multi-DM (Quasar): get_num_threads()/get_my_thread_id() partition receivers.
//  Hart tid owns { r | r % P == tid }. Collective and per-receiver APIs skip
//  non-owned receivers, so the same Flow A/B/C/D source works for P=1 and P>1.
//  (Sharing one receiver ring across sender DMs — per-hart STRIDE credits — is
//  deferred; with R < P some harts idle.)
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
//    for r in 0..num_receivers:
//      reserve_back_for_receiver(r, n);  // no-op if !owns_receiver(r)
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
//  Mid-flight resize (sender):
//    set_entry_size(E2);           // snap forward + publish pad credits; no drain
//    // then continue with E2-sized pushes, or signal host to launch a new consumer Program
//
// ═══════════════════════════════════════════════════════════════════════
//  RECEIVER FLOW
// ═══════════════════════════════════════════════════════════════════════
//
//  Multi-DM receiver: Quasar lane credits. Active lane count P is the receiver
//  KernelSpec's num_threads (a relay DFB's num_producers must equal it); it reaches the
//  device in the program's kernel-config slot. The persistent page reserves
//  PREFETCHER_PIPE_MAX_CREDIT_LANES slots per receiver. Hart tid owns lane tid;
//  wait_front(n)/pop_front(n) are n owned strides (entries tid, tid+P, …). Sender
//  stripes pages_sent to lane (entry_idx % P). With P=1, behavior matches a single
//  (sent,acked) pair.
//
//  Standard receiver (DM consumes data):
//    wait_front(n);
//    auto lock = scoped_read_lock(n);
//    auto rd = lock.get_ptr();  // CoreLocalMem at fifo front
//    // process data at rd.get_address() / rd.get_unsafe_ptr() ...
//    pop_front(n);
//
//  Mid-flight resize (receiver):
//    set_receiver_entry_size(E2);  // use on receiver cores only
//
// ═══════════════════════════════════════════════════════════════════════
//  RELAY DFB FLOW — bridging PrefetcherPipe to Compute
// ═══════════════════════════════════════════════════════════════════════
//
//  Compute cannot issue NOC atomics. Data is bridged via a host-declared local
//  DataflowBuffer that aliases the PrefetcherPipe ring. Pipe consumers are the
//  relay producers (num_producers may be >1); TRISC/DM consumers use the normal
//  local DFB API (num_consumers / cap).
//
//  Host: a DataflowBufferSpec with prefetcher_pipe_relays naming the pipe(s), produced by
//  the receiver DM kernel that binds those pipes and consumed by the compute kernel.
//  DM deliberately receives no relay binding token and must use bind_relay().
//
//  DM (receiver kernel) — single producer:
//    PrefetcherPipe pipe(pipe::in);
//    auto relay = pipe.bind_relay();
//    while (has_more) {
//        relay.reserve_back(n);
//        pipe.wait_front(n);
//        relay.push_back(n);
//        pipe.pop_front(n);  // wait for relay consumers, then NOC-ack sender
//    }
//
//  DM — multi producer (P=num_threads, pap=STRIDED, pipe lanes=P):
//    relay.reserve_back(n);
//    pipe.wait_front(n);   // n owned strides (lane credits)
//    relay.push_back(n);
//    pipe.pop_front(n);
//
//  Compute kernel (reads relay DFB, no PrefetcherPipe or NOC knowledge):
//    DataflowBuffer relay(dfb::relay);  // RelayDFBBindingToken from kernel_bindings_generated.h
//    // construction snaps the borrowed iface to the durable checkpoint (O(1) launch-msg
//    // slot lookup)
//    relay.wait_front(n);
//    // consume ...
//    relay.pop_front(n);
//
class PrefetcherPipe {
public:
    FORCE_INLINE explicit PrefetcherPipe(uint8_t prefetcher_pipe_id) : prefetcher_pipe_id_(prefetcher_pipe_id) {
        const uint32_t launch_index = *GET_MAILBOX_ADDRESS_DEV(launch_msg_rd_ptr);
        const auto* launch_msg = GET_MAILBOX_ADDRESS_DEV(launch[launch_index]);
        const auto& kernel_config = launch_msg->kernel_config;
        ASSERT(kernel_config.prefetcher_pipe_offset != REMOTE_DFB_OFFSET_NONE);

        const uint32_t kernel_config_base = kernel_config.kernel_config_base[PROGRAMMABLE_CORE_TYPE];
        volatile tt_l1_ptr uint32_t* region =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(kernel_config_base + kernel_config.prefetcher_pipe_offset);
        ASSERT(prefetcher_pipe_id < load_prefetcher_pipe_config_word(region, 0));

        volatile tt_l1_ptr uint32_t* slot =
            region + REMOTE_DFB_REGION_HEADER_WORDS + prefetcher_pipe_id * UINT32_WORDS_PER_REMOTE_DFB_CONFIG;
        const uint32_t config_page_addr = load_prefetcher_pipe_config_word(slot, 0);
        const uint32_t dense_entry_size = load_prefetcher_pipe_config_word(slot, 1);
        const uint32_t relay_word = load_prefetcher_pipe_config_word(slot, 2);
        setup_prefetcher_pipe_interface(interface_, config_page_addr, dense_entry_size, relay_word);

        // Fixed for the kernel's lifetime; read once here so the credit hot path never
        // re-loads them from the (uncached) config page.
        volatile tt_l1_ptr uint32_t* l1_config =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(interface_.sender.config_ptr);
        fifo_size_ = load_prefetcher_pipe_config_word(l1_config, REMOTE_DFB_CFG_FIFO_SIZE);
#ifdef ARCH_QUASAR
        // Active lanes P come with the program (kernel-config slot), not from the persistent
        // page, so every kernel of one program sees the same P in dispatch order.
        num_credit_lanes_ = static_cast<uint8_t>(prefetcher_pipe_slot_credit_lanes(relay_word));
#else
        num_credit_lanes_ = 1;  // capacity is 1 lane; host never packs a P field for WH/BH
#endif
        ASSERT(num_credit_lanes_ >= 1 && num_credit_lanes_ <= credit_lane_capacity());

#if !defined(COMPILE_FOR_TRISC)

        const bool is_sender = static_cast<bool>(load_prefetcher_pipe_config_word(l1_config, REMOTE_DFB_CFG_IS_SENDER));
        const uint32_t applied_entry_size =
            load_prefetcher_pipe_config_word(l1_config, PREFETCHER_PIPE_CFG_APPLIED_ENTRY_SIZE);
        // Same-epoch relaunch: setup already restored the checkpoint + this program's
        // entry size. Skip resize so a producer can relaunch and keep
        // filling free space while outstanding credits wait for an offline consumer.
        // A changed entry size snaps this endpoint and publishes/consumes pad credits;
        // it does not drain payload already in flight at the peer's old entry size.
        // Multi-DM: only tid 0 mutates shared config; others re-setup after the barrier.
        // Drop any cached copy of this core's credit words before anyone touches them, so a
        // host reset / the previous program's flushed values in TL1 are what every hart sees.
        // Not multi-DM specific: it is the Quasar cost of reading credit words through the
        // cache at all (P == 1 included). Compiles out on WH/BH, and sync_threads() returns
        // immediately when the kernel has one thread, so the single-thread path only pays
        // the one-time invalidate here and the flush in the dtor.
        invalidate_local_credit_lines(/*lane_offset_bytes=*/0);
        sync_threads();
        if (dense_entry_size != applied_entry_size) {
            if (get_my_thread_id() == 0) {
                const uint8_t noc_id = noc_index;
                if (is_sender) {
                    resize_sender_interface<true>(dense_entry_size, noc_id);
                    store_prefetcher_pipe_config_word(
                        l1_config, PREFETCHER_PIPE_CFG_APPLIED_ENTRY_SIZE, interface_.sender.fifo_page_size);
                } else {
                    resize_receiver_interface<true>(dense_entry_size, noc_id);
                    store_prefetcher_pipe_config_word(
                        l1_config, PREFETCHER_PIPE_CFG_APPLIED_ENTRY_SIZE, interface_.receiver.fifo_page_size);
                }
            }
            sync_threads();
            if (get_my_thread_id() != 0) {
                setup_prefetcher_pipe_interface(interface_, config_page_addr, dense_entry_size, relay_word);
            }
        }
        if (!is_sender) {
            bind_receiver_credit_lane();
        }
#endif
    }

    // Metal 2.0: construct from a `pipe::<accessor>` token (kernel_bindings_generated.h). The
    // token is the program slot id; on every node the slot record names the pipe present there.
    FORCE_INLINE explicit PrefetcherPipe(PrefetcherPipeBindingToken token) :
        PrefetcherPipe(token.prefetcher_pipe_id()) {}

    FORCE_INLINE ~PrefetcherPipe() {
        commit();
#if defined(ARCH_QUASAR) && !defined(COMPILE_FOR_TRISC)
        // Write this core's credit words back to TL1 so they outlive the kernel (next program's
        // ctor invalidates and re-reads them; the host may peek at them). After every hart's
        // last cached store; one flush covers all harts (L2 flush probes every DM L1 D$).
        sync_threads();
        if (get_my_thread_id() == 0) {
            flush_local_credit_lines(receiver_lane_offset_bytes());
        }
#endif
#if defined(ARCH_BLACKHOLE) && !defined(COMPILE_FOR_TRISC)
        // BH forces noc_fast_atomic_increment to non-posted; drain only the NOCs that
        // issued PrefetcherPipe credit atomics (may differ from compile-time noc_index).
        if (atomic_noc_mask_ & 0x1u) {
            noc_async_atomic_barrier(0);
        }
        if (atomic_noc_mask_ & 0x2u) {
            noc_async_atomic_barrier(1);
        }
#endif
    }

    // Persist durable cursors. Multi-DM: only tid 0 writes the shared checkpoint word.
    FORCE_INLINE void commit() {
        if (get_my_thread_id() != 0) {
            return;
        }
        volatile tt_l1_ptr uint32_t* l1_config =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(interface_.sender.config_ptr);
        const bool is_sender = static_cast<bool>(load_prefetcher_pipe_config_word(l1_config, REMOTE_DFB_CFG_IS_SENDER));
        const uint32_t epoch_fifo_start = load_prefetcher_pipe_config_word(l1_config, REMOTE_DFB_CFG_FIFO_START);
        const uint32_t epoch_entry_size =
            load_prefetcher_pipe_config_word(l1_config, PREFETCHER_PIPE_CFG_APPLIED_ENTRY_SIZE);
        if (is_sender) {
            CrossNodeSenderDFBInterface& iface = interface_.sender;
            if (iface.fifo_start_addr == epoch_fifo_start && iface.fifo_page_size == epoch_entry_size) {
                store_prefetcher_pipe_config_word(
                    l1_config,
                    PREFETCHER_PIPE_CFG_FIFO_PTR_CHECKPOINT,
                    iface.fifo_start_addr + sender_wr_offset(iface, 0));
            }
        } else {
            CrossNodeReceiverDFBInterface& iface = interface_.receiver;
            if (iface.fifo_start_addr == epoch_fifo_start && iface.fifo_page_size == epoch_entry_size) {
                store_prefetcher_pipe_config_word(
                    l1_config, PREFETCHER_PIPE_CFG_FIFO_PTR_CHECKPOINT, iface.fifo_rd_ptr);
            }
        }
    }

#if !defined(COMPILE_FOR_TRISC)
    // Change the size of subsequent sender operations. Each receiver's stored write
    // cursor is snapped forward and the skipped bytes are published as pad credits,
    // matching GlobalCB. Outstanding old-size payload need not drain.
    FORCE_INLINE void set_entry_size(uint32_t entry_size) {
        volatile tt_l1_ptr uint32_t* l1_config =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(interface_.sender.config_ptr);
        ASSERT(static_cast<bool>(load_prefetcher_pipe_config_word(l1_config, REMOTE_DFB_CFG_IS_SENDER)));
        const uint8_t noc_id = noc_index;
        resize_sender_interface<true>(entry_size, noc_id);
        store_prefetcher_pipe_config_word(
            l1_config, PREFETCHER_PIPE_CFG_APPLIED_ENTRY_SIZE, interface_.sender.fifo_page_size);
    }

    // Change the size of subsequent receiver operations after this receiver has
    // consumed its E1 entries. Consume the sender's matching pad credits, without
    // requiring the sender's newer E2 payload to drain.
    // If this core has a relay, also rewrite the local relay DFB iface (page size,
    // usable limit, rd/wr) so a RelayView from bind_relay() is not left on E1.
    // TRISC must already have consumed the previous-size tiles (relay-backed pop_front);
    // its local CB is a separate object aligned at DataflowBuffer construction.
    FORCE_INLINE void set_receiver_entry_size(uint32_t entry_size) {
        volatile tt_l1_ptr uint32_t* l1_config =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(interface_.receiver.config_ptr);
        ASSERT(!static_cast<bool>(load_prefetcher_pipe_config_word(l1_config, REMOTE_DFB_CFG_IS_SENDER)));
        const uint8_t noc_id = noc_index;
        resize_receiver_interface<true>(entry_size, noc_id);
        store_prefetcher_pipe_config_word(
            l1_config, PREFETCHER_PIPE_CFG_APPLIED_ENTRY_SIZE, interface_.receiver.fifo_page_size);
        const CrossNodeReceiverDFBInterface& iface = interface_.receiver;
        if (iface.relay_id != RELAY_DFB_INVALID) {
            align_local_dfb_to_prefetcher_pipe_receiver_iface(iface.relay_id, iface);
        }
    }
#endif

    // -----------------------------------------------------------------------
    // Sender-side API (same as CrossNodeDFB)
    // -----------------------------------------------------------------------

    // Spin until this hart's owned receivers have space for num_entries entries.
    // Multi-DM: each hart covers { r | r % P == tid }; together they cover all receivers.
    FORCE_INLINE void reserve_back(uint32_t num_entries) {
        WAYPOINT("GSRW");
        CrossNodeSenderDFBInterface& iface = interface_.sender;
        const uint32_t num_recv = cross_node_dfb_num_receivers(iface.num_receivers_and_remote_pages_sent_ptr);
        const uint32_t P = num_credit_lanes();

        for (uint32_t i = 0; i < num_recv; ++i) {
            if (!owns_receiver(i)) {
                continue;
            }
            const uint32_t wr_offset = sender_wr_offset(iface, i);
            assert_contiguous_write(iface, wr_offset, num_entries);
            if (P == 1) {
                const uint32_t num_units = fifo_size() / L1_ALIGNMENT;
                volatile tt_l1_ptr uint32_t* sent_ptr = local_sent_ptr(iface, i, 0);
                volatile tt_l1_ptr uint32_t* acked_ptr = local_acked_ptr(iface, i, 0);
                const uint32_t total_units_needed = units_for_write(iface, wr_offset, num_entries);
                uint32_t sent = 0;
                uint32_t acked = 0;
                do {
                    sent = *sent_ptr;
                    acked = load_remote_l1_credit(acked_ptr);
                } while ((num_units - (sent - acked)) < total_units_needed);
            } else {
                // Lane free space like tile counters: count entries per lane in this push.
                const uint32_t entry_size = iface.fifo_page_size;
                const uint32_t upe = units_per_entry(iface);
                const uint32_t cap = lane_capacity_units(iface);
                uint32_t counts[PREFETCHER_PIPE_MAX_CREDIT_LANES] = {};
                ASSERT(P <= PREFETCHER_PIPE_MAX_CREDIT_LANES);
                uint32_t entry_idx = wr_offset / entry_size;
                for (uint32_t e = 0; e < num_entries; ++e) {
                    counts[(entry_idx + e) % P] += upe;
                }
                for (uint32_t lane = 0; lane < P; ++lane) {
                    if (counts[lane] == 0) {
                        continue;
                    }
                    volatile tt_l1_ptr uint32_t* sent_ptr = local_sent_ptr(iface, i, lane);
                    volatile tt_l1_ptr uint32_t* acked_ptr = local_acked_ptr(iface, i, lane);
                    uint32_t sent = 0;
                    uint32_t acked = 0;
                    do {
                        sent = *sent_ptr;
                        acked = load_remote_l1_credit(acked_ptr);
                    } while ((cap - (sent - acked)) < counts[lane]);
                }
            }
        }
        WAYPOINT("GSRD");
    }

    // Spin until a SINGLE receiver (receiver_idx) has space for num_entries entries.
    // Use this for Flow C (per-receiver credit) to avoid blocking on unrelated receivers.
    // Multi-DM: no-op when this hart does not own receiver_idx.
    FORCE_INLINE void reserve_back_for_receiver(uint32_t receiver_idx, uint32_t num_entries) {
        if (!owns_receiver(receiver_idx)) {
            return;
        }
        WAYPOINT("GSRW");
        CrossNodeSenderDFBInterface& iface = interface_.sender;
        const uint32_t P = num_credit_lanes();
        const uint32_t wr_offset = sender_wr_offset(iface, receiver_idx);
        assert_contiguous_write(iface, wr_offset, num_entries);
        if (P == 1) {
            const uint32_t num_units = fifo_size() / L1_ALIGNMENT;
            volatile tt_l1_ptr uint32_t* sent_ptr = local_sent_ptr(iface, receiver_idx, 0);
            volatile tt_l1_ptr uint32_t* acked_ptr = local_acked_ptr(iface, receiver_idx, 0);
            const uint32_t total_units_needed = units_for_write(iface, wr_offset, num_entries);
            uint32_t sent = 0;
            uint32_t acked = 0;
            do {
                sent = *sent_ptr;
                acked = load_remote_l1_credit(acked_ptr);
            } while ((num_units - (sent - acked)) < total_units_needed);
        } else {
            const uint32_t entry_size = iface.fifo_page_size;
            const uint32_t upe = units_per_entry(iface);
            const uint32_t cap = lane_capacity_units(iface);
            uint32_t counts[PREFETCHER_PIPE_MAX_CREDIT_LANES] = {};
            ASSERT(P <= PREFETCHER_PIPE_MAX_CREDIT_LANES);
            uint32_t entry_idx = wr_offset / entry_size;
            for (uint32_t e = 0; e < num_entries; ++e) {
                counts[(entry_idx + e) % P] += upe;
            }
            for (uint32_t lane = 0; lane < P; ++lane) {
                if (counts[lane] == 0) {
                    continue;
                }
                volatile tt_l1_ptr uint32_t* sent_ptr = local_sent_ptr(iface, receiver_idx, lane);
                volatile tt_l1_ptr uint32_t* acked_ptr = local_acked_ptr(iface, receiver_idx, lane);
                uint32_t sent = 0;
                uint32_t acked = 0;
                do {
                    sent = *sent_ptr;
                    acked = load_remote_l1_credit(acked_ptr);
                } while ((cap - (sent - acked)) < counts[lane]);
            }
        }
        WAYPOINT("GSRD");
    }

#if !defined(COMPILE_FOR_TRISC)

    // ------------------------------------------------------------------
    // Write primitives — kick off NoC writes at each receiver's stored write cursor.
    // They do NOT credit that receiver, and crediting is what advances the cursor, so
    // repeating a write before crediting overwrites the same slots.
    // Call push_back() or push_back_to_receiver() after all writes.
    // Multi-DM: only owned receivers are written (same partition as reserve/push).
    // ------------------------------------------------------------------

    // Interleaved scatter
    // Writes rows from src interleaved across num_receivers destinations.
    // Each receiver i gets rows at src + i * (num_rows * coalesced_page_size),
    // written at that receiver's write position.
    template <typename Src>
    FORCE_INLINE void write_strided(
        const Noc& noc,
        const Src& src,
        uint32_t num_rows,
        uint32_t coalesced_num_pages_per_row,
        uint32_t coalesced_page_size,
        const typename noc_traits_t<Src>::src_args_type& src_args = {}) {
        CrossNodeSenderDFBInterface& iface = interface_.sender;
        const uint32_t num_recv = cross_node_dfb_num_receivers(iface.num_receivers_and_remote_pages_sent_ptr);

        const uint32_t row_bytes_per_recv = coalesced_num_pages_per_row * coalesced_page_size;
        const uint32_t bytes_per_recv = num_rows * row_bytes_per_recv;
        const uint32_t row_stride_in_stage = row_bytes_per_recv * num_recv;

        uint32_t src_addr = noc_traits_t<Src>::template src_addr<Noc::AddressType::LOCAL_L1>(src, noc, src_args);
        uint32_t recv_src_offset = 0;
        for (uint32_t i = 0; i < num_recv; ++i) {
            if (owns_receiver(i)) {
                const uint32_t wr_offset = sender_wr_offset(iface, i);
                assert_contiguous_bytes(iface, wr_offset, bytes_per_recv);

                uint32_t current_src_addr = src_addr + recv_src_offset;
                destination_offset_bytes_ = 0;
#ifdef ARCH_QUASAR
                // Non-posted: flush_writes waits for completion before pages_sent (posted can reorder).
                noc.set_async_write_state(*this, coalesced_page_size, {.receiver_idx = i});
#else
                noc.set_async_write_state<NocOptions::POSTED>(*this, coalesced_page_size, {.receiver_idx = i});
#endif
                for (uint32_t h = 0; h < num_rows; ++h) {
                    const uint32_t row_src_start = current_src_addr;
                    for (uint32_t w = 0; w < coalesced_num_pages_per_row; ++w) {
#ifdef ARCH_QUASAR
                        noc.async_write_with_state(
                            CoreLocalMem<uint32_t>(current_src_addr),
                            *this,
                            coalesced_page_size,
                            {},
                            {.receiver_idx = i});
#else
                        noc.async_write_with_state<NocOptions::POSTED>(
                            CoreLocalMem<uint32_t>(current_src_addr),
                            *this,
                            coalesced_page_size,
                            {},
                            {.receiver_idx = i});
#endif
                        current_src_addr += coalesced_page_size;
                        destination_offset_bytes_ += coalesced_page_size;
                    }
                    current_src_addr = row_src_start + row_stride_in_stage;
                }
            }
            recv_src_offset += row_bytes_per_recv;
        }
        destination_offset_bytes_ = 0;
    }

    // Broadcast: write n entries of identical data from src to owned receivers
    // at their current write position. Uses loop-unicast (hardware NOC multicast requires
    // a rectangular destination grid). For different bytes per receiver, use
    // write_to_receiver / write_strided instead.
    template <typename Src>
    FORCE_INLINE void write_broadcast(
        const Noc& noc,
        const Src& src,
        uint32_t num_entries,
        const typename noc_traits_t<Src>::src_args_type& src_args = {}) {
        CrossNodeSenderDFBInterface& iface = interface_.sender;
        const uint32_t entry_size = iface.fifo_page_size;
        ASSERT(entry_size != 0);
        const uint32_t len_bytes = num_entries * entry_size;
        ASSERT(len_bytes != 0 || num_entries == 0);
        if (num_entries == 0) {
            return;
        }
        const uint32_t num_recv = cross_node_dfb_num_receivers(iface.num_receivers_and_remote_pages_sent_ptr);

        destination_offset_bytes_ = 0;
        for (uint32_t i = 0; i < num_recv; ++i) {
            if (!owns_receiver(i)) {
                continue;
            }
            const uint32_t wr_offset = sender_wr_offset(iface, i);
            assert_contiguous_write(iface, wr_offset, num_entries);
#ifdef ARCH_QUASAR
            noc.async_write(src, *this, len_bytes, src_args, {.receiver_idx = i});
#else
            noc.async_write<NocOptions::POSTED>(src, *this, len_bytes, src_args, {.receiver_idx = i});
#endif
        }
    }

    // Write n entries from src to a single receiver (receiver_idx) at that
    // receiver's write position.  Does NOT increment credits.  Pair with push_back()
    // (collective credit after all per-receiver writes) or push_back_to_receiver()
    // (per-receiver credit) as appropriate.
    // Multi-DM: no-op when this hart does not own receiver_idx.
    template <typename Src>
    FORCE_INLINE void write_to_receiver(
        const Noc& noc,
        uint32_t receiver_idx,
        const Src& src,
        uint32_t num_entries,
        const typename noc_traits_t<Src>::src_args_type& src_args = {}) {
        if (!owns_receiver(receiver_idx)) {
            return;
        }
        CrossNodeSenderDFBInterface& iface = interface_.sender;
        const uint32_t entry_size = iface.fifo_page_size;
        const uint32_t len_bytes = num_entries * entry_size;
        const uint32_t wr_offset = sender_wr_offset(iface, receiver_idx);
        assert_contiguous_write(iface, wr_offset, num_entries);
        destination_offset_bytes_ = 0;
#ifdef ARCH_QUASAR
        noc.async_write(src, *this, len_bytes, src_args, {.receiver_idx = receiver_idx});
#else
        noc.async_write<NocOptions::POSTED>(src, *this, len_bytes, src_args, {.receiver_idx = receiver_idx});
#endif
    }

    // Flush payload writes from this core before publishing pages_sent.
    // WH/BH: posted flush (depart queue). Quasar: write barrier so destination TL1 is
    // complete before pages_sent — posted payload + credit can otherwise reorder for
    // interleaved STRIDED relay consumers under same-program backpressure.
    FORCE_INLINE void flush_writes(const Noc& noc = Noc{}) {
#ifdef ARCH_QUASAR
        noc.async_write_barrier();
#else
        noc.async_writes_flushed<NocOptions::POSTED>();
#endif
    }

    // Credit-only: NOC-inc pages_sent on owned receivers by num_entries. Crediting a
    // receiver is also what advances its stored write cursor.
    // Call after all write_* for this slot.
    FORCE_INLINE void push_back(uint32_t num_entries, const Noc& noc = Noc{}) {
        CrossNodeSenderDFBInterface& iface = interface_.sender;
        const uint32_t num_recv = cross_node_dfb_num_receivers(iface.num_receivers_and_remote_pages_sent_ptr);
        const uint8_t noc_id = noc.get_noc_id();
        for (uint32_t i = 0; i < num_recv; ++i) {
            if (!owns_receiver(i)) {
                continue;
            }
            const uint32_t wr_offset = sender_wr_offset(iface, i);
            assert_contiguous_write(iface, wr_offset, num_entries);
            increment_sender_credits_for_receiver<detail::default_noc_mode>(
                iface, i, num_entries, wr_offset, noc_id, true, write_at_cmd_buf);
        }
    }

    // Credit-only for one receiver: NOC-inc pages_sent on receiver_idx by num_entries,
    // which also advances that receiver's stored write cursor. Used for round-robin /
    // uneven per-receiver credit distribution (caller manages receiver index).
    // Multi-DM: no-op when this hart does not own receiver_idx.
    FORCE_INLINE void push_back_to_receiver(uint32_t receiver_idx, uint32_t num_entries, const Noc& noc = Noc{}) {
        if (!owns_receiver(receiver_idx)) {
            return;
        }
        CrossNodeSenderDFBInterface& iface = interface_.sender;

        const uint32_t wr_offset = sender_wr_offset(iface, receiver_idx);
        assert_contiguous_write(iface, wr_offset, num_entries);
        const uint8_t noc_id = noc.get_noc_id();
        increment_sender_credits_for_receiver<detail::default_noc_mode>(
            iface, receiver_idx, num_entries, wr_offset, noc_id, true, write_at_cmd_buf);
    }

#endif  // !COMPILE_FOR_TRISC

    // Wait until owned receivers have acked all pages_sent (drains this hart's pipeline).
    FORCE_INLINE void barrier() {
        WAYPOINT("CNBW");
        CrossNodeSenderDFBInterface& iface = interface_.sender;
        const uint32_t num_recv = cross_node_dfb_num_receivers(iface.num_receivers_and_remote_pages_sent_ptr);
        const uint32_t P = num_credit_lanes();
        for (uint32_t i = 0; i < num_recv; ++i) {
            if (!owns_receiver(i)) {
                continue;
            }
            for (uint32_t lane = 0; lane < P; ++lane) {
                volatile tt_l1_ptr uint32_t* sent_ptr = local_sent_ptr(iface, i, lane);
                volatile tt_l1_ptr uint32_t* acked_ptr = local_acked_ptr(iface, i, lane);
                while (true) {
                    if (load_remote_l1_credit(acked_ptr) == *sent_ptr) {
                        break;
                    }
                }
            }
        }
        WAYPOINT("CNBD");
    }

    // -----------------------------------------------------------------------
    // Receiver-side API
    // -----------------------------------------------------------------------

    // Spin until this hart's lane has space for num_entries owned slots.
    // P=1: contiguous FIFO credits (legacy, may include trailing gap units).
    // P>1: lane credits; num_entries are owned strides (e.g. tid0: slots 0,P,2P,…).
    FORCE_INLINE void wait_front(uint32_t num_entries) {
        WAYPOINT("CNWF");
        CrossNodeReceiverDFBInterface& iface = interface_.receiver;
        const uint32_t P = num_credit_lanes();
        const uint32_t entry_size = iface.fifo_page_size;
        volatile tt_l1_ptr uint32_t* acked_ptr =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(iface.aligned_pages_acked_ptr);
        volatile tt_l1_ptr uint32_t* sent_ptr =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(iface.aligned_pages_sent_ptr);
        uint32_t units_needed = 0;
        if (P == 1) {
            const uint32_t payload_bytes = num_entries * entry_size;
            // A read window may straddle the wrap (the caller keeps a lookahead over the ring end);
            // only asking for more than the ring holds is a bug. units_for_read is already wrap-aware,
            // and the GlobalCB twin (remote_cb_wait_front) allows the same straddle.
            ASSERT(payload_bytes <= iface.fifo_limit_page_aligned - iface.fifo_start_addr);
            const uint32_t rd_offset = iface.fifo_rd_ptr - iface.fifo_start_addr;
            units_needed = units_for_read(iface, rd_offset, payload_bytes);
        } else {
            units_needed = num_entries * units_per_entry(iface);
            (void)entry_size;
        }
        uint32_t sent = 0;
        uint32_t acked = 0;
        do {
            sent = load_remote_l1_credit(sent_ptr);
            acked = *acked_ptr;
        } while ((sent - acked) < units_needed);
        WAYPOINT("CNWD");
    }

#if !defined(COMPILE_FOR_TRISC)
    // Advance read pointer and NOC-inc pages_acked on sender.
    // If bind_relay() was called, waits until compute has consumed num_entries first.
    FORCE_INLINE void pop_front(uint32_t num_entries, const Noc& noc = Noc{}) {
        if (interface_.receiver.relay_id != RELAY_DFB_INVALID) {
            ASSERT(relay_dfb_.has_value());
            wait_relay_consumed(num_entries);
        }
        pop_front_impl(num_entries, noc);
    }
#endif  // !COMPILE_FOR_TRISC

    // -----------------------------------------------------------------------
    // Accessors
    // -----------------------------------------------------------------------

    // Number of receivers connected to this PrefetcherPipe (sender participant cores only).
    FORCE_INLINE uint32_t num_receivers() {
        const CrossNodeSenderDFBInterface& iface = interface_.sender;
        return cross_node_dfb_num_receivers(iface.num_receivers_and_remote_pages_sent_ptr);
    }

    // Partition-R ownership from the kernel thread config (get_num_threads / get_my_thread_id).
    // Hart tid owns receivers where r % P == tid. WH/BH report P=1, tid=0: every receiver is owned.
    FORCE_INLINE bool owns_receiver(uint32_t receiver_idx) const {
        return (receiver_idx % get_num_threads()) == get_my_thread_id();
    }

    // Lock entries at the receiver's current front for direct CPU access. The pointer covers
    // num_entries contiguous entries, so with P > 1 lanes (where this hart's entries are P
    // entries apart) only one entry may be locked at a time: lock, read, pop_front(1), repeat.
    [[nodiscard]] FORCE_INLINE auto scoped_read_lock(uint32_t num_entries = 1) {
        const CrossNodeReceiverDFBInterface& iface = interface_.receiver;
        ASSERT(num_entries == 1 || num_credit_lanes() == 1);
        ASSERT(iface.fifo_rd_ptr + num_entries * iface.fifo_page_size <= iface.fifo_limit_page_aligned);
#ifdef ARCH_QUASAR
        // Quasar DM L2 does not snoop NOC→TL1 fills; invalidate then hand out the uncached alias.
        invalidate_l2_cache_range(iface.fifo_rd_ptr, num_entries * iface.fifo_page_size);
        return make_dfb_scoped_lock<false>(iface.fifo_rd_ptr + MEM_L1_UNCACHED_BASE, []() {});
#else
        return make_dfb_scoped_lock<false>(iface.fifo_rd_ptr, []() {});
#endif
    }

    FORCE_INLINE uint32_t get_entry_size() { return interface_.sender.fifo_page_size; }

    // True while the sender has published entries this receiver has not yet popped. Reads both
    // halves of this receiver's counter pair -- entries_sent sits one L1_ALIGNMENT below
    // entries_acked, the relationship wait_front() relies on -- so the pair layout stays inside
    // the class. Receiver participants only.
    FORCE_INLINE bool has_unconsumed_entries() {
        auto* acked_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(interface_.receiver.aligned_pages_acked_ptr);
        auto* sent_ptr =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(interface_.receiver.aligned_pages_acked_ptr - L1_ALIGNMENT);
        // pages_sent arrives via NOC; pages_acked is written by this receiver.
        return load_remote_l1_credit(sent_ptr) != load_local_l1_credit(acked_ptr);
    }

#if !defined(COMPILE_FOR_TRISC)
    // -----------------------------------------------------------------------
    // Host-declared relay DFB (DM → compute)
    // -----------------------------------------------------------------------

    // Producer-only view over the pipe-owned local DFB. Reserve/push stay here;
    // pop_front waits on that same DFB so WH/BH and Quasar share one object.
    class RelayView {
    public:
        FORCE_INLINE void reserve_back(uint16_t num_entries) { dfb_.reserve_back(num_entries); }
        FORCE_INLINE void push_back(uint16_t num_entries) { dfb_.push_back(num_entries); }

    private:
        friend class PrefetcherPipe;
        FORCE_INLINE explicit RelayView(DataflowBuffer& dfb) : dfb_(dfb) {}
        DataflowBuffer& dfb_;
    };

    // Open the relay DFB the host registered for this slot (DataflowBufferSpec::
    // prefetcher_pipe_relays). Constructs the
    // local DataflowBuffer (Quasar: dfb_ensure_ready), then snaps TC/CB slots to the
    // current receiver cursor/page size (TRISC is aligned separately in
    // DataflowBuffer(RelayDFBBindingToken)). A later set_receiver_entry_size() refreshes
    // that same local iface in place.
    FORCE_INLINE RelayView bind_relay() {
        const CrossNodeReceiverDFBInterface& iface = interface_.receiver;
        ASSERT(iface.relay_id != RELAY_DFB_INVALID);
        ASSERT(!relay_dfb_.has_value());
        // Construct first: Quasar DataflowBuffer ctor runs dfb_ensure_ready (resets slots).
        // Multi-producer: each pipe-consumer hart gets its own DataflowBuffer view of the
        // same relay id; STRIDED pap partitions producer slots.
        relay_dfb_.emplace(RelayDFBBindingToken{iface.relay_id});
        sync_threads();
        // Align shared local iface once after all producers have constructed.
        if (get_my_thread_id() == 0) {
            align_local_dfb_to_prefetcher_pipe_receiver_iface(iface.relay_id, iface);
#ifndef ARCH_QUASAR
            const uintptr_t entries_acked_ptr = reinterpret_cast<uintptr_t>(get_cb_tiles_acked_ptr(iface.relay_id));
            relay_entries_acked_checkpoint_ = static_cast<uint16_t>(reg_read(entries_acked_ptr));
#endif
        }
        sync_threads();
#ifndef ARCH_QUASAR
        if (get_my_thread_id() != 0) {
            const uintptr_t entries_acked_ptr = reinterpret_cast<uintptr_t>(get_cb_tiles_acked_ptr(iface.relay_id));
            relay_entries_acked_checkpoint_ = static_cast<uint16_t>(reg_read(entries_acked_ptr));
        }
#endif
        return RelayView(*relay_dfb_);
    }
#endif

private:
#if !defined(COMPILE_FOR_TRISC)
    friend struct ::noc_traits_t<PrefetcherPipe>;
#endif

    CrossNodeDFBInterface interface_;
    uint8_t prefetcher_pipe_id_ = 0;
    uint8_t num_credit_lanes_ = 1;  // active lanes P (kernel-config slot), read once in the ctor
    uint32_t fifo_size_ = 0;        // full ring allocation (config word[3]); credit modulus
    uint32_t destination_offset_bytes_ = 0;

#if !defined(COMPILE_FOR_TRISC)
    std::optional<DataflowBuffer> relay_dfb_;
#ifndef ARCH_QUASAR
    uint16_t relay_entries_acked_checkpoint_ = 0;
#endif
#if defined(ARCH_BLACKHOLE)
    // Bit i set => this pipe issued a (forced non-posted) credit atomic on NOC i.
    uint8_t atomic_noc_mask_ = 0;
#endif

    FORCE_INLINE void note_credit_atomic_noc(uint8_t noc_id) {
#if defined(ARCH_BLACKHOLE)
        atomic_noc_mask_ |= static_cast<uint8_t>(1u << noc_id);
#endif
    }

    // [base, base + size) of the credit words this core writes through the cached view: the
    // whole SENT block on the sender, this receiver's ACKED slots (all lanes) on a receiver.
    // lane_offset_bytes is tid * L1_ALIGNMENT once bind_receiver_credit_lane has run, else 0.
    FORCE_INLINE void local_credit_range(uint32_t lane_offset_bytes, uint32_t& base, uint32_t& size) const {
        volatile tt_l1_ptr uint32_t* l1_config =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(interface_.sender.config_ptr);
        const bool is_sender = static_cast<bool>(load_prefetcher_pipe_config_word(l1_config, REMOTE_DFB_CFG_IS_SENDER));
        const uint32_t cap = credit_lane_capacity();
        if (is_sender) {
            const CrossNodeSenderDFBInterface& iface = interface_.sender;
            base = iface.aligned_pages_sent_ptr;
            size = cross_node_dfb_num_receivers(iface.num_receivers_and_remote_pages_sent_ptr) * cap * L1_ALIGNMENT;
        } else {
            base = interface_.receiver.aligned_pages_acked_ptr - lane_offset_bytes;
            size = cap * L1_ALIGNMENT;
        }
    }

    FORCE_INLINE uint32_t receiver_lane_offset_bytes() const {
        volatile tt_l1_ptr uint32_t* l1_config =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(interface_.sender.config_ptr);
        const bool is_sender = static_cast<bool>(load_prefetcher_pipe_config_word(l1_config, REMOTE_DFB_CFG_IS_SENDER));
        if (is_sender || num_credit_lanes() == 1) {
            return 0;
        }
        return get_my_thread_id() * L1_ALIGNMENT;
    }

    // Quasar: discard cached copies (L1 D$ of this hart + shared L2) of the local credit
    // lines. Only safe when nothing is dirty, i.e. after the previous pipe's dtor flushed.
    FORCE_INLINE void invalidate_local_credit_lines(uint32_t lane_offset_bytes) const {
#ifdef ARCH_QUASAR
        uint32_t base = 0;
        uint32_t size = 0;
        local_credit_range(lane_offset_bytes, base, size);
        constexpr uint32_t line = 64;
        for (uint32_t addr = base & ~(line - 1); addr < base + size; addr += line) {
            invalidate_l1_dcache(addr);
        }
        invalidate_l2_cache_range(base, size);
#else
        (void)lane_offset_bytes;
#endif
    }

    // Quasar: write the local credit lines back to TL1 (probes every DM hart's L1 D$).
    FORCE_INLINE void flush_local_credit_lines(uint32_t lane_offset_bytes) const {
#ifdef ARCH_QUASAR
        uint32_t base = 0;
        uint32_t size = 0;
        local_credit_range(lane_offset_bytes, base, size);
        flush_l2_cache_range(base, size);
#else
        (void)lane_offset_bytes;
#endif
    }

    FORCE_INLINE void wait_relay_consumed(uint32_t num_entries) {
        ASSERT(num_entries <= relay_dfb_->get_local_num_entries());
        WAYPOINT("PDCW");
#ifdef ARCH_QUASAR
        // After the relay.push_back(N), HW posted already marks those entries;
        // wait until consumer acked catches posted on every TC.
        (void)num_entries;
        relay_dfb_->wait_relay_consumer_caught_up();
#else
        const uint16_t relay_dfb_id = relay_dfb_->get_id();
        const uintptr_t entries_acked_ptr = reinterpret_cast<uintptr_t>(get_cb_tiles_acked_ptr(relay_dfb_id));
        uint16_t entries_acked;
        do {
            invalidate_l1_cache();
            entries_acked = static_cast<uint16_t>(reg_read(entries_acked_ptr));
        } while (static_cast<uint16_t>(entries_acked - relay_entries_acked_checkpoint_) < num_entries);
        relay_entries_acked_checkpoint_ = static_cast<uint16_t>(relay_entries_acked_checkpoint_ + num_entries);
#endif
        WAYPOINT("PDCD");
    }
#endif

#if !defined(COMPILE_FOR_TRISC)
    FORCE_INLINE void pop_front_impl(uint32_t num_entries, const Noc& noc) {
        CrossNodeReceiverDFBInterface& iface = interface_.receiver;
        const uint32_t P = num_credit_lanes();
        const uint32_t entry_size = iface.fifo_page_size;
        uint32_t num_units = 0;
        if (P == 1) {
            const uint32_t payload_bytes = num_entries * entry_size;
            // A pop may straddle the wrap, matching wait_front and the GlobalCB twin
            // (remote_cb_pop_front); only popping more than the ring holds is a bug.
            ASSERT(payload_bytes <= iface.fifo_limit_page_aligned - iface.fifo_start_addr);
            const uint32_t rd_offset = iface.fifo_rd_ptr - iface.fifo_start_addr;
            num_units = units_for_read(iface, rd_offset, payload_bytes);
            // Carry the remainder past the wrap rather than snapping to the base: a batched pop that
            // crosses the usable limit resumes that many bytes into the ring. units_for_read has
            // already credited the trailing gap this crossing skips.
            const uint32_t next_rd_ptr = iface.fifo_rd_ptr + payload_bytes;
            iface.fifo_rd_ptr = next_rd_ptr >= iface.fifo_limit_page_aligned
                                    ? iface.fifo_start_addr + (next_rd_ptr - iface.fifo_limit_page_aligned)
                                    : next_rd_ptr;
        } else {
            const uint32_t stride = entry_size * P;
            num_units = num_entries * units_per_entry(iface);
            for (uint32_t e = 0; e < num_entries; ++e) {
                iface.fifo_rd_ptr += stride;
                if (iface.fifo_rd_ptr >= iface.fifo_limit_page_aligned) {
                    iface.fifo_rd_ptr = iface.fifo_start_addr + get_my_thread_id() * entry_size;
                }
            }
        }

        const uint8_t noc_id = noc.get_noc_id();
        note_credit_atomic_noc(noc_id);
        detail::update_pages_acked<detail::default_noc_mode>(
            reinterpret_cast<const RemoteReceiverCBInterface&>(iface), num_units, noc_id, true, write_at_cmd_buf);
    }
#endif

#ifdef PREFETCHER_PIPE_TEST_HELPERS
    friend void test_stale_commit_after_resize(
        PrefetcherPipe&, uint32_t new_entry_size, uint32_t stale_entry_size, uint32_t poison_wr_ptr);
#endif

    // Poll a credit word updated by a remote NOC atomic / write.
    // Quasar DM L2 does not snoop NOC→TL1; WH/BH needs invalidate_l1_cache.
    FORCE_INLINE static uint32_t load_remote_l1_credit(volatile tt_l1_ptr uint32_t* ptr) {
#ifdef ARCH_QUASAR
        return *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(reinterpret_cast<uintptr_t>(ptr) + MEM_L1_UNCACHED_BASE);
#else
        invalidate_l1_cache();
        return *ptr;
#endif
    }

    // Credit words this core owns (sent / wr_cursor on the sender, acked on the receiver) are
    // plain cached loads/stores. That is only safe because the config page keeps them in cache
    // lines that hold no NoC-written word (SENT and ACKED blocks, see
    // remote_dfb_config_layout.h): Quasar's DM L2 writes back whole 64B lines, so a dirty line
    // may only ever contain words this core wrote. Peer-written words are always read through
    // load_remote_l1_credit. Cached copies are dropped at construction
    // (invalidate_local_credit_lines) so a host reset in TL1 is observed, and written back at
    // destruction (flush_local_credit_lines) so the values persist into the next program.

    // Byte offset of (receiver_idx, lane)'s slot within either credit block. Layout stride is
    // credit_lane_capacity() (allocated slots); active P only selects which are used.
    FORCE_INLINE static uint32_t credit_slot_offset(uint32_t receiver_idx, uint32_t lane) {
        ASSERT(lane < credit_lane_capacity());
        return (receiver_idx * credit_lane_capacity() + lane) * L1_ALIGNMENT;
    }

    // Local entries_sent for receiver_idx's lane (default lane 0 = write-cursor slot).
    FORCE_INLINE static volatile tt_l1_ptr uint32_t* local_sent_ptr(
        const CrossNodeSenderDFBInterface& iface, uint32_t receiver_idx, uint32_t lane = 0) {
        return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
            iface.aligned_pages_sent_ptr + credit_slot_offset(receiver_idx, lane));
    }

    // Local entries_acked for receiver_idx's lane (NoC-atomic target of that receiver hart).
    FORCE_INLINE static volatile tt_l1_ptr uint32_t* local_acked_ptr(
        const CrossNodeSenderDFBInterface& iface, uint32_t receiver_idx, uint32_t lane = 0) {
        return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
            iface.aligned_pages_acked_ptr + credit_slot_offset(receiver_idx, lane));
    }

    // Receiver-side entries_sent slot on receiver_idx's core (same offset as local_sent_ptr).
    FORCE_INLINE static uint32_t remote_sent_ptr(
        const CrossNodeSenderDFBInterface& iface, uint32_t receiver_idx, uint32_t lane = 0) {
        return cross_node_dfb_remote_pages_sent_ptr(iface.num_receivers_and_remote_pages_sent_ptr) +
               credit_slot_offset(receiver_idx, lane);
    }

    // Active pipe-consumer lanes P (kernel-config slot, cached in the ctor).
    FORCE_INLINE uint32_t num_credit_lanes() const { return num_credit_lanes_; }

    // Slots reserved per receiver in the config page (may exceed active lanes).
    FORCE_INLINE static constexpr uint32_t credit_lane_capacity() {
#ifdef ARCH_QUASAR
        return PREFETCHER_PIPE_MAX_CREDIT_LANES;
#else
        return 1u;
#endif
    }

    FORCE_INLINE static uint32_t units_per_entry(const CrossNodeSenderDFBInterface& iface) {
        return iface.fifo_page_size / L1_ALIGNMENT;
    }

    FORCE_INLINE static uint32_t units_per_entry(const CrossNodeReceiverDFBInterface& iface) {
        return iface.fifo_page_size / L1_ALIGNMENT;
    }

    FORCE_INLINE uint32_t lane_capacity_units(const CrossNodeSenderDFBInterface& iface) const {
        const uint32_t P = num_credit_lanes();
        const uint32_t ring_entries = usable_offset(iface) / iface.fifo_page_size;
        ASSERT(ring_entries % P == 0);
        return (ring_entries / P) * units_per_entry(iface);
    }

#if !defined(COMPILE_FOR_TRISC)
    // Point this hart at its credit lane and STRIDED rd_ptr (no-op when P==1).
    FORCE_INLINE void bind_receiver_credit_lane() {
        CrossNodeReceiverDFBInterface& iface = interface_.receiver;
        const uint32_t P = num_credit_lanes();
        const uint32_t tid = get_my_thread_id();
        ASSERT(tid < P);
        if (P == 1) {
            return;
        }
        iface.aligned_pages_sent_ptr += tid * L1_ALIGNMENT;
        iface.aligned_pages_acked_ptr += tid * L1_ALIGNMENT;
        iface.remote_pages_acked_ptr += tid * L1_ALIGNMENT;
        const uint32_t entry_size = iface.fifo_page_size;
        const uint32_t stride = entry_size * P;
        uint32_t next = iface.fifo_start_addr + align(iface.fifo_rd_ptr - iface.fifo_start_addr, entry_size);
        if (next >= iface.fifo_limit_page_aligned) {
            next = iface.fifo_start_addr;
        }
        uint32_t lane_base = iface.fifo_start_addr + tid * entry_size;
        if (next > lane_base) {
            const uint32_t delta = next - lane_base;
            lane_base = lane_base + ((delta + stride - 1) / stride) * stride;
        }
        if (lane_base >= iface.fifo_limit_page_aligned) {
            lane_base = iface.fifo_start_addr + tid * entry_size;
        }
        iface.fifo_rd_ptr = lane_base;
    }
#endif

    // Local entries_sent counter for one receiver (L1_ALIGNMENT units; written only by
    // this core, remotely mirrored on the receiver). Lane 0 holds the wr cursor.

    // Full allocation is the stable credit modulus across entry-size changes.
    FORCE_INLINE static uint32_t fifo_size(uint32_t config_ptr) {
        return load_prefetcher_pipe_config_word(
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(config_ptr), REMOTE_DFB_CFG_FIFO_SIZE);
    }

    // Same value cached in the ctor for the hot path.
    FORCE_INLINE uint32_t fifo_size() const { return fifo_size_; }

    FORCE_INLINE static uint32_t usable_offset(const CrossNodeSenderDFBInterface& iface) {
        return iface.fifo_limit_page_aligned - iface.fifo_start_addr;
    }

    // A receiver's write cursor: a byte offset from the ring base, stored beside that receiver's
    // credit counters and advanced with them (see advance_wr_offset). Durable across programs for
    // the same reason the counters are -- it lives in the config page -- and independent of them
    // for the reason PREFETCHER_PIPE_SLOT_WR_OFFSET_WORD gives: entries_sent wraps at 2^32, which
    // only preserves a (sent % ring_units) derivation when ring_units is a power of two. Receivers
    // advance independently (Flow C), so each one carries its own cursor.
    FORCE_INLINE static volatile tt_l1_ptr uint32_t* local_wr_offset_ptr(
        const CrossNodeSenderDFBInterface& iface, uint32_t receiver_idx) {
        return local_sent_ptr(iface, receiver_idx) + PREFETCHER_PIPE_SLOT_WR_OFFSET_WORD;
    }

    FORCE_INLINE static uint32_t sender_wr_offset(const CrossNodeSenderDFBInterface& iface, uint32_t receiver_idx) {
        return *local_wr_offset_ptr(iface, receiver_idx);
    }

    // Advance a receiver's cursor by the units just credited to it. Payload and any trailing gap
    // are both in `units` (units_for_write), so a lap is exactly the full allocation, and the
    // contiguity rule caps a single credit at one lap: one conditional subtract is enough.
    FORCE_INLINE void advance_wr_offset(
        const CrossNodeSenderDFBInterface& iface, uint32_t receiver_idx, uint32_t units) {
        volatile tt_l1_ptr uint32_t* offset_ptr = local_wr_offset_ptr(iface, receiver_idx);
        const uint32_t ring_bytes = fifo_size();
        uint32_t next = *offset_ptr + units * L1_ALIGNMENT;
        ASSERT(next <= ring_bytes);
        if (next >= ring_bytes) {
            next -= ring_bytes;
        }
        *offset_ptr = next;
    }

    // Producer writes must be contiguous (same rule as local CBs): wr_offset + len must
    // land at or before the limit. Crossing the wrap in one call is illegal.
    FORCE_INLINE static void assert_contiguous_bytes(
        const CrossNodeSenderDFBInterface& iface, uint32_t wr_offset, uint32_t len_bytes) {
        ASSERT(wr_offset + len_bytes <= usable_offset(iface));
    }

    FORCE_INLINE static void assert_contiguous_write(
        const CrossNodeSenderDFBInterface& iface, uint32_t wr_offset, uint32_t num_entries) {
        assert_contiguous_bytes(iface, wr_offset, num_entries * iface.fifo_page_size);
    }

    // Credits include the trailing allocation gap when this payload reaches the
    // page-aligned limit, so one lap of credits is exactly the full allocation and both
    // endpoints come back to the ring base on the same lap.
    FORCE_INLINE uint32_t
    units_for_read(const CrossNodeReceiverDFBInterface& iface, uint32_t offset, uint32_t payload_bytes) {
        uint32_t credited_bytes = payload_bytes;
        const uint32_t usable = iface.fifo_limit_page_aligned - iface.fifo_start_addr;
        if (offset + payload_bytes >= usable) {
            credited_bytes += fifo_size() - usable;
        }
        return credited_bytes / L1_ALIGNMENT;
    }

    FORCE_INLINE uint32_t
    units_for_write(const CrossNodeSenderDFBInterface& iface, uint32_t wr_offset, uint32_t num_entries) {
        const uint32_t payload_bytes = num_entries * iface.fifo_page_size;
        uint32_t credited_bytes = payload_bytes;
        if (wr_offset + payload_bytes >= usable_offset(iface)) {
            credited_bytes += fifo_size() - usable_offset(iface);
        }
        return credited_bytes / L1_ALIGNMENT;
    }

#if !defined(COMPILE_FOR_TRISC)
    // Stripe num_entries of credits across lanes (entry_idx % P); advance wr cursor by
    // the contiguous units_for_write total. P==1 matches the legacy single-pair path.
    template <uint8_t nm>
    FORCE_INLINE void increment_sender_credits_for_receiver(
        CrossNodeSenderDFBInterface& iface,
        uint32_t receiver_idx,
        uint32_t num_entries,
        uint32_t wr_offset,
        uint8_t noc,
        bool posted,
        uint8_t cmd_buf) {
        if (num_entries == 0) {
            return;
        }
        const uint32_t P = num_credit_lanes();
        const uint32_t total_units = units_for_write(iface, wr_offset, num_entries);
        note_credit_atomic_noc(noc);
        volatile tt_l1_ptr uint32_t* xy =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(iface.receiver_noc_xy_ptr) + 2 * receiver_idx;

        if (P == 1) {
            volatile tt_l1_ptr uint32_t* sent_ptr = local_sent_ptr(iface, receiver_idx, 0);
            const uint64_t remote_sent_noc_addr =
                noc_address_backend::worker_address(xy[0], xy[1], remote_sent_ptr(iface, receiver_idx, 0), noc);
            *sent_ptr += total_units;
            advance_wr_offset(iface, receiver_idx, total_units);
            noc_fast_atomic_increment<nm>(
                noc,
                cmd_buf,
                remote_sent_noc_addr,
                NOC_UNICAST_WRITE_VC,
                total_units,
                31 /*wrap*/,
                false /*linked*/,
                posted,
                MEM_NOC_ATOMIC_RET_VAL_ADDR);
            return;
        }

        // P > 1: entries stripe across lanes (entry_idx % P). The receiver only ever compares
        // sent - acked per lane, so credits for the same lane in one push are summed and
        // published with a single atomic: push_back(n) costs min(n, P) atomics, not n.
        const uint32_t entry_size = iface.fifo_page_size;
        const uint32_t upe = units_per_entry(iface);
        const uint32_t usable = usable_offset(iface);
        const uint32_t gap_units = (fifo_size() - usable) / L1_ALIGNMENT;
        uint32_t counts[PREFETCHER_PIPE_MAX_CREDIT_LANES] = {};
        ASSERT(P <= PREFETCHER_PIPE_MAX_CREDIT_LANES);
        uint32_t off = wr_offset;
        uint32_t lane = (off / entry_size) % P;
        for (uint32_t e = 0; e < num_entries; ++e) {
            counts[lane] += upe + ((off + entry_size >= usable) ? gap_units : 0);
            off += entry_size;
            if (off >= usable) {
                off = 0;  // ring_entries % P == 0 (lane_capacity_units), so lane wraps to 0 too
            }
            lane = (lane + 1 == P) ? 0 : lane + 1;
        }
        for (lane = 0; lane < P; ++lane) {
            if (counts[lane] == 0) {
                continue;
            }
            *local_sent_ptr(iface, receiver_idx, lane) += counts[lane];
            const uint64_t remote_sent_noc_addr =
                noc_address_backend::worker_address(xy[0], xy[1], remote_sent_ptr(iface, receiver_idx, lane), noc);
            noc_fast_atomic_increment<nm>(
                noc,
                cmd_buf,
                remote_sent_noc_addr,
                NOC_UNICAST_WRITE_VC,
                counts[lane],
                31 /*wrap*/,
                false /*linked*/,
                posted,
                MEM_NOC_ATOMIC_RET_VAL_ADDR);
        }
        advance_wr_offset(iface, receiver_idx, total_units);
    }

    // Contiguous unit bump for mid-flight resize pad credits (single credit lane only).
    template <uint8_t nm>
    FORCE_INLINE void increment_sender_credits_units_for_receiver(
        CrossNodeSenderDFBInterface& iface,
        uint32_t receiver_idx,
        uint32_t adjustment,
        uint8_t noc,
        bool posted,
        uint8_t cmd_buf) {
        if (adjustment == 0) {
            return;
        }
        ASSERT(num_credit_lanes() == 1);
        note_credit_atomic_noc(noc);
        volatile tt_l1_ptr uint32_t* sent_ptr = local_sent_ptr(iface, receiver_idx, 0);
        volatile tt_l1_ptr uint32_t* xy =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(iface.receiver_noc_xy_ptr) + 2 * receiver_idx;
        const uint64_t remote_sent_noc_addr =
            noc_address_backend::worker_address(xy[0], xy[1], remote_sent_ptr(iface, receiver_idx, 0), noc);

        *sent_ptr += adjustment;
        advance_wr_offset(iface, receiver_idx, adjustment);
        noc_fast_atomic_increment<nm>(
            noc,
            cmd_buf,
            remote_sent_noc_addr,
            NOC_UNICAST_WRITE_VC,
            adjustment,
            31 /*wrap*/,
            false /*linked*/,
            posted,
            MEM_NOC_ATOMIC_RET_VAL_ADDR);
    }

    // Snap every receiver's stored cursor forward to the new page grid and publish the
    // skipped bytes as pad credits. update_remote_over_noc is not defaulted: with it false
    // only this core's page grid changes and every stored cursor stays on the old grid with
    // no pad credits published, which desynchronizes the receivers. That is for a caller
    // manipulating the local epoch alone, and nothing but the tests should want it.
    template <bool update_remote_over_noc>
    FORCE_INLINE void resize_sender_interface(
        uint32_t page_size,
        uint8_t noc,
        uint8_t nm = detail::default_noc_mode,
        bool posted = true,
        uint8_t cmd_buf = detail::default_cmd_buf) {
        CrossNodeSenderDFBInterface& sender_cb_interface = interface_.sender;
        ASSERT(static_cast<bool>(load_prefetcher_pipe_config_word(
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sender_cb_interface.config_ptr), REMOTE_DFB_CFG_IS_SENDER)));
        ASSERT(page_size % REMOTE_CIRCULAR_BUFFER_ALIGNED_PAGE_SIZE == 0);
        const uint32_t allocation_size = fifo_size();
        ASSERT(page_size != 0 && page_size <= allocation_size);
        uint32_t fifo_start_addr = sender_cb_interface.fifo_start_addr;
        uint32_t cb_size_page_aligned = allocation_size - allocation_size % page_size;
        uint32_t fifo_limit_page_aligned = fifo_start_addr + cb_size_page_aligned;
        if constexpr (update_remote_over_noc) {
            const uint32_t num_recv =
                cross_node_dfb_num_receivers(sender_cb_interface.num_receivers_and_remote_pages_sent_ptr);
            for (uint32_t i = 0; i < num_recv; ++i) {
                const uint32_t current_offset = sender_wr_offset(sender_cb_interface, i);
                uint32_t next_offset = align(current_offset, page_size);
                uint32_t adjustment_bytes = next_offset - current_offset;
                if (next_offset >= cb_size_page_aligned) {
                    next_offset = 0;
                    adjustment_bytes = allocation_size - current_offset;
                }
                const uint32_t adjustment = adjustment_bytes / L1_ALIGNMENT;
                if (nm == DM_DYNAMIC_NOC) {
#ifdef ARCH_QUASAR
                    // Quasar has one NOC; do not instantiate DM_DYNAMIC_NOC templates.
                    ASSERT(false);
#else
                    increment_sender_credits_units_for_receiver<DM_DYNAMIC_NOC>(
                        sender_cb_interface, i, adjustment, noc, posted, cmd_buf);
#endif
                } else {
                    increment_sender_credits_units_for_receiver<DM_DEDICATED_NOC>(
                        sender_cb_interface, i, adjustment, noc, posted, cmd_buf);
                }
            }
        }
        sender_cb_interface.fifo_limit_page_aligned = fifo_limit_page_aligned;
        sender_cb_interface.fifo_page_size = page_size;
    }

    // Consume the pad credits that the sender published when it snapped its cursor to the
    // new page grid. Not defaulted for the same reason resize_sender_interface is not: with
    // update_remote_over_noc false the read cursor moves without acking those credits.
    template <bool update_remote_over_noc>
    FORCE_INLINE void resize_receiver_interface(
        uint32_t page_size,
        uint8_t noc,
        uint8_t nm = detail::default_noc_mode,
        bool posted = true,
        uint8_t cmd_buf = detail::default_cmd_buf) {
        CrossNodeReceiverDFBInterface& receiver_cb_interface = interface_.receiver;
        ASSERT(!static_cast<bool>(load_prefetcher_pipe_config_word(
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(receiver_cb_interface.config_ptr),
            REMOTE_DFB_CFG_IS_SENDER)));
        ASSERT(page_size % REMOTE_CIRCULAR_BUFFER_ALIGNED_PAGE_SIZE == 0);
        const uint32_t allocation_size = fifo_size();
        ASSERT(page_size != 0 && page_size <= allocation_size);
        uint32_t fifo_start_addr = receiver_cb_interface.fifo_start_addr;
        uint32_t fifo_rd_ptr = receiver_cb_interface.fifo_rd_ptr;
        uint32_t cb_size_page_aligned = allocation_size - allocation_size % page_size;
        uint32_t fifo_limit_page_aligned = fifo_start_addr + cb_size_page_aligned;

        const uint32_t current_offset = fifo_rd_ptr - fifo_start_addr;
        uint32_t next_offset = align(current_offset, page_size);
        uint32_t adjustment_bytes = next_offset - current_offset;
        if (next_offset >= cb_size_page_aligned) {
            next_offset = 0;
            adjustment_bytes = allocation_size - current_offset;
        }
        if constexpr (update_remote_over_noc) {
            volatile tt_l1_ptr uint32_t* pages_acked_ptr =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(receiver_cb_interface.aligned_pages_acked_ptr);
            const uint32_t adjustment = adjustment_bytes / L1_ALIGNMENT;
            if (adjustment != 0) {
                uint32_t pages_acked = 0;
                uint32_t pages_sent = 0;
                uint32_t num_pages_recv = 0;
                volatile tt_l1_ptr uint32_t* pages_sent_ptr =
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(receiver_cb_interface.aligned_pages_sent_ptr);
                do {
                    // pages_acked is local; pages_sent arrives via NOC from the sender.
                    pages_acked = *pages_acked_ptr;
                    pages_sent = load_remote_l1_credit(pages_sent_ptr);
                    num_pages_recv = pages_sent - pages_acked;
                } while (num_pages_recv < adjustment);

                note_credit_atomic_noc(noc);
                if (nm == DM_DYNAMIC_NOC) {
#ifdef ARCH_QUASAR
                    // Quasar has one NOC; do not instantiate DM_DYNAMIC_NOC templates.
                    ASSERT(false);
#else
                    detail::update_pages_acked<DM_DYNAMIC_NOC>(
                        reinterpret_cast<const RemoteReceiverCBInterface&>(receiver_cb_interface),
                        adjustment,
                        noc,
                        posted,
                        cmd_buf);
#endif
                } else {
                    detail::update_pages_acked<DM_DEDICATED_NOC>(
                        reinterpret_cast<const RemoteReceiverCBInterface&>(receiver_cb_interface),
                        adjustment,
                        noc,
                        posted,
                        cmd_buf);
                }
            }
        }
        receiver_cb_interface.fifo_rd_ptr = fifo_start_addr + next_offset;
        receiver_cb_interface.fifo_limit_page_aligned = fifo_limit_page_aligned;
        receiver_cb_interface.fifo_page_size = page_size;
    }
#endif  // !COMPILE_FOR_TRISC
};

}  // namespace experimental

#if !defined(COMPILE_FOR_TRISC)
template <Noc::AddressType address_type>
FORCE_INLINE uint32_t noc_traits_t<experimental::PrefetcherPipe>::src_addr(
    const experimental::PrefetcherPipe& src, const Noc&, const src_args_type&) {
    static_assert(address_type == Noc::AddressType::LOCAL_L1, "PrefetcherPipe can only be used as a local L1 source");
    ASSERT(!static_cast<bool>(experimental::load_prefetcher_pipe_config_word(
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(src.interface_.receiver.config_ptr), REMOTE_DFB_CFG_IS_SENDER)));
    return src.interface_.receiver.fifo_rd_ptr;
}

template <Noc::AddressType address_type>
FORCE_INLINE uint64_t noc_traits_t<experimental::PrefetcherPipe>::dst_addr(
    const experimental::PrefetcherPipe& dst, const Noc& noc, const dst_args_type& args) {
    static_assert(address_type == Noc::AddressType::NOC, "PrefetcherPipe can only be used as a NoC destination");
    const CrossNodeSenderDFBInterface& iface = dst.interface_.sender;
    ASSERT(static_cast<bool>(experimental::load_prefetcher_pipe_config_word(
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(iface.config_ptr), REMOTE_DFB_CFG_IS_SENDER)));
    ASSERT(args.receiver_idx < cross_node_dfb_num_receivers(iface.num_receivers_and_remote_pages_sent_ptr));
    volatile tt_l1_ptr uint32_t* xy =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(iface.receiver_noc_xy_ptr) + 2 * args.receiver_idx;
    const uint32_t local_address = iface.fifo_start_addr +
                                   experimental::PrefetcherPipe::sender_wr_offset(iface, args.receiver_idx) +
                                   dst.destination_offset_bytes_;
    return noc_address_backend::worker_address(xy[0], xy[1], local_address, noc.get_noc_id());
}
#endif
