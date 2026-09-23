// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// Device verbs for UVA: put into this core's D2H socket, drain this core's H2D ring.
// Kernel-only; pulls in dataflow_api.h and socket_api.h.
#pragma once

#include <stdint.h>

#include "risc_common.h"
#include "api/dataflow/dataflow_api.h"
#include "api/socket_api.h"

#include <tt-metalium/experimental/sockets/host_uva_frame.hpp>

namespace tt::tt_metal::experimental {

namespace detail {

// One socket per core and one kernel per core, so this is the right lifetime. A kernel is
// a single translation unit, so `static` gives exactly one instance in .bss.
inline SocketSenderInterface g_socket;
inline uint32_t g_stage_addr = 0;
inline uint32_t g_origin = 0;
// Where the host publishes what the FAR device has pulled, and what we have put.
inline uint32_t g_consumed_addr = 0;
inline uint32_t g_posted = 0;
// Signal addresses go on the wire as offsets from this, so a target with a different
// allocator base still resolves them.
inline uint32_t g_l1_base = 0;

inline SocketReceiverInterface g_rx;
inline uint64_t g_ring = 0;  // host ring anchor; read_ptr is an offset against it
inline uint32_t g_rx_pcie_xy_enc = 0;
inline uint32_t g_landing = 0;
inline uint32_t g_page_size = 0;
// The span ABOVE l1_base, not the size of L1: a signal store lands at l1_base + sig_off, so
// bounding the offset alone would let it run l1_base bytes past the end.
inline uint32_t g_sig_span = 0;
inline uint32_t g_rx_l1_base = 0;
inline bool g_tx_on = false;
inline bool g_rx_on = false;

// Exceeding a single NOC transaction's cap writes NOTHING, so the chunk loop is
// unconditional. Per-arch: a hardcoded 8 KiB used to split every page needlessly.
constexpr uint32_t kMaxNocWrite = NOC_MAX_BURST_SIZE;

inline uint64_t page_addr() {
    // write_ptr added in 64 bits, after the halves are joined: the 32-bit add drops the
    // carry when fifo_addr sits within fifo_size of a 4 GiB boundary.
    return ((static_cast<uint64_t>(g_socket.d2h.data_addr_hi) << 32) |
            static_cast<uint64_t>(g_socket.downstream_fifo_addr)) +
           g_socket.write_ptr;
}

inline void push(uint32_t src, uint64_t dst, uint32_t bytes) {
    const uint32_t enc = g_socket.d2h.pcie_xy_enc;
    while (bytes > 0) {
        const uint32_t chunk = bytes < kMaxNocWrite ? bytes : kMaxNocWrite;
        noc_wwrite_with_state<noc_mode, write_cmd_buf, CQ_NOC_SNDL, CQ_NOC_SEND, CQ_NOC_WAIT, true, false>(
            noc_index, src, enc, dst, chunk, 1);
        src += chunk;
        dst += chunk;
        bytes -= chunk;
    }
}

// Low half first: reading it latches the high half, so the other order can pair a fresh
// low with a stale high across a 32-bit rollover.
inline uint64_t wall_clock() {
    volatile uint32_t tt_reg_ptr* lo = reinterpret_cast<volatile uint32_t tt_reg_ptr*>(RISCV_DEBUG_REG_WALL_CLOCK_L);
    volatile uint32_t tt_reg_ptr* hi = reinterpret_cast<volatile uint32_t tt_reg_ptr*>(RISCV_DEBUG_REG_WALL_CLOCK_H);
    const uint32_t l = lo[0];
    return static_cast<uint64_t>(l) | (static_cast<uint64_t>(hi[0]) << 32);
}

// Payload, then trailer, then the caller decides whether to commit. The signal fields are
// written unconditionally: the staging slot is reused, so stale bytes would read as an op.
inline void stage(uint32_t src_l1, tt_uva_t dst, uint32_t bytes, uint32_t sig_off, uint32_t sig_val, uint32_t sig_op) {
    // Bracketed separately from the write below: this is the wait for the host to retire a
    // page, which is the slot round trip and not a cost of sending.
    const uint64_t t_pre = wall_clock();
    socket_reserve_pages(g_socket, 1);
    const uint64_t page = page_addr();
    const uint64_t t0 = wall_clock();

    push(src_l1, page, bytes);
    noc_async_write_barrier();

    volatile tt_l1_ptr FrameTrailer* t = reinterpret_cast<volatile tt_l1_ptr FrameTrailer*>(g_stage_addr);
    t->guard = tt_uva_frame_guard(kFrameVersion);
    t->dst = tt_uva_bits(dst);
    t->length = bytes;
    t->origin = g_origin;
    t->elapsed = tt_uva_frame_elapsed_pack(wall_clock() - t0, t0 - t_pre);
    t->sig_off = sig_off;
    t->sig_val = sig_val;
    t->sig_op = sig_op;

    // Page TAIL, not page + bytes: D2HLeg::poll() always reads the trailer at
    // page_size - kFrameTrailerBytes, and cannot know `bytes` before it has the trailer.
    push(g_stage_addr, page + g_page_size - kFrameTrailerBytes, kFrameTrailerBytes);
    noc_async_write_barrier();

    socket_push_pages(g_socket, 1);
    ++g_posted;
}

// A single NOC transaction has a burst limit and exceeding it reads NOTHING, so the chunk
// loop is unconditional rather than a limit every caller must remember.
inline void pull(uint64_t src, uint32_t dst_l1, uint32_t bytes) {
    while (bytes) {
        const uint32_t chunk = bytes > NOC_MAX_BURST_SIZE ? NOC_MAX_BURST_SIZE : bytes;
        noc_read_with_state<noc_mode, read_cmd_buf, CQ_NOC_SNDL, CQ_NOC_SEND, CQ_NOC_WAIT>(
            NOC_INDEX, g_rx_pcie_xy_enc, src, dst_l1, chunk);
        src += chunk;
        dst_l1 += chunk;
        bytes -= chunk;
    }
}

// No atomic: this core's puller is the only writer of any signal word on this core, however
// many peers are sending. The senders contend in the ring, not here.
inline void apply_signal(const volatile FrameTrailer* t) {
    if (t->sig_op == kSignalNone) {
        return;
    }
    // The offset came from a peer, so it is bounded before it becomes a store.
    if (!tt_uva_frame_signal_ok(t->sig_op, t->sig_off, g_sig_span)) {
        return;
    }
    volatile tt_l1_ptr uint32_t* const w = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(g_rx_l1_base + t->sig_off);
    *w = t->sig_op == kSignalAdd ? (*w + t->sig_val) : t->sig_val;
}

// The whole ordering contract in one place: the bytes land, the barrier retires the read,
// and only then does anything -- credit or signal -- advertise them.
inline void land_one() {
    pull(g_ring + (g_rx.read_ptr - g_rx.fifo_addr), g_landing, g_page_size);
    noc_async_read_barrier();

    socket_pop_pages(g_rx, 1);
    socket_notify_sender(g_rx);

    apply_signal(reinterpret_cast<const volatile FrameTrailer*>(g_landing + g_page_size - kFrameTrailerBytes));
}

// Lands one frame if one is waiting. The bounded wait is what keeps a stop flag reachable:
// an unbounded socket_wait_for_pages on an idle socket never returns.
inline bool poll_one(uint32_t polls) {
    if (!socket_wait_for_pages(g_rx, 1, polls)) {
        invalidate_l1_cache();
        return false;
    }
    land_one();
    return true;
}

}  // namespace detail

// The device wall clock, for a kernel timing its own loop. Same source the frame trailer
// stamps, so a loop total and a per-frame cost are on one timebase.
inline uint64_t tt_uva_clock() { return detail::wall_clock(); }

// This core's selector, derived from its own coordinates so no argument can forge it.
inline uint32_t tt_uva_self(uint32_t grid_width, uint32_t host, uint32_t chip, uint32_t chips_per_host) {
    const uint32_t core = get_absolute_logical_y() * grid_width + get_absolute_logical_x();
    return tt_uva_t6_global_selector(host, chip, core, chips_per_host);
}

// A zero config address means this RISC does not own that direction. Exactly one RISC may
// drive a socket, so the half this one does not own is left unarmed rather than raced.
inline void tt_uva_ini(
    uint32_t tx_config_addr,
    uint32_t rx_config_addr,
    uint32_t page_size,
    uint32_t stage_addr,
    uint32_t origin,
    uint32_t landing,
    uint32_t l1_base,
    uint32_t l1_size,
    uint32_t consumed_addr = 0) {
    detail::g_tx_on = tx_config_addr != 0;
    detail::g_rx_on = rx_config_addr != 0;
    // Both legs share it, and stage() needs it to place the trailer at the page tail.
    detail::g_page_size = page_size;
    if (detail::g_tx_on) {
        detail::g_socket = create_sender_socket_interface(tx_config_addr);
        set_sender_socket_page_size(detail::g_socket, page_size);
        detail::g_stage_addr = stage_addr;
        detail::g_origin = origin;
        detail::g_consumed_addr = consumed_addr;
        detail::g_posted = 0;
        detail::g_l1_base = l1_base;
        noc_write_init_state<write_cmd_buf>(noc_index, NOC_UNICAST_WRITE_VC);
    }
    if (detail::g_rx_on) {
        detail::g_rx = create_receiver_socket_interface(rx_config_addr);
        set_receiver_socket_page_size(detail::g_rx, page_size);
        // In DEVICE_PULL the ring is pinned HOST memory: fifo_addr is a logical anchor and
        // read_ptr an offset against it, not an L1 address.
        detail::g_ring = (static_cast<uint64_t>(detail::g_rx.h2d.data_addr_hi) << 32) |
                         static_cast<uint64_t>(detail::g_rx.h2d.data_addr_lo);
        detail::g_rx_pcie_xy_enc = detail::g_rx.h2d.pcie_xy_enc;
        detail::g_landing = landing;
        detail::g_rx_l1_base = l1_base;
        detail::g_sig_span = l1_size > l1_base ? l1_size - l1_base : 0;
        // Required before noc_read_with_state on the same buffer: it is what sets
        // RESP_MARKED, without which detail::pull()'s read never retires.
        noc_read_init_state<read_cmd_buf>(NOC_INDEX);
    }
}

// Store `bytes` from L1 to `dst`, and publish it. Blocks only if the FIFO is full.
inline void tt_uva_put(uint32_t src_l1, tt_uva_t dst, uint32_t bytes) {
    detail::stage(src_l1, dst, bytes, 0, 0, kSignalNone);
    socket_notify_receiver(detail::g_socket);
}

// As tt_uva_put, then updates `sig_addr` on the target once the payload has landed there.
// A sibling, not a wrapper: stage() commits, so a later signal would cost a second message.
inline void tt_uva_put_signal(
    uint32_t src_l1, tt_uva_t dst, uint32_t bytes, uint32_t sig_addr, uint32_t sig_val, uint32_t sig_op) {
    detail::stage(src_l1, dst, bytes, sig_addr - detail::g_l1_base, sig_val, sig_op);
    socket_notify_receiver(detail::g_socket);
}

// Returns once THIS host has released every frame this core put, which the H2H leg flushes
// before it acks -- so they are in the peer host's window. One-sided: no far device needed.
inline void tt_uva_quiet() {
    if (detail::g_tx_on) {
        socket_barrier(detail::g_socket);
    }
}

// Returns once the FAR device has pulled everything this core put. A full round trip, so
// calling it per message is the latency shape, not the bandwidth one.
inline void tt_uva_sync() {
    if (detail::g_consumed_addr == 0) {
        return;
    }
    volatile tt_l1_ptr uint32_t* const consumed =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(detail::g_consumed_addr);
    // Signed difference, so the comparison survives the counters wrapping at 2^32.
    while (static_cast<int32_t>(*consumed - detail::g_posted) < 0) {
        invalidate_l1_cache();
    }
}

// Consumer side. A signal word is written by another agent, so every read invalidates:
// the value changes with no local store.

// The verb for a consumer that does NOT own this core's socket -- one on another RISC, which
// must not touch read_ptr. It spins on the fetch itself; a second driver would race the ring.
inline uint32_t tt_uva_signal_fetch(uint32_t sig_addr) {
    invalidate_l1_cache();
    return *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sig_addr);
}

// Reports the signal without committing to reach it, landing at most one frame trying. Tests
// before draining, so a satisfied signal never pulls a frame the caller did not ask for.
inline bool tt_uva_test(uint32_t sig_addr, uint32_t cmp_value, uint32_t polls = 4096) {
    if (static_cast<int32_t>(tt_uva_signal_fetch(sig_addr) - cmp_value) >= 0) {
        return true;
    }
    (void)detail::poll_one(polls);
    return static_cast<int32_t>(tt_uva_signal_fetch(sig_addr) - cmp_value) >= 0;
}

// Drains what this RISC put, then writes back only the sockets it armed: a config write for
// a socket another RISC drives would publish stale pointers over that RISC's progress.
inline void tt_uva_fin() {
    if (detail::g_tx_on) {
        socket_barrier(detail::g_socket);
        update_socket_config(detail::g_socket);
    }
    if (detail::g_rx_on) {
        update_socket_config(detail::g_rx);
    }
    noc_async_write_barrier();
}

}  // namespace tt::tt_metal::experimental
