// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Device-side helpers shared by the DRISC and ERISC microbenchmark kernels.

#pragma once

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "internal/tt-1xx/risc_common.h"
#include "noc_overlay_parameters.h"
#include "drisc_mb_common.hpp"

// The benchmark runs as separate programs, so each kernel fits the DRISC's code region (~11 KB): part 1 measures
// per-operation costs, NoC operations and round trips; part 2 runs the pipelines; part 3 measures the DRISC's
// command buffers and NoC ports under load; part 4 runs two DRISCs on one Ethernet tile. The host sets the part.
#if !defined(DRISC_MB_PART) || DRISC_MB_PART < 1 || DRISC_MB_PART > 4
#error "DRISC_MB_PART must be 1, 2, 3 or 4"
#endif

namespace drisc_mb {

FORCE_INLINE uint32_t now() {
#if defined(DRISC_MB_WALL_CLOCK)
    return get_timestamp_32b();
#else
    uint32_t cycles;
    asm volatile("csrr %0, mcycle" : "=r"(cycles));
    return cycles;
#endif
}

template <typename T>
FORCE_INLINE volatile tt_l1_ptr T* l1(uint32_t addr) {
    return reinterpret_cast<volatile tt_l1_ptr T*>(addr);
}

// Stream registers as counters: remote writers add to them, the owner reads them.
FORCE_INLINE uint32_t stream_value(uint32_t stream) {
    return NOC_STREAM_READ_REG(stream, STREAM_REMOTE_DEST_BUF_SPACE_AVAILABLE_REG_INDEX) &
           ((1u << REMOTE_DEST_WORDS_FREE_WIDTH) - 1);
}
FORCE_INLINE void stream_add_local(uint32_t stream, int32_t delta) {
    NOC_STREAM_WRITE_REG(
        stream,
        STREAM_REMOTE_DEST_BUF_SPACE_AVAILABLE_UPDATE_REG_INDEX,
        static_cast<uint32_t>(delta) << REMOTE_DEST_BUF_WORDS_FREE_INC);
}
// Writing the buffer size also sets the space-available counter (tests/tt_metal/tt_metal/api/test_noc.cpp).
FORCE_INLINE void stream_reset(uint32_t stream) {
    NOC_STREAM_WRITE_REG(stream, STREAM_REMOTE_DEST_BUF_SIZE_REG_INDEX, 0);
}
FORCE_INLINE uint64_t stream_update_addr(uint32_t x, uint32_t y, uint32_t stream, uint8_t noc) {
    return get_noc_addr(x, y, STREAM_REG_ADDR(stream, STREAM_REMOTE_DEST_BUF_SPACE_AVAILABLE_UPDATE_REG_INDEX), noc);
}
// A router-style doorbell: one inline write that adds `delta` to a stream register on another core.
FORCE_INLINE void stream_add_remote(uint64_t update_addr, uint32_t delta, uint8_t noc) {
    noc_inline_dw_write<InlineWriteDst::REG>(update_addr, delta << REMOTE_DEST_BUF_WORDS_FREE_INC, 0xF, noc);
}

struct Results {
    volatile tt_l1_ptr uint32_t* words;
    uint32_t error_entry;

    FORCE_INLINE void put(uint32_t entry, uint32_t cycles, uint32_t ops) {
        words[2 * entry] = cycles;
        words[2 * entry + 1] = ops;
    }
    // Records the first phase whose wait timed out; the kernel then returns.
    FORCE_INLINE void fail(uint32_t phase) {
        if (words[2 * error_entry] == 0) {
            words[2 * error_entry] = phase + 1;
        }
    }
};

// Polls `done` until it holds or kSpinLimit polls pass.
template <typename F>
FORCE_INLINE bool spin_until(F&& done) {
    for (uint32_t i = 0; i < kSpinLimit; i++) {
        if (done()) {
            return true;
        }
    }
    return false;
}

FORCE_INLINE uint32_t next_slot(uint32_t slot, uint32_t num_slots) { return slot + 1 == num_slots ? 0 : slot + 1; }

// Transaction ids 1-15 by slot: slot + 1, wrapping so slots 0 and 15 share an id, which only makes a sent check
// conservative. A compare and an add, as a router would compute it (a modulo costs a multiply chain).
static_assert(kMaxDriscSlots < 30 && kEthLandingSlots < 30);
FORCE_INLINE uint32_t slot_trid(uint32_t slot) { return slot < 15 ? slot + 1 : slot - 14; }

// Lean stateful issue, for a core with one fixed destination per command buffer. The stock transaction-id write
// (noc_async_write_one_packet_with_trid_with_state) polls the command buffer twice: once to set the id, then again,
// right after that register write, to issue. One poll is enough: nothing else issues on the buffer in between, and
// writing the id does not make it busy (the argument ncrisc_noc_write_with_state gives for skip_cmdbuf_chk). The
// length stays in the state, so each write programs only the id, source, destination and go.
// The source is this core: firmware's noc_init sets that only on the write and register-write command buffers, and an
// inline write on any buffer overwrites it with its destination, so set it here.
FORCE_INLINE void lean_write_set_state(
    uint64_t dst_noc_addr, uint32_t size, uint8_t noc, uint32_t cmd_buf = write_cmd_buf) {
    ncrisc_noc_write_set_state</*posted=*/false, /*one_packet=*/true>(
        noc, cmd_buf, dst_noc_addr, size, NOC_UNICAST_WRITE_VC);
    const uint64_t self = NOC_XY_ADDR(my_x[noc], my_y[noc], 0);
    NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_TARG_ADDR_MID, 0);
    NOC_CMD_BUF_WRITE_REG(
        noc, cmd_buf, NOC_TARG_ADDR_COORDINATE, (uint32_t)(self >> NOC_ADDR_COORD_SHIFT) & NOC_COORDINATE_MASK);
}

FORCE_INLINE void lean_trid_write(
    uint32_t src, uint32_t dst_l1, uint32_t trid, uint8_t noc, uint32_t cmd_buf = write_cmd_buf) {
    while (!noc_cmd_buf_ready(noc, cmd_buf));
    NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_PACKET_TAG, NOC_PACKET_TAG_TRANSACTION_ID(trid));
    ncrisc_noc_write_with_state<
        DM_DEDICATED_NOC,
        /*posted=*/false,
        /*update_counter=*/true,
        /*one_packet=*/true,
        /*skip_cmdbuf_chk=*/true>(noc, cmd_buf, src, dst_l1);
}

// A doorbell (stream register += 1) with destination and value set once: each one is a poll and a go.
FORCE_INLINE void lean_doorbell_set_state(uint64_t update_addr, uint8_t noc, uint8_t cmd_buf = write_at_cmd_buf) {
    noc_inline_dw_write_set_state</*posted=*/false, /*set_val=*/true>(
        update_addr, 1u << REMOTE_DEST_BUF_WORDS_FREE_INC, 0xF, cmd_buf, noc);
}

FORCE_INLINE void lean_doorbell(uint8_t noc, uint8_t cmd_buf = write_at_cmd_buf) {
    noc_inline_dw_write_with_state<
        /*update_addr_lo=*/false,
        /*update_counter=*/true,
        /*posted=*/false,
        /*update_addr_hi=*/false,
        /*update_val=*/false,
        InlineWriteDst::REG>(0, 0, cmd_buf, noc);
}

// A stream register += `count` to the destination set by lean_doorbell_set_state: a poll, the value and a go.
FORCE_INLINE void lean_doorbell_add(uint32_t count, uint8_t noc, uint8_t cmd_buf) {
    noc_inline_dw_write_with_state<
        /*update_addr_lo=*/false,
        /*update_counter=*/true,
        /*posted=*/false,
        /*update_addr_hi=*/false,
        /*update_val=*/true,
        InlineWriteDst::REG>(count << REMOTE_DEST_BUF_WORDS_FREE_INC, 0, cmd_buf, noc);
}

// Values for one operation, computed before its timer starts.
struct OpArgs {
    uint32_t src;
    uint32_t dst;     // address at the destination (low half of the NoC address)
    uint32_t dst_hi;  // high half of the NoC address, for stateless calls
    uint32_t trid;
    uint64_t noc_dst() const { return (static_cast<uint64_t>(dst_hi) << 32) | dst; }
};

FORCE_INLINE OpArgs no_args(uint32_t) { return OpArgs{}; }

// Total cycles of `n` operations, each issued with the NoC idle. `prep(i)` computes the operation's values and the
// wait for the previous operation's acks happens before the timer starts; only `issue(args)` is timed, with its
// values already in registers, as a core keeping them there would pay. This is the CPU cost when the port is free;
// it includes one timer read. `issue` should capture by value, so it does not reload captures from the stack.
template <typename Prep, typename Issue>
FORCE_INLINE uint32_t time_alone(uint32_t n, uint8_t noc, Prep&& prep, Issue&& issue) {
    uint32_t total = 0;
    for (uint32_t i = 0; i < n; i++) {
        OpArgs a = prep(i);
        noc_async_write_barrier(noc);
        asm volatile("" : "+r"(a.src), "+r"(a.dst), "+r"(a.dst_hi), "+r"(a.trid));
        const uint32_t t0 = now();
        issue(a);
        total += now() - t0;
    }
    noc_async_write_barrier(noc);
    return total;
}

// Phase barrier with the other core: bump its sync stream, wait for ours to reach the number of barriers passed.
// Callers wait for their own NoC writes of the previous phase before calling it.
struct Peer {
    uint64_t sync_addr;
    uint32_t my_sync_stream;
    uint8_t noc;
    uint32_t passed;

    FORCE_INLINE bool barrier() {
        passed++;
        stream_add_remote(sync_addr, 1, noc);
        noc_async_write_barrier(noc);
        const uint32_t target = passed;
        const uint32_t stream = my_sync_stream;
        return spin_until([&] { return stream_value(stream) >= target; });
    }
};

// Startup: reset our streams first, then tell the peer we are ready and wait for the same from it. The host zeroes
// both ready words before launch, so neither core can increment a stream the other has not reset yet.
FORCE_INLINE bool handshake(uint32_t my_base, uint64_t peer_ready_addr, uint8_t noc) {
    *l1<uint32_t>(my_base + kHandshakeSrcOffset) = kMagicReady;
    noc_async_write_one_packet(my_base + kHandshakeSrcOffset, peer_ready_addr, sizeof(uint32_t), noc);
    noc_async_write_barrier(noc);
    volatile tt_l1_ptr uint32_t* ready = l1<uint32_t>(my_base + kReadyOffset);
    return spin_until([&] {
        invalidate_l1_cache();
        return *ready == kMagicReady;
    });
}

// Per-op costs of local loads and stream register reads, measured the same way on both cores.
struct LocalEntries {
    uint32_t timer, l1_dep, l1_indep4, stream_read, stream_read_indep4, sink;
};
FORCE_INLINE void measure_local(
    Results& res, const LocalEntries& e, uint32_t scratch, const uint32_t (&streams)[4], uint32_t n) {
    uint32_t sink = 0;

    uint32_t start = now();
    for (uint32_t i = 0; i < n; i++) {
        sink += now();
    }
    res.put(e.timer, now() - start, n);

    // Dependent loads: a pointer chase through one self-referencing word.
    *l1<uint32_t>(scratch) = scratch;
    uint32_t addr = scratch;
    start = now();
    for (uint32_t i = 0; i < n; i++) {
        addr = *l1<uint32_t>(addr);
    }
    res.put(e.l1_dep, now() - start, n);
    sink += addr;

    // Four independent loads issued back to back, used afterwards: shows whether loads overlap.
    volatile tt_l1_ptr uint32_t* w = l1<uint32_t>(scratch + 16);
    start = now();
    for (uint32_t i = 0; i < n; i++) {
        const uint32_t a = w[0];
        const uint32_t b = w[1];
        const uint32_t c = w[2];
        const uint32_t d = w[3];
        sink += (a ^ b) + (c ^ d);
    }
    res.put(e.l1_indep4, now() - start, 4 * n);

    start = now();
    for (uint32_t i = 0; i < n; i++) {
        sink += stream_value(streams[0]);
    }
    res.put(e.stream_read, now() - start, n);

    start = now();
    for (uint32_t i = 0; i < n; i++) {
        const uint32_t a = NOC_STREAM_READ_REG(streams[0], STREAM_REMOTE_DEST_BUF_SPACE_AVAILABLE_REG_INDEX);
        const uint32_t b = NOC_STREAM_READ_REG(streams[1], STREAM_REMOTE_DEST_BUF_SPACE_AVAILABLE_REG_INDEX);
        const uint32_t c = NOC_STREAM_READ_REG(streams[2], STREAM_REMOTE_DEST_BUF_SPACE_AVAILABLE_REG_INDEX);
        const uint32_t d = NOC_STREAM_READ_REG(streams[3], STREAM_REMOTE_DEST_BUF_SPACE_AVAILABLE_REG_INDEX);
        sink += (a ^ b) + (c ^ d);
    }
    res.put(e.stream_read_indep4, now() - start, 4 * n);

    res.put(e.sink, sink, 0);
}

}  // namespace drisc_mb
