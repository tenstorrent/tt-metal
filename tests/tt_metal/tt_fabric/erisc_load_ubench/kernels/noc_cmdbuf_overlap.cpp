// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// NoC command buffers on the router's receiving core (see test_erisc_load_overlap.cpp, NocCmdBufOverlap, and
// noc_cmdbuf_overlap_common.hpp for the groups). Runs on ERISC1 with NOC1, as the router's receiver does. Writes are
// issued as its local writes are: each buffer preset once with the write's NOC_CTRL and this core as the source
// (noc_write_init_state; noc_init sets the source coordinate only on buffers 0 and 2), then per write the transaction
// ID and source, length, destination, coordinate and go (noc_write_with_state). Table 1 reads the buffer's ready state
// before each write (inside ncrisc_noc_set_transaction_id); the ready-mask cases read it only when the mask has no
// ready bit for the buffer, and then write the transaction ID directly (the router would use the NoC API's
// skip-check flag, which the tree's API does not have). The buffers' original state is saved at the start and
// restored at the end. Every wait of this kernel's own is bounded: a case that never completes ends the run with an
// error instead of hanging the core.

#include <atomic>
#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_stream_regs.hpp"
#include "tests/tt_metal/tt_fabric/erisc_load_ubench/kernels/noc_cmdbuf_overlap_common.hpp"

using namespace noc_cmdbuf_overlap;

#ifndef NOC_CMDBUF_GROUP
#define NOC_CMDBUF_GROUP 0
#endif

volatile uint32_t sink;  // keeps the table-3 samples

namespace {

constexpr uint32_t kNoc = noc_index;
constexpr uint32_t kSpinLimit = 1u << 24;  // about 0.1 s
uint32_t self_coord;

FORCE_INLINE uint32_t cycles() {
    uint32_t c;
    asm volatile("csrr %0, mcycle" : "=r"(c));
    return c;
}

// Bounded waits.
FORCE_INLINE bool nonposted_acked() {
    for (uint32_t i = 0; i < kSpinLimit; i++) {
        if (ncrisc_noc_nonposted_writes_flushed(kNoc)) {
            return true;
        }
    }
    return false;
}

FORCE_INLINE bool nonposted_sent() {
    for (uint32_t i = 0; i < kSpinLimit; i++) {
        if (ncrisc_noc_nonposted_writes_sent(kNoc)) {
            return true;
        }
    }
    return false;
}

FORCE_INLINE bool posted_sent() {
    for (uint32_t i = 0; i < kSpinLimit; i++) {
        if (ncrisc_noc_posted_writes_sent(kNoc)) {
            return true;
        }
    }
    return false;
}

FORCE_INLINE bool buffer_ready(uint32_t buf) {
    for (uint32_t i = 0; i < kSpinLimit; i++) {
        if (noc_cmd_buf_ready(kNoc, buf)) {
            return true;
        }
    }
    return false;
}

// A buffer set up for writes: NOC_CTRL preset (non-posted as the local writes, or posted as the forwards), and this
// core as the source.
template <uint32_t BUF, enum CQNocCmdFlags FLAGS = CQ_NOC_mkp>
void preset_write() {
    buffer_ready(BUF);
    noc_write_init_state<BUF, FLAGS>(kNoc, kVc);
    noc_cmd_buf_set_targ_addr_coordinate(kNoc, BUF, self_coord);
    noc_cmd_buf_clear_targ_addr_mid(kNoc, BUF);
    noc_cmd_buf_clear_ret_addr_mid(kNoc, BUF);
}

struct Rings {
    uint32_t src_base;
    uint64_t dst_base;  // NoC address of the destination ring
};

FORCE_INLINE uint32_t src_of(const Rings& r, uint32_t slot) { return r.src_base + slot * kSlotBytes; }
FORCE_INLINE uint64_t dst_of(const Rings& r, uint32_t slot) { return r.dst_base + slot * kSlotBytes; }

// Per-case results.
struct Times {
    uint32_t issue_sum = 0, issue_min = ~0u, ready_cycles = 0, first_min = ~0u;
    uint32_t sent_sum = 0, sent_min = ~0u, acked_sum = 0, acked_min = ~0u, reads = 0;

    FORCE_INLINE void add(uint32_t issue, uint32_t first, uint32_t sent, uint32_t acked) {
        issue_sum += issue;
        issue_min = issue < issue_min ? issue : issue_min;
        first_min = first < first_min ? first : first_min;
        sent_sum += sent;
        sent_min = sent < sent_min ? sent : sent_min;
        acked_sum += acked;
        acked_min = acked < acked_min ? acked : acked_min;
    }
    void store(volatile uint32_t* out) const {
        out[ISSUE] = issue_sum;
        out[ISSUE_MIN] = issue_min;
        out[TAG_WAIT] = ready_cycles;
        out[FIRST_MIN] = first_min;
        out[SENT] = sent_sum;
        out[SENT_MIN] = sent_min;
        out[ACKED] = acked_sum;
        out[ACKED_MIN] = acked_min;
        out[READS] = reads;
    }
};

// The ready state of buffers [0, N) as a mask: their registers read as one group (the loads back to back between two
// signal fences, the tests after them).
template <uint32_t N>
FORCE_INLINE uint32_t sample_ready() {
    uint32_t raw[N];
    std::atomic_signal_fence(std::memory_order_seq_cst);
#pragma GCC unroll 4
    for (uint32_t b = 0; b < N; b++) {
        raw[b] = NOC_CMD_BUF_READ_REG(kNoc, b, NOC_CMD_CTRL);
    }
    std::atomic_signal_fence(std::memory_order_seq_cst);
    uint32_t m = 0;
#pragma GCC unroll 4
    for (uint32_t b = 0; b < N; b++) {
        m |= (raw[b] == NOC_CTRL_STATUS_READY ? 1u : 0u) << b;
    }
    return m;
}

// The ready mask: buffers known ready. A buffer seen ready stays ready until this core issues on it.
struct Mask {
    uint32_t ready = 0;
};

template <uint32_t N>
FORCE_INLINE void resample(Mask& m, Times& t) {
    const uint32_t t0 = cycles();
    m.ready = sample_ready<N>();
    t.ready_cycles += cycles() - t0;
    t.reads += N;
}

// Makes sure buffer BUF is ready: from the mask, else a new sample of the pool [0, N), else a wait on BUF. Then
// clears its bit (the command about to be issued makes it busy).
template <uint32_t N, uint32_t BUF>
FORCE_INLINE bool take(Mask& m, Times& t) {
    if (!(m.ready & (1u << BUF))) {
        resample<N>(m, t);
        if (!(m.ready & (1u << BUF))) {
            const uint32_t t0 = cycles();
            const bool ok = buffer_ready(BUF);
            t.ready_cycles += cycles() - t0;
            t.reads += 1;
            if (!ok) {
                return false;
            }
        }
    }
    m.ready &= ~(1u << BUF);
    return true;
}

// A write on a buffer known ready: no ready read.
template <uint32_t BUF, bool POSTED>
FORCE_INLINE void write_known_ready(uint32_t src, uint64_t dst, uint32_t size, uint32_t trid) {
    NOC_CMD_BUF_WRITE_REG(kNoc, BUF, NOC_PACKET_TAG, NOC_PACKET_TAG_TRANSACTION_ID(trid));
    noc_write_with_state<DM_DEDICATED_NOC, BUF, CQ_NOC_SNDL, CQ_NOC_SEND, CQ_NOC_wait, true, POSTED>(
        kNoc, src, dst, size);
}

// A write as the router's local write: the transaction ID call waits for the buffer (a ready read each time).
template <uint32_t BUF, bool POSTED>
FORCE_INLINE void write_checked(Times& t, uint32_t src, uint64_t dst, uint32_t size, uint32_t trid) {
    const uint32_t t0 = cycles();
    ncrisc_noc_set_transaction_id(kNoc, BUF, trid);
    t.ready_cycles += cycles() - t0;
    t.reads += 1;
    noc_write_with_state<DM_DEDICATED_NOC, BUF, CQ_NOC_SNDL, CQ_NOC_SEND, CQ_NOC_wait, true, POSTED>(
        kNoc, src, dst, size);
}

// A write on buffer BUF of the pool [0, N), through the mask.
template <uint32_t N, uint32_t BUF, bool POSTED>
FORCE_INLINE bool write_masked(Mask& m, Times& t, uint32_t src, uint64_t dst, uint32_t size, uint32_t trid) {
    if (!take<N, BUF>(m, t)) {
        return false;
    }
    write_known_ready<BUF, POSTED>(src, dst, size, trid);
    return true;
}

// A write on the lowest buffer of the pool [0, N) the mask has as ready (after a new sample if it has none; buffer
// `next` waited for if the sample finds none either). `next` advances past the buffer used.
template <uint32_t N>
FORCE_INLINE bool write_any(Mask& m, Times& t, uint32_t& next, uint32_t src, uint64_t dst, uint32_t size, uint32_t trid) {
    if ((m.ready & ((1u << N) - 1)) == 0) {
        resample<N>(m, t);
    }
    const uint32_t b = m.ready ? static_cast<uint32_t>(__builtin_ctz(m.ready)) : next;
    next = b + 1 == N ? 0 : b + 1;
    switch (b) {
        case 0: return write_masked<N, 0, false>(m, t, src, dst, size, trid);
        case 1:
            if constexpr (N > 1) {
                return write_masked<N, 1, false>(m, t, src, dst, size, trid);
            }
            break;
        case 2:
            if constexpr (N > 2) {
                return write_masked<N, 2, false>(m, t, src, dst, size, trid);
            }
            break;
        default:
            if constexpr (N > 3) {
                return write_masked<N, 3, false>(m, t, src, dst, size, trid);
            }
            break;
    }
    return false;
}

#if NOC_CMDBUF_GROUP == 0

// Table 1: four writes, write k on buffer k % NBUF, from slot `slot`, each reading its buffer's ready state.
template <uint32_t NBUF>
FORCE_INLINE void checked4(Times& t, const Rings& r, uint32_t slot, uint32_t size, uint32_t& first) {
    const uint32_t t0 = cycles();
    write_checked<0, false>(t, src_of(r, slot), dst_of(r, slot), size, slot);
    first = cycles() - t0;
    write_checked<1 % NBUF, false>(t, src_of(r, slot + 1), dst_of(r, slot + 1), size, slot + 1);
    write_checked<2 % NBUF, false>(t, src_of(r, slot + 2), dst_of(r, slot + 2), size, slot + 2);
    write_checked<3 % NBUF, false>(t, src_of(r, slot + 3), dst_of(r, slot + 3), size, slot + 3);
}

template <uint32_t NBUF>
bool table1_case(volatile uint32_t* out, const Rings& r, uint32_t shape, uint32_t size) {
    const uint32_t groups = (shape == BURST ? kBurstWrites : kStreamWrites) / 4;
    const uint32_t reps = shape == BURST ? kBurstReps : kStreamReps;
    Times t;
    for (uint32_t rep = 0; rep < reps; rep++) {
        if (!nonposted_acked()) {
            return false;
        }
        uint32_t first = 0, group_first = 0;
        const uint32_t t0 = cycles();
        for (uint32_t g = 0; g < groups; g++) {
            checked4<NBUF>(t, r, (g * 4) % kSlots, size, group_first);
            first = g == 0 ? group_first : first;
        }
        const uint32_t t1 = cycles();
        if (!nonposted_sent()) {
            return false;
        }
        const uint32_t t2 = cycles();
        if (!nonposted_acked()) {
            return false;
        }
        const uint32_t t3 = cycles();
        t.add(t1 - t0, first, t2 - t0, t3 - t0);
    }
    t.store(out);
    return true;
}

#elif NOC_CMDBUF_GROUP == 1

// Table 2: four writes through the mask, write k on buffer k % N (ROT) or on the lowest ready one (ANY).
template <uint32_t N, bool ANY>
FORCE_INLINE bool masked4(
    Mask& m, Times& t, uint32_t& next, const Rings& r, uint32_t slot, uint32_t size, uint32_t& first) {
    const uint32_t t0 = cycles();
    bool ok;
    if constexpr (ANY) {
        ok = write_any<N>(m, t, next, src_of(r, slot), dst_of(r, slot), size, slot);
        first = cycles() - t0;
        ok = ok && write_any<N>(m, t, next, src_of(r, slot + 1), dst_of(r, slot + 1), size, slot + 1);
        ok = ok && write_any<N>(m, t, next, src_of(r, slot + 2), dst_of(r, slot + 2), size, slot + 2);
        ok = ok && write_any<N>(m, t, next, src_of(r, slot + 3), dst_of(r, slot + 3), size, slot + 3);
    } else {
        ok = write_masked<N, 0, false>(m, t, src_of(r, slot), dst_of(r, slot), size, slot);
        first = cycles() - t0;
        ok = ok && write_masked<N, 1 % N, false>(m, t, src_of(r, slot + 1), dst_of(r, slot + 1), size, slot + 1);
        ok = ok && write_masked<N, 2 % N, false>(m, t, src_of(r, slot + 2), dst_of(r, slot + 2), size, slot + 2);
        ok = ok && write_masked<N, 3 % N, false>(m, t, src_of(r, slot + 3), dst_of(r, slot + 3), size, slot + 3);
    }
    return ok;
}

// The mask is sampled once before each repetition (the previous pass's end) and after each group of four writes but
// the last (each pass's end); the sample before the repetition is counted in the ready cycles, not in the issue time.
template <uint32_t N, bool ANY>
bool table2_case(volatile uint32_t* out, const Rings& r, uint32_t shape, uint32_t size) {
    const uint32_t groups = (shape == BURST ? kBurstWrites : kStreamWrites) / 4;
    const uint32_t reps = shape == BURST ? kBurstReps : kStreamReps;
    Times t;
    for (uint32_t rep = 0; rep < reps; rep++) {
        if (!nonposted_acked()) {
            return false;
        }
        Mask m;
        uint32_t next = 0, first = 0, group_first = 0;
        resample<N>(m, t);
        const uint32_t t0 = cycles();
        for (uint32_t g = 0; g < groups; g++) {
            if (g != 0) {
                resample<N>(m, t);
            }
            if (!masked4<N, ANY>(m, t, next, r, (g * 4) % kSlots, size, group_first)) {
                return false;
            }
            first = g == 0 ? group_first : first;
        }
        const uint32_t t1 = cycles();
        if (!nonposted_sent()) {
            return false;
        }
        const uint32_t t2 = cycles();
        if (!nonposted_acked()) {
            return false;
        }
        const uint32_t t3 = cycles();
        t.add(t1 - t0, first, t2 - t0, t3 - t0);
    }
    t.store(out);
    return true;
}

#else

// Table 3: cycles for kSampleIters grouped samples of N idle buffers.
template <uint32_t N>
uint32_t table3_case() {
    uint32_t acc = 0;
    const uint32_t t0 = cycles();
    for (uint32_t i = 0; i < kSampleIters; i++) {
        acc += sample_ready<N>();
    }
    const uint32_t t1 = cycles();
    sink = acc;
    return t1 - t0;
}

// Table 4: kPairs forward-like pairs per repetition: a posted data write, then a posted inline write of `val` to the
// destination core's stream register (the forward's credit).
struct Credit {
    uint64_t addr;
    uint32_t val;
};

template <uint32_t MODE>
FORCE_INLINE bool pair(Mask& m, Times& t, const Rings& r, const Credit& c, uint32_t slot, uint32_t size) {
    const uint32_t src = src_of(r, slot);
    const uint64_t dst = dst_of(r, slot);
    if constexpr (MODE == PAIR_SHARED) {
        // One buffer: the inline write leaves its destination as the buffer's target, so the data write restores
        // this core as its source first, and every command is programmed in full.
        const uint32_t t0 = cycles();
        const bool ok = buffer_ready(0);
        t.ready_cycles += cycles() - t0;
        t.reads += 2;  // and the inline write's own check
        if (!ok) {
            return false;
        }
        noc_cmd_buf_set_targ_addr_coordinate(kNoc, 0, self_coord);
        ncrisc_noc_fast_write<DM_DEDICATED_NOC, true, true>(
            kNoc, 0, src, dst, size, kVc, false, false, 1, false, true, slot);
        noc_fast_write_dw_inline<DM_DEDICATED_NOC, InlineWriteDst::REG>(kNoc, 0, c.val, c.addr, 0xF, kVc, false, true);
    } else if constexpr (MODE == PAIR_FULL) {
        write_checked<0, true>(t, src, dst, size, slot);
        t.reads += 1;
        noc_fast_write_dw_inline<DM_DEDICATED_NOC, InlineWriteDst::REG>(kNoc, 3, c.val, c.addr, 0xF, kVc, false, true);
    } else if constexpr (MODE == PAIR_PRESET) {
        write_checked<0, true>(t, src, dst, size, slot);
        t.reads += 1;
        noc_fast_write_dw_inline_with_state<DM_DEDICATED_NOC, false, false, false, true, true, InlineWriteDst::REG>(
            kNoc, 1);
    } else {
        // Ready mask over buffers 0 and 1 (and 2 when the data rotates over 0 and 2).
        constexpr uint32_t N = MODE == PAIR_PRESET_MASK_ROT ? 3 : 2;
        bool ok;
        if (MODE == PAIR_PRESET_MASK_ROT && (slot & 1)) {
            ok = write_masked<N, (N > 2 ? 2 : 0), true>(m, t, src, dst, size, slot);
        } else {
            ok = write_masked<N, 0, true>(m, t, src, dst, size, slot);
        }
        if (!ok || !take<N, 1>(m, t)) {
            return false;
        }
        NOC_CMD_BUF_WRITE_REG(kNoc, 1, NOC_CMD_CTRL, NOC_CTRL_SEND_REQ);  // the preset credit, buffer known ready
        noc_posted_writes_num_issued[kNoc] += 1;
    }
    return true;
}

template <uint32_t MODE>
bool table4_case(volatile uint32_t* out, const Rings& r, const Credit& c, uint32_t size) {
    // Presets for this mode (the shared mode changes buffer 0).
    preset_write<0, CQ_NOC_mkP>();
    preset_write<2, CQ_NOC_mkP>();
    if constexpr (MODE >= PAIR_PRESET) {
        noc_fast_write_dw_inline_set_state<true, true>(kNoc, 1, c.addr, 0xF, kVc, c.val);
    }
    constexpr uint32_t N = MODE == PAIR_PRESET_MASK_ROT ? 3 : 2;
    Times t;
    for (uint32_t rep = 0; rep < kPairReps; rep++) {
        if (!posted_sent()) {
            return false;
        }
        Mask m;
        if constexpr (MODE >= PAIR_PRESET_MASK) {
            resample<N>(m, t);
        }
        uint32_t first = 0;
        const uint32_t t0 = cycles();
        for (uint32_t p = 0; p < kPairs; p++) {
            if constexpr (MODE >= PAIR_PRESET_MASK) {
                if (p != 0) {
                    resample<N>(m, t);
                }
            }
            if (!pair<MODE>(m, t, r, c, p % kSlots, size)) {
                return false;
            }
            first = p == 0 ? cycles() - t0 : first;
        }
        const uint32_t t1 = cycles();
        if (!posted_sent()) {
            return false;
        }
        const uint32_t t2 = cycles();
        t.add(t1 - t0, first, t2 - t0, t2 - t0);
    }
    t.store(out);
    return true;
}

#endif

}  // namespace

void kernel_main() {
    const uint32_t results = get_arg_val<uint32_t>(0);
    const uint32_t src_base = get_arg_val<uint32_t>(1);
    const uint32_t dst_x = get_arg_val<uint32_t>(2);
    const uint32_t dst_y = get_arg_val<uint32_t>(3);
    const uint32_t dst_l1 = get_arg_val<uint32_t>(4);
    volatile uint32_t* out = reinterpret_cast<volatile uint32_t*>(results);
    const Rings r{src_base, get_noc_addr(dst_x, dst_y, dst_l1, kNoc)};

    NocCmdBufState saved[4];
    for (uint32_t b = 0; b < 4; b++) {
        noc_cmd_buf_save_state(kNoc, b, &saved[b]);
    }
    ncrisc_noc_counters_init();
    self_coord =
        static_cast<uint32_t>(NOC_XY_ADDR(my_x[kNoc], my_y[kNoc], 0) >> NOC_ADDR_COORD_SHIFT) & NOC_COORDINATE_MASK;

    // The case that timed out, if any: (table) << 24 | (shape + 1) << 16 | (size index + 1) << 8 | (variant + 1).
    uint32_t failed = 0;
#if NOC_CMDBUF_GROUP == 0
    preset_write<0>();
    preset_write<1>();
    preset_write<2>();
    preset_write<3>();
    for (uint32_t shape = 0; shape < kNumShapes && !failed; shape++) {
        for (uint32_t s = 0; s < kNumSizes && !failed; s++) {
            const uint32_t code = 1u << 24 | (shape + 1) << 16 | (s + 1) << 8;
            if (!table1_case<1>(out + t1_index(shape, s, 0), r, shape, kSizes[s])) {
                failed = code | 1;
            } else if (!table1_case<2>(out + t1_index(shape, s, 1), r, shape, kSizes[s])) {
                failed = code | 2;
            } else if (!table1_case<4>(out + t1_index(shape, s, 2), r, shape, kSizes[s])) {
                failed = code | 3;
            }
        }
    }
#elif NOC_CMDBUF_GROUP == 1
    preset_write<0>();
    preset_write<1>();
    preset_write<2>();
    preset_write<3>();
    for (uint32_t shape = 0; shape < kNumShapes && !failed; shape++) {
        for (uint32_t s = 0; s < kNumSizes && !failed; s++) {
            const uint32_t code = 2u << 24 | (shape + 1) << 16 | (s + 1) << 8;
            const uint32_t size = kSizes[s];
            if (!table2_case<1, false>(out + t2_index(shape, s, MASK_ROT1), r, shape, size)) {
                failed = code | (MASK_ROT1 + 1);
            } else if (!table2_case<2, false>(out + t2_index(shape, s, MASK_ROT2), r, shape, size)) {
                failed = code | (MASK_ROT2 + 1);
            } else if (!table2_case<4, false>(out + t2_index(shape, s, MASK_ROT4), r, shape, size)) {
                failed = code | (MASK_ROT4 + 1);
            } else if (!table2_case<2, true>(out + t2_index(shape, s, MASK_ANY2), r, shape, size)) {
                failed = code | (MASK_ANY2 + 1);
            } else if (!table2_case<4, true>(out + t2_index(shape, s, MASK_ANY4), r, shape, size)) {
                failed = code | (MASK_ANY4 + 1);
            }
        }
    }
#else
    out[kT3 + 0] = table3_case<1>();
    out[kT3 + 1] = table3_case<2>();
    out[kT3 + 2] = table3_case<3>();
    out[kT3 + 3] = table3_case<4>();
    const Credit c{
        get_noc_addr(dst_x, dst_y, get_stream_reg_write_addr<kCreditStream>(), kNoc),
        static_cast<uint32_t>(pack_value_for_inc_on_write_stream_reg_write(1))};
    for (uint32_t s = 0; s < kNumPairSizes && !failed; s++) {
        const uint32_t code = 4u << 24 | 1u << 16 | (s + 1) << 8;
        const uint32_t size = kPairSizes[s];
        if (!table4_case<PAIR_SHARED>(out + t4_index(s, PAIR_SHARED), r, c, size)) {
            failed = code | (PAIR_SHARED + 1);
        } else if (!table4_case<PAIR_FULL>(out + t4_index(s, PAIR_FULL), r, c, size)) {
            failed = code | (PAIR_FULL + 1);
        } else if (!table4_case<PAIR_PRESET>(out + t4_index(s, PAIR_PRESET), r, c, size)) {
            failed = code | (PAIR_PRESET + 1);
        } else if (!table4_case<PAIR_PRESET_MASK>(out + t4_index(s, PAIR_PRESET_MASK), r, c, size)) {
            failed = code | (PAIR_PRESET_MASK + 1);
        } else if (!table4_case<PAIR_PRESET_MASK_ROT>(out + t4_index(s, PAIR_PRESET_MASK_ROT), r, c, size)) {
            failed = code | (PAIR_PRESET_MASK_ROT + 1);
        }
    }
#endif
    out[kFailedBase + NOC_CMDBUF_GROUP] = failed;

    nonposted_acked();
    posted_sent();
    for (uint32_t b = 0; b < 4; b++) {
        noc_cmd_buf_restore_state(kNoc, b, &saved[b]);
    }
    out[kDoneBase + NOC_CMDBUF_GROUP] = kDone;
}
