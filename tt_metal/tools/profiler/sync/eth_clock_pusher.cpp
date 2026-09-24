// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Resident idle-eth clock tracker: the chip's AICLK wall clock modelled against the eth tile's free-running 50 MHz
// counter, as instants the host places records by.
//
// Runs on one idle ethernet core per chip for the life of the profiling session, and does nothing but sample: the
// sampler below and the local clock model. Its instants go to a ring in its L1 (kSyncRingAddr, tail in its
// control block) that the drainer on a second idle core (eth_clock_drainer.cpp) ships to the host over the NoC;
// that core also ships this one's firmware markers and the active eth cores' rings.

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "tools/profiler/kernel_profiler.hpp"
#include "internal/ethernet/eth_ptp_clock.hpp"

constexpr uint32_t kPointTicks = get_compile_time_arg_val(0);  // refclk between the open segment's points (50/us)
constexpr uint32_t kCtrlAddr =
    get_compile_time_arg_val(1);  // done +0, heartbeat +4, go +8, sync tail +12, head +16, stop +64
constexpr uint32_t kRingAddr = get_compile_time_arg_val(2);      // model::kRingSamples raw samples, for transitions
constexpr uint32_t kSyncRingAddr = get_compile_time_arg_val(3);  // this core's instants, kSyncRingRecords of them

namespace kp = kernel_profiler;
namespace eth_ptp = tt::tt_metal::eth_ptp;

#if defined(PROFILE_KERNEL)

// AICLK is a PLL multiple of the crystal the refclk counts: k8/8 wall ticks per refclk tick, k8 an integer, exact
// between DVFS steps (every run longer than 50 ms measured sits on its multiple to <0.005 ppm). So over one rate the
// wall clock is a line whose slope is known once k8 is, and whose only free parameter is the phase of the refclk's
// increment against the wall ticks -- the intercept. This core measures both, and the host receives POINTS of the
// line, never samples to fit: a point is (refclk r, the line's wall at r in eighths of a tick), tagged with k8 and
// the sample count behind it; a new k8, or a CLOSE point, starts the next segment, and two consecutive segments meet
// where their lines cross.
//
// A sample is one advance of the refclk the ERISC reads (it moves in steps of 4 ticks, 80 ns apart) caught between
// two consecutive refclk reads of the sampler below, and placed at the centre of that pair: the wall time of one
// update to within half the pair's width, with no quantisation noise, and unbiased, the update being uniform over
// the pair. About one update in three or four is caught; a fresh segment's intercept is at a quarter of a tick after
// 16 of them and keeps improving as 1/sqrt(n), and the host never places a record on a line drawn through a handful
// of points.
//
// A step shows as kConfirm consecutive samples off the line by more than kOffTicks. The old segment's last on-line
// sample closes it; the new slope is locked once the newest kWinTicks of samples lie on one line, which rejects the
// PLL's glide. Nothing here is a typed-in correction: the pairs' widths are measured, k8 is integer arithmetic, and
// the intercept is a mean.

// This core's instants, in a ring of kSyncRingRecords the drainer reads over the NoC: the tail is published in the
// control block, the drainer writes the count it consumed back beside it. An instant the ring has no room for is
// dropped; this loop never waits for the drainer.
namespace sync {
static uint32_t g_tail = 0;
inline volatile tt_l1_ptr uint32_t* rec(uint32_t i) {
    return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
        kSyncRingAddr + (i % kp::kSyncRingRecords) * kp::kSyncRecordWords * 4u);
}
// `resid8`: the largest residual, in eighths of a wall tick, of the samples behind this point against its line.
inline void write(uint32_t kind, uint32_t role, uint32_t round, uint64_t value, uint64_t wall, uint32_t resid8) {
    const uint32_t head = *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(kCtrlAddr + 16);
    if (g_tail - head >= kp::kSyncRingRecords) {
        return;
    }
    volatile tt_l1_ptr uint32_t* r = rec(g_tail);
    r[kp::SYNC_META] = (kind << 8) | role;
    r[kp::SYNC_ROUND] = round;
    r[kp::SYNC_VALUE_LO] = static_cast<uint32_t>(value);
    r[kp::SYNC_VALUE_HI] = static_cast<uint32_t>(value >> 32);
    r[kp::SYNC_WALL_LO] = static_cast<uint32_t>(wall);
    r[kp::SYNC_WALL_HI] = static_cast<uint32_t>(wall >> 32);
    r[kp::SYNC_REF_LO] = resid8;
    r[kp::SYNC_REF_HI] = 0;
    asm volatile("fence" ::: "memory");
    *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(kCtrlAddr + 12) = ++g_tail;
}
}  // namespace sync

namespace model {
constexpr uint32_t kRingSamples = 512;  // raw samples kept, ~0.3 us apart: ~150 us deep (kEthRingBytes on the host)
constexpr uint32_t kConfirm = 4;        // consecutive off-line samples that make a step
// A sample sits at the centre of the pair of refclk reads that caught its advance, within half a cycle or a cycle
// and a half (kernel_main); the thresholds below sit well above that. Off the line by kOffTicks is off: a 1/8 step
// gets there in ~1 us.
constexpr int64_t kOffTicks = 6;
// The intercept follows the samples: once this many are behind the line it moves by 1/2^kEmaShift of each residue
// (the mean is kept scaled by 2^kEmaShift so sub-tick residues are not lost to the shift), so the phase's wander and
// a frequency a few ppm off the k8 grid walk the intercept instead of the residues, and the points stay on the true
// line. 32 samples is ~15 us; a one-notch step walks the residue ~3 ticks a sample, far past kOffTicks, so it shows.
constexpr uint32_t kEmaShift = 5;
constexpr uint32_t kAcqTicks = 1024;       // refclk after a departure before the first lock test (20 us): past any ramp
constexpr uint32_t kWinTicks = 1024;       // the window that must lie on one line to lock its slope (~45 samples)
constexpr uint32_t kWinSpreadTicks = 14;   // one line's samples spread less than this; a glide inside bends more
constexpr uint32_t kAcqTestEvery = 64;     // samples between lock tests: a scan of the window every ~30 us
constexpr uint32_t kFirstPointN = 16;      // samples behind a segment's first point: a quarter of a tick
constexpr uint32_t kLastDoublingN = 4096;  // points at every doubling of the count up to here, then every kPointTicks
// While no line holds (a step's acquisition, which a PLL glide keeps failing), every kRawMean samples go out as one
// point, their mean, k8 0: the map then bends through the glide instead of bridging it with one chord (whose error is
// the frequency change times the seam length over eight).
constexpr uint32_t kCountMax = 1u << 22;  // the residue sum stops here, and it stays in 32 bits
// A point lies this far behind the newest sample (164 us): a departure is confirmed within ~25 us of samples, and
// a sweep can hold sampling for ~100 us, so no point ever lands on a step the model has not yet seen.
constexpr uint32_t kPointLagTicks = 8192;
constexpr uint64_t kReanchorTicks = 1ull << 30;
static_assert((kRingSamples & (kRingSamples - 1)) == 0);

// A sample: its refclk and the wall clock at it in eighths of a cycle, the unit of every wall quantity below.
struct Raw {
    uint32_t r_lo, r_hi, w8_lo, w8_hi;
};
inline volatile tt_l1_ptr Raw* ring() { return reinterpret_cast<volatile tt_l1_ptr Raw*>(kRingAddr); }
inline uint64_t raw_r(const volatile tt_l1_ptr Raw& e) { return (static_cast<uint64_t>(e.r_hi) << 32) | e.r_lo; }
inline uint64_t raw_w8(const volatile tt_l1_ptr Raw& e) { return (static_cast<uint64_t>(e.w8_hi) << 32) | e.w8_lo; }

struct Model {
    uint32_t k8 = 0;          // wall ticks per refclk tick in eighths; 0 while a slope is being acquired
    uint64_t ra = 0, wa8 = 0;  // anchor: the line passes wa8 + c8 at ra
    int32_t sum = 0;           // residues (w8 - wa8) - k8*(r - ra) summed over the counted samples
    uint32_t n = 0;
    int32_t c8 = 0;          // the residues' running mean: sum / n until 2^kEmaShift samples, then ema >> kEmaShift
    int32_t ema = 0;         // that mean times 2^kEmaShift
    uint64_t r_last_on = 0;  // newest sample on the line
    uint32_t off = 0;        // consecutive samples off the line
    uint64_t r_dep = 0;      // the first of them
    uint64_t r_acq0 = 0;     // where the acquisition began
    uint32_t acq_count = 0;
    uint64_t r_lock = 0;  // where the segment's line begins: the oldest sample of the window that locked it
    uint64_t r_last_point = 0;
    uint32_t max_d8 = 0;  // the largest |residual| of an on-line sample since the last point, in eighths
    uint32_t ring_n = 0;  // ring entries written; the newest is ring()[(ring_n - 1) & (kRingSamples - 1)]
    uint32_t win_i = 0;   // the oldest ring entry within kWinTicks of the newest sample (try_lock's window)
    // The last kWin samples, in this RISC's local memory (the L1 ring above is for the lock test; reading it back
    // costs an L1 miss per word). A raw instant is the mean of kRawMean of them, each sample in one instant. The mean
    // of samples spread over a bending stretch sits above the curve by the curvature times the spread's variance over
    // two: four samples over ~0.75 us put an instant ~0.2 cycles off in the steepest glide at 0.8 GHz, while their
    // mean halves a sample's noise (within half a cycle to a cycle and a half of the truth). A mean must not span a
    // hole in the samples (it would average across the glide's bend): win_from is the first sample after the last one.
    static constexpr uint32_t kWin = 16;
    uint64_t win_r[kWin] = {}, win_w8[kWin] = {};
    uint32_t win_n = 0;     // samples pushed; the newest is win_r[(win_n - 1) & (kWin - 1)]
    uint32_t win_from = 0;  // the first sample a mean may include
    uint32_t since = 0;     // samples since the last instant
};

inline __attribute__((always_inline)) void ring_push(Model& m, uint64_t r, uint64_t w8) {
    volatile tt_l1_ptr Raw& e = ring()[m.ring_n & (kRingSamples - 1)];
    e.r_lo = static_cast<uint32_t>(r);
    e.r_hi = static_cast<uint32_t>(r >> 32);
    e.w8_lo = static_cast<uint32_t>(w8);
    e.w8_hi = static_cast<uint32_t>(w8 >> 32);
    m.ring_n++;
}

// The line's wall at refclk r.
inline int64_t line_w8(const Model& m, uint64_t r) {
    return static_cast<int64_t>(m.wa8) + m.c8 + static_cast<int64_t>(m.k8) * static_cast<int64_t>(r - m.ra);
}

// A point of the line at refclk r, in eighths of a tick, never before the segment's own start: the host keeps
// segments disjoint in refclk.
inline void write_point(Model& m, uint64_t r, uint32_t role) {
    r = r > m.r_lock ? r : m.r_lock;
    m.r_last_point = r;
    const uint32_t n = m.n < kCountMax ? m.n : kCountMax - 1;
    sync::write(kp::kSyncKindLocal, role, m.k8 | (n << 8), r, static_cast<uint64_t>(line_w8(m, r)), m.max_d8);
    m.max_d8 = 0;
}

// A sample as a point: k8 0 tells the host it is an instant of the wall clock, on no line.
inline void write_raw_point(Model& m, uint64_t r, uint64_t w8) {
    m.r_last_point = r;
    sync::write(kp::kSyncKindLocal, kp::kSyncLocalPoint, 0, r, w8, 0u);
}
inline void begin_acquire(Model& m, uint64_t r_from) {
    m.k8 = 0;
    m.n = 0;
    m.sum = 0;
    m.r_acq0 = r_from;
    m.acq_count = 0;
    m.since = 0;
    m.win_i = m.ring_n;
}
constexpr uint32_t kRawMean = 4;
constexpr uint32_t kGroupGapTicks = 125;  // 2.5 us
static_assert((kRawMean & (kRawMean - 1)) == 0);
constexpr uint32_t kRawMeanShift = __builtin_ctz(kRawMean);
inline __attribute__((always_inline)) void win_push(Model& m, uint64_t r, uint64_t w8) {
    if (m.win_n != 0 && static_cast<uint32_t>(r - m.win_r[(m.win_n - 1) & (Model::kWin - 1)]) > kGroupGapTicks) {
        m.win_from = m.win_n;
    }
    m.win_r[m.win_n & (Model::kWin - 1)] = r;
    m.win_w8[m.win_n & (Model::kWin - 1)] = w8;
    m.win_n++;
}
// The mean of the kRawMean samples ending at (exclusive) window index `end`: the refclk exact (the ERISC sees the
// count move in fours), the wall rounded to its eighth.
inline void win_mean(const Model& m, uint32_t end, uint64_t& mr, uint64_t& mw8) {
    uint64_t sr = 0, sw = 0;
    for (uint32_t i = end - kRawMean; i < end; i++) {
        sr += m.win_r[i & (Model::kWin - 1)];
        sw += m.win_w8[i & (Model::kWin - 1)];
    }
    mr = sr >> kRawMeanShift;
    mw8 = (sw + (kRawMean / 2)) >> kRawMeanShift;
}
__attribute__((noinline)) void raw_sample(Model& m) {
    if (++m.since < kRawMean) {
        return;
    }
    m.since = 0;
    if (m.win_n - m.win_from < kRawMean) {
        return;
    }
    uint64_t mr, mw;
    win_mean(m, m.win_n, mr, mw);
    write_raw_point(m, mr, mw);
}

// Locks the slope from the newest kWinTicks of the ring when those samples lie on one line: the slope from the
// window's ends, then every kLockStride-th entry's residue against it within one refclk tick of phase. A window
// across a glide bends more than that and the test is retried after kAcqTestEvery samples. The window's oldest
// entry is carried in m.win_i, advanced a step at a time as samples arrive, so the test walks ~17 entries: whole,
// over every entry, it held the sampler 4-6 us at +40 and +60 us of every seam, holes a glide bent across.
constexpr uint32_t kLockStride = 4;
inline __attribute__((always_inline)) void advance_window(Model& m, uint64_t r_now) {
    while (m.win_i + 1 < m.ring_n && r_now - raw_r(ring()[m.win_i & (kRingSamples - 1)]) > kWinTicks) {
        m.win_i++;
    }
}
__attribute__((noinline)) bool try_lock(Model& m, uint64_t r_now) {
    const uint32_t cnt = m.ring_n - m.win_i;
    if (cnt < 8 || cnt > kRingSamples) {
        return false;
    }
    const volatile tt_l1_ptr Raw& oldest = ring()[m.win_i & (kRingSamples - 1)];
    const volatile tt_l1_ptr Raw& newest = ring()[(m.ring_n - 1) & (kRingSamples - 1)];
    const uint64_t r_old = raw_r(oldest), w_old8 = raw_w8(oldest);
    const uint32_t dr = static_cast<uint32_t>(raw_r(newest) - r_old);
    const uint32_t dw8 = static_cast<uint32_t>(raw_w8(newest) - w_old8);
    if (dr < kWinTicks / 2) {
        return false;
    }
    const uint32_t k8 = (dw8 + dr / 2u) / dr;
    int32_t lo = 0, hi = 0, sum = 0;
    uint32_t n = 0;
    for (uint32_t i = m.win_i; i < m.ring_n; i += kLockStride, n++) {
        const volatile tt_l1_ptr Raw& e = ring()[i & (kRingSamples - 1)];
        const int32_t res = static_cast<int32_t>(static_cast<uint32_t>(raw_w8(e) - w_old8)) -
                            static_cast<int32_t>(k8 * static_cast<uint32_t>(raw_r(e) - r_old));
        lo = n == 0 || res < lo ? res : lo;
        hi = n == 0 || res > hi ? res : hi;
        sum += res;
    }
    if (static_cast<uint32_t>(hi - lo) > 8u * kWinSpreadTicks) {
        return false;
    }
    m.k8 = k8;
    m.ra = r_old;
    m.wa8 = w_old8;
    m.r_lock = r_old;
    m.n = n;
    m.sum = sum;
    m.c8 = sum / static_cast<int32_t>(n);
    m.ema = m.c8 << kEmaShift;
    m.r_last_on = r_now;
    m.off = 0;
    return true;
}

// A confirmed step. The glide is under way before any sample is off the line by kOffTicks: the samples drift
// within it for a microsecond, then kConfirm of them are counted off. The line closes kPreSamples before the first
// off sample, where the samples still sat on it, the slope acquisition restarts from the departure, and those
// samples and the confirming ones open the seam, so the onset is placed from measurements, not from the line.
constexpr uint32_t kPreSamples = 8;
__attribute__((noinline, cold)) void step(Model& m) {
    constexpr uint32_t back = kConfirm + kPreSamples;
    static_assert(back % kRawMean == 0);
    static_assert(back + 1 <= Model::kWin);
    write_point(m, m.win_r[(m.win_n - back - 1) & (Model::kWin - 1)], kp::kSyncLocalClose);
    begin_acquire(m, m.r_dep);
    // The seam's first instants, the means over those samples, from local memory: every cycle here is a sample
    // not taken at the onset, the glide's steepest microseconds.
    for (uint32_t end = m.win_n - back + kRawMean; end <= m.win_n; end += kRawMean) {
        if (end - kRawMean < m.win_from) {
            continue;
        }
        uint64_t mr, mw;
        win_mean(m, end, mr, mw);
        write_raw_point(m, mr, mw);
    }
}

// A sample while no line holds: the ring (the lock test's window) gets it, and the seam its instants.
__attribute__((noinline)) void acquire(Model& m, uint64_t r, uint64_t w8) {
    ring_push(m, r, w8);
    advance_window(m, r);
    if (r - m.r_acq0 >= kAcqTicks && ++m.acq_count >= kAcqTestEvery) {
        m.acq_count = 0;
        if (try_lock(m, r)) {
            return;
        }
    }
    raw_sample(m);
}
__attribute__((noinline)) void off_line(Model& m, uint64_t r) {
    if (m.off++ == 0) {
        m.r_dep = r;
    }
    if (m.off >= kConfirm) {
        step(m);
    }
}
__attribute__((noinline)) void count_point(Model& m, uint64_t r) {
    if (m.n <= (1u << kEmaShift)) {
        m.c8 = m.sum / static_cast<int32_t>(m.n);
        m.ema = m.c8 << kEmaShift;
    }
    if (m.n >= kFirstPointN && m.n <= kLastDoublingN) {
        write_point(m, r - kPointLagTicks, kp::kSyncLocalPoint);
    }
}
__attribute__((noinline)) void line_housekeeping(Model& m, uint64_t r) {
    if (r - m.r_last_point >= kPointTicks) {
        write_point(m, r - kPointLagTicks, kp::kSyncLocalPoint);
    }
    if (r - m.ra >= kReanchorTicks) {
        m.ra += kReanchorTicks;
        m.wa8 += static_cast<uint64_t>(m.k8) * kReanchorTicks;
    }
}
// On a line the residue is computed in 32 bits: it is small, and the line's terms wrap alike. The ring is left alone
// (only a lock test reads it, and a step restarts it).
inline __attribute__((always_inline)) void feed(Model& m, uint64_t r, uint64_t w8) {
    win_push(m, r, w8);
    if (m.k8 == 0) {
        acquire(m, r, w8);
        return;
    }
    const int32_t e = static_cast<int32_t>(
        static_cast<uint32_t>(w8 - m.wa8) -
        m.k8 * static_cast<uint32_t>(static_cast<uint32_t>(r) - static_cast<uint32_t>(m.ra)));
    const int32_t d = e - m.c8;
    if (d > 8 * kOffTicks || d < -8 * kOffTicks) {
        off_line(m, r);
        return;
    }
    m.off = 0;
    m.r_last_on = r;
    const uint32_t ad = static_cast<uint32_t>(d < 0 ? -d : d);
    m.max_d8 = ad > m.max_d8 ? ad : m.max_d8;
    if (m.n >= (1u << kEmaShift)) {
        m.ema += e - (m.ema >> kEmaShift);
        m.c8 = (m.ema + (1 << (kEmaShift - 1))) >> kEmaShift;
    }
    if (m.n < kCountMax) {
        m.sum += e;
        m.n++;
        if ((m.n & (m.n - 1)) == 0) {
            count_point(m, r);
        }
    }
    if (static_cast<uint32_t>(r) - static_cast<uint32_t>(m.r_last_point) >= kPointTicks ||
        static_cast<uint32_t>(r) - static_cast<uint32_t>(m.ra) >= kReanchorTicks) {
        line_housekeeping(m, r);
    }
}
}  // namespace model

// Both clocks' high words and the low words they were last seen at.
struct Carry {
    uint32_t r_hi, w_hi, prev_r_lo, prev_w_lo;
};
// A caught advance: the new count and the wall of its place, in eighths, go to the model.
inline __attribute__((always_inline)) void sample(model::Model& m, Carry& c, uint32_t r_lo, uint32_t w, uint32_t pos8) {
    c.r_hi += r_lo < c.prev_r_lo;
    c.w_hi += w < c.prev_w_lo;
    c.prev_r_lo = r_lo;
    c.prev_w_lo = w;
    model::feed(
        m, (static_cast<uint64_t>(c.r_hi) << 32) | r_lo, (((static_cast<uint64_t>(c.w_hi) << 32) | w) << 3) + pos8);
}

// The sampler. A pass is kBlocks + 2 blocks of [refclk, refclk, wall, refclk] in one asm block: the load unit takes
// four loads in four consecutive cycles and idles two, and each block's check of the block before it sits in those two
// (measured: sixteen such blocks take 96 cycles, as sixteen without the checks do, and a check one block behind its
// value never waits for it). The refclk reads run unbroken for longer than an advance period at any AICLK, so a pass
// catches the first advance after it starts whatever its phase, and no phase of the loop against the advances can
// starve the sampler. A block's pairs are the one from the block before across its idle slots, the one between its
// first two reads and the one across its wall read; the catch counts only if the wall reads of its block and of the
// blocks either side are exactly one block period apart, so every read of its pair issued in its place, and a stall
// anywhere else in the pass (a fetch after the model's code evicted it) cannot move it. The branch predictor is off for
// the pass: a check taken on the last pass is otherwise mispredicted on this one. A 0-5 cycle pad before each pass puts
// the advance at a uniform phase against the blocks, so a catch is uniform within its pair and the pair's centre places
// it without bias; the pairs' widths are measured before the go word from the catches' share of each, the reads placed
// from the one a cycle before the wall read.
namespace sampler {
constexpr uint32_t kBlocks = 20;
inline __attribute__((always_inline)) uint32_t pass(uint32_t (&w)[3], uint32_t (&o)[4]) {
    uint32_t st, w0, w1, w2, o0, o1, o2, o3, x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11;
    asm volatile(
        ".option push\n\t"
        ".option norvc\n\t"
        "csrrsi zero, 0x7c0, 2\n\t"
        "lw %[s2_1], 0(%[cfr])\n\t"
        "lw %[s2_2], 0(%[cfr])\n\t"
        "lw %[s2_0], 0(%[wall])\n\t"
        "lw %[s2_3], 0(%[cfr])\n\t"
        "nop\n\t"
        "nop\n\t"
        "lw %[s0_1], 0(%[cfr])\n\t"
        "lw %[s0_2], 0(%[cfr])\n\t"
        "lw %[s0_0], 0(%[wall])\n\t"
        "lw %[s0_3], 0(%[cfr])\n\t"
        "nop\n\t"
        "nop\n\t"
        "lw %[s1_1], 0(%[cfr])\n\t"
        "lw %[s1_2], 0(%[cfr])\n\t"
        "lw %[s1_0], 0(%[wall])\n\t"
        "lw %[s1_3], 0(%[cfr])\n\t"
        "bne %[s0_3], %[s2_3], .Lh%=_0\n\t"
        "nop\n\t"
        "lw %[s2_1], 0(%[cfr])\n\t"
        "lw %[s2_2], 0(%[cfr])\n\t"
        "lw %[s2_0], 0(%[wall])\n\t"
        "lw %[s2_3], 0(%[cfr])\n\t"
        "bne %[s1_3], %[s0_3], .Lh%=_1\n\t"
        "nop\n\t"
        "lw %[s0_1], 0(%[cfr])\n\t"
        "lw %[s0_2], 0(%[cfr])\n\t"
        "lw %[s0_0], 0(%[wall])\n\t"
        "lw %[s0_3], 0(%[cfr])\n\t"
        "bne %[s2_3], %[s1_3], .Lh%=_2\n\t"
        "nop\n\t"
        "lw %[s1_1], 0(%[cfr])\n\t"
        "lw %[s1_2], 0(%[cfr])\n\t"
        "lw %[s1_0], 0(%[wall])\n\t"
        "lw %[s1_3], 0(%[cfr])\n\t"
        "bne %[s0_3], %[s2_3], .Lh%=_3\n\t"
        "nop\n\t"
        "lw %[s2_1], 0(%[cfr])\n\t"
        "lw %[s2_2], 0(%[cfr])\n\t"
        "lw %[s2_0], 0(%[wall])\n\t"
        "lw %[s2_3], 0(%[cfr])\n\t"
        "bne %[s1_3], %[s0_3], .Lh%=_4\n\t"
        "nop\n\t"
        "lw %[s0_1], 0(%[cfr])\n\t"
        "lw %[s0_2], 0(%[cfr])\n\t"
        "lw %[s0_0], 0(%[wall])\n\t"
        "lw %[s0_3], 0(%[cfr])\n\t"
        "bne %[s2_3], %[s1_3], .Lh%=_5\n\t"
        "nop\n\t"
        "lw %[s1_1], 0(%[cfr])\n\t"
        "lw %[s1_2], 0(%[cfr])\n\t"
        "lw %[s1_0], 0(%[wall])\n\t"
        "lw %[s1_3], 0(%[cfr])\n\t"
        "bne %[s0_3], %[s2_3], .Lh%=_6\n\t"
        "nop\n\t"
        "lw %[s2_1], 0(%[cfr])\n\t"
        "lw %[s2_2], 0(%[cfr])\n\t"
        "lw %[s2_0], 0(%[wall])\n\t"
        "lw %[s2_3], 0(%[cfr])\n\t"
        "bne %[s1_3], %[s0_3], .Lh%=_7\n\t"
        "nop\n\t"
        "lw %[s0_1], 0(%[cfr])\n\t"
        "lw %[s0_2], 0(%[cfr])\n\t"
        "lw %[s0_0], 0(%[wall])\n\t"
        "lw %[s0_3], 0(%[cfr])\n\t"
        "bne %[s2_3], %[s1_3], .Lh%=_8\n\t"
        "nop\n\t"
        "lw %[s1_1], 0(%[cfr])\n\t"
        "lw %[s1_2], 0(%[cfr])\n\t"
        "lw %[s1_0], 0(%[wall])\n\t"
        "lw %[s1_3], 0(%[cfr])\n\t"
        "bne %[s0_3], %[s2_3], .Lh%=_9\n\t"
        "nop\n\t"
        "lw %[s2_1], 0(%[cfr])\n\t"
        "lw %[s2_2], 0(%[cfr])\n\t"
        "lw %[s2_0], 0(%[wall])\n\t"
        "lw %[s2_3], 0(%[cfr])\n\t"
        "bne %[s1_3], %[s0_3], .Lh%=_10\n\t"
        "nop\n\t"
        "lw %[s0_1], 0(%[cfr])\n\t"
        "lw %[s0_2], 0(%[cfr])\n\t"
        "lw %[s0_0], 0(%[wall])\n\t"
        "lw %[s0_3], 0(%[cfr])\n\t"
        "bne %[s2_3], %[s1_3], .Lh%=_11\n\t"
        "nop\n\t"
        "lw %[s1_1], 0(%[cfr])\n\t"
        "lw %[s1_2], 0(%[cfr])\n\t"
        "lw %[s1_0], 0(%[wall])\n\t"
        "lw %[s1_3], 0(%[cfr])\n\t"
        "bne %[s0_3], %[s2_3], .Lh%=_12\n\t"
        "nop\n\t"
        "lw %[s2_1], 0(%[cfr])\n\t"
        "lw %[s2_2], 0(%[cfr])\n\t"
        "lw %[s2_0], 0(%[wall])\n\t"
        "lw %[s2_3], 0(%[cfr])\n\t"
        "bne %[s1_3], %[s0_3], .Lh%=_13\n\t"
        "nop\n\t"
        "lw %[s0_1], 0(%[cfr])\n\t"
        "lw %[s0_2], 0(%[cfr])\n\t"
        "lw %[s0_0], 0(%[wall])\n\t"
        "lw %[s0_3], 0(%[cfr])\n\t"
        "bne %[s2_3], %[s1_3], .Lh%=_14\n\t"
        "nop\n\t"
        "lw %[s1_1], 0(%[cfr])\n\t"
        "lw %[s1_2], 0(%[cfr])\n\t"
        "lw %[s1_0], 0(%[wall])\n\t"
        "lw %[s1_3], 0(%[cfr])\n\t"
        "bne %[s0_3], %[s2_3], .Lh%=_15\n\t"
        "nop\n\t"
        "lw %[s2_1], 0(%[cfr])\n\t"
        "lw %[s2_2], 0(%[cfr])\n\t"
        "lw %[s2_0], 0(%[wall])\n\t"
        "lw %[s2_3], 0(%[cfr])\n\t"
        "bne %[s1_3], %[s0_3], .Lh%=_16\n\t"
        "nop\n\t"
        "lw %[s0_1], 0(%[cfr])\n\t"
        "lw %[s0_2], 0(%[cfr])\n\t"
        "lw %[s0_0], 0(%[wall])\n\t"
        "lw %[s0_3], 0(%[cfr])\n\t"
        "bne %[s2_3], %[s1_3], .Lh%=_17\n\t"
        "nop\n\t"
        "lw %[s1_1], 0(%[cfr])\n\t"
        "lw %[s1_2], 0(%[cfr])\n\t"
        "lw %[s1_0], 0(%[wall])\n\t"
        "lw %[s1_3], 0(%[cfr])\n\t"
        "bne %[s0_3], %[s2_3], .Lh%=_18\n\t"
        "nop\n\t"
        "lw %[s2_1], 0(%[cfr])\n\t"
        "lw %[s2_2], 0(%[cfr])\n\t"
        "lw %[s2_0], 0(%[wall])\n\t"
        "lw %[s2_3], 0(%[cfr])\n\t"
        "bne %[s1_3], %[s0_3], .Lh%=_19\n\t"
        "nop\n\t"
        "li %[st], 0\n\t"
        "j .Le%=\n\t"
        ".Lh%=_0:\n\t"
        "li %[st], 1\n\t"
        "j .Lt%=_0\n\t"
        ".Lh%=_1:\n\t"
        "li %[st], 2\n\t"
        "j .Lt%=_1\n\t"
        ".Lh%=_2:\n\t"
        "li %[st], 3\n\t"
        "j .Lt%=_2\n\t"
        ".Lh%=_3:\n\t"
        "li %[st], 4\n\t"
        "j .Lt%=_0\n\t"
        ".Lh%=_4:\n\t"
        "li %[st], 5\n\t"
        "j .Lt%=_1\n\t"
        ".Lh%=_5:\n\t"
        "li %[st], 6\n\t"
        "j .Lt%=_2\n\t"
        ".Lh%=_6:\n\t"
        "li %[st], 7\n\t"
        "j .Lt%=_0\n\t"
        ".Lh%=_7:\n\t"
        "li %[st], 8\n\t"
        "j .Lt%=_1\n\t"
        ".Lh%=_8:\n\t"
        "li %[st], 9\n\t"
        "j .Lt%=_2\n\t"
        ".Lh%=_9:\n\t"
        "li %[st], 10\n\t"
        "j .Lt%=_0\n\t"
        ".Lh%=_10:\n\t"
        "li %[st], 11\n\t"
        "j .Lt%=_1\n\t"
        ".Lh%=_11:\n\t"
        "li %[st], 12\n\t"
        "j .Lt%=_2\n\t"
        ".Lh%=_12:\n\t"
        "li %[st], 13\n\t"
        "j .Lt%=_0\n\t"
        ".Lh%=_13:\n\t"
        "li %[st], 14\n\t"
        "j .Lt%=_1\n\t"
        ".Lh%=_14:\n\t"
        "li %[st], 15\n\t"
        "j .Lt%=_2\n\t"
        ".Lh%=_15:\n\t"
        "li %[st], 16\n\t"
        "j .Lt%=_0\n\t"
        ".Lh%=_16:\n\t"
        "li %[st], 17\n\t"
        "j .Lt%=_1\n\t"
        ".Lh%=_17:\n\t"
        "li %[st], 18\n\t"
        "j .Lt%=_2\n\t"
        ".Lh%=_18:\n\t"
        "li %[st], 19\n\t"
        "j .Lt%=_0\n\t"
        ".Lh%=_19:\n\t"
        "li %[st], 20\n\t"
        "j .Lt%=_1\n\t"
        ".Lt%=_0:\n\t"
        "mv %[a], %[s2_0]\n\t"
        "mv %[b], %[s0_0]\n\t"
        "mv %[c], %[s1_0]\n\t"
        "mv %[o0], %[s2_3]\n\t"
        "mv %[o1], %[s0_1]\n\t"
        "mv %[o2], %[s0_2]\n\t"
        "mv %[o3], %[s0_3]\n\t"
        "j .Le%=\n\t"
        ".Lt%=_1:\n\t"
        "mv %[a], %[s0_0]\n\t"
        "mv %[b], %[s1_0]\n\t"
        "mv %[c], %[s2_0]\n\t"
        "mv %[o0], %[s0_3]\n\t"
        "mv %[o1], %[s1_1]\n\t"
        "mv %[o2], %[s1_2]\n\t"
        "mv %[o3], %[s1_3]\n\t"
        "j .Le%=\n\t"
        ".Lt%=_2:\n\t"
        "mv %[a], %[s1_0]\n\t"
        "mv %[b], %[s2_0]\n\t"
        "mv %[c], %[s0_0]\n\t"
        "mv %[o0], %[s1_3]\n\t"
        "mv %[o1], %[s2_1]\n\t"
        "mv %[o2], %[s2_2]\n\t"
        "mv %[o3], %[s2_3]\n\t"
        ".Le%=:\n\t"
        "csrrci zero, 0x7c0, 2\n\t"
        ".option pop\n\t"
        : [st] "=&r"(st),
          [a] "=&r"(w0),
          [b] "=&r"(w1),
          [c] "=&r"(w2),
          [o0] "=&r"(o0),
          [o1] "=&r"(o1),
          [o2] "=&r"(o2),
          [o3] "=&r"(o3),
          [s0_0] "=&r"(x0),
          [s0_1] "=&r"(x1),
          [s0_2] "=&r"(x2),
          [s0_3] "=&r"(x3),
          [s1_0] "=&r"(x4),
          [s1_1] "=&r"(x5),
          [s1_2] "=&r"(x6),
          [s1_3] "=&r"(x7),
          [s2_0] "=&r"(x8),
          [s2_1] "=&r"(x9),
          [s2_2] "=&r"(x10),
          [s2_3] "=&r"(x11)
        : [wall] "r"(eth_ptp::kWallClockLo), [cfr] "r"(eth_ptp::kPtpCfrLo)
        : "memory");
    w[0] = w0;
    w[1] = w1;
    w[2] = w2;
    o[0] = o0;
    o[1] = o1;
    o[2] = o2;
    o[3] = o3;
    return st;
}
__attribute__((noinline)) uint32_t pass_out_of_line(uint32_t (&w)[3], uint32_t (&o)[4]) { return pass(w, o); }
inline __attribute__((always_inline)) void pad(uint32_t& walk) {
    walk = walk * 1103515245u + 12345u;
    const uint32_t n = ((walk >> 16) * 6u) >> 16;
    asm volatile(
        ".option push\n\t"
        ".option norvc\n\t"
        "la t0, 1f\n\t"
        "slli t1, %0, 2\n\t"
        "sub t0, t0, t1\n\t"
        "jr t0\n\t"
        ".rept 5\n\t"
        "nop\n\t"
        ".endr\n"
        "1:\n\t"
        ".option pop"
        :
        : "r"(n)
        : "t0", "t1", "memory");
}
// The block period, and pair p's place past the catch's reference wall read in eighths of a cycle (four cycles
// before its block's wall read, so every place is positive): 0 the pair from the block before, 1 the pair between
// the block's first two reads, 2 the pair across its wall read. Typed until measured.
struct Table {
    uint32_t period = 6;
    uint32_t pos8[3] = {32 - 28, 32 - 12, 32};
};
struct Catch {
    uint32_t r, w, pos8, pair;
};
template <uint32_t (*Pass)(uint32_t (&)[3], uint32_t (&)[4])>
inline __attribute__((always_inline)) bool take(uint32_t& walk, Catch& c, const Table& t) {
    uint32_t w[3], o[4];
    pad(walk);
    if (Pass(w, o) == 0 || w[1] - w[0] != t.period || w[2] - w[1] != t.period) {
        return false;
    }
    const uint32_t p = o[1] != o[0] ? 0u : o[2] != o[1] ? 1u : 2u;
    c.r = o[p + 1];
    c.w = w[1] - 4u;
    c.pos8 = t.pos8[p];
    c.pair = p;
    return true;
}
// Before the go word: the block period, the mode of the catching block's wall step over the first 1024 catches, then
// the catches counted by pair until the go word, and at least 2^16 of them.
Table calibrate(volatile tt_l1_ptr uint32_t* go, volatile tt_l1_ptr uint32_t* stop, volatile tt_l1_ptr uint32_t* hb) {
    Table t;
    static uint32_t hist[64];
    uint32_t walk = eth_ptp::rd(eth_ptp::kWallClockLo) | 1u;
    for (uint32_t i = 0; i < 1024; i++) {
        uint32_t w[3], o[4];
        pad(walk);
        if (pass_out_of_line(w, o) != 0) {
            hist[(w[1] - w[0]) & 63u]++;
        }
    }
    for (uint32_t i = 1; i < 64; i++) {
        t.period = hist[i] > hist[t.period] ? i : t.period;
    }
    uint32_t pairs[3] = {}, total = 0;
    for (uint32_t i = 1; *stop == 0u; i++) {
        Catch c;
        if (take<pass_out_of_line>(walk, c, t)) {
            pairs[c.pair]++;
            total++;
        }
        if ((i & 1023u) == 0u) {
            (*hb)++;
            invalidate_l1_cache();
            if ((total >= (1u << 16) && *go != 0u) || total >= (1u << 26)) {
                break;
            }
        }
    }
    uint32_t k = 0;
    while ((total >> k) >= (1u << 20)) {
        k++;
    }
    const uint32_t n = total >> k;
    if (n == 0) {
        return t;
    }
    int32_t g64[3];
    for (uint32_t p = 0; p < 3; p++) {
        g64[p] = static_cast<int32_t>((64u * t.period * (pairs[p] >> k)) / n);
    }
    const int32_t r1 = 4 * 64 - 64;  // the read before the wall read, in 64ths past the reference
    const int32_t c64[3] = {r1 - g64[1] - g64[0] / 2, r1 - g64[1] / 2, r1 + g64[2] / 2};
    for (uint32_t p = 0; p < 3; p++) {
        t.pos8[p] = static_cast<uint32_t>((c64[p] + 4) >> 3);
    }
    return t;
}
}  // namespace sampler

#endif

void kernel_main() {
#if defined(PROFILE_KERNEL)
    volatile tt_l1_ptr uint32_t* done = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(kCtrlAddr);
    volatile tt_l1_ptr uint32_t* hb = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(kCtrlAddr + 4);
    volatile tt_l1_ptr uint32_t* go = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(kCtrlAddr + 8);
    volatile tt_l1_ptr uint32_t* stop = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(kCtrlAddr + 64);
    *done = 0;
    *hb = 0;
    *go = 0;
    *stop = 0;

    const sampler::Table table = sampler::calibrate(go, stop, hb);
    // Sampling waits for the host's go word, written once the receiver's ingest threads are up.
    while (*go == 0u && *stop == 0u) {
        (*hb)++;
        invalidate_l1_cache();
    }

    // Both clocks' high words are carried from the low words' wraps, which at this rate no sweep can hide (86 s and
    // 3.2 s periods).
    const eth_ptp::Instant start = eth_ptp::read_instant();
    Carry carry{
        static_cast<uint32_t>(start.refclk >> 32), start.wall_hi, static_cast<uint32_t>(start.refclk), start.wall_lo};
    model::Model m;
    model::begin_acquire(m, start.refclk);
    uint32_t iter = 0, walk = start.wall_lo | 1u;
    while (true) {
        sampler::Catch c;
        if (sampler::take<sampler::pass>(walk, c, table)) {
            sample(m, carry, c.r, c.w, c.pos8);
        }
        if ((++iter & 255u) != 0u) {
            continue;
        }
        (*hb)++;
        invalidate_l1_cache();
        // Teardown: the relay stop word, written by the host at quiesce. The streaming control layout has no
        // terminate slot; this word is the only stop signal a resident eth kernel gets. The drainer, stopped after
        // this core, ships what the ring still holds.
        if (*stop != 0u) {
            break;
        }
    }
    *done = kp::kRelayDoneWord;
#endif
}
