// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Resident idle-eth clock tracker: the chip's AICLK wall clock modelled against the eth tile's free-running 50 MHz
// counter, as instants the host places records by.
//
// Runs on one idle ethernet core per chip for the life of the profiling session, and does nothing but sample: the
// bracket loop below and the local clock model. Its instants go to a ring in its L1 (kSyncRingAddr, tail in its
// control block) that the drainer on a second idle core (eth_clock_drainer.cpp) ships to the host over the NoC;
// that core also ships this one's firmware markers and the active eth cores' rings. Anything this loop did besides
// sampling was a hole in the samples that a frequency glide bent across.

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
// A sample brackets one wall-clock read between two refclk reads and counts only when the refclk changed inside
// the bracket. The three loads pipeline, so the bracket is a couple of AICLK cycles wide (measured: the counter
// changes inside it on 2 % of iterations), and the refclk the ERISC reads advances in steps of 4 ticks, 80 ns
// apart: a counted sample is the wall time of one such update to within a cycle, with no quantisation noise, and
// the update is uniform over the bracket, which is symmetric about the wall read, so there is no read-latency
// term either. About one update in fifty is caught, one every ~5 us; a fresh segment's intercept is at a quarter
// of a tick after 16 of them and keeps improving as 1/sqrt(n), and the host never places a record on a line drawn
// through a handful of points.
//
// A step shows as kConfirm consecutive samples off the line by more than kOffTicks. The old segment's last on-line
// sample closes it; the new slope is locked once the newest kWinTicks of samples lie on one line, which rejects the
// PLL's glide. Nothing here is a typed-in correction: the bracket cancels the read latency by construction, k8 is
// integer arithmetic, and the intercept is a mean.

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
// A sample's wall read sits midway between refclk reads ~9 cycles apart, so its pairing error is under 5 cycles;
// the thresholds below sit above that. Off the line by kOffTicks is off: a 1/8 step gets there in ~1 us.
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
// While no line holds (a step's acquisition, which a PLL glide of tens of ms keeps failing), the bracketed samples
// go out as points, k8 0, as few as keep the host's chord through them within kRawEpsTicks of every sample: the map
// then bends through the glide instead of bridging it with one chord (whose error is the frequency change times the
// seam length over eight -- hundreds of us for a 7% ramp over 30 ms). A slow ramp costs a point every ~80 us, a
// fast one a point a sample while it lasts.
constexpr int64_t kRawEpsTicks = 1;       // 0.7 ns: the means it applies to carry ~1 cycle of noise themselves
constexpr uint32_t kCountMax = 1u << 22;  // the residue sum stops here, and it stays in 32 bits
// A point lies this far behind the newest sample (164 us): a departure is confirmed within ~25 us of samples, and
// a sweep can hold sampling for ~100 us, so no point ever lands on a step the model has not yet seen.
constexpr uint32_t kPointLagTicks = 8192;
constexpr uint64_t kReanchorTicks = 1ull << 30;  // multiple of 8: the anchor moves along the exact line
static_assert((kRingSamples & (kRingSamples - 1)) == 0 && kReanchorTicks % 8 == 0);

struct Raw {
    uint32_t r_lo, r_hi, w_lo, w_hi;
};
inline volatile tt_l1_ptr Raw* ring() { return reinterpret_cast<volatile tt_l1_ptr Raw*>(kRingAddr); }
inline uint64_t raw_r(const volatile tt_l1_ptr Raw& e) { return (static_cast<uint64_t>(e.r_hi) << 32) | e.r_lo; }
inline uint64_t raw_w(const volatile tt_l1_ptr Raw& e) { return (static_cast<uint64_t>(e.w_hi) << 32) | e.w_lo; }

struct Model {
    uint32_t k8 = 0;          // wall ticks per refclk tick in eighths; 0 while a slope is being acquired
    uint64_t ra = 0, wa = 0;  // anchor: the line passes 8*wa + c8 eighths at ra
    int32_t sum = 0;          // residues 8*(w - wa) - k8*(r - ra) summed over the counted samples
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
    // Acquisition's raw points: the anchor (the last point sent), the previous sample, and the cone of chord slopes
    // from the anchor that keep every sample since within kRawEpsTicks, as fractions n/d with d > 0.
    uint64_t raw_r0 = 0, raw_w0 = 0, raw_rp = 0, raw_wp = 0;
    int32_t lo_n = 0, lo_d = 1, hi_n = 0, hi_d = 1;  // 32-bit: a seam's spans stay under 2^31; the products are 64
    bool cone = false;
    // The last kWin samples, in this RISC's local memory (the L1 ring above is for the lock test; reading it back
    // costs an L1 miss per word). A raw instant is the mean of the last kRawMean of them, taken every kRawStride:
    // one sample's pairing error is wider than kRawEpsTicks and would break the cone at every one, the mean over
    // ~1 us bends by under a cycle in any glide seen, and the stride keeps the instants ~0.6 us apart (one a sample
    // doubled the records in a seam and the tail with them). A mean must not span a hole in the samples (it would
    // average across the glide's bend): win_from is the first sample after the last one.
    static constexpr uint32_t kWin = 16;
    uint64_t win_r[kWin] = {}, win_w[kWin] = {};
    uint32_t win_n = 0;     // samples pushed; the newest is win_r[(win_n - 1) & (kWin - 1)]
    uint32_t win_from = 0;  // the first sample a mean may include
    uint32_t onset = 0;     // means still to send as they form: a seam's first, where the glide is steepest
};

inline __attribute__((always_inline)) void ring_push(Model& m, uint64_t r, uint64_t w) {
    volatile tt_l1_ptr Raw& e = ring()[m.ring_n & (kRingSamples - 1)];
    e.r_lo = static_cast<uint32_t>(r);
    e.r_hi = static_cast<uint32_t>(r >> 32);
    e.w_lo = static_cast<uint32_t>(w);
    e.w_hi = static_cast<uint32_t>(w >> 32);
    m.ring_n++;
}

// The line's wall at refclk r, in eighths of a tick.
inline int64_t line_w8(const Model& m, uint64_t r) {
    return 8 * static_cast<int64_t>(m.wa) + m.c8 + static_cast<int64_t>(m.k8) * static_cast<int64_t>(r - m.ra);
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

// A sample as a point, in eighths of a tick: k8 0 tells the host it is an instant of the wall clock, on no line.
inline void write_raw_point(Model& m, uint64_t r, uint64_t w) {
    m.r_last_point = r;
    sync::write(kp::kSyncKindLocal, kp::kSyncLocalPoint, 0, r, w << 3, 8u * static_cast<uint32_t>(kRawEpsTicks));
}
// One acquisition sample: the seam's first becomes a point and the anchor; then the cone narrows with each sample,
// and the sample that empties it makes the previous one the next point and anchor.
__attribute__((noinline)) void raw_feed(Model& m, uint64_t r, uint64_t w) {
    if (m.raw_r0 == 0) {
        write_raw_point(m, r, w);
        m.raw_r0 = m.raw_rp = r;
        m.raw_w0 = m.raw_wp = w;
        m.cone = false;
        return;
    }
    const int32_t dr = static_cast<int32_t>(r - m.raw_r0), dw = static_cast<int32_t>(w - m.raw_w0);
    const int32_t lo = dw - kRawEpsTicks, hi = dw + kRawEpsTicks;
    if (!m.cone) {
        m.lo_n = lo;
        m.hi_n = hi;
        m.lo_d = m.hi_d = dr;
        m.cone = true;
    } else {
        if (static_cast<int64_t>(lo) * m.lo_d > static_cast<int64_t>(m.lo_n) * dr) {
            m.lo_n = lo;
            m.lo_d = dr;
        }
        if (static_cast<int64_t>(hi) * m.hi_d < static_cast<int64_t>(m.hi_n) * dr) {
            m.hi_n = hi;
            m.hi_d = dr;
        }
        if (static_cast<int64_t>(m.lo_n) * m.hi_d > static_cast<int64_t>(m.hi_n) * m.lo_d) {
            write_raw_point(m, m.raw_rp, m.raw_wp);
            m.raw_r0 = m.raw_rp;
            m.raw_w0 = m.raw_wp;
            const int32_t dr2 = static_cast<int32_t>(r - m.raw_r0), dw2 = static_cast<int32_t>(w - m.raw_w0);
            m.lo_n = dw2 - kRawEpsTicks;
            m.hi_n = dw2 + kRawEpsTicks;
            m.lo_d = m.hi_d = dr2;
        }
    }
    m.raw_rp = r;
    m.raw_wp = w;
}
inline void begin_acquire(Model& m, uint64_t r_from) {
    m.k8 = 0;
    m.n = 0;
    m.sum = 0;
    m.r_acq0 = r_from;
    m.acq_count = 0;
    m.raw_r0 = 0;
    m.cone = false;
    m.win_i = m.ring_n;
}
constexpr uint32_t kRawMean = 4;
constexpr uint32_t kRawStride = 2;
constexpr uint32_t kTightCycles = 8;      // the wall reads either side of a bracket four back-to-back loads apart
constexpr uint32_t kGroupGapTicks = 125;  // 2.5 us
static_assert(kRawMean == 4 && (kRawMean % kRawStride) == 0);
inline __attribute__((always_inline)) void win_push(Model& m, uint64_t r, uint64_t w) {
    if (m.win_n != 0 && r - m.win_r[(m.win_n - 1) & (Model::kWin - 1)] > kGroupGapTicks) {
        m.win_from = m.win_n;
    }
    m.win_r[m.win_n & (Model::kWin - 1)] = r;
    m.win_w[m.win_n & (Model::kWin - 1)] = w;
    m.win_n++;
}
// The mean of the kRawMean samples ending at (exclusive) window index `end`.
inline void win_mean(const Model& m, uint32_t end, uint64_t& mr, uint64_t& mw) {
    uint64_t sr = 0, sw = 0;
    for (uint32_t i = end - kRawMean; i < end; i++) {
        sr += m.win_r[i & (Model::kWin - 1)];
        sw += m.win_w[i & (Model::kWin - 1)];
    }
    mr = sr >> 2;
    mw = sw >> 2;
}
inline void send_onset_mean(Model& m, uint64_t mr, uint64_t mw) {
    write_raw_point(m, mr, mw);
    m.raw_r0 = m.raw_rp = mr;
    m.raw_w0 = m.raw_wp = mw;
    m.cone = false;
}
__attribute__((noinline)) void raw_sample(Model& m) {
    if (m.win_n - m.win_from < kRawMean || (m.onset == 0 && (m.win_n % kRawStride) != 0)) {
        return;
    }
    uint64_t mr, mw;
    win_mean(m, m.win_n, mr, mw);
    if (m.onset == 0) {
        raw_feed(m, mr, mw);
        return;
    }
    // The cone would hold each of these back until the next one arrived, and the stride would halve them; here
    // every 0.3 us matters.
    m.onset--;
    send_onset_mean(m, mr, mw);
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
    const uint64_t r_old = raw_r(oldest), w_old = raw_w(oldest);
    const uint32_t dr = static_cast<uint32_t>(raw_r(newest) - r_old);
    const uint32_t dw = static_cast<uint32_t>(raw_w(newest) - w_old);
    if (dr < kWinTicks / 2) {
        return false;
    }
    const uint32_t k8 = (8u * dw + dr / 2u) / dr;
    int32_t lo = 0, hi = 0, sum = 0;
    uint32_t n = 0;
    for (uint32_t i = m.win_i; i < m.ring_n; i += kLockStride, n++) {
        const volatile tt_l1_ptr Raw& e = ring()[i & (kRingSamples - 1)];
        const int32_t res = 8 * static_cast<int32_t>(static_cast<uint32_t>(raw_w(e) - w_old)) -
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
    m.wa = w_old;
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
constexpr uint32_t kOnsetMeans = 14;  // live means sent as they form after the replay's, one a sample, ~4 us
__attribute__((noinline, cold)) void step(Model& m) {
    constexpr uint32_t back = kConfirm + kPreSamples;
    static_assert(back % kRawMean == 0);
    static_assert(back + 1 <= Model::kWin);
    write_point(m, m.win_r[(m.win_n - back - 1) & (Model::kWin - 1)], kp::kSyncLocalClose);
    begin_acquire(m, m.r_dep);
    // The seam's first instants, the means over those samples, from local memory: every cycle here is a sample
    // not taken at the onset, the glide's steepest microseconds.
    for (uint32_t end = m.win_n - back + kRawMean; end <= m.win_n; end += kRawStride) {
        if (end - kRawMean < m.win_from) {
            continue;
        }
        uint64_t mr, mw;
        win_mean(m, end, mr, mw);
        send_onset_mean(m, mr, mw);
    }
    m.onset = kOnsetMeans;
}

inline __attribute__((always_inline)) void feed(Model& m, uint64_t r, uint64_t w) {
    ring_push(m, r, w);
    win_push(m, r, w);
    if (m.k8 == 0) {
        advance_window(m, r);
        if (r - m.r_acq0 >= kAcqTicks && ++m.acq_count >= kAcqTestEvery) {
            m.acq_count = 0;
            if (try_lock(m, r)) {
                return;
            }
        }
        raw_sample(m);
        return;
    }
    const int64_t e = 8 * static_cast<int64_t>(w - m.wa) - static_cast<int64_t>(m.k8) * static_cast<int64_t>(r - m.ra);
    const int64_t d = e - m.c8;
    if (d > 8 * kOffTicks || d < -8 * kOffTicks) {
        if (m.off++ == 0) {
            m.r_dep = r;
        }
        if (m.off >= kConfirm) {
            step(m);
        }
        return;
    }
    m.off = 0;
    m.r_last_on = r;
    const uint32_t ad = static_cast<uint32_t>(d < 0 ? -d : d);
    m.max_d8 = ad > m.max_d8 ? ad : m.max_d8;
    if (m.n >= (1u << kEmaShift)) {
        m.ema += static_cast<int32_t>(e) - (m.ema >> kEmaShift);
        m.c8 = (m.ema + (1 << (kEmaShift - 1))) >> kEmaShift;
    }
    if (m.n < kCountMax) {
        m.sum += static_cast<int32_t>(e);
        m.n++;
        if ((m.n & (m.n - 1)) == 0) {
            if (m.n <= (1u << kEmaShift)) {
                m.c8 = m.sum / static_cast<int32_t>(m.n);
                m.ema = m.c8 << kEmaShift;
            }
            if (m.n >= kFirstPointN && m.n <= kLastDoublingN) {
                write_point(m, r - kPointLagTicks, kp::kSyncLocalPoint);
            }
        }
    }
    if (r - m.r_last_point >= kPointTicks) {
        write_point(m, r - kPointLagTicks, kp::kSyncLocalPoint);
    }
    if (r - m.ra >= kReanchorTicks) {
        m.ra += kReanchorTicks;
        m.wa += static_cast<uint64_t>(m.k8) * (kReanchorTicks / 8u);
    }
}
}  // namespace model

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

    // Sampling waits for the host's go word, written once the receiver's ingest threads are up.
    while (*go == 0u && *stop == 0u) {
        (*hb)++;
        invalidate_l1_cache();
    }

    // Each iteration is a group of eight brackets, r0 w1 r1 w2 r2 ... w8 r8: seventeen loads issued back to back with
    // nothing between them (stores or compares in the group widen it), one ~35-cycle window in which a refclk advance
    // is caught by the bracket it fell in -- at most one fits, advances being 64-108 cycles apart -- with the wall
    // read midway between the two refclk reads that saw it; the wall reads either side of that bracket time it,
    // and only a tight one (loads that issued back to back) is kept. About one iteration in three samples. Each
    // iteration is padded by 0-3 pseudo-random cycles so the advance's phase against the loop walks instead of
    // locking. Both clocks' high words are carried from the low words' wraps, which at this rate no sweep can hide
    // (86 s and 3.2 s periods).
    const eth_ptp::Instant start = eth_ptp::read_instant();
    uint32_t r_hi = static_cast<uint32_t>(start.refclk >> 32), prev_r_lo = static_cast<uint32_t>(start.refclk);
    uint32_t w_hi = start.wall_hi, prev_w_lo = start.wall_lo;
    model::Model m;
    model::begin_acquire(m, start.refclk);
    uint32_t iter = 0, walk = start.wall_lo | 1u;
    while (true) {
        walk = walk * 1103515245u + 12345u;
        for (uint32_t d = walk >> 30; d != 0; d--) {
            asm volatile("nop");
        }
        // All loads ahead of every compare: a compare waiting on an early load would hold the later loads.
        uint32_t w0 = eth_ptp::rd(eth_ptp::kWallClockLo), r0 = eth_ptp::rd(eth_ptp::kPtpCfrLo);
        uint32_t w1 = eth_ptp::rd(eth_ptp::kWallClockLo), r1 = eth_ptp::rd(eth_ptp::kPtpCfrLo);
        uint32_t w2 = eth_ptp::rd(eth_ptp::kWallClockLo), r2 = eth_ptp::rd(eth_ptp::kPtpCfrLo);
        uint32_t w3 = eth_ptp::rd(eth_ptp::kWallClockLo), r3 = eth_ptp::rd(eth_ptp::kPtpCfrLo);
        uint32_t w4 = eth_ptp::rd(eth_ptp::kWallClockLo), r4 = eth_ptp::rd(eth_ptp::kPtpCfrLo);
        uint32_t w5 = eth_ptp::rd(eth_ptp::kWallClockLo), r5 = eth_ptp::rd(eth_ptp::kPtpCfrLo);
        uint32_t w6 = eth_ptp::rd(eth_ptp::kWallClockLo), r6 = eth_ptp::rd(eth_ptp::kPtpCfrLo);
        uint32_t w7 = eth_ptp::rd(eth_ptp::kWallClockLo), r7 = eth_ptp::rd(eth_ptp::kPtpCfrLo);
        uint32_t w8 = eth_ptp::rd(eth_ptp::kWallClockLo), r8 = eth_ptp::rd(eth_ptp::kPtpCfrLo);
        asm volatile("" : "+r"(w0), "+r"(r0), "+r"(w1), "+r"(r1), "+r"(w2), "+r"(r2), "+r"(w3), "+r"(r3), "+r"(w4));
        asm volatile("" : "+r"(r4), "+r"(w5), "+r"(r5), "+r"(w6), "+r"(r6), "+r"(w7), "+r"(r7), "+r"(w8), "+r"(r8));
        uint32_t rb_lo = r8, w_lo = w8;
        // The wall reads either side of the bracket that caught the advance time the bracket itself: the loads
        // issue back to back while the load queue takes them (the group's first few) and at its drain rate after,
        // so brackets range from a cycle to ~20 wide, and the width is the pairing error a sample carries.
        uint32_t width = 2u * (w8 - w7);
        if (r8 == r0) {
            goto housekeeping;
        }
        if (r1 != r0) {
            rb_lo = r1, w_lo = w1, width = w2 - w0;
        } else if (r2 != r1) {
            rb_lo = r2, w_lo = w2, width = w3 - w1;
        } else if (r3 != r2) {
            rb_lo = r3, w_lo = w3, width = w4 - w2;
        } else if (r4 != r3) {
            rb_lo = r4, w_lo = w4, width = w5 - w3;
        } else if (r5 != r4) {
            rb_lo = r5, w_lo = w5, width = w6 - w4;
        } else if (r6 != r5) {
            rb_lo = r6, w_lo = w6, width = w7 - w5;
        } else if (r7 != r6) {
            rb_lo = r7, w_lo = w7, width = w8 - w6;
        }
        // Only the tight brackets are samples: a wide one pairs the wall read with the advance at an offset of its
        // own, and mixing them biases whatever averages over them.
        if (width > model::kTightCycles) {
            goto housekeeping;
        }
        r_hi += rb_lo < prev_r_lo;
        w_hi += w_lo < prev_w_lo;
        prev_r_lo = rb_lo;
        prev_w_lo = w_lo;
        model::feed(m, (static_cast<uint64_t>(r_hi) << 32) | rb_lo, (static_cast<uint64_t>(w_hi) << 32) | w_lo);
    housekeeping:
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
