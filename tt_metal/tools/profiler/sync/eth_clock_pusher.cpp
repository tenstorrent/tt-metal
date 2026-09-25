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
#include "internal/tt-1xx/blackhole/arc_pll_regs.h"

constexpr uint32_t kPointTicks = get_compile_time_arg_val(0);  // refclk between the open segment's points (50/us)
constexpr uint32_t kCtrlAddr =
    get_compile_time_arg_val(1);  // done +0, heartbeat +4, go +8, sync tail +12, head +16, stop +64
constexpr uint32_t kPllAddr = get_compile_time_arg_val(2);       // a 64 B-aligned L1 scratch for the PLL reads
constexpr uint32_t kSyncRingAddr = get_compile_time_arg_val(3);  // this core's instants, kSyncRingRecords of them
constexpr uint32_t kArcXy = get_compile_time_arg_val(4);         // the ARC tile, x | y << 16

namespace kp = kernel_profiler;
namespace eth_ptp = tt::tt_metal::eth_ptp;

#if defined(PROFILE_KERNEL)

// AICLK is PLL0's multiple of the crystal the refclk counts (arc_pll_regs.h): the wall clock gains k8/8 ticks per
// refclk tick, k8 = 8 * FBDIV / (REFDIV * postdiv0), for as long as FBDIV holds. This core reads FBDIV from PLL0 over
// the NoC and sends the host POINTS of the wall clock against the refclk, which the host joins with straight lines.
//
// A sample is one advance of the refclk the ERISC reads (it moves in steps of 4 ticks, 80 ns apart) caught between
// two consecutive refclk reads of the sampler below, and placed at the centre of that pair: the wall time of one
// update to within half the pair's width, with no quantisation noise, and unbiased, the update being uniform over
// the pair. About one update in three or four is caught.
//
// A point is the centroid of a window of consecutive samples at one FBDIV, moved along that FBDIV's exact slope to
// the nearest whole refclk tick (the refclk moves in fours, so a centroid's own refclk is fractional). Windows restart
// at every change of FBDIV and double from one sample, closing at kPointTicks: right after a change the points follow
// the clock sample by sample while the PLL settles, and on a steady rate each is a mean of a few thousand samples.
// The centroid of samples on a line is on the line, so the chord between two points is exact wherever the clock is
// straight, and nothing lags the clock.
//
// One read of PLL0 is in flight at a time. A sample joins a window once a read issued after it has returned that
// window's FBDIV. A read that returns another FBDIV closes the window, and the samples taken while the change could
// have happened go out alone.

// This core's instants, in a ring of kSyncRingRecords the drainer reads over the NoC: the tail is published in the
// control block, the drainer writes the count it consumed back beside it. An instant the ring has no room for is
// dropped; this loop never waits for the drainer.
namespace sync {
static uint32_t g_tail = 0;
inline volatile tt_l1_ptr uint32_t* rec(uint32_t i) {
    return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
        kSyncRingAddr + (i % kp::kSyncRingRecords) * kp::kSyncRecordWords * 4u);
}
inline void emit(uint32_t meta, uint32_t round, uint64_t value, uint64_t wall, uint32_t ref_lo, uint32_t ref_hi) {
    const uint32_t head = *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(kCtrlAddr + 16);
    if (g_tail - head >= kp::kSyncRingRecords) {
        return;
    }
    volatile tt_l1_ptr uint32_t* r = rec(g_tail);
    r[kp::SYNC_META] = meta;
    r[kp::SYNC_ROUND] = round;
    r[kp::SYNC_VALUE_LO] = static_cast<uint32_t>(value);
    r[kp::SYNC_VALUE_HI] = static_cast<uint32_t>(value >> 32);
    r[kp::SYNC_WALL_LO] = static_cast<uint32_t>(wall);
    r[kp::SYNC_WALL_HI] = static_cast<uint32_t>(wall >> 32);
    r[kp::SYNC_REF_LO] = ref_lo;
    r[kp::SYNC_REF_HI] = ref_hi;
    asm volatile("fence" ::: "memory");
    *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(kCtrlAddr + 12) = ++g_tail;
}
}  // namespace sync

// Points go out kSyncLocalPoints to a LOCAL record (hostdev/streaming_profiler_common.h): the first whole, the rest as
// a refclk step and an offset from the record's slope; a point whose step or offset does not fit starts the next one.
namespace pack {
static uint32_t g_n = 0, g_meta = 0, g_round = 0, g_d[kp::kSyncLocalPoints - 1] = {};
static uint64_t g_r0 = 0, g_w0 = 0;
inline void flush() {
    if (g_n != 0) {
        sync::emit(g_meta | g_n, g_round, g_r0, g_w0, g_d[0], g_d[1]);
        g_n = 0;
    }
}
__attribute__((noinline)) void add(uint64_t r, uint64_t w8, uint32_t k8, bool close, uint32_t slope) {
    if (g_n != 0) {
        if (r - g_r0 <= 0xFFFFu) {
            const uint32_t dr = static_cast<uint32_t>(r - g_r0);
            const int32_t off =
                static_cast<int32_t>(static_cast<uint32_t>(w8) - static_cast<uint32_t>(g_w0) - (g_round >> 24) * dr);
            if (off >= -32768 && off <= 32767) {
                g_d[g_n - 1] = dr | (static_cast<uint32_t>(off) << 16);
                g_round |= k8 << (8 * g_n);
                g_meta |= static_cast<uint32_t>(close) << (2 + g_n);
                if (++g_n == kp::kSyncLocalPoints) {
                    flush();
                }
                return;
            }
        }
        flush();
    }
    g_r0 = r;
    g_w0 = w8;
    g_round = k8 | (slope << 24);
    g_meta = (kp::kSyncKindLocal << 8) | (static_cast<uint32_t>(close) << 2);
    g_d[0] = g_d[1] = 0;
    g_n = 1;
}
}  // namespace pack

namespace model {
struct Model {
    uint32_t cntl1 = 0;  // PLL0 CNTL_1 of the open window
    uint32_t den = 8;    // REFDIV * postdiv0
    uint32_t k8 = 0;     // wall ticks per refclk tick in eighths
    // The open window: its first sample, the sums past it of the samples' refclk and of their residues against the
    // slope, (w8 - w0) - k8 * (r - r0), its count, and the count that closes it.
    uint64_t r0 = 0, w0 = 0;
    uint32_t sr = 0;
    int32_t se = 0;
    uint32_t cnt = 0, size = 1;
    // The samples not yet in a window or sent, in this RISC's local memory: a read's round trip is ~2 samples, so
    // kWin holds several reads' worth.
    static constexpr uint32_t kWin = 32;
    uint64_t win_r[kWin] = {}, win_w8[kWin] = {};
    uint32_t win_n = 0;  // samples pushed; sample i is at i & (kWin - 1)
    uint32_t done = 0;   // samples before this one are in a window or sent
};

inline __attribute__((always_inline)) void win_push(Model& m, uint64_t r, uint64_t w8) {
    if (m.win_n - m.done >= Model::kWin) {
        m.done = m.win_n - Model::kWin + 1;
    }
    m.win_r[m.win_n & (Model::kWin - 1)] = r;
    m.win_w8[m.win_n & (Model::kWin - 1)] = w8;
    m.win_n++;
}

// The open window's centroid as a point, tagged with k8 and the count behind it.
__attribute__((noinline)) void close_window(Model& m, uint32_t role) {
    const uint32_t dr = (m.sr + m.cnt / 2u) / m.cnt;
    const int32_t half = static_cast<int32_t>(m.cnt / 2u);
    const int32_t e = (m.se + (m.se < 0 ? -half : half)) / static_cast<int32_t>(m.cnt);
    pack::add(
        m.r0 + dr,
        m.w0 + static_cast<uint64_t>(m.k8) * dr + static_cast<int64_t>(e),
        m.k8,
        role == kp::kSyncLocalClose,
        m.k8);
    m.cnt = 0;
    m.size = m.size < (1u << 23) ? m.size * 2u : m.size;
}

inline __attribute__((always_inline)) void add(Model& m, uint64_t r, uint64_t w8) {
    if (m.cnt == 0) {
        m.r0 = r;
        m.w0 = w8;
        m.sr = 0;
        m.se = 0;
    }
    const uint32_t dr = static_cast<uint32_t>(r - m.r0);
    m.sr += dr;
    m.se += static_cast<int32_t>(static_cast<uint32_t>(w8 - m.w0) - m.k8 * dr);
    if (++m.cnt == m.size || dr >= kPointTicks) {
        close_window(m, kp::kSyncLocalPoint);
    }
}

// A sample alone as a point: k8 0 tells the host it is a single reading.
inline void write_sample(const Model& m, uint32_t i) {
    pack::add(m.win_r[i & (Model::kWin - 1)], m.win_w8[i & (Model::kWin - 1)], 0, false, m.k8);
}

// A completed read of PLL0 CNTL_1: `issued` and `completed` are the sample counts when it was issued and when its
// data was seen, so samples before `issued` were taken before the register was read, and samples from `completed` on
// after it.
__attribute__((noinline)) void on_read(Model& m, uint32_t cntl1, uint32_t issued, uint32_t completed) {
    if (cntl1 == m.cntl1) {
        for (; static_cast<int32_t>(issued - m.done) > 0; m.done++) {
            add(m, m.win_r[m.done & (Model::kWin - 1)], m.win_w8[m.done & (Model::kWin - 1)]);
        }
        return;
    }
    if (m.cnt != 0) {
        close_window(m, kp::kSyncLocalClose);
    }
    for (uint32_t i = m.done; static_cast<int32_t>(completed - i) > 0; i++) {
        write_sample(m, i);
    }
    m.done = completed;
    m.cntl1 = cntl1;
    m.k8 = ((cntl1 >> 16) * 8u) / m.den;
    m.size = 1;
}
}  // namespace model

// The reads of PLL0 CNTL_1 on NoC 0, one in flight. The destination sits at the register's offset modulo 64 B.
namespace pll {
constexpr uint32_t kCntl1 = ARC_PLL0_BASE + ARC_PLL_CNTL_1;
constexpr uint32_t kDst = kPllAddr + (kCntl1 & 63u);
inline uint64_t src(uint32_t reg) { return get_noc_addr(kArcXy & 0xFFFFu, kArcXy >> 16, reg, 0); }
// A blocking read for the setup, which gives up rather than hang the core.
inline bool read(uint32_t reg, uint32_t& v) {
    const uint32_t dst = kPllAddr + (reg & 63u);
    noc_async_read(src(reg), dst, 4, 0);
    for (uint32_t spin = 0; !ncrisc_noc_reads_flushed(0); spin++) {
        if (spin == (1u << 24)) {
            return false;
        }
    }
    v = *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dst);
    return true;
}
struct Poll {
    uint32_t issued = 0;
    bool out = false;
};
inline __attribute__((always_inline)) void step(Poll& p, model::Model& m) {
    if (!p.out) {
        noc_async_read(src(kCntl1), kDst, 4, 0);
        p.issued = m.win_n;
        p.out = true;
    } else if (ncrisc_noc_reads_flushed(0)) {
        p.out = false;
        model::on_read(m, *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(kDst), p.issued, m.win_n);
    }
}
// REFDIV * postdiv0 from PLL0, and the CNTL_1 the first line opens with; false if the divider does not divide eight
// FBDIVs into whole k8 (the line's slope would not be exact) or the ARC tile did not answer.
bool setup(model::Model& m) {
    uint32_t c1, c5, use;
    if (!read(ARC_PLL0_BASE + ARC_PLL_CNTL_1, c1) || !read(ARC_PLL0_BASE + ARC_PLL_CNTL_5, c5) ||
        !read(ARC_PLL0_BASE + ARC_PLL_USE_POSTDIV, use)) {
        return false;
    }
    const uint32_t pd = c5 & 0xFFu;
    const uint32_t post = (use & 1u) == 0u ? 1u : pd <= 16u ? pd + 1u : (pd + 1u) * 2u;
    m.den = (c1 & 0xFFu) * post;
    if (m.den == 0u || 8u % m.den != 0u) {
        return false;
    }
    m.cntl1 = c1;
    m.k8 = ((c1 >> 16) * 8u) / m.den;
    return true;
}
}  // namespace pll

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
    model::win_push(
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

    model::Model m;
    if (!pll::setup(m)) {
        return;
    }
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
    pll::Poll poll;
    uint32_t iter = 0, walk = start.wall_lo | 1u;
    while (true) {
        sampler::Catch c;
        if (sampler::take<sampler::pass>(walk, c, table)) {
            sample(m, carry, c.r, c.w, c.pos8);
        }
        pll::step(poll, m);
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
    if (m.cnt != 0) {
        model::close_window(m, kp::kSyncLocalPoint);
    }
    pack::flush();
    *done = kp::kRelayDoneWord;
#endif
}
