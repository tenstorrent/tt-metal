// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Resident idle-eth clock tracker and sync pusher, which drains its chip's active eth cores too.
//
// Runs on one idle ethernet core per chip for the life of the profiling session. It keeps the chip's AICLK wall
// clock modelled against the eth tile's free-running 50 MHz counter (the local clock model below). Points of that
// model, and the link ends' stamp averages read from the ring at the end of each active eth core's link L1, go to
// the host as sync frames over the SYNC socket (hostdev/streaming_profiler_common.h, kSyncRecordWords), which the
// sync engine reads itself. Between samples it also acts as a mini-relay for the chip's ACTIVE eth cores: those run
// the fabric router and can spend no cycles on egress, so this core NoC-reads their control vectors and rings, packs
// a frame stamped with THEIR coordinate, pushes it on the PROFILER socket, and writes their heads back -- the decoder
// resolves a frame's core by its XY, so that socket carries every core this pusher serves exactly as one relay
// socket carries its cores.
//
// WHY THE ETH CORE PUSHES: the DRAM relay is unrolled for exactly five Tensix RISCs per core and an eth core has
// two. Putting eth in the relay roster meant a heterogeneous frame format through the relay, the frame sizing and
// the decoder lane indexing. Instead the host enumerates every eth core here as a standard 5-lane core whose
// sibling lanes are simply always empty (the decoder skips a lane whose extent is 0 exactly as it does an idle
// TRISC), and the relay never sees an eth core.
//
// WIRE: identical to the relay (hostdev/streaming_profiler_common.h): w0 | payload_words | 5 heads | xy |
// control-vector words 16..31 | per-lane runs preceded by spsc_span_pack_pad. The same three run shapes as the
// relay (flat, near-full wrap image, two-piece wrap), chosen by the SHARED spsc_span_wrap_image predicate, so the
// decoder linearises exactly what was shipped.
//
// EGRESS ORDERING: every data write is flushed before bytes_sent is announced (socket_notify_receiver), the rule
// the relay enforces: the PCIe tile keeps no order between packets, and a 4 B notify has been seen landing ahead
// of the data it announces.

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/socket_api.h"
#include "tools/profiler/kernel_profiler.hpp"
#include "internal/ethernet/eth_ptp_clock.hpp"

constexpr uint32_t kPointTicks = get_compile_time_arg_val(0);        // refclk between the open segment's points (50/us)
constexpr uint32_t kSocketConfigAddr = get_compile_time_arg_val(1);  // D2HSocket config in this core's L1
constexpr uint32_t kStageAddr = get_compile_time_arg_val(2);         // one frame slot in this core's L1
constexpr uint32_t kCtrlAddr = get_compile_time_arg_val(3);          // done +0, heartbeat +4, go +8, stop +64
constexpr uint32_t kMyXy = get_compile_time_arg_val(4);              // y << 16 | x, this core's frame identity
// Scratch for a linked core: its control vector (256 B) at +0, then its two ring images (2 KiB each) -- 4608 B,
// separate from the frame slot, which is sized for the packed payload alone.
constexpr uint32_t kScratchAddr = get_compile_time_arg_val(5);
constexpr uint32_t kTileTableAddr = get_compile_time_arg_val(6);  // hostdev EthTileTable
constexpr uint32_t kMeasureOnly = get_compile_time_arg_val(7);    // write the tile table and exit
constexpr uint32_t kRingAddr = get_compile_time_arg_val(8);       // model::kRingSamples raw samples, for transitions
constexpr uint32_t kSyncCfgAddr = get_compile_time_arg_val(9);    // the sync socket's D2HSocket config
constexpr uint32_t kSyncRingAddr = get_compile_time_arg_val(10);  // this core's points, kSyncRingRecords of them
constexpr uint32_t kLinkRingAddr =
    get_compile_time_arg_val(11);  // the link ends' sync ring, one address on every active eth core; 0 = none

namespace kp = kernel_profiler;

// Frame geometry: the standard 5-slot core the host enumerated every eth core as.
constexpr uint32_t kNumRisc = kp::PROFILER_SPSC_TENSIX_RISC;
constexpr uint32_t kNumEthRisc = 2;  // DM0, DM1: the lanes that physically exist on an eth core
static_assert(kNumEthRisc <= kNumRisc, "eth lanes must fit the standard slot count");
constexpr uint32_t kRingWords = kp::PROFILER_L1_VECTOR_SIZE;
constexpr uint32_t kRingBytes = kRingWords * 4u;
constexpr uint32_t kCvWords = kp::PROFILER_L1_CONTROL_VECTOR_SIZE;
constexpr uint32_t kPrefix = kp::SPSC_SPAN_PREFIX_WORDS;
constexpr uint32_t kWireCtrl = kp::SPSC_SPAN_WIRE_CTRL_WORDS;
constexpr uint32_t kPageWords = kp::SPSC_SPAN_PAGE_WORDS;
constexpr uint32_t kPageBytes = kPageWords * 4u;
constexpr uint32_t kLenWord = 1;
// Ship once the live lane holds this many words, or after this much refclk regardless, so a trickle still reaches
// the host within ~1 ms. The linked cores are swept on the same cadence.
constexpr uint32_t kShipWords = kRingWords / 4u;
constexpr uint32_t kSweepTicks = 50000;  // 1 ms
constexpr uint32_t kMaxLinked = 16;      // BH has 14 eth cores

namespace eth_ptp = tt::tt_metal::eth_ptp;

#if defined(PROFILE_KERNEL)

// AICLK is a PLL multiple of the crystal the refclk counts: k8/8 wall ticks per refclk tick, k8 an integer, exact
// between DVFS steps (every run longer than 50 ms measured sits on its multiple to <0.005 ppm). So over one rate the
// wall clock is a line whose slope is known once k8 is, and whose only free parameter is the phase of the refclk's
// increment against the wall ticks -- the intercept. This core measures both, and the host receives POINTS of the
// line, never samples to fit: a point is (refclk r, the line's wall at r), tagged with k8 and the sample count
// behind it; a new k8, or a CLOSE point, starts the next segment, and two consecutive segments meet where their
// lines cross.
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

// This core's points, in a ring of kSyncRingRecords the sweep drains into sync frames; a point the ring has no room
// for is skipped, never waited for: the next one carries the same line.
namespace sync {
static uint32_t g_head = 0, g_tail = 0;
inline volatile tt_l1_ptr uint32_t* rec(uint32_t i) {
    return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
        kSyncRingAddr + (i % kp::kSyncRingRecords) * kp::kSyncRecordWords * 4u);
}
inline void write(uint32_t kind, uint32_t role, uint32_t round, uint64_t value, uint64_t wall) {
    if (g_tail - g_head == kp::kSyncRingRecords) {
        return;
    }
    volatile tt_l1_ptr uint32_t* r = rec(g_tail);
    r[kp::SYNC_META] = (kind << 8) | role;
    r[kp::SYNC_ROUND] = round;
    r[kp::SYNC_VALUE_LO] = static_cast<uint32_t>(value);
    r[kp::SYNC_VALUE_HI] = static_cast<uint32_t>(value >> 32);
    r[kp::SYNC_WALL_LO] = static_cast<uint32_t>(wall);
    r[kp::SYNC_WALL_HI] = static_cast<uint32_t>(wall >> 32);
    g_tail++;
}
}  // namespace sync

namespace model {
constexpr uint32_t kRingSamples = 128;  // raw samples kept, ~5 us apart: ~600 us deep
constexpr uint32_t kConfirm = 4;           // consecutive off-line samples that make a step
constexpr int64_t kOffTicks = 16;          // off the line by this much is off: a 1/8 step gets there in 2.6 us
constexpr uint32_t kAcqTicks = 4096;       // refclk after a departure before the first lock test (82 us)
constexpr uint32_t kWinTicks = 4096;       // the window that must lie on one line to lock its slope (~16 samples)
constexpr uint32_t kWinSpreadTicks = 8;    // one line's samples spread less than this; a glide inside bends more
constexpr uint32_t kAcqTestEvery = 8;      // samples between lock tests
constexpr uint32_t kFirstPointN = 16;      // samples behind a segment's first point: a quarter of a tick
constexpr uint32_t kLastDoublingN = 4096;  // points at every doubling of the count up to here, then every kPointTicks
constexpr uint32_t kCountMax = 1u << 22;   // the residue sum stops here, and it stays in 32 bits
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
    int32_t c8 = 0;          // sum / n, refreshed at each doubling of n
    uint64_t r_last_on = 0;  // newest sample on the line
    uint32_t off = 0;        // consecutive samples off the line
    uint64_t r_dep = 0;      // the first of them
    uint64_t r_acq0 = 0;     // where the acquisition began
    uint32_t acq_count = 0;
    uint64_t r_lock = 0;  // where the segment's line begins: the oldest sample of the window that locked it
    uint64_t r_last_point = 0;
    uint32_t ring_n = 0;  // ring entries written; the newest is ring()[(ring_n - 1) & (kRingSamples - 1)]
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

// A point of the line at refclk r, never before the segment's own start: the host keeps segments disjoint in refclk.
inline void write_point(Model& m, uint64_t r, uint32_t role) {
    r = r > m.r_lock ? r : m.r_lock;
    m.r_last_point = r;
    const uint64_t w = static_cast<uint64_t>((line_w8(m, r) + 4) >> 3);
    const uint32_t n = m.n < kCountMax ? m.n : kCountMax - 1;
    sync::write(kp::kSyncKindLocal, role, m.k8 | (n << 8), r, w);
}

inline void begin_acquire(Model& m, uint64_t r_from) {
    m.k8 = 0;
    m.n = 0;
    m.sum = 0;
    m.r_acq0 = r_from;
    m.acq_count = 0;
}

// Locks the slope from the newest kWinTicks of the ring when those samples lie on one line: the slope from the
// window's ends, then every entry's residue against it within one refclk tick of phase. A window across a glide
// bends more than that and the test is retried after kAcqTestEvery samples.
__attribute__((noinline)) bool try_lock(Model& m, uint64_t r_now) {
    const uint32_t avail = m.ring_n < kRingSamples ? m.ring_n : kRingSamples;
    uint32_t cnt = 0;
    for (uint32_t i = 1; i <= avail; i++) {
        if (r_now - raw_r(ring()[(m.ring_n - i) & (kRingSamples - 1)]) > kWinTicks) {
            break;
        }
        cnt = i;
    }
    if (cnt < 8) {
        return false;
    }
    const volatile tt_l1_ptr Raw& oldest = ring()[(m.ring_n - cnt) & (kRingSamples - 1)];
    const volatile tt_l1_ptr Raw& newest = ring()[(m.ring_n - 1) & (kRingSamples - 1)];
    const uint64_t r_old = raw_r(oldest), w_old = raw_w(oldest);
    const uint32_t dr = static_cast<uint32_t>(raw_r(newest) - r_old);
    const uint32_t dw = static_cast<uint32_t>(raw_w(newest) - w_old);
    if (dr < kWinTicks / 2) {
        return false;
    }
    const uint32_t k8 = (8u * dw + dr / 2u) / dr;
    int32_t lo = 0, hi = 0, sum = 0;
    for (uint32_t i = 1; i <= cnt; i++) {
        const volatile tt_l1_ptr Raw& e = ring()[(m.ring_n - i) & (kRingSamples - 1)];
        const int32_t res = 8 * static_cast<int32_t>(static_cast<uint32_t>(raw_w(e) - w_old)) -
                            static_cast<int32_t>(k8 * static_cast<uint32_t>(raw_r(e) - r_old));
        lo = i == 1 || res < lo ? res : lo;
        hi = i == 1 || res > hi ? res : hi;
        sum += res;
    }
    if (static_cast<uint32_t>(hi - lo) > 8u * kWinSpreadTicks) {
        return false;
    }
    m.k8 = k8;
    m.ra = r_old;
    m.wa = w_old;
    m.r_lock = r_old;
    m.n = cnt;
    m.sum = sum;
    m.c8 = sum / static_cast<int32_t>(cnt);
    m.r_last_on = r_now;
    m.off = 0;
    return true;
}

// A confirmed step: the old line closes at its last on-line sample and the slope acquisition restarts from the
// departure.
__attribute__((noinline, cold)) void step(Model& m) {
    write_point(m, m.r_last_on, kp::kSyncLocalClose);
    begin_acquire(m, m.r_dep);
}

inline __attribute__((always_inline)) void feed(Model& m, uint64_t r, uint64_t w) {
    ring_push(m, r, w);
    if (m.k8 == 0) {
        if (r - m.r_acq0 >= kAcqTicks && ++m.acq_count >= kAcqTestEvery) {
            m.acq_count = 0;
            try_lock(m, r);
        }
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
    if (m.n < kCountMax) {
        m.sum += static_cast<int32_t>(e);
        m.n++;
        if ((m.n & (m.n - 1)) == 0) {
            m.c8 = m.sum / static_cast<int32_t>(m.n);
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

// Each Tensix tile keeps its own wall clock. They tick on the one AICLK, but the reset that starts them reaches the
// die in rings, 5 ticks per ring with the centre 20 ticks behind the edge (measured identical on eight p150s), so a
// worker record needs its own tile's integer to land in the eth wall domain. One raw 4 B NoC read of the tile's
// WALL_CLOCK_L bracketed by this core's wall clock samples (tile - midpoint) plus a path term: on the torus the two
// NoCs swap request and response hop counts, so their average carries no hop term, and the same read of this core's
// own register measures this end of the path. What a one-way read cannot split, the far end, stays in the reading
// and is the host's to bound; no number is assumed for it here.
constexpr uint32_t kTileReps = 32;
constexpr uint32_t kTileWallLo = 0xFFB121F0u;      // RISCV_DEBUG_REG_WALL_CLOCK_L, latches the high word
constexpr uint32_t kTileWallHiLive = 0xFFB121F4u;  // RISCV_DEBUG_REG_WALL_CLOCK_1, the live high word

inline uint32_t tile_coord(uint32_t xy) {
    return static_cast<uint32_t>(get_noc_addr(xy & 0xFFFFu, xy >> 16, 0) >> NOC_ADDR_COORD_SHIFT) & NOC_COORDINATE_MASK;
}

// A 4 B read of `addr` on tile `coord`, landing at the scratch word congruent to it: programmed first, then sent
// and waited for as one step, so a bracket around the send alone holds the packet's flight and the two NIUs'
// handling and none of the programming.
inline volatile tt_l1_ptr uint32_t* tile_arm(uint32_t noc, uint32_t coord, uint32_t addr, uint32_t bytes) {
    const uint32_t dst = kScratchAddr + (addr & 0x3Fu);
    NOC_CMD_BUF_WRITE_REG(noc, NCRISC_RD_CMD_BUF, NOC_RET_ADDR_LO, dst);
    NOC_CMD_BUF_WRITE_REG(noc, NCRISC_RD_CMD_BUF, NOC_TARG_ADDR_LO, addr);
    NOC_CMD_BUF_WRITE_REG(noc, NCRISC_RD_CMD_BUF, NOC_TARG_ADDR_MID, 0);
    NOC_CMD_BUF_WRITE_REG(noc, NCRISC_RD_CMD_BUF, NOC_TARG_ADDR_COORDINATE, coord);
    NOC_CMD_BUF_WRITE_REG(noc, NCRISC_RD_CMD_BUF, NOC_AT_LEN_BE, bytes);
    return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dst);
}

inline void tile_send_wait(uint32_t noc) {
    NOC_CMD_BUF_WRITE_REG(noc, NCRISC_RD_CMD_BUF, NOC_CMD_CTRL, NOC_CTRL_SEND_REQ);
    noc_reads_num_issued[noc] += 1;
    while (!ncrisc_noc_reads_flushed(noc)) {
    }
}

inline uint32_t tile_read(uint32_t noc, uint32_t coord, uint32_t addr) {
    volatile tt_l1_ptr uint32_t* land = tile_arm(noc, coord, addr, 4);
    tile_send_wait(noc);
    invalidate_l1_cache();
    return *land;
}

// Median over kTileReps of 2 * (tile wall - bracket midpoint) on one NoC, in wall ticks. Upper median, as the
// calibration used. The request is programmed once and re-sent per rep (the NIU
// clears only NOC_CMD_CTRL on acceptance), and the bracket holds nothing but the send store and the poll: every
// instruction inside it is time the bound has to allow the far end. The poll watches the word land in L1, which
// is closer than the NIU's response counter; the sentinel is the previous reading with its top bit flipped, a value
// the next reading can only take after advancing exactly 2^31 ticks, so a halted or slow clock cannot make it spin.
// The low words differ by the eth-minus-tensix offset modulo 2^32, anywhere in [-2^31, 2^31), so the doubled and
// summed quarter ticks need 64 bits: in 32 they wrap for 3/4 of the offsets and the table lands 2^30 ticks off.
__attribute__((noinline, cold)) int64_t tile_bracket(uint32_t noc, uint32_t coord) {
    uint32_t prev = tile_read(noc, coord, kTileWallLo);  // also leaves the request programmed
    volatile tt_l1_ptr uint32_t* const land =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(kScratchAddr + (kTileWallLo & 0x3Fu));
    volatile uint32_t* const ctrl = reinterpret_cast<volatile uint32_t*>(
        (NCRISC_RD_CMD_BUF << NOC_CMD_BUF_OFFSET_BIT) + (noc << NOC_INSTANCE_OFFSET_BIT) + NOC_CMD_CTRL);
    volatile uint32_t* const wall = reinterpret_cast<volatile uint32_t*>(eth_ptp::kWallClockLo);
    int64_t d[kTileReps];
    for (uint32_t i = 0; i < kTileReps; i++) {
        noc_reads_num_issued[noc] += 1;
        const uint32_t sentinel = prev ^ 0x80000000u;
        *land = sentinel;
        const uint32_t w0 = *wall;
        *ctrl = NOC_CTRL_SEND_REQ;
        do {
            invalidate_l1_cache();
        } while (*land == sentinel);
        const uint32_t w1 = *wall;
        const uint32_t v = *land;
        prev = v;
        const int64_t di = 2 * static_cast<int64_t>(static_cast<int32_t>(v - w0)) - static_cast<int32_t>(w1 - w0);
        uint32_t j = i;
        for (; j > 0 && d[j - 1] > di; j--) {
            d[j] = d[j - 1];
        }
        d[j] = di;
    }
    return d[kTileReps / 2];
}

inline int64_t round_q(int64_t q) { return (q + 2) >> 2; }

// Fills the host's tile table: per tile, this core's wall tick minus the tile's as a 64-bit integer whose low word
// is the two NoCs' bracket result, less this core's own loopback reads on both NoCs, and whose high word comes from
// the tile's live high word read on both sides of its low word (retaken across a wrap) against this core's own.
__attribute__((noinline, cold)) void measure_tiles() {
    volatile tt_l1_ptr uint32_t* tab = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(kTileTableAddr);
    invalidate_l1_cache();
    const uint32_t n = tab[kp::ETH_TILE_N];
    const uint32_t me = tile_coord(kMyXy);
    const int64_t loop_q = tile_bracket(0, me) + tile_bracket(1, me);
    for (uint32_t i = 0; i < n; i++) {
        const uint32_t coord = tile_coord(tab[kp::ETH_TILE_XY_0 + i]);
        const int64_t fine = -round_q(tile_bracket(0, coord) + tile_bracket(1, coord) - loop_q);
        uint32_t hi_a, lo, hi_b;
        do {
            hi_a = tile_read(0, coord, kTileWallHiLive);
            lo = tile_read(0, coord, kTileWallLo);
            hi_b = tile_read(0, coord, kTileWallHiLive);
        } while (hi_a != hi_b);
        const uint32_t my_lo = eth_ptp::rd(eth_ptp::kWallClockLo);
        const uint32_t my_hi = eth_ptp::rd(eth_ptp::kWallClockHi);
        const int64_t coarse = static_cast<int64_t>((static_cast<uint64_t>(my_hi) << 32) | my_lo) -
                               static_cast<int64_t>((static_cast<uint64_t>(hi_a) << 32) | lo);
        const int64_t full = coarse + static_cast<int32_t>(static_cast<uint32_t>(fine) - static_cast<uint32_t>(coarse));
        const uint32_t w = kp::eth_tile_out_word(n, i);
        tab[w] = static_cast<uint32_t>(static_cast<uint64_t>(full));
        tab[w + 1] = static_cast<uint32_t>(static_cast<uint64_t>(full) >> 32);
    }
    tab[kp::ETH_TILE_READY] = kp::kEthTileReadyWord | n;
}


inline void write_to_host(const SocketSenderInterface& s, uint32_t src_l1, uint64_t dst_pcie, uint32_t size) {
    noc_wwrite_with_state<noc_mode, write_cmd_buf, CQ_NOC_SNDL, CQ_NOC_SEND, CQ_NOC_WAIT, true, false>(
        NOC_INDEX, src_l1, s.d2h.pcie_xy_enc, dst_pcie, size, 1);
}

// A frame that crosses the FIFO wrap splits into two writes; socket_push_pages only wraps the pointer.
inline void push_fifo(const SocketSenderInterface& s, uint32_t src, uint32_t dst, uint32_t len) {
    const uint32_t fifo_size = s.downstream_fifo_curr_size;
    if (dst >= fifo_size) {
        dst -= fifo_size;
    }
    const uint64_t base = (static_cast<uint64_t>(s.d2h.data_addr_hi) << 32) | s.downstream_fifo_addr;
    const uint32_t first = (dst + len > fifo_size) ? fifo_size - dst : len;
    write_to_host(s, src, base + dst, first);
    if (first < len) {
        write_to_host(s, src + first, base, len - first);
    }
}

inline void copy_words(volatile tt_l1_ptr uint32_t* dst, const volatile tt_l1_ptr uint32_t* src, uint32_t n) {
    for (uint32_t i = 0; i < n; i++) {
        dst[i] = src[i];
    }
}

inline void ship(SocketSenderInterface& s, uint32_t bytes) {
    const uint32_t pages = bytes / kPageBytes;
    socket_reserve_pages(s, pages);
    push_fifo(s, kStageAddr, s.write_ptr, bytes);
    socket_push_pages(s, pages);
    // Data lands before its announcement: flush, then bytes_sent.
    noc_async_writes_flushed();
    socket_notify_receiver(s);
}


// Places one lane's run [start, start+take) from `ring` (its L1 image, already local) into the frame at `off`,
// in the shape the decoder expects for (start, take). Returns the new offset.
inline uint32_t place_run(
    volatile tt_l1_ptr uint32_t* frame,
    const volatile tt_l1_ptr uint32_t* ring,
    uint32_t start,
    uint32_t take,
    uint32_t off) {
    const uint32_t hm = start & (kRingWords - 1u);
    if (hm + take <= kRingWords) {
        off += kp::spsc_span_pack_pad(start, off);
        copy_words(frame + off, ring + hm, take);
        return off + take;
    }
    if (kp::spsc_span_wrap_image(start, take, kRingWords)) {
        off += kp::spsc_span_pack_pad(0u, off);
        copy_words(frame + off, ring, kRingWords);
        return off + kRingWords;
    }
    off += kp::spsc_span_pack_pad(start, off);
    const uint32_t first = kRingWords - hm;
    copy_words(frame + off, ring + hm, first);
    copy_words(frame + off + first, ring, take - first);
    return off + take;
}

inline uint32_t finish_frame(volatile tt_l1_ptr uint32_t* frame, uint32_t xy, uint32_t off) {
    frame[0] = kp::spsc_span_w0();
    frame[kp::SPSC_PREFIX_XY] = xy;
    frame[kLenWord] = off - kPrefix;
    const uint32_t bytes = off * 4u;
    return (bytes + kPageBytes - 1u) & ~(kPageBytes - 1u);
}

// A sync frame for core `xy`: records [first, first + n) of the ring of `ring_records` at `recs`.
inline uint32_t pack_sync_frame(
    uint32_t xy, const volatile tt_l1_ptr uint32_t* recs, uint32_t first, uint32_t n, uint32_t ring_records) {
    volatile tt_l1_ptr uint32_t* frame = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(kStageAddr);
    for (uint32_t r = 0; r < kNumRisc; r++) {
        frame[kp::SPSC_PREFIX_HEAD_0 + r] = 0;
    }
    frame[kp::SPSC_PREFIX_HEAD_0] = n;
    uint32_t off = kPrefix;
    for (uint32_t i = 0; i < n; i++) {
        copy_words(frame + off, recs + ((first + i) % ring_records) * kp::kSyncRecordWords, kp::kSyncRecordWords);
        off += kp::kSyncRecordWords;
    }
    while (off < kPrefix + kWireCtrl) {
        frame[off++] = 0;
    }
    return finish_frame(frame, xy, off);
}

// This core: rings and control vector are local. Advances its own heads -- this RISC is their consumer.
inline uint32_t pack_own_frame() {
    volatile tt_l1_ptr uint32_t* cv = kp::profiler_control_buffer;
    volatile tt_l1_ptr uint32_t* frame = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(kStageAddr);
    uint32_t off = kPrefix + kWireCtrl;
    bool live = false;
    invalidate_l1_cache();
    copy_words(frame + kPrefix, cv + kp::SPSC_WIRE_CV_BASE, kWireCtrl);
    for (uint32_t r = 0; r < kNumRisc; r++) {
        uint32_t start = cv[kp::SPSC_RING_HEAD_0 + r];
        const uint32_t tail = r < kNumEthRisc ? cv[kp::SPSC_RING_TAIL_0 + r] : start;
        uint32_t take = tail - start;
        if (take > kRingWords) {
            // Lapped: a producer wrote past its consumer. Only the last ring image is still intact; ship that and
            // never index past it (an unclamped take would read beyond the image into this core's own L1).
            start = tail - kRingWords;
            take = kRingWords;
        }
        frame[kp::SPSC_PREFIX_HEAD_0 + r] = start;
        if (take == 0) {
            continue;
        }
        live = true;
        off = place_run(frame, kp::profiler_data_buffer[r].data, start, take, off);
        cv[kp::SPSC_RING_HEAD_0 + r] = tail;
    }
    return live ? finish_frame(frame, kMyXy, off) : 0u;
}

// A linked (active) eth core: its control vector and live rings are NoC-read into the scratch, the frame is
// packed from those images, and its heads are written back over the NoC once the frame is staged.
inline uint32_t pack_linked_frame(uint32_t xy, uint32_t prof_l1) {
    constexpr uint32_t kCvScratch = kScratchAddr;
    constexpr uint32_t img_base = kScratchAddr + kp::PROFILER_L1_CONTROL_BUFFER_SIZE;
    volatile tt_l1_ptr uint32_t* cv = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(kCvScratch);
    volatile tt_l1_ptr uint32_t* frame = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(kStageAddr);
    const uint32_t x = xy & 0xFFFFu;
    const uint32_t y = xy >> 16;
    // Control vector first: a tail observed here bounds the ring words read after it.
    noc_async_read(get_noc_addr(x, y, prof_l1), kCvScratch, kCvWords * 4u);
    noc_async_read_barrier();
    bool live = false;
    uint32_t starts[kNumEthRisc];
    uint32_t takes[kNumEthRisc];
    for (uint32_t r = 0; r < kNumEthRisc; r++) {
        const uint32_t tail = cv[kp::SPSC_RING_TAIL_0 + r];
        starts[r] = cv[kp::SPSC_RING_HEAD_0 + r];
        takes[r] = tail - starts[r];
        if (takes[r] > kRingWords) {
            // Lapped (see pack_own_frame): ship the last intact ring image, never index past it.
            starts[r] = tail - kRingWords;
            takes[r] = kRingWords;
        }
        live = live || takes[r] != 0;
    }
    if (!live) {
        return 0;
    }
    for (uint32_t r = 0; r < kNumEthRisc; r++) {
        if (takes[r] == 0) {
            continue;
        }
        const uint32_t ring_l1 = prof_l1 + kp::PROFILER_L1_CONTROL_BUFFER_SIZE + r * kRingBytes;
        const uint32_t img = img_base + r * kRingBytes;
        const uint32_t hm = starts[r] & (kRingWords - 1u);
        if (hm + takes[r] <= kRingWords) {
            noc_async_read(get_noc_addr(x, y, ring_l1 + hm * 4u), img + hm * 4u, takes[r] * 4u);
        } else if (kp::spsc_span_wrap_image(starts[r], takes[r], kRingWords)) {
            noc_async_read(get_noc_addr(x, y, ring_l1), img, kRingBytes);
        } else {
            const uint32_t first = kRingWords - hm;
            noc_async_read(get_noc_addr(x, y, ring_l1 + hm * 4u), img + hm * 4u, first * 4u);
            noc_async_read(get_noc_addr(x, y, ring_l1), img, (takes[r] - first) * 4u);
        }
    }
    noc_async_read_barrier();
    uint32_t off = kPrefix + kWireCtrl;
    copy_words(frame + kPrefix, cv + kp::SPSC_WIRE_CV_BASE, kWireCtrl);
    for (uint32_t r = 0; r < kNumRisc; r++) {
        const uint32_t start = r < kNumEthRisc ? starts[r] : cv[kp::SPSC_RING_HEAD_0 + r];
        frame[kp::SPSC_PREFIX_HEAD_0 + r] = start;
        if (r >= kNumEthRisc || takes[r] == 0) {
            continue;
        }
        const volatile tt_l1_ptr uint32_t* img =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(img_base + r * kRingBytes);
        off = place_run(frame, img, start, takes[r], off);
        // Head write-back (to the tail observed above): the producer on that core sees its ring drain, as it
        // would from a relay.
        noc_inline_dw_write(get_noc_addr(x, y, prof_l1 + (kp::SPSC_RING_HEAD_0 + r) * 4u), start + takes[r], 0xF);
    }
    return finish_frame(frame, xy, off);
}

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

    // Linked cores: rt args [0] = n, then (xy, profiler L1 base) per core.
    const uint32_t n_linked_arg = get_arg_val<uint32_t>(0);
    const uint32_t n_linked = n_linked_arg < kMaxLinked ? n_linked_arg : kMaxLinked;
    uint32_t linked_xy[kMaxLinked];
    uint32_t linked_l1[kMaxLinked];
    for (uint32_t i = 0; i < n_linked; i++) {
        linked_xy[i] = get_arg_val<uint32_t>(1 + 2 * i);
        linked_l1[i] = get_arg_val<uint32_t>(2 + 2 * i);
    }

    if constexpr (kMeasureOnly != 0) {
        measure_tiles();
        return;
    }

    SocketSenderInterface sender = create_sender_socket_interface(kSocketConfigAddr);
    set_sender_socket_page_size(sender, kPageBytes);
    SocketSenderInterface sync_sender = create_sender_socket_interface(kSyncCfgAddr);
    set_sender_socket_page_size(sync_sender, kPageBytes);
    noc_write_init_state<write_cmd_buf>(NOC_INDEX, NOC_UNICAST_WRITE_VC);

    const auto sweep_sync = [&]() {
        while (sync::g_head != sync::g_tail) {
            const uint32_t left = sync::g_tail - sync::g_head;
            const uint32_t n = left < kp::kSyncFrameRecords ? left : kp::kSyncFrameRecords;
            ship(sync_sender, pack_sync_frame(kMyXy, sync::rec(0), sync::g_head, n, kp::kSyncRingRecords));
            sync::g_head += n;
        }
    };
    // A linked core's sync ring image, read into the scratch past its ring images.
    constexpr uint32_t kLinkScratch = kScratchAddr + kp::PROFILER_L1_CONTROL_BUFFER_SIZE + kNumEthRisc * kRingBytes;
    constexpr uint32_t kLinkRingBytes = kp::kLinkSyncRingRecords * kp::kSyncRecordWords * 4u;
    constexpr uint32_t kTailBlock = (kp::SPSC_LINK_SYNC_TAIL * 4u) & ~63u;
    static_assert(kLinkScratch + kLinkRingBytes <= kScratchAddr + 4608);
    uint32_t link_cursor[kMaxLinked] = {};
    // Linked core i's records [cursor, tail), the tail from the control vector pack_linked_frame just read. The end
    // never waits for this core, so a tail more than a ring ahead means the oldest are gone. A round's two records
    // land back to back, so a run reaching into the ring's last two slots re-reads the tail after the image and drops
    // what the end may have overwritten meanwhile.
    const auto ship_link_sync = [&](uint32_t i) {
        if (kLinkRingAddr == 0) {
            return;
        }
        volatile tt_l1_ptr uint32_t* cv = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(kScratchAddr);
        const uint32_t tail = cv[kp::SPSC_LINK_SYNC_TAIL];
        if (tail == link_cursor[i]) {
            return;
        }
        const uint32_t x = linked_xy[i] & 0xFFFFu, y = linked_xy[i] >> 16;
        uint32_t first =
            tail - link_cursor[i] > kp::kLinkSyncRingRecords ? tail - kp::kLinkSyncRingRecords : link_cursor[i];
        noc_async_read(get_noc_addr(x, y, kLinkRingAddr), kLinkScratch, kLinkRingBytes);
        noc_async_read_barrier();
        if (tail - first > kp::kLinkSyncRingRecords - 2) {
            noc_async_read(get_noc_addr(x, y, linked_l1[i] + kTailBlock), kScratchAddr + kTailBlock, 64);
            noc_async_read_barrier();
            const uint32_t now = cv[kp::SPSC_LINK_SYNC_TAIL];
            if (now - first > kp::kLinkSyncRingRecords) {
                first = now - kp::kLinkSyncRingRecords;
            }
        }
        if (first < tail) {
            ship(
                sync_sender,
                pack_sync_frame(
                    linked_xy[i],
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(kLinkScratch),
                    first,
                    tail - first,
                    kp::kLinkSyncRingRecords));
        }
        link_cursor[i] = tail;
    };
    const auto sweep = [&]() {
        uint32_t bytes = pack_own_frame();
        if (bytes != 0) {
            ship(sender, bytes);
        }
        for (uint32_t i = 0; i < n_linked; i++) {
            bytes = pack_linked_frame(linked_xy[i], linked_l1[i]);
            if (bytes != 0) {
                ship(sender, bytes);
            }
            ship_link_sync(i);
        }
        sweep_sync();
    };

    // Sampling waits for the host's go word, written once the receiver's ingest threads drain this socket. Started
    // at launch, the pusher fills its 1 MiB FIFO with pre-capture samples and then laps the consumers' first read.
    while (*go == 0u && *stop == 0u) {
        (*hb)++;
        invalidate_l1_cache();
    }

    // The loop reads three registers per sample -- refclk, wall, refclk -- and carries both clocks' high words
    // itself from the low words' wraps, which at this rate no sweep can hide (86 s and 3.2 s periods).
    const eth_ptp::Instant start = eth_ptp::read_instant();
    uint32_t r_hi = static_cast<uint32_t>(start.refclk >> 32), prev_r_lo = static_cast<uint32_t>(start.refclk);
    uint32_t w_hi = start.wall_hi, prev_w_lo = start.wall_lo;
    model::Model m;
    model::begin_acquire(m, start.refclk);
    uint64_t r_sweep = start.refclk;
    uint32_t iter = 0;
    while (true) {
        const uint32_t ra_lo = eth_ptp::rd(eth_ptp::kPtpCfrLo);
        const uint32_t w_lo = eth_ptp::rd(eth_ptp::kWallClockLo);
        const uint32_t rb_lo = eth_ptp::rd(eth_ptp::kPtpCfrLo);
        r_hi += rb_lo < prev_r_lo;
        w_hi += w_lo < prev_w_lo;
        prev_r_lo = rb_lo;
        prev_w_lo = w_lo;
        const uint64_t r = (static_cast<uint64_t>(r_hi) << 32) | rb_lo;
        if (rb_lo != ra_lo) {
            model::feed(m, r, (static_cast<uint64_t>(w_hi) << 32) | w_lo);
        }
        if ((++iter & 255u) != 0u) {
            continue;
        }
        (*hb)++;
        invalidate_l1_cache();
        const uint32_t fill = kp::profiler_control_buffer[kp::TAIL_INDEX] - kp::profiler_control_buffer[kp::HEAD_INDEX];
        if (fill >= kShipWords || r - r_sweep >= kSweepTicks) {
            r_sweep = r;
            sweep();
        }
        // Teardown: the relay stop word, written by the host at quiesce. The streaming control layout has no
        // terminate slot; this word is the only stop signal a resident eth kernel gets.
        if (*stop != 0u) {
            break;
        }
    }
    // Final sweep, then the relay done protocol: Drained once the last page is pushed, Done once every byte acked.
    sweep();
    *done = kp::kRelayDrainedWord;
    socket_barrier(sender);
    socket_barrier(sync_sender);
    noc_async_writes_flushed();
    update_socket_config(sender);
    update_socket_config(sync_sender);
    *done = kp::kRelayDoneWord;
#endif
}
