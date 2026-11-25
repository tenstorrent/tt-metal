// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Combine lab writer (the flat expert's y writer, producer stage): writer w of W owns bytes [w SEG, (w + 1) SEG) of
// every row. All writers walk the same row sequence - row r goes to pair p = (r K) % NP of this chip's (home,
// accumulator) pairs (home index p / A, accumulator p % A: a pseudo-random interleave of homes, as production's token
// order) - in tiles of 32 rows. A row needs a slot of its port's injection ring: the port's reader grants slots
// (cumulative GRANT[pid] here); the writer writes its segment into slot (rows issued to the port) % ISLOTS at offset
// w SEG (writer 0 also the row's 32 B trailer: home, accumulator, arrival flag), and once a unit's writes are flushed
// adds the unit's rows to the port's LANDED[w]. A unit is the rows issued since the last flush; a row whose port has no
// grant ends the unit (flushed, counted) before the writer waits (head of line, as production's y writer). PACE > 0:
// a tile is started no earlier than PACE cycles after the previous one (the experts' rate).
//
// RT: 0 me, 1 w, 2 NH homes, NH x (home block index, first-hop direction index), then the ports' cores (x | y << 8)
// by port id (direction index L + link).
#include "api/dataflow/dataflow_api.h"
#ifdef LAB_TIMING
#include "api/debug/kernel_profiler.hpp"
#endif

constexpr uint32_t A = get_named_compile_time_arg_val("A");
constexpr uint32_t L = get_named_compile_time_arg_val("L");
constexpr uint32_t N = get_named_compile_time_arg_val("N");
constexpr uint32_t W = get_named_compile_time_arg_val("W");
constexpr uint32_t SEG = get_named_compile_time_arg_val("SEG");
constexpr uint32_t ROWB = get_named_compile_time_arg_val("ROWB");  // the row's bytes (its trailer at ROWB)
constexpr uint32_t SLOT = get_named_compile_time_arg_val("SLOT");
constexpr uint32_t ISLOTS = get_named_compile_time_arg_val("ISLOTS");
constexpr uint32_t NROWS = get_named_compile_time_arg_val("NROWS");  // rows a call (this chip's)
constexpr uint32_t NP = get_named_compile_time_arg_val("NP");        // (home, accumulator) pairs
constexpr uint32_t K = get_named_compile_time_arg_val("K");          // the interleave's stride (coprime with NP)
constexpr uint32_t PACE = get_named_compile_time_arg_val("PACE");
// PACE_G: tiles a burst (production's 128-row sub-blocks: 4) - a burst's tiles go back to back, the next burst no
// earlier than PACE_G x PACE cycles after the previous one
constexpr uint32_t PACE_G = get_named_compile_time_arg_val("PACE_G");
constexpr uint32_t ST = get_named_compile_time_arg_val("STATE");
constexpr uint32_t IRING = get_named_compile_time_arg_val("IRING");  // the ports' injection rings
constexpr uint32_t GRANT = ST + get_named_compile_time_arg_val("O_GRANT");    // [8] grants from each port
constexpr uint32_t ISSUED = ST + get_named_compile_time_arg_val("O_ISSUED");  // [8] rows issued to each port
constexpr uint32_t TRS = ST + get_named_compile_time_arg_val("O_TRSTAGE");    // [32] x 32 B trailer stages
constexpr uint32_t SRCSEG = ST + get_named_compile_time_arg_val("O_SRC");     // the segment's bytes
constexpr uint32_t LANDED = ST + get_named_compile_time_arg_val("O_LANDED");  // (on the ports) [W]
constexpr uint32_t NPORT = 4 * L;
// SL > 1 (SPLIT): writer w works for part g = w / (W / SL) of every row - the link-g port of its direction, at offset
// (w % (W / SL)) SEG of that part (ROWB: a part's bytes, its trailer there); else every writer for every row
constexpr uint32_t SL = get_named_compile_time_arg_val("SL");
constexpr uint32_t WG = W / SL;  // writers a part
// NOWRITE: the hand-off's protocol without its payload (no segment writes; writer 0 still writes the trailers)
constexpr uint32_t NOWRITE = get_named_compile_time_arg_val("NOWRITE");
// WMODE: the segment writes - 0 noc_async_write (generic), 1 noc_async_write_one_packet (a piece fits one packet)
constexpr uint32_t WMODE = get_named_compile_time_arg_val("WMODE");
// WDEST 1 (diagnosis): the payload into this core's own L1 - the same issue path, no route to the port
constexpr uint32_t WDEST = get_named_compile_time_arg_val("WDEST");

void kernel_main() {
    const uint32_t me = get_arg_val<uint32_t>(0), w = get_arg_val<uint32_t>(1), nh = get_arg_val<uint32_t>(2);
    uint8_t home_idx[N], home_dir[N];
    for (uint32_t i = 0; i < nh; ++i) {
        home_idx[i] = get_arg_val<uint32_t>(3 + 2 * i);
        home_dir[i] = get_arg_val<uint32_t>(4 + 2 * i);
    }
    uint32_t port_xy[NPORT];
    for (uint32_t i = 0; i < NPORT; ++i) {
        port_xy[i] = get_arg_val<uint32_t>(3 + 2 * nh + i);
    }
    volatile tt_l1_ptr uint32_t* grant = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(GRANT);
    volatile tt_l1_ptr uint32_t* issued = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ISSUED);
    uint32_t unit[NPORT];
    for (uint32_t i = 0; i < NPORT; ++i) {
        unit[i] = 0;
    }
    uint32_t stage = 0;  // (trailer stages used in this unit)
#ifdef LAB_TIMING
    uint32_t t_retire = 0;
#endif
    auto retire = [&]() {
#ifdef LAB_TIMING
        const uint32_t tr0 = get_timestamp_32b();
#endif
        noc_async_writes_flushed();
        for (uint32_t i = 0; i < NPORT; ++i) {
            if (unit[i]) {
                noc_semaphore_inc(get_noc_addr(port_xy[i] & 0xFF, port_xy[i] >> 8, LANDED + 4 * (w % WG)), unit[i]);
                unit[i] = 0;
            }
        }
        stage = 0;
#ifdef LAB_TIMING
        t_retire += get_timestamp_32b() - tr0;
#endif
    };
#ifdef LAB_TIMING
    uint32_t t_grant = 0, t_write = 0;
#endif
    const uint32_t t0 = get_timestamp_32b();
    for (uint32_t r0 = 0; r0 < NROWS; r0 += 32) {
        if constexpr (PACE > 0) {
            while (get_timestamp_32b() - t0 < (r0 / 32) / PACE_G * PACE_G * PACE) {
            }
        }
        for (uint32_t r = r0; r < r0 + 32 && r < NROWS; ++r) {
            const uint32_t p = (r * K) % NP, hi = p / A, o = p % A, h = home_idx[hi];
            const uint32_t pid = home_dir[hi] * L + (SL > 1 ? w / WG : (o + me + h) % L);
            const uint32_t k = issued[pid];
            invalidate_l1_cache();
            if (grant[pid] - k < 1) {
                retire();  // (head of line: the unit so far goes out before the wait)
#ifdef LAB_TIMING
                const uint32_t tg = get_timestamp_32b();
#endif
                do {
                    invalidate_l1_cache();
                } while (grant[pid] - k < 1);
#ifdef LAB_TIMING
                t_grant += get_timestamp_32b() - tg;
#endif
            }
            const uint32_t slot = IRING + (k % ISLOTS) * SLOT;
            const uint32_t px = port_xy[pid] & 0xFF, py = port_xy[pid] >> 8;
#ifdef LAB_TIMING
            const uint32_t tw = get_timestamp_32b();
#endif
            if constexpr (!NOWRITE && WDEST == 1) {
                noc_async_write(SRCSEG, get_noc_addr(SRCSEG + SEG), SEG);  // (a loopback into its own L1)
            } else if constexpr (!NOWRITE && WMODE == 1) {
                noc_async_write_one_packet(SRCSEG, get_noc_addr(px, py, slot + (w % WG) * SEG), SEG);
            } else if constexpr (!NOWRITE) {
                noc_async_write(SRCSEG, get_noc_addr(px, py, slot + (w % WG) * SEG), SEG);
            }
#ifdef LAB_TIMING
            t_write += get_timestamp_32b() - tw;
#endif
            if (w % WG == 0) {
                volatile tt_l1_ptr uint32_t* tr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(TRS + stage * 32);
                tr[0] = h;
                tr[1] = o;
                for (uint32_t i = 2; i < 7; ++i) {
                    tr[i] = 0;
                }
                tr[7] = 1;  // (the arrival flag: the packet's last word)
                asm volatile("fence" ::: "memory");
                noc_async_write(TRS + stage * 32, get_noc_addr(px, py, slot + ROWB), 32);
                ++stage;
            }
            issued[pid] = k + 1;
            ++unit[pid];
        }
        retire();
    }
    noc_async_atomic_barrier();
#ifdef LAB_TIMING
    DeviceTimestampedData("LAB-WGRANT", t_grant);
    DeviceTimestampedData("LAB-WEND", get_timestamp_32b() - t0);
    DeviceTimestampedData("LAB-WRETIRE", t_retire);
    DeviceTimestampedData("LAB-WWRITE", t_write);
#endif
}
