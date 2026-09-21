// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// One tile of the tile clock network, on whichever RISC the tile has. The tile announces itself with the host's
// nonce, waits for its turn, reads each partner tile's wall clock over the NoC the host named (`reps` brackets
// each) and writes per partner the median and the quartile spread of 2 * (partner wall - bracket midpoint) in the
// clocks' low words, the median round trip, and one coarse whole-clock difference that says which 2^32-tick turn
// the median sits in, then announces done with the nonce inverted, waits to be released and zeroes its scratch.
// Partners share this tile's row or column, NoC 0 towards higher coordinates and NoC 1 towards lower, so a pair's
// two readings cross the same links in opposite directions.
//
// The samples of one partner go into two histograms (offsets, round trips) centred on a short warm-up's medians, so
// the scratch is the same size for any number of reps; a sample outside the window lands in its edge bin, which
// moves no median while such samples stay in the minority.
//
// A DRAM tile's NIU answers NoC traffic from GDDR unless it is in stream mode, so a DRISC holds stream mode from its
// announcement until its release: only then can the others read its wall clock and its own reads land in its L1.
//
// Arguments: scratch address, reps, nonce, partner count, then a partner per argument as noc << 31 | y << 16 | x.
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "hostdev/streaming_profiler_common.h"
#include "tools/profiler/sync/tile_read.hpp"
#if defined(COMPILE_FOR_DRISC)
#include "experimental/drisc_mode.h"
#endif

namespace kp = kernel_profiler;

constexpr uint32_t kWarmup = 16;

// The bin whose running count first reaches `rank` samples.
inline int32_t quantile(const volatile tt_l1_ptr uint32_t* hist, uint32_t rank) {
    uint32_t seen = 0;
    for (uint32_t b = 0; b < kp::kTileNetBins; b++) {
        seen += hist[b];
        if (seen >= rank) {
            return static_cast<int32_t>(b) - static_cast<int32_t>(kp::kTileNetBins / 2);
        }
    }
    return static_cast<int32_t>(kp::kTileNetBins / 2) - 1;
}

void kernel_main() {
    const uint32_t scratch = get_arg_val<uint32_t>(0);
    const uint32_t reps = get_arg_val<uint32_t>(1);
    const uint32_t nonce = get_arg_val<uint32_t>(2);
    const uint32_t n = get_arg_val<uint32_t>(3);
    volatile tt_l1_ptr uint32_t* tab = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch + kp::kTileNetTable);
    volatile tt_l1_ptr uint32_t* hist_d = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch + kp::kTileNetHist);
    volatile tt_l1_ptr uint32_t* hist_r = hist_d + kp::kTileNetBins;
    volatile uint32_t* const wall = reinterpret_cast<volatile uint32_t*>(tile_read::kWallLo);
#if defined(COMPILE_FOR_DRISC)
    experimental::drisc_set_stream_mode_all();
#endif
    tab[kp::TILE_NET_GO] = 0;
    tab[kp::TILE_NET_READY] = nonce;
    uint32_t go;
    do {
        invalidate_l1_cache();
        go = tab[kp::TILE_NET_GO];
    } while (go == 0);
    if (go == kp::kTileNetGoMeasure) {
        for (uint32_t k = 0; k < n; k++) {
            const uint32_t w = get_arg_val<uint32_t>(4 + k);
            for (uint32_t b = 0; b < 2 * kp::kTileNetBins; b++) {
                hist_d[b] = 0;
            }
            // The windows are centred on the medians of a short warm-up: the first brackets run cold and land wide.
            const uint32_t noc = w >> 31, coord = tile_read::coord(w & 0x7FFFFFFFu);
            int64_t warm_d[kWarmup], warm_r[kWarmup];
            tile_read::bracket(noc, coord, scratch, wall, kWarmup, [&](uint32_t i, int64_t d, int64_t r) {
                uint32_t j = i;
                for (; j > 0 && warm_d[j - 1] > d; j--) {
                    warm_d[j] = warm_d[j - 1];
                }
                warm_d[j] = d;
                for (j = i; j > 0 && warm_r[j - 1] > r; j--) {
                    warm_r[j] = warm_r[j - 1];
                }
                warm_r[j] = r;
            });
            const int64_t centre_d = warm_d[kWarmup / 2], centre_r = warm_r[kWarmup / 2];
            tile_read::bracket(noc, coord, scratch, wall, reps, [&](uint32_t, int64_t d, int64_t r) {
                const int64_t bd = d - centre_d + kp::kTileNetBins / 2, br = r - centre_r + kp::kTileNetBins / 2;
                hist_d[bd < 0 ? 0 : bd >= kp::kTileNetBins ? kp::kTileNetBins - 1 : bd]++;
                hist_r[br < 0 ? 0 : br >= kp::kTileNetBins ? kp::kTileNetBins - 1 : br]++;
            });
            const uint64_t coarse = tile_read::wall64(noc, coord, scratch) - tile_read::own_wall64();
            const uint32_t out = kp::TILE_NET_OUT_0 + kp::TILE_NET_OUT_WORDS * k;
            tab[out] = static_cast<uint32_t>(static_cast<int32_t>(centre_d + quantile(hist_d, reps / 2 + 1)));
            tab[out + 1] = static_cast<uint32_t>(quantile(hist_d, 3 * reps / 4 + 1) - quantile(hist_d, reps / 4 + 1));
            tab[out + 2] = static_cast<uint32_t>(static_cast<int32_t>(centre_r + quantile(hist_r, reps / 2 + 1)));
            tab[out + 3] = static_cast<uint32_t>(coarse);
            tab[out + 4] = static_cast<uint32_t>(coarse >> 32);
        }
        tab[kp::TILE_NET_READY] = ~nonce;
        do {
            invalidate_l1_cache();
        } while (tab[kp::TILE_NET_GO] != kp::kTileNetGoExit);
    }
    // On a dispatch core the scratch is the profiler's ring space, which its first frame must find zero.
    volatile tt_l1_ptr uint32_t* p = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch);
    for (uint32_t i = 0; i < kp::kTileNetScratchBytes / 4; i++) {
        p[i] = 0;
    }
#if defined(COMPILE_FOR_DRISC)
    experimental::drisc_set_noc2axi_mode_all();
#endif
}
