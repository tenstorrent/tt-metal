// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// Resident on one idle eth core per chip. At each checkpoint the host asks for, it reads every target tile's wall
// clock over NoC 0 with the tile clock network's estimator (tools/profiler/sync/tile_sync.cpp): per target the median
// and quartile spread of 2 * (target wall - bracket midpoint), the median round trip, and the coarse whole-clock
// difference that places the median in its 2^32-tick turn. Between checkpoints it reads nothing off its own tile.
//
// Arguments: scratch address, target count, reps. Scratch: the reads' 64 B landing block, the go word at +64 (the
// checkpoint to take, kExit to leave), the done word at +68, the targets from +128 as y << 16 | x, then kOutWords per
// target, then the two histograms.
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "tools/profiler/sync/tile_read.hpp"

constexpr uint32_t kMaxTargets = 256;
constexpr uint32_t kOutWords = 5;
constexpr uint32_t kBins = 128;
constexpr uint32_t kWarmup = 16;
constexpr uint32_t kExit = 0xFFFFFFFFu;

inline int32_t quantile(const volatile tt_l1_ptr uint32_t* hist, uint32_t rank) {
    uint32_t seen = 0;
    for (uint32_t b = 0; b < kBins; b++) {
        seen += hist[b];
        if (seen >= rank) {
            return static_cast<int32_t>(b) - static_cast<int32_t>(kBins / 2);
        }
    }
    return static_cast<int32_t>(kBins / 2) - 1;
}

void kernel_main() {
    const uint32_t scratch = get_arg_val<uint32_t>(0);
    const uint32_t n = get_arg_val<uint32_t>(1);
    const uint32_t reps = get_arg_val<uint32_t>(2);
    volatile tt_l1_ptr uint32_t* ctl = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch + 64);
    volatile tt_l1_ptr uint32_t* targets = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch + 128);
    volatile tt_l1_ptr uint32_t* out = targets + kMaxTargets;
    volatile tt_l1_ptr uint32_t* hist_d = out + kOutWords * kMaxTargets;
    volatile tt_l1_ptr uint32_t* hist_r = hist_d + kBins;
    volatile uint32_t* const wall = reinterpret_cast<volatile uint32_t*>(tile_read::kWallLo);
    for (;;) {
        uint32_t go;
        do {
            invalidate_l1_cache();
            go = ctl[0];
        } while (go == ctl[1]);
        if (go == kExit) {
            break;
        }
        for (uint32_t k = 0; k < n; k++) {
            for (uint32_t b = 0; b < 2 * kBins; b++) {
                hist_d[b] = 0;
            }
            const uint32_t coord = tile_read::coord(targets[k]);
            int64_t warm_d[kWarmup], warm_r[kWarmup];
            tile_read::bracket(0, coord, scratch, wall, kWarmup, [&](uint32_t i, int64_t d, int64_t r) {
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
            tile_read::bracket(0, coord, scratch, wall, reps, [&](uint32_t, int64_t d, int64_t r) {
                const int64_t bd = d - centre_d + kBins / 2, br = r - centre_r + kBins / 2;
                hist_d[bd < 0 ? 0 : bd >= kBins ? kBins - 1 : bd]++;
                hist_r[br < 0 ? 0 : br >= kBins ? kBins - 1 : br]++;
            });
            const uint64_t coarse = tile_read::wall64(0, coord, scratch) - tile_read::own_wall64();
            volatile tt_l1_ptr uint32_t* o = out + kOutWords * k;
            o[0] = static_cast<uint32_t>(static_cast<int32_t>(centre_d + quantile(hist_d, reps / 2 + 1)));
            o[1] = static_cast<uint32_t>(quantile(hist_d, 3 * reps / 4 + 1) - quantile(hist_d, reps / 4 + 1));
            o[2] = static_cast<uint32_t>(static_cast<int32_t>(centre_r + quantile(hist_r, reps / 2 + 1)));
            o[3] = static_cast<uint32_t>(coarse);
            o[4] = static_cast<uint32_t>(coarse >> 32);
        }
        ctl[1] = go;
    }
}
