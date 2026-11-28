// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// OpenVDN frame scan, per unit one 32-row block of one head's state in one direction (one or more units per core):
//
//     state <- state @ T[f] + J[f]      over the frames in order (forward) or in reverse, from the start state
//
// The rows of the [128, 128] state evolve independently, so each core carries its 32 rows (4 tiles) through all
// frames; the state stays in L1. Same arithmetic as `linear_ops_openvdn.bidirectional_scan`: the fp32 state and T are
// split into bf16 hi/lo parts, the three leading products accumulate in fp32 DST at HiFi4, and J is added in fp32 on
// the SFPU. Every new state block is also packed for the writer (the prefix or suffix bank).
//
// Compile-time args: has_start. Runtime args: number of frames, number of units.

#include <cstdint>

#include "api/compute/cb_api.h"
#include "api/compute/common.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"

namespace {

constexpr uint32_t NT = 4;
constexpr uint32_t TILES = NT * NT;

// fp32 (unpacked to DST)
constexpr uint32_t cb_t = 0;      // T[f], streamed
constexpr uint32_t cb_j = 1;      // this block's rows of J[f], streamed
constexpr uint32_t cb_start = 2;  // this block's rows of the start state
constexpr uint32_t cb_tmp = 6;    // T staging for the split
// The state block and its split parts, two sets (current and next) used alternately.
constexpr uint32_t cb_s32[2] = {7, 20};
constexpr uint32_t cb_s_hi[2] = {10, 21};
constexpr uint32_t cb_s_lo[2] = {11, 22};
// bf16
constexpr uint32_t cb_t_hi = 8;
constexpr uint32_t cb_t_lo = 9;
// output, fp32
constexpr uint32_t cb_out = 16;

FORCE_INLINE void copy_to(uint32_t cb, uint32_t tile, uint32_t dst) {
    reconfig_data_format_srca(cb);
    copy_init(cb);
    copy_tile(cb, tile, dst);
}

FORCE_INLINE void pack_to(uint32_t dst, uint32_t cb) {
    cb_reserve_back(cb, 1);
    pack_reconfig_data_format(cb);
    pack_tile(dst, cb);
    cb_push_back(cb, 1);
}

// lo[t] = f32[base + t] - hi[base + t], t < count, pushed to `lo`.
FORCE_INLINE void make_lo(uint32_t f32, uint32_t hi, uint32_t lo, uint32_t base, uint32_t count) {
    for (uint32_t t = 0; t < count; ++t) {
        tile_regs_acquire();
        copy_to(f32, base + t, 0);
        copy_to(hi, base + t, 1);
        sub_binary_tile_init();
        sub_binary_tile(0, 1, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_to(0, lo);
        tile_regs_release();
    }
}

// The first state block: copied from `in` (start state or J[first frame]); with `to_out` also written out.
FORCE_INLINE void seed_state(uint32_t in, bool to_out) {
    for (uint32_t j = 0; j < NT; ++j) {
        cb_wait_front(in, 1);
        tile_regs_acquire();
        copy_to(in, 0, 0);
        tile_regs_commit();
        tile_regs_wait();
        if (to_out) {
            pack_to(0, cb_out);
        }
        pack_to(0, cb_s32[0]);
        pack_to(0, cb_s_hi[0]);
        tile_regs_release();
        cb_pop_front(in, 1);
    }
    cb_wait_front(cb_s32[0], NT);
    cb_wait_front(cb_s_hi[0], NT);
    make_lo(cb_s32[0], cb_s_hi[0], cb_s_lo[0], 0, NT);
    cb_wait_front(cb_s_lo[0], NT);
}

// One unit: one state block through all frames.
template <bool has_start>
void scan_unit(uint32_t frames) {
    uint32_t cur = 0;
    uint32_t step = 0;
    if constexpr (has_start) {
        seed_state(cb_start, false);
    } else {
        seed_state(cb_j, true);  // state after the first frame is J[first]
        step = 1;
    }

    for (; step < frames; ++step) {
        // T[f], split.
        for (uint32_t t = 0; t < TILES; ++t) {
            cb_wait_front(cb_t, 1);
            tile_regs_acquire();
            copy_to(cb_t, 0, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_to(0, cb_tmp);
            pack_to(0, cb_t_hi);
            tile_regs_release();
            cb_pop_front(cb_t, 1);
        }
        cb_wait_front(cb_tmp, TILES);
        cb_wait_front(cb_t_hi, TILES);
        make_lo(cb_tmp, cb_t_hi, cb_t_lo, 0, TILES);
        cb_pop_front(cb_tmp, TILES);
        cb_wait_front(cb_t_lo, TILES);

        // next = state @ T + J, into the other set.
        const uint32_t s_hi = cb_s_hi[cur], s_lo = cb_s_lo[cur];
        const uint32_t ns32 = cb_s32[1 - cur], ns_hi = cb_s_hi[1 - cur], ns_lo = cb_s_lo[1 - cur];
        for (uint32_t j = 0; j < NT; ++j) {
            tile_regs_acquire();
            reconfig_data_format<SrcOrder::Reverse>(s_hi, cb_t_hi);
            matmul_init(s_hi, cb_t_hi);
            for (uint32_t k = 0; k < NT; ++k) {
                matmul_tiles(s_hi, cb_t_hi, k, k * NT + j, 0);
                matmul_tiles(s_hi, cb_t_lo, k, k * NT + j, 0);
                matmul_tiles(s_lo, cb_t_hi, k, k * NT + j, 0);
            }
            cb_wait_front(cb_j, 1);
            copy_to(cb_j, 0, 1);
            add_binary_tile_init();
            add_binary_tile(0, 1, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_to(0, cb_out);
            pack_to(0, ns32);
            pack_to(0, ns_hi);
            tile_regs_release();
            cb_pop_front(cb_j, 1);
        }
        cb_pop_front(cb_t_hi, TILES);
        cb_pop_front(cb_t_lo, TILES);
        cb_pop_front(cb_s32[cur], NT);
        cb_pop_front(s_hi, NT);
        cb_pop_front(s_lo, NT);
        cb_wait_front(ns32, NT);
        cb_wait_front(ns_hi, NT);
        make_lo(ns32, ns_hi, ns_lo, 0, NT);
        cb_wait_front(ns_lo, NT);
        cur = 1 - cur;
    }
    cb_pop_front(cb_s32[cur], NT);
    cb_pop_front(cb_s_hi[cur], NT);
    cb_pop_front(cb_s_lo[cur], NT);
}

}  // namespace

void kernel_main() {
    constexpr bool has_start = get_compile_time_arg_val(0) != 0;
    const uint32_t frames = get_arg_val<uint32_t>(0);
    const uint32_t units = get_arg_val<uint32_t>(1);

    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_s_hi[0], cb_t_hi, cb_s32[0]);
    for (uint32_t u = 0; u < units; ++u) {
        scan_unit<has_start>(frames);
    }
}
