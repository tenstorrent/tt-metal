// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Test kernel of copy_dest_rows / copy_dest_row (tests/tt_metal/tt_metal/llk/test_copy_dest_rows.cpp). Inputs c_0 (tile
// A) and c_1 (tile B), one tile each, never popped; output c_16, drained tile by tile by the writer.
//   MODE 0  DEST lane-map probe: A into slot 0, then every sfpi vector v of the slot is overwritten with
//           1000 * v + vConstTileId (tile id = 2 * lane) and the slot is packed: one tile.
//   MODE 1  bands: two reference tiles (A from slot 0, B from slot 1, no op), then per case k in [0, NUM_CASES):
//           in_band = k / 8, out_band = k % 8, the target PAIR (0: A's band into A itself, 1: A's band into B); A and B
//           are reloaded for every case and both slots are packed after the op: two tiles per case.
//   MODE 2  rows: as MODE 1 with from_row = k / 32, to_row = k % 32.

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/copy_dest_rows.h"
#include "api/compute/copy_dest_values.h"
#include "api/compute/pack.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/circular_buffer.h"
#if defined(TRISC_MATH)
#include "llk_math_eltwise_sfpu_common.h"
#include "sfpi.h"
#endif

namespace {
constexpr uint32_t MODE = get_compile_time_arg_val(0);
constexpr uint32_t NUM_CASES = get_compile_time_arg_val(1);
constexpr uint32_t PAIR = get_compile_time_arg_val(2);  // the target slot of MODE 1 and 2: 0 = A itself, 1 = B

constexpr uint32_t cb_a = tt::CBIndex::c_0;
constexpr uint32_t cb_b = tt::CBIndex::c_1;
constexpr uint32_t cb_out = tt::CBIndex::c_16;

#ifdef TRISC_MATH
// Slot 0 <- the identity of its DEST geometry: vector v, lane l holds 1000 * v + 2 * l (vConstTileId = 2 * lane).
inline void probe_fill_slot0() {
    _llk_math_eltwise_sfpu_start_(0);
#pragma GCC unroll 0
    for (uint32_t v = 0; v < 32; v++) {
        // the tile id as a float through the 2^23 trick: bits(2^23) + id reinterpreted is 2^23 + id exactly
        const sfpi::vFloat id = sfpi::as<sfpi::vFloat>(sfpi::vConstTileId + 0x4B000000) - 8388608.0f;
        sfpi::dst_reg[v] = id + static_cast<float>(1000 * v);
    }
    _llk_math_eltwise_sfpu_done_();
}
#endif

inline void load_ab() {
    copy_init(cb_a);
    copy_tile(cb_a, 0, 0);
    copy_init(cb_b);
    copy_tile(cb_b, 0, 1);
}
}  // namespace

void kernel_main() {
    compute_kernel_hw_startup(cb_a, cb_out);
    copy_dest_values_init();
    CircularBuffer a(cb_a), b(cb_b), out(cb_out);
    a.wait_front(1);
    b.wait_front(1);

    if constexpr (MODE == 0) {
        tile_regs_acquire();
        copy_init(cb_a);
        copy_tile(cb_a, 0, 0);
        MATH((probe_fill_slot0()));
        tile_regs_commit();
        tile_regs_wait();
        out.reserve_back(1);
        pack_tile(0, cb_out);
        out.push_back(1);
        tile_regs_release();
        return;
    }

    auto pack_both = [&]() {
        tile_regs_commit();
        tile_regs_wait();
        out.reserve_back(2);
        pack_tile<true>(0, cb_out, 0);
        pack_tile<true>(1, cb_out, 1);
        out.push_back(2);
        tile_regs_release();
    };

    // The reference tiles: A and B as the device holds them in DEST, packed untouched.
    tile_regs_acquire();
    load_ab();
    pack_both();

    for (uint32_t k = 0; k < NUM_CASES; k++) {
        tile_regs_acquire();
        load_ab();
        if constexpr (MODE == 1) {
            copy_dest_rows(PAIR, k % 8, 0, k / 8);
        } else {
            copy_dest_row(PAIR, k % 32, 0, k / 32);
        }
        pack_both();
    }
}
