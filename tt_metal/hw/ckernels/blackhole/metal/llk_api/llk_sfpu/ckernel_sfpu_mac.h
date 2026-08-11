// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_defs.h"
#include "lltt.h"
#include "sfpi.h"

namespace ckernel::sfpu {

// mac: out = a * b + c, computed in FP32 accumulator via SFPMAD.
//
// The replay sequence is recorded once in mac_init (with fixed dest offsets 0, 64, 128
// matching tile indices 0, 1, 2) and replayed ITERATIONS times here.
// ADDR_MOD_6 on SFPSTORE auto-advances the dest base register by 2 rows per
// replay, so the next replay's SFPLOADs read the next row group automatically.
// This avoids the explicit sfpi::dst_reg++ used in a plain for-loop, which
// only advances the write counter and not the read counter.
//
// The dst_index_* parameters are accepted for signature compatibility with the
// ternary SFPU dispatch but are NOT used: the dest offsets are baked into the
// recorded replay sequence, so the operand tiles are always (0, 1, 2) and the
// result is always written to tile 0.  Passing anything else has no effect.
//
// Because the replay slots (0..6) are shared with other SFPU ops, this must be
// replayed while mac_init's recording is still the resident one - i.e. a
// mac_tile call is only valid immediately after mac_tile_init.
template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en, DataFormat data_format, int ITERATIONS>
inline void calculate_mac(
    [[maybe_unused]] const uint dst_index_in0,  // input a  (fixed at 0)
    [[maybe_unused]] const uint dst_index_in1,  // input b  (fixed at 1)
    [[maybe_unused]] const uint dst_index_in2,  // input c  (fixed at 2)
    [[maybe_unused]] const uint dst_index_out) {  // output  (fixed at 0)
    static_assert(
        data_format == DataFormat::Float32 || data_format == DataFormat::Float16_b,
        "Unsupported data format for calculate_mac(). Supported data formats are: Float32, Float16_b.");

    // The replay buffer was recorded in mac_init with fixed offsets (0, 64, 128, 0).
    // All call sites use tile indices (0, 1, 2, 0), so the recorded sequence matches.
    constexpr int num_instrs = is_fp32_dest_acc_en ? 6 : 7;
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        lltt::replay(0, num_instrs);
    }
}

}  // namespace ckernel::sfpu
