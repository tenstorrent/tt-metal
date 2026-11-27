// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "cmath_common.h"
#include "llk_defs.h"
#include "lltt.h"
#include "sfpi.h"

namespace ckernel::sfpu {

// Replay slot and length shared by mac_init() (which records) and calculate_mac()
// (which replays).  lltt::record() captures exactly `length` instructions, so the two
// must agree: an under-count leaves the tail of the body out of the buffer, an
// over-count swallows whatever the caller emits next.
//
// The lengths below are the instructions sfpi emits for mac_init()'s two-row body:
//   6x SFPLOAD, 2x SFPMAD, [2x SFPSTOCHRND for bf16,] 2x SFPSTORE
// If mac_init()'s body changes, re-check them against the generated kernel.
inline constexpr int MAC_REPLAY_SLOT = 0;
template <bool is_fp32_dest_acc_en>
inline constexpr int mac_replay_len = is_fp32_dest_acc_en ? 10 : 12;

template <bool APPROXIMATE, bool is_fp32_dest_acc_en, DataFormat data_format>
inline void mac_init() {
    // Slots from replay_buf_offset up belong to the FPU; the recording must stay below them.
    static_assert(
        MAC_REPLAY_SLOT + mac_replay_len<is_fp32_dest_acc_en> <= ckernel::math::replay_buf_offset,
        "the recorded mac body must fit in the SFPU replay slots");

    // The ternary SFPU dispatch sets addr_mod_base=1 (via set_addr_mod_base()),
    // which shifts instruction addr_mod fields 0..3 to hardware slots 4..7.
    // So SFPSTORE with ADDR_MOD_2 in the instruction actually uses hardware
    // slot 6 at runtime.  We must configure hardware slot 6 (ADDR_MOD_6), not
    // slot 2, for the dest auto-advance to work correctly.
    // (SfpuType::where uses the same slot, but with an increment of 2.)
    addr_mod_t{
        .srca = {.incr = 0},
        .srcb = {.incr = 0},
        .dest = {.incr = 4},
    }
        .set(ADDR_MOD_6);

    // Record the replay sequence once at init time with fixed dest offsets.
    // All callers use tile indices (0, 1, 2, 0), i.e. dst_reg[0], dst_reg[32], dst_reg[64]
    // (sfpi indexes dst_reg in rows; the row stride is 2, so tile n starts at row n * 32).
    //
    // The body covers two rows (dst_reg[n] and dst_reg[n + 1]): a one-row body spends an
    // SFPNOP on the SFPMAD write latency that nothing can fill, whereas the second row's
    // SFPMAD fills it. The second row's store carries the auto-advance, which therefore
    // moves 4 (two rows of stride 2), not 2. Same instructions on the same operands, so
    // the results are bit-identical to the one-row body.
    //
    // The SFPLOAD/SFPSTORE instruction mod describes the DST register layout, which is
    // 32-bit only when fp32 dest accumulation is enabled - it is not a property of the
    // input tensor's data format.  Keying it off data_format would issue FP32-mode
    // accesses against a 16-bit DST whenever the inputs are fp32 but the output dtype
    // (which is what drives fp32_dest_acc_en) is bf16, and 16-bit accesses against a
    // 32-bit DST in the mirrored case.  DataLayout::FSrcB is mod 0, i.e. "whatever
    // format DST is configured with".
    //
    // sfpi defaults the loads and the first row's store to the arch's no-increment addr_mod
    // (ADDR_MOD_3 here), which is what we want; only the second row's store overrides it,
    // with the auto-advancing mod set above.
    constexpr sfpi::DataLayout dst_fmt = is_fp32_dest_acc_en ? sfpi::DataLayout::F32 : sfpi::DataLayout::FSrcB;

    lltt::record<lltt::NoExec>(MAC_REPLAY_SLOT, mac_replay_len<is_fp32_dest_acc_en>);
    sfpi::vFloat a0 = sfpi::dst_reg[0].mode<dst_fmt>();
    sfpi::vFloat b0 = sfpi::dst_reg[32].mode<dst_fmt>();
    sfpi::vFloat c0 = sfpi::dst_reg[64].mode<dst_fmt>();
    sfpi::vFloat a1 = sfpi::dst_reg[1].mode<dst_fmt>();
    sfpi::vFloat b1 = sfpi::dst_reg[33].mode<dst_fmt>();
    sfpi::vFloat c1 = sfpi::dst_reg[65].mode<dst_fmt>();
    sfpi::vFloat result0 = a0 * b0 + c0;
    sfpi::vFloat result1 = a1 * b1 + c1;
    if constexpr (!is_fp32_dest_acc_en) {
        // SFPMAD accumulates in fp32; an SFPSTORE into a 16-bit DST truncates the
        // mantissa, so round to nearest-even on the way down to bf16 instead.
        result0 = sfpi::convert<sfpi::vFloat16b>(result0, sfpi::RoundMode::Nearest);
        result1 = sfpi::convert<sfpi::vFloat16b>(result1, sfpi::RoundMode::Nearest);
    }
    sfpi::dst_reg[0].mode<dst_fmt>() = result0;
    sfpi::dst_reg[1].mode<dst_fmt>(ADDR_MOD_2) = result1;
}

// mac: out = a * b + c, computed in FP32 accumulator via SFPMAD.
//
// The replay sequence is recorded once in mac_init (with fixed dest offsets 0, 64, 128
// matching tile indices 0, 1, 2), covers two rows, and is replayed ITERATIONS / 2 times here.
// ADDR_MOD_2 on the second row's SFPSTORE auto-advances the dest base register by 4
// (two rows of stride 2) per replay, so the next replay's SFPLOADs read the next row
// pair automatically.
// This avoids the explicit sfpi::dst_reg++ used in a plain for-loop, which
// only advances the write counter and not the read counter.
//
// The dst_index_* parameters are accepted for signature compatibility with the
// ternary SFPU dispatch but are NOT used: the dest offsets are baked into the
// recorded replay sequence, so the operand tiles are always (0, 1, 2) and the
// result is always written to tile 0.  Passing anything else has no effect.
//
// Because the replay slots mac_init records into are shared with other SFPU ops, this must be
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

    static_assert(ITERATIONS % 2 == 0, "the recorded body covers two rows");

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d += 2) {
        lltt::replay(MAC_REPLAY_SLOT, mac_replay_len<is_fp32_dest_acc_en>);
    }
}

}  // namespace ckernel::sfpu
