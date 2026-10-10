// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// SFPU helpers (math thread): left-shift a 128-float chunk by one float, A'[j] = A[j + 1].
// Lane layout (measured, checks/sfpu_shuffle.py): vector 2v holds the even floats of [64v, 64v + 64),
// vector 2v + 1 the odd ones, as 4 rows x 8 lanes (row r lane m <-> float 16r + 2m (+1 for odd)).
// subvec_shflror1 rotates each row right by one lane; subvec_transp transposes rows across 4 registers.
#pragma once
inline __attribute__((always_inline)) sfpi::vFloat sh_rotl1(
    sfpi::vFloat x) {  // row-wise rotate left by one lane = 7 right rotations
    x = sfpi::vFloat(sfpi::subvec_shflror1(x));
    x = sfpi::vFloat(sfpi::subvec_shflror1(x));
    x = sfpi::vFloat(sfpi::subvec_shflror1(x));
    x = sfpi::vFloat(sfpi::subvec_shflror1(x));
    x = sfpi::vFloat(sfpi::subvec_shflror1(x));
    x = sfpi::vFloat(sfpi::subvec_shflror1(x));
    x = sfpi::vFloat(sfpi::subvec_shflror1(x));
    return x;
}
inline __attribute__((always_inline)) sfpi::vFloat sh_rows_up(
    sfpi::vFloat y, sfpi::vFloat yn) {  // [y.r1, y.r2, y.r3, yn.r0]
    sfpi::vFloat ync = yn, a = 0.0f, b = 0.0f;
    sfpi::subvec_transp(y, yn, a, b);    // y = [y.r0 ..], yn = [y.r1, yn.r1, ..], a = [y.r2 ..], b = [y.r3 ..]
    sfpi::subvec_transp(yn, a, b, ync);  // yn = [y.r1, y.r2, y.r3, ync.r0]
    return yn;
}
// odd vector of the shifted chunk: e rotated left one lane, lane 7 of each row taken from the next row's lane 0
inline __attribute__((always_inline)) sfpi::vFloat sh_odd(sfpi::vFloat e, sfpi::vFloat en) {
    sfpi::vFloat m = e;
    sfpi::vFloat lane0 = sfpi::vFloat(sfpi::subvec_shflshr1(sfpi::vFloat(1.0f)));  // 0 at lane 0, 1 elsewhere
    sfpi::vFloat ru = sh_rows_up(e, en);
    v_if(lane0 == 0.0f) { m = ru; }
    v_endif;
    return sh_rotl1(m);
}
// chunk at vector offset SRC (4 vectors) -> shifted chunk at vector offset DST (relative to the current dst_reg base)
template <uint32_t SRC, uint32_t DST>
inline __attribute__((always_inline)) void sh_chunk() {
    sfpi::vFloat e0 = sfpi::dst_reg[SRC + 0], e1 = sfpi::dst_reg[SRC + 2];
    sfpi::dst_reg[DST + 0] = sfpi::vFloat(sfpi::dst_reg[SRC + 1]);
    sfpi::dst_reg[DST + 2] = sfpi::vFloat(sfpi::dst_reg[SRC + 3]);
    sfpi::dst_reg[DST + 1] = sh_odd(e0, e1);
    sfpi::dst_reg[DST + 3] = sh_odd(e1, e1);  // the last float of the chunk is patched later (next block)
}
