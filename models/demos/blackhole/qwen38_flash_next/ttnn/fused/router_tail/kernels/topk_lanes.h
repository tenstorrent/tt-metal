// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Router tail: the single-core topk LLK's local sort with a pass mask.
//
// `_bitonic_topk_phases_steps` (tt_metal/tt-llk/tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_topk.h) sorts the 64
// rows of the two DST value tiles (with their index tiles) in four passes, (face, col) = (0, 0), (0, 1), (1, 0),
// (1, 1) at DST address offsets 0, 2, 16 and 18.  Each pass is the complete bitonic network for eight of the tile's
// 32 columns (the tokens of the untransposed row tile); the passes touch disjoint DST rows and share nothing but the
// replay buffer.  Skipping a pass leaves its eight columns unsorted and every other column bit for bit as the
// four-pass sort leaves it, so a caller that needs only some tokens runs only their passes: the same instructions on
// the same DST rows in the same order for every live token.
//
// `_bitonic_topk_phases_steps_lanes` is the LLK function copied verbatim plus the guard
// `if (pass_mask & (1u << (face * 2 + col)))` around each pass's phase loop (bit p = pass p = face p / 2, col p % 2).
// tests/test_fused_router_tail_static.py regenerates the copy from the LLK source and pins it, so an LLK change shows
// up as a test failure, not as a silent divergence.  `topk_local_sort_lanes` is `topk_local_sort` with the mask.

#pragma once

#include <cstdint>

#include "api/compute/compute_kernel_api.h"

// The compute source is built once per TRISC; the SFPU LLK (and so every name the copy uses) exists on the math
// TRISC only, exactly as compute_kernel_api.h includes ckernel_sfpu_topk.h under TRISC_MATH.
#ifdef TRISC_MATH
namespace ckernel {
namespace sfpu {

// ---- copied from ckernel_sfpu_topk.h: _bitonic_topk_phases_steps, with the pass guard ----
template <
    bool APPROXIMATION_MODE,
    bool is_fp32_dest_acc_en,
    bool STABLE_SORT = false,
    bool FUSED = false,
    bool RANK_STAMPED = false,
    TopkTieOrder TIE_ORDER = TopkTieOrder::Unset>
inline void _bitonic_topk_phases_steps_lanes(
    const int idir,
    const int i_end_phase,
    const int i_start_phase,
    const int i_end_step,
    const int i_start_step,
    const std::uint32_t pass_mask) {
    // NOTE (stable sort): TIE_ORDER is the GLOBAL sort order, not this call's idir. Callers may run
    // this network with a flipped idir to build bitonic sequences; the tie polarity must not follow it.
    // If more than 1 phase is requested, do all the steps from all phases
    // If 1 phase is requested, use i_start_step/i_end_step parameters

    // UInt16-in-32b-DEST: clear garbage high bits before compare-swap (#50215).
    topk_uint16_clear_value_tiles_high_bits();

    topk_validate_mode_combo<is_fp32_dest_acc_en, STABLE_SORT, FUSED, RANK_STAMPED, TIE_ORDER>();
    // Fused packed keys halve the load/store footprint; replay window bases stay put
    // (slots 4-7 / 12-15 simply go unused in fused mode).
    constexpr int ldst_count = FUSED ? 4 : 8;

    if constexpr (STABLE_SORT) {
        // Establish the lanes-on/flags-true CC entry invariant once; every stable comparator
        // body re-establishes it via its trailing SFPENCC, and the intervening loads/stores/
        // transposes/SFPCONFIG writes preserve CC state.
        TOPK_SFPENCC_ALL_LANES_ON();
    }

    // init the replay buffer for local sort if uninitialized
    bool init_load = (topk_replay_init >= 0) ? true : false;
    bool init_store = (topk_replay_init >= 0) ? true : false;
    bool init_phase;

    std::uint32_t dst_addr_offset = 0;
    for (int face = 0; face < 2; face++) {
        for (int col = 0; col < 2; col++) {
            if (pass_mask & (1u << (face * 2 + col))) {
                bool dir = idir;
                for (int ph = i_start_phase; ph < (i_end_phase + 1); ph++) {
                    init_phase = true;  // init each new phase of local sort in replay buffer

                    TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);
                    switch (ph) {
                        case 0: {
                            for (int d = 0; d < 4; d++) {
                                // Groups of 16 datums being sorted at the same time
                                if (init_load) {
                                    load_replay_buf<Exec>(0, ldst_count, [] {
                                        bitonic_topk_load16<is_fp32_dest_acc_en, FUSED, RANK_STAMPED>(4, 8);
                                    });
                                    init_load = false;
                                } else {
                                    lltt::replay(0, ldst_count);
                                }
                                if constexpr (STABLE_SORT) {
                                    // Stable sequence exceeds the replay window; issue inline.
                                    bitonic_topk_ph0_st1_to_1<STABLE_SORT, TIE_ORDER>();
                                    init_phase = false;
                                } else {
                                    constexpr int replay_count = 4;
                                    if (init_phase) {
                                        load_replay_buf<Exec>(16, replay_count, [] {
                                            bitonic_topk_ph0_st1_to_1<STABLE_SORT, TIE_ORDER>();
                                        });
                                        init_phase = false;
                                    } else {
                                        lltt::replay(16, replay_count);
                                    }
                                }
                                if (init_store) {
                                    load_replay_buf<Exec>(8, ldst_count, [] {
                                        bitonic_topk_store16<is_fp32_dest_acc_en, true, FUSED, RANK_STAMPED>(4, 8);
                                    });
                                    init_store = false;
                                } else {
                                    lltt::replay(8, ldst_count);
                                }
                            }
                            break;
                        }
                        case 1: {
                            // Groups of 16 datums being sorted at the same time
                            for (int d = 0; d < 4; d++) {
                                lltt::replay(0, ldst_count);
                                if constexpr (STABLE_SORT) {
                                    // Stable sequence exceeds the replay window; issue inline.
                                    bitonic_topk_ph1_st2_to_1<STABLE_SORT, TIE_ORDER>();
                                    init_phase = false;
                                } else {
                                    constexpr int replay_count = 6;
                                    if (init_phase) {
                                        load_replay_buf<Exec>(16, replay_count, [] {
                                            bitonic_topk_ph1_st2_to_1<STABLE_SORT, TIE_ORDER>();
                                        });
                                        init_phase = false;
                                    } else {
                                        lltt::replay(16, replay_count);
                                    }
                                }
                                lltt::replay(8, ldst_count);
                            }
                            break;
                        }
                        case 2: {
                            for (int d = 0; d < 4; d++) {
                                lltt::replay(0, ldst_count);
                                if constexpr (STABLE_SORT) {
                                    // Stable sequence exceeds the replay window; issue inline.
                                    bitonic_topk_ph2_st3_to_1<STABLE_SORT, TIE_ORDER>();
                                    init_phase = false;
                                } else {
                                    constexpr int replay_count = 9;
                                    if (init_phase) {
                                        load_replay_buf<Exec>(16, replay_count, [] {
                                            bitonic_topk_ph2_st3_to_1<STABLE_SORT, TIE_ORDER>();
                                        });
                                        init_phase = false;
                                    } else {
                                        lltt::replay(16, replay_count);
                                    }
                                }
                                lltt::replay(8, ldst_count);
                            }
                            break;
                        }
                        case 3:
                            for (int d = 0; d < 4; d++) {
                                lltt::replay(0, ldst_count);
                                bitonic_topk_ph3_st4_to_1<STABLE_SORT, FUSED, TIE_ORDER>(dir, init_phase, 16);
                                lltt::replay(8, ldst_count);
                                dir = !dir;
                            }
                            break;
                        default:
                            std::uint32_t num_steps = ph + 1;
                            std::uint32_t start_step = (i_start_phase == i_end_phase) ? i_start_step : num_steps;
                            std::uint32_t end_step = (i_start_phase == i_end_phase) ? i_end_step : 4;
                            std::uint32_t sorted_seq_length = 1 << num_steps;
                            std::uint32_t datums_compared = 0;
                            std::uint32_t total_datums_to_compare = 64;
                            for (std::uint32_t ss = start_step; ss > end_step; ss--) {
                                // Steps N to 5
                                TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);
                                dir = idir;
                                std::uint32_t dist = (ss == 5) ? 16 : 32;
                                std::uint32_t inner_d =
                                    dist >>
                                    3;  // How many loops to sort the sequence of length (2^ss / 16). Each loop sorts 16
                                datums_compared = 0;
                                std::uint32_t dst_offset = 0;
                                // Record this step's load16/store16 on the first
                                // iteration (which also executes them), replay after.
                                bool init_step_replay = true;
                                while (datums_compared < total_datums_to_compare) {
                                    for (std::uint32_t ii = 0; ii < inner_d; ii++) {
                                        // FUSED elides the four index loads/stores, so the recorded
                                        // sequence is half as long.
                                        constexpr int step_io_replay_count = FUSED ? 4 : 8;
                                        if (init_step_replay) {
                                            load_replay_buf<Exec>(
                                                TOPK_STEP_LOAD_REPLAY_START, step_io_replay_count, [dist] {
                                                    bitonic_topk_load16<is_fp32_dest_acc_en, FUSED, RANK_STAMPED>(
                                                        4, 2 * dist);
                                                });
                                        } else {
                                            lltt::replay(TOPK_STEP_LOAD_REPLAY_START, step_io_replay_count);
                                        }
                                        bitonic_topk_step_N<STABLE_SORT, TIE_ORDER>(dir);
                                        if (init_step_replay) {
                                            load_replay_buf<
                                                Exec>(TOPK_STEP_STORE_REPLAY_START, step_io_replay_count, [dist] {
                                                bitonic_topk_store16<is_fp32_dest_acc_en, false, FUSED, RANK_STAMPED>(
                                                    4, 2 * dist);
                                            });
                                            init_step_replay = false;
                                        } else {
                                            lltt::replay(TOPK_STEP_STORE_REPLAY_START, step_io_replay_count);
                                        }
                                        std::uint32_t dst_inc = 8;
                                        dst_offset += dst_inc;
                                        bool dst_cr = false;
                                        if (ii == (inner_d - 1)) {
                                            dst_cr = true;
                                            dst_inc = 4 * dist;
                                            dst_offset = 2 * dist;
                                        } else if (dst_offset == 16) {
                                            dst_cr = true;
                                            dst_inc = 32;
                                        }
                                        bitonic_topk_inc_x8_dest(dst_inc, dst_cr);
                                        datums_compared += 16;
                                    }
                                    dir = (datums_compared == sorted_seq_length) ? !dir : dir;
                                }
                            }
                            // steps 4 to 1
                            dir = idir;
                            TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);
                            datums_compared = 0;
                            while (datums_compared < total_datums_to_compare) {
                                lltt::replay(0, ldst_count);
                                bitonic_topk_ph3_st4_to_1<STABLE_SORT, FUSED, TIE_ORDER>(dir, init_phase, 16);
                                lltt::replay(8, ldst_count);
                                datums_compared += 16;
                                dir = (datums_compared == sorted_seq_length) ? !dir : dir;
                            }
                    }
                }
            }
            dst_addr_offset += 2;
            set_dst_write_addr(dst_addr_offset);
        }
        dst_addr_offset = 16;
        set_dst_write_addr(dst_addr_offset);
    }
    topk_replay_init = -1;
}
// ---- end of the copy ----

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en, bool STABLE_SORT = false>
inline void calculate_bitonic_topk_phases_steps_lanes(
    int idir, int i_end_phase, int i_start_phase, int i_end_step, int i_start_step, std::uint32_t pass_mask) {
    _bitonic_topk_phases_steps_lanes<APPROXIMATION_MODE, is_fp32_dest_acc_en, STABLE_SORT>(
        idir, i_end_phase, i_start_phase, i_end_step, i_start_step, pass_mask);
}

}  // namespace sfpu
}  // namespace ckernel
#endif  // TRISC_MATH

// topk_local_sort (compute_kernel_api.h) with the pass mask: bit p sorts pass p's eight token columns.  The phase /
// step window (i_start_phase, i_end_step, i_start_step) is the LLK's, defaulted as topk_local_sort defaults it; the
// kernel's timing knob narrows it.
template <bool stable_sort = false, bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void topk_local_sort_lanes(
    uint32_t idst,
    int idir,
    int i_end_phase,
    uint32_t pass_mask,
    int i_start_phase = 0,
    int i_end_step = 0,
    int i_start_step = 0) {
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        is_fp32_dest_acc_en,
        calculate_bitonic_topk_phases_steps_lanes,
        (true /* APPROXIMATE */, is_fp32_dest_acc_en, stable_sort),
        idst,
        VectorMode::RC_custom,
        idir,
        i_end_phase,
        i_start_phase,
        i_end_step,
        i_start_step,
        pass_mask));
}
