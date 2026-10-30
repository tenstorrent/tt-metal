// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// GDN decode step 1a, K2's gtest kernel: runs one pass of fused_recurrent_gdn_math.hpp, or one whole token, REPS
// times on one core with the CB ids of the contract (section 3), and stamps the MATH wall clock into L1 (runtime
// arg 0): [0] start, [1] end of the REPS loop; [2..5] cycles accumulated per pass (constants, read, vector, update)
// in the token variant.
//
// Compile-time args: VARIANT, REPS, KT, VT, EMIT_STATE.
//   1  read_pass       in: qk16 (c_0), S (c_6)                        out: r16 -> c_16
//   2  update_pass     in: kT16 (c_1), u16 (c_13), S (c_6), ecol32 (c_9)   out: S_next -> c_16 [, state_out -> c_17]
//   3  constants_pass  in: gcol32 (c_4), qk16 (c_0), qcol32 (c_2), ones32 (c_7); craw16 = c_11   out: ecol32 -> c_16,
//                      ccol32 -> c_17
//   4  decode_token    in: c_0..c_7; internal c_9..c_13; S_cur = c_6, S_next = c_14 (recycled per repetition)
//                      out: out16 -> c_16, state_out -> c_17
// The inputs are never popped, so every repetition sees the same tiles; the outputs of all repetitions stay in their
// CBs (sized by the host) so the pack thread never waits for the writer.

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/dataflow/circular_buffer.h"
#if defined(TRISC_MATH)
#include "internal/tt-1xx/risc_common.h"
#endif

#include "../../../../../ttnn/cpp/ttnn/operations/transformer/fused_recurrent_gated_delta_rule/device/kernels/compute/fused_recurrent_gdn_math.hpp"

namespace {
constexpr uint32_t VARIANT = get_compile_time_arg_val(0);
constexpr uint32_t REPS = get_compile_time_arg_val(1);
constexpr uint32_t KT = get_compile_time_arg_val(2);
constexpr uint32_t VT = get_compile_time_arg_val(3);
constexpr bool EMIT_STATE = get_compile_time_arg_val(4) != 0;
// 1: compute_kernel_hw_startup<SrcOrder::Reverse>(in0, in1, out) (SrcA <- in1, SrcB <- in0, the matmul's mapping);
// 0: the regular order today's op kernel uses. The passes must give the same bits either way.
constexpr bool STARTUP_REVERSE = get_compile_time_arg_val(5) != 0;
// Token variant: 1 = start up on the utd S (c_6) as today's op kernel does, 0 = on its plain alias (c_19).
constexpr bool STARTUP_ON_UTD = get_compile_time_arg_val(6) != 0;
constexpr uint32_t KV = KT * VT;

template <uint32_t In0, uint32_t In1, uint32_t Out>
inline void startup() {
    if constexpr (STARTUP_REVERSE) {
        compute_kernel_hw_startup<SrcOrder::Reverse>(In0, In1, Out);
    } else {
        compute_kernel_hw_startup(In0, In1, Out);
    }
}

namespace cb {
constexpr uint32_t qk16 = 0, kT16 = 1, qcol32 = 2, v16 = 3, gcol32 = 4, bcol16 = 5, S = 6, ones32 = 7, zero32 = 8;
constexpr uint32_t ecol32 = 9, r16 = 10, craw16 = 11, ccol32 = 12, u16 = 13, s2 = 14, s3 = 15, out16 = 16;
constexpr uint32_t state_out = 17;
// The vector pass's constants tile, and the plain (non-utd) aliases of S and s2 over the same L1 memory for the
// read-pass matmul (the ids of the op's fused_recurrent_gdn_cb.hpp).
constexpr uint32_t ebc16 = 19, S_mm = 20, s2_mm = 21;
}  // namespace cb

#ifdef TRISC_MATH
volatile tt_l1_ptr uint32_t* g_stats = nullptr;
uint32_t g_last = 0;
#endif

inline void stamp(uint32_t slot) {
#ifdef TRISC_MATH
    g_stats[slot] = get_timestamp_32b();
#endif
}

inline void pass_clock_start() {
#ifdef TRISC_MATH
    g_last = get_timestamp_32b();
#endif
}

struct StampHook {
    void operator()(uint32_t pass_done) const {
#ifdef TRISC_MATH
        const uint32_t now = get_timestamp_32b();
        g_stats[2 + pass_done] += now - g_last;
        g_last = now;
#endif
    }
};
}  // namespace

void kernel_main() {
#ifdef TRISC_MATH
    g_stats = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_arg_val<uint32_t>(0));
    for (uint32_t i = 2; i < 8; i++) {
        g_stats[i] = 0;
    }
#endif

    if constexpr (VARIANT == 1 || VARIANT == 17) {
        // 17: S arrives utd under c_6 and the matmul reads it through the plain alias c_20 (the token's form).
        constexpr uint32_t s_mm = (VARIANT == 17) ? cb::S_mm : cb::S;
        startup<cb::qk16, s_mm, cb::out16>();
        stamp(0);
        for (uint32_t r = 0; r < REPS; r++) {
            gdn_decode::read_pass<KT, VT>(cb::qk16, s_mm, cb::out16);
        }
        stamp(1);
    } else if constexpr (VARIANT == 2) {
        startup<cb::kT16, cb::u16, cb::out16>();
        stamp(0);
        for (uint32_t r = 0; r < REPS; r++) {
            gdn_decode::update_pass<KT, VT>(
                cb::kT16,
                cb::u16,
                cb::S,
                cb::ecol32,
                gdn_decode::StateCbs{cb::out16, cb::out16},
                cb::state_out,
                EMIT_STATE);
        }
        stamp(1);
    } else if constexpr (VARIANT == 5 || VARIANT == 6) {
        // Update-pass cost decomposition (experiment, same loop shape as update_pass): 5 = seed + rank-1 + pack (no
        // decay), 6 = seed + decay + pack (no rank-1). S_next -> c_16.
        startup<cb::kT16, cb::u16, cb::out16>();
        CircularBuffer kT(cb::kT16), u(cb::u16), s(cb::S), ecol(cb::ecol32), sn(cb::out16);
        kT.wait_front(KT);
        u.wait_front(VT);
        s.wait_front(KV);
        ecol.wait_front(1);
        stamp(0);
        for (uint32_t r = 0; r < REPS; r++) {
            sn.reserve_back(KV);
            reconfig_full_operand<SrcOrder::Reverse>(cb::kT16, cb::u16);
            for (uint32_t i = 0; i < KT; i++) {
                for (uint32_t j0 = 0; j0 < VT; j0 += gdn_decode::kUpdateGroup) {
                    const uint32_t nn = (VT - j0 < gdn_decode::kUpdateGroup) ? (VT - j0) : gdn_decode::kUpdateGroup;
                    tile_regs_acquire();
                    copy_init(cb::S);
                    for (uint32_t jj = 0; jj < nn; jj++) {
                        copy_tile(cb::S, i * VT + j0 + jj, jj);
                    }
                    if constexpr (VARIANT == 6) {
                        copy_tile(cb::ecol32, 0, gdn_decode::kUpdateGroup);
                        for (uint32_t jj = 0; jj < nn; jj++) {
                            gdn_decode_sfpu::mul_col_bcast<32>(jj, gdn_decode::kUpdateGroup);
                        }
                    } else {
                        matmul_init(cb::kT16, cb::u16, 0);
                        for (uint32_t jj = 0; jj < nn; jj++) {
                            matmul_tiles(cb::kT16, cb::u16, i, j0 + jj, jj);
                        }
                    }
                    tile_regs_commit();
                    tile_regs_wait();
                    pack_init(cb::out16);
                    for (uint32_t jj = 0; jj < nn; jj++) {
                        pack_tile<true>(jj, cb::out16, i * VT + j0 + jj);
                    }
                    tile_regs_release();
                }
            }
            sn.push_back(KV);
        }
        stamp(1);
    } else if constexpr (VARIANT == 3) {
        // ecol32 (c_9) is internal (the pass reads it back for ebc16); ebc16 goes to the output CB c_16.
        startup<cb::qk16, cb::qcol32, cb::out16>();
        CircularBuffer ecol(cb::ecol32);
        stamp(0);
        for (uint32_t r = 0; r < REPS; r++) {
            gdn_decode::constants_pass<KT>(
                cb::gcol32, cb::qk16, cb::qcol32, cb::ones32, cb::bcol16, cb::ecol32, cb::craw16, cb::out16);
            ecol.wait_front(1);
            ecol.pop_front(1);
        }
        stamp(1);
    } else if constexpr (VARIANT == 4 || VARIANT == 7) {
        // 7: the fused read + vector form (read_vector_pass), otherwise the two-pass form.
        startup<cb::qk16, STARTUP_ON_UTD ? cb::S : cb::S_mm, cb::out16>();
        constexpr gdn_decode::TokenCbs cbs{
            .qk16 = cb::qk16,
            .kT16 = cb::kT16,
            .qcol32 = cb::qcol32,
            .v16 = cb::v16,
            .gcol32 = cb::gcol32,
            .bcol16 = cb::bcol16,
            .ones32 = cb::ones32,
            .ecol32 = cb::ecol32,
            .r16 = cb::r16,
            .craw16 = cb::craw16,
            .ccol32 = cb::ccol32,
            .u16 = cb::u16,
            .out16 = cb::out16,
            .state_out = cb::state_out,
            .ebc16 = cb::ebc16};
        constexpr gdn_decode::StateCbs s_cur{cb::S, cb::S_mm}, s_next{cb::s2, cb::s2_mm};
        CircularBuffer sn(cb::s2), sn_mm(cb::s2_mm);
        stamp(0);
        for (uint32_t r = 0; r < REPS; r++) {
            pass_clock_start();
            gdn_decode::decode_token<KT, VT, (VARIANT == 7)>(
                cbs, s_cur, s_next, EMIT_STATE, /*pop_inputs=*/false, StampHook{});
            // Every repetition starts from the same S (c_6): recycle the ring slot the token wrote (both ids).
            sn.wait_front(KV);
            sn.pop_front(KV);
            sn_mm.wait_front(KV);
            sn_mm.pop_front(KV);
        }
        stamp(1);
    }
}
