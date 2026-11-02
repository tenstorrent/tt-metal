// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Compute for conv_silu (GDN milestone 1): depthwise causal conv over K taps, then SiLU.
//
// conv[c] = sum_j tap[j][c] * state[j][c].
//
// mul_tiles and add_tiles CLEAR Dst (the LLK passes clear_fp32_dst_acc=true), so a mul+add chain
// cannot accumulate. Measured: an accumulating chain matches the true sum to 4.0e-02 while
// clear-then-add lands 7.7 off, keeping only the last term. What accumulates is a matmul, so each tap
// is applied as a matmul against a DIAGONAL on the right:
//     (state @ diag(tap))[r][c] = state[r][c] * tap[c]     (B[k][c] = tap[c] iff k == c)
// One matmul_tiles per tap into the same Dst slot accumulates the K-term sum with no packing and no
// L1 round trip. The diagonals are precomputed on the host from the static taps, so the kernel never
// builds them.

#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/matmul.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/typecast.h"
#include "api/dataflow/circular_buffer.h"

constexpr uint32_t cb_qkv = 0, cb_state = 1, cb_diag = 2, cb_work = 3;
constexpr uint32_t cb_q = 4, cb_k = 5, cb_v = 6, cb_shift = 7;

namespace {
constexpr uint32_t GROUP = 4;  // fp32 Dst slots per acquire
}

void kernel_main() {
    constexpr uint32_t kd_t = get_compile_time_arg_val(0);
    constexpr uint32_t v_t = get_compile_time_arg_val(1);
    constexpr uint32_t K = get_compile_time_arg_val(2);
    constexpr uint32_t per_core = get_compile_time_arg_val(3);
    // CHAIN_ROUND=1: round the running sum to bf16 after every tap, which is what the model's ttnn chain
    // (mul, then K-1 macs, each writing a bf16 tensor) does, so the op can be bit-exact with it.
    constexpr uint32_t chain_round = get_compile_time_arg_val(4);
    constexpr uint32_t do_silu = get_compile_time_arg_val(5);  // 0 only for the stage-by-stage equivalence test
    constexpr uint32_t F32 = static_cast<uint32_t>(DataFormat::Float32);
    constexpr uint32_t BF16 = static_cast<uint32_t>(DataFormat::Float16_b);

    compute_kernel_hw_startup(cb_state, cb_diag, cb_work);
    silu_tile_init();
    matmul_init(cb_state, cb_diag);

    // The reader pushes state and diagonals; wait for them before reading, or the compute side reads
    // uninitialised L1 (this produced inf in the conv output on the first run).
    CircularBuffer state(cb_state);
    CircularBuffer diag(cb_diag);
    CircularBuffer work(cb_work);
    state.wait_front(K * per_core);
    diag.wait_front(K * per_core);
    // Push one q|k|v group at a time so the writer can drain it while compute runs on: reserving all
    // per_core pages in one call and holding them until the end is what deadlocked iteration 8
    // (writer waits cb_state -> compute waits cb_work -> cycle).
    for (uint32_t i0 = 0; i0 < per_core; i0 += GROUP) {
        const uint32_t nn = (per_core - i0 < GROUP) ? (per_core - i0) : GROUP;
        work.reserve_back(nn);
        tile_regs_acquire();
        if constexpr (chain_round) {
            for (uint32_t j = 0; j < K; j++) {
                for (uint32_t i = 0; i < nn; i++) {
                    matmul_tiles(cb_state, cb_diag, j * per_core + i0 + i, j * per_core + i0 + i, i);
                }
                typecast_tile_init<F32, BF16>();
                for (uint32_t i = 0; i < nn; i++) {
                    typecast_tile<F32, BF16>(i);
                }
            }
            silu_tile_init();
        } else {
            for (uint32_t i = 0; i < nn; i++) {
                const uint32_t c = i0 + i;
                // tap 0 seeds Dst[i] (a matmul clears the slot); taps 1..K-1 accumulate into it.
                for (uint32_t j = 0; j < K; j++) {
                    matmul_tiles(cb_state, cb_diag, j * per_core + c, j * per_core + c, i);
                }
            }
        }
        // SiLU is a MATH-thread (SFPU) op, so it has to run inside the acquire..commit window. Running
        // it after tile_regs_wait executes it on the PACK thread, which is a correctness bug, not just
        // a scheduling one.
        if constexpr (do_silu) {
            for (uint32_t i = 0; i < nn; i++) {
                silu_tile(i);
            }
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t i = 0; i < nn; i++) {
            pack_tile<true>(i, cb_work, i);
        }
        tile_regs_release();
        work.push_back(nn);
    }

    state.pop_front(K * per_core);
    diag.pop_front(K * per_core);
    (void)kd_t;
    (void)v_t;
}
