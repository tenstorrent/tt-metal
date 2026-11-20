// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// The compute kernel of a merge core in program P2 at 16 token rows: attn_merge_compute.cpp's chain run once per
// sub-pass in which the core's row is live. The data-movement kernel (attn_merge16_dm.cpp) says how many chains in
// cb_ctl before the first one and feeds each chain exactly as attn_merge_dm.cpp feeds the one. What follows is that
// file's description.
//
// The compute kernel of a merge core in program P2 of the attention half (fused/layer_attn/program.py), derived
// from fused/merges/kernels/lse_merge_compute.cpp: a chain of pairwise log-sum-exp merges of partials (O, max, sum)
// of one user and one head group with the flash-attention tail LLK (api/compute/experimental/sdpa.h sdpa_tail):
//
//     m = max(m1, m2);  P1 = exp(scale (m1 - m));  P2 = exp(scale (m2 - m));  s = s1 P1 + s2 P2;  O = O1 P1 + O2 P2
//
//   column stage   the running partial (this chip's own partial at first) with n_col partials of the chips of the
//                  column, in the order the data-movement kernel delivers them (the order they landed in)
//   row stage      then with n_row partials of the chips of the row; the column result is in cb_send (the
//                  data-movement kernel reads it for the row's sender before it delivers the first row partial)
//   sink           then with the sink peer, and the result divided by the sum: ctx [8 heads, 512] -> cb_final
// A partial arrives as [max / sum tile | 16 O tiles] in cb_in: the one buffer is the LLK's max / sum input and its
// O input. What differs from the merges piece's kernel: no landing with two parities and a slot per chip (the
// partials come one by one through cb_in, so a merge starts when its partial is there), and NO MERGE IN PLACE.
//
// A circular buffer has ONE producer. The pack thread counts the tiles it pushed in a counter of its own and
// publishes that count (llk_io_pack.h llk_push_to_brisc), so a buffer the data-movement kernel has pushed into
// (the chip's own partial) and the pack thread then pushes into as well shows the unpack thread fewer tiles than it
// holds, and the next cb_wait_front never returns: this hung the first two-chip run of this program (log
// mk8a_p2_parse_pair_col), and the merges piece's kernel has the same in-place merge. So the own partial stays in
// its buffer (cb_local, written by the data-movement kernel only), and the results of the merges alternate between
// two buffers only the pack thread writes (cb_t0, cb_t1); cb_send is written by the pack thread when there is a
// column stage, by the data-movement kernel when there is none.
//
// Compile-time args: n_col, n_row, cb_local_ms, cb_local_o, cb_send_ms, cb_send_o, cb_sink, cb_final, cb_in,
// cb_t0_ms, cb_t0_o, cb_t1_ms, cb_t1_o, scale (fp32 bits), cb_ctl.

#include <cstdint>

#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_unary/exp.h"
#include "api/compute/experimental/sdpa.h"

constexpr uint32_t n_col = get_compile_time_arg_val(0);
constexpr uint32_t n_row = get_compile_time_arg_val(1);
constexpr uint32_t cb_local_ms = get_compile_time_arg_val(2);
constexpr uint32_t cb_local_o = get_compile_time_arg_val(3);
constexpr uint32_t cb_send_ms = get_compile_time_arg_val(4);
constexpr uint32_t cb_send_o = get_compile_time_arg_val(5);
constexpr uint32_t cb_sink = get_compile_time_arg_val(6);
constexpr uint32_t cb_final = get_compile_time_arg_val(7);
constexpr uint32_t cb_in = get_compile_time_arg_val(8);
constexpr uint32_t cb_t_ms[2] = {get_compile_time_arg_val(9), get_compile_time_arg_val(11)};
constexpr uint32_t cb_t_o[2] = {get_compile_time_arg_val(10), get_compile_time_arg_val(12)};
constexpr uint32_t scale_fp32 = get_compile_time_arg_val(13);
constexpr uint32_t cb_ctl = get_compile_time_arg_val(14);
constexpr bool exp_approx_mode = false;
constexpr uint32_t O_TILES = 16;
// With a 16-bit DEST (the default of the merges piece and of sparse_mla) the SFPU stores the new running sum and the
// two weights into DEST truncated to BF16 while the running O goes through the FPU and the packer: every
// unnormalised merge then loses a little of the sum and O / sum drifts up (+1.8 % after 7 + 3 merges, measured:
// tests/test_p2.py). With a Float32 DEST (fp32_dest_acc_en) sums and weights stay Float32 and only the packed
// partial is rounded to BF16; the LLK then takes blocks of 4 tiles (models/demos/deepseek_v3_b1 flash_mla's
// setting for it) instead of 8. Either way a merge takes and leaves 1 + 16 tiles per partial.
constexpr uint32_t num_blocks = DST_ACCUM_MODE ? 4 : 2;
constexpr uint32_t block_size = O_TILES / num_blocks;
static_assert(block_size * num_blocks == O_TILES, "a partial's O is 16 tiles whatever the block size");
static_assert(cb_local_o == 0, "the hardware startup is seeded from circular buffer 0");

namespace {
// worker (a delivered partial or the sink) merged into prev; the result goes to cur.
template <bool normalize>
ALWI void merge(uint32_t worker, uint32_t prev_ms, uint32_t prev_o, uint32_t cur_ms, uint32_t cur_o) {
    sdpa_tail<exp_approx_mode, normalize, block_size, num_blocks, scale_fp32, VectorMode::C>(
        worker, prev_ms, cur_ms, worker, prev_o, cur_o);
}
}  // namespace

void kernel_main() {
    deepseek_compute_kernel_init();
    cb_wait_front(cb_ctl, 1);
    const uint32_t chains = ckernel::read_tile_value(cb_ctl, 0, 0);
    cb_pop_front(cb_ctl, 1);
    for (uint32_t chain = 0; chain < chains; ++chain) {
        reconfig_full_operand_srca(cb_sink);
        exp_tile_init<exp_approx_mode, scale_fp32>();

        // where the running partial is; without a column stage but with a row stage the own partial is in cb_send
        constexpr bool local_in_send = n_col == 0 && n_row > 0;
        uint32_t run_ms = local_in_send ? cb_send_ms : cb_local_ms;
        uint32_t run_o = local_in_send ? cb_send_o : cb_local_o;
        uint32_t t = 0;
        for (uint32_t i = 0; i < n_col; ++i) {
            const bool to_send = n_row > 0 && i + 1 == n_col;
            const uint32_t cur_ms = to_send ? cb_send_ms : cb_t_ms[t];
            const uint32_t cur_o = to_send ? cb_send_o : cb_t_o[t];
            merge<false>(cb_in, run_ms, run_o, cur_ms, cur_o);
            run_ms = cur_ms;
            run_o = cur_o;
            t ^= to_send ? 0u : 1u;
        }
        for (uint32_t i = 0; i < n_row; ++i) {
            merge<false>(cb_in, run_ms, run_o, cb_t_ms[t], cb_t_o[t]);
            run_ms = cb_t_ms[t];
            run_o = cb_t_o[t];
            t ^= 1u;
        }
        merge<true>(cb_sink, run_ms, run_o, run_ms, cb_final);
    }
}
