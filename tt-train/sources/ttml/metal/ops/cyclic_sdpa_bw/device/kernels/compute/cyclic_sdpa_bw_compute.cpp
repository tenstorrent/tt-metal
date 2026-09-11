// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The cyclic backward pass on one core: the schedule's T + 1 block pairs,
// each contributing to one row gradient and two column gradients.
//
// Every operand of a timestep arrives from DRAM and every result goes back to
// DRAM, including the column gradients. The paper keeps (K_j, V_j, dK_j,
// dV_j) resident on the owning core and changes columns twice per core, which
// is where its DRAM-traffic argument comes from; this step deliberately does
// not, and reloads them each timestep instead. The arithmetic and the
// schedule are unaffected, and the residency rules are validated separately
// by the transport probe -- which poisons the column pages in DRAM so that a
// first visit reading them is caught. Restoring residency belongs with the
// relay, which is where it pays.
//
// What that buys is a compute kernel with no state carried between
// timesteps: each timestep is the single block pair of
// cyclic_pair_compute.cpp, which is separately tested, wrapped in a loop.
// Every gradient uses the same three buffers -- a seed from the reader, an
// accumulator, and an output the writer takes -- because
//
//   * sdpa_bw's accumulate path packs without reserving, and only a
//     reserve/push cycle by *this* kernel leaves the packer's write pointer
//     where that expects; a push from the reader does not, since the write
//     pointers are per RISC. The copy from seed to accumulator is what
//     establishes it;
//   * a buffer the reader fills and this kernel accumulates onto would let
//     the writer's cb_wait_front be satisfied by the reader's push, and it
//     would write the value back unaccumulated. The copy out gives the writer
//     a buffer pushed exactly once per result.
//
// Only dQ needs the chip-wide barrier. Each column belongs to exactly one
// core, so dK and dV are never contended.

#include <api/compute/cb_api.h>
#include <api/compute/pack.h>
#include <api/compute/reconfig_data_format.h>
#include <api/compute/reg_api.h>
#include <hostdevcommon/kernel_structs.h>
#include <tensix.h>

#include <cstdint>

#include "api/compute/bcast.h"
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/copy_dest_values.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/exp.h"
#include "api/compute/eltwise_unary/sfpu_split_includes.h"
#include "api/compute/mask.h"
#include "api/compute/matmul.h"
#include "api/compute/reduce.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/transpose_dest.h"
#include "tt-train/sources/ttml/metal/ops/cyclic_sdpa_bw/device/cyclic_schedule.hpp"
#include "tt-train/sources/ttml/metal/ops/sdpa_bw/device/kernels/compute/sdpa_bw_compute_utils.hpp"

namespace {

constexpr uint32_t kCores = get_compile_time_arg_val(0);
constexpr uint32_t qWt = get_compile_time_arg_val(1);
constexpr uint32_t vWt = get_compile_time_arg_val(2);
constexpr uint32_t scaler_bits = get_compile_time_arg_val(3);
constexpr uint32_t minus_one_bits = get_compile_time_arg_val(4);
constexpr uint32_t custom_inf_bits = get_compile_time_arg_val(5);
constexpr uint32_t block_size = get_compile_time_arg_val(6);

// Operands, all per timestep.
constexpr uint32_t cb_query = tt::CBIndex::c_0;
constexpr uint32_t cb_key = tt::CBIndex::c_1;
constexpr uint32_t cb_value = tt::CBIndex::c_2;
constexpr uint32_t cb_grad_output = tt::CBIndex::c_3;
constexpr uint32_t cb_lse = tt::CBIndex::c_4;
constexpr uint32_t cb_u_scalar = tt::CBIndex::c_5;
constexpr uint32_t cb_attn_mask = tt::CBIndex::c_6;

// Intermediates.
constexpr uint32_t cb_attention_weights = tt::CBIndex::c_10;
constexpr uint32_t cb_grad_attn_weights = tt::CBIndex::c_11;
constexpr uint32_t cb_grad_scores = tt::CBIndex::c_12;
constexpr uint32_t cb_grad_scores_transposed = tt::CBIndex::c_13;
constexpr uint32_t cb_attn_weights_transposed = tt::CBIndex::c_14;

// Each gradient: what the reader loaded, the accumulator, the writer's copy.
constexpr uint32_t cb_grad_query_seed = tt::CBIndex::c_15;
constexpr uint32_t cb_grad_query_accum = tt::CBIndex::c_16;
constexpr uint32_t cb_grad_query_out = tt::CBIndex::c_17;
constexpr uint32_t cb_grad_key_seed = tt::CBIndex::c_18;
constexpr uint32_t cb_grad_key_accum = tt::CBIndex::c_19;
constexpr uint32_t cb_grad_key_out = tt::CBIndex::c_20;
constexpr uint32_t cb_grad_value_seed = tt::CBIndex::c_21;
constexpr uint32_t cb_grad_value_accum = tt::CBIndex::c_22;
constexpr uint32_t cb_grad_value_out = tt::CBIndex::c_23;

// dS, dS^T and P^T out of one acquire region. dS must not be written to a CB
// and read back to be transposed: the transposed buffers unpack in Default
// mode, because matmul Src registers do not support Float32 unpack, so a
// copy_tile out of one lands in DST in a layout the 32-bit transpose_dest
// scrambles. copy_dest_values duplicates dS inside DST instead.
void grad_scores_and_transposes() {
    cb_wait_front(cb_grad_attn_weights, onetile);
    cb_wait_front(cb_attention_weights, onetile);
    cb_wait_front(cb_u_scalar, onetile);

    constexpr uint32_t grad_reg = 0;
    constexpr uint32_t attn_reg = 1;
    constexpr uint32_t grad_keep_reg = 2;

    tile_regs_acquire();
    reconfig_data_format(cb_grad_attn_weights, cb_u_scalar);
    sub_bcast_cols_init(cb_grad_attn_weights, cb_u_scalar);
    sub_tiles_bcast_cols(cb_grad_attn_weights, cb_u_scalar, 0, 0, grad_reg);

    reconfig_data_format_srca(cb_grad_attn_weights, cb_attention_weights);
    copy_init(cb_attention_weights);
    copy_tile(cb_attention_weights, 0, attn_reg);

    mul_binary_tile_init();
    mul_binary_tile(grad_reg, attn_reg, grad_reg);
    binop_with_scalar_tile_init();
    mul_unary_tile(grad_reg, scaler_bits);

    copy_dest_values_init();
    copy_dest_values<DataFormat::Float32>(grad_reg, grad_keep_reg);

    transpose_dest_init</* is_32bit */ true>(cb_attention_weights);
    transpose_dest</* is_32bit */ true>(grad_reg);
    transpose_dest</* is_32bit */ true>(attn_reg);

    tile_regs_commit();
    tile_regs_wait();
    cb_reserve_back(cb_grad_scores, onetile);
    cb_reserve_back(cb_grad_scores_transposed, onetile);
    cb_reserve_back(cb_attn_weights_transposed, onetile);
    pack_reconfig_data_format(cb_grad_attn_weights, cb_grad_scores);
    pack_tile(grad_keep_reg, cb_grad_scores);
    pack_reconfig_data_format(cb_grad_scores, cb_grad_scores_transposed);
    pack_tile(grad_reg, cb_grad_scores_transposed);
    pack_reconfig_data_format(cb_grad_scores_transposed, cb_attn_weights_transposed);
    pack_tile(attn_reg, cb_attn_weights_transposed);
    tile_regs_release();
    cb_push_back(cb_grad_scores, onetile);
    cb_push_back(cb_grad_scores_transposed, onetile);
    cb_push_back(cb_attn_weights_transposed, onetile);
}

}  // namespace

void kernel_main() {
    const uint32_t my_core = get_arg_val<uint32_t>(0);

    using ttml::metal::ops::cyclic_sdpa_bw::CyclicSchedule;
    constexpr CyclicSchedule sched(kCores);
    constexpr uint32_t kTimesteps = 2u * kCores + 1u;

    compute_kernel_hw_startup(cb_query, cb_key, cb_attention_weights);
    copy_init(cb_query);
    matmul_init(cb_query, cb_key);
    cb_wait_front(cb_attn_mask, onetile);

    for (uint32_t t = 0; t < kTimesteps; ++t) {
        const auto pair = sched.pair(my_core, t);
        const bool diagonal = pair.i == pair.j;

        cb_wait_front(cb_query, qWt);
        cb_wait_front(cb_key, qWt);
        cb_wait_front(cb_value, vWt);
        cb_wait_front(cb_grad_output, vWt);
        cb_wait_front(cb_lse, onetile);

        // ---- S = Q K^T / sqrt(d), masked when i == j, then P = exp(S - L)
        constexpr uint32_t scores_reg = 0;
        reconfig_data_format(cb_query, cb_key);
        matmul_init(cb_query, cb_key, /* transpose */ 1);
        tile_regs_acquire();
        for (uint32_t k = 0; k < qWt; ++k) {
            matmul_tiles(cb_query, cb_key, k, k, scores_reg);
        }
        if (diagonal) {
            apply_mask_on_reg(scores_reg, cb_attn_mask, scaler_bits, minus_one_bits, custom_inf_bits);
        } else {
            binop_with_scalar_tile_init();
            mul_unary_tile(scores_reg, scaler_bits);
        }
        apply_softmax_statistics_on_dst(scores_reg, cb_lse);
        tile_regs_commit();
        tile_regs_wait();
        cb_reserve_back(cb_attention_weights, onetile);
        pack_reconfig_data_format(cb_attention_weights);
        pack_tile(scores_reg, cb_attention_weights);
        tile_regs_release();
        cb_push_back(cb_attention_weights, onetile);

        // ---- dP = dO V^T, then dS with its transposes
        compute_grad_attn_weights(
            cb_grad_output, cb_value, vWt, cb_grad_attn_weights, cb_attention_weights, scaler_bits);
        grad_scores_and_transposes();

        // ---- dQ_i = (dQ_i from DRAM) + dS K_j
        pack_tiles_to_output(cb_grad_query_seed, cb_grad_query_accum, qWt);
        update_grad_query(
            cb_grad_scores, cb_key, cb_grad_query_accum, qWt, block_size, /* accumulate */ true);
        pack_tiles_to_output(cb_grad_query_accum, cb_grad_query_out, qWt);

        // ---- dV_j = (dV_j from DRAM) + P^T dO_i
        pack_tiles_to_output(cb_grad_value_seed, cb_grad_value_accum, vWt);
        update_grad_value(
            cb_attn_weights_transposed,
            cb_grad_output,
            cb_grad_value_accum,
            vWt,
            block_size,
            /* accumulate */ true);
        pack_tiles_to_output(cb_grad_value_accum, cb_grad_value_out, vWt);

        // ---- dK_j = (dK_j from DRAM) + dS^T Q_i
        pack_tiles_to_output(cb_grad_key_seed, cb_grad_key_accum, qWt);
        // The last two arguments must name what the *previous* operation
        // actually left in the packer and in SrcA, because the reconfigs they
        // drive are conditional and skip when the formats match. The
        // preceding pack_tiles_to_output packed to cb_grad_value_out and read
        // cb_grad_value_accum, both Float32; naming cb_grad_output here
        // instead -- which is what sdpa_bw's own call site names, because
        // there the previous operation is different -- makes the SrcA
        // reconfig look unnecessary and leaves the unpacker in Float32 while
        // the matmul needs Q in bfloat16. dK then comes out wrong while dQ
        // and dV are fine.
        update_grad_key(
            cb_grad_scores_transposed,
            cb_query,
            cb_grad_key_accum,
            qWt,
            block_size,
            /* cb_prev_pack */ cb_grad_value_out,
            /* cb_prev_srca */ cb_grad_value_accum,
            /* accumulate */ true);
        pack_tiles_to_output(cb_grad_key_accum, cb_grad_key_out, qWt);

        cb_pop_front(cb_query, qWt);
        cb_pop_front(cb_key, qWt);
        cb_pop_front(cb_value, vWt);
        cb_pop_front(cb_grad_output, vWt);
        cb_pop_front(cb_lse, onetile);
        cb_pop_front(cb_u_scalar, onetile);
        cb_pop_front(cb_attention_weights, onetile);
        cb_pop_front(cb_grad_attn_weights, onetile);
    }
}
