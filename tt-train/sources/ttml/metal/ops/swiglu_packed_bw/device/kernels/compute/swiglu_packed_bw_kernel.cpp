// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Fused packed-SwiGLU backward: reads gate|up from the packed forward input, writes dgate|dup into
// the two halves of one dpacked tensor. Per-block math shared with swiglu_elemwise_bw (see header).

#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/matmul.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/transpose.h"
#include "tt-train/sources/ttml/metal/common/swiglu_gate_bw_compute.hpp"

constexpr uint32_t num_blocks_per_core = get_compile_time_arg_val(0);
constexpr uint32_t block_size = get_compile_time_arg_val(1);
constexpr uint32_t scale_bits = get_compile_time_arg_val(2);  // TRANSPOSED_GRAD: the LoRA tail's scale
constexpr uint32_t Ht = get_compile_time_arg_val(3);          // TRANSPOSED_GRAD: token tile rows (column-major blocks)

constexpr uint32_t cb_gate = tt::CBIndex::c_0;   // gate branch (silu'd)
constexpr uint32_t cb_up = tt::CBIndex::c_1;     // up branch
constexpr uint32_t cb_dh = tt::CBIndex::c_2;     // upstream grad dL/dh
constexpr uint32_t cb_dgate = tt::CBIndex::c_3;  // out: grad wrt gate branch
constexpr uint32_t cb_dup = tt::CBIndex::c_4;    // out: grad wrt up branch
constexpr uint32_t cb_sigmoid = tt::CBIndex::c_5;
constexpr uint32_t cb_scratch = tt::CBIndex::c_6;
constexpr uint32_t cb_silu_grad = tt::CBIndex::c_7;
#ifdef TRANSPOSED_GRAD
constexpr uint32_t cb_dh_t = tt::CBIndex::c_8;   // dh^T tiles as read
constexpr uint32_t cb_a = tt::CBIndex::c_9;      // lora_a tiles of the block's columns
constexpr uint32_t cb_tail = tt::CBIndex::c_10;  // the row's s_dh^T tile
constexpr uint32_t cb_sdh = tt::CBIndex::c_11;   // s_dh: the tail tile transposed and scaled
constexpr uint32_t one_bits = 0x3F800000U;

// Builds the token-major dh block (iteration 38): dh(i, j) = (dh^T(j, i))^T + s_dh(i) @ lora_a(j): each tile
// transposed into fp32 DEST, the low-rank term matmul-accumulated onto it (exact, as transpose_split's fused
// pass), one DEST pass per tile, block_size tiles per acquire. Blocks run column-major, so lora_a's tiles stay
// in cb_a for a whole column block (new_col: the previous ones popped, the next ones waited on); s_dh (the row's
// tail tile transposed and scaled) is rebuilt every block (one tile).
inline void build_dh_block(bool new_col, bool first) {
    if (new_col) {
        if (!first) {
            cb_pop_front(cb_a, block_size);
        }
        cb_wait_front(cb_a, block_size);
    }
    cb_wait_front(cb_tail, 1);
    tile_regs_acquire();
    reconfig_data_format_srca(cb_tail);
    transpose_init(cb_tail);
    transpose_tile(cb_tail, 0, 0);
    if constexpr (scale_bits != one_bits) {
        binop_with_scalar_tile_init();
        mul_unary_tile(0, scale_bits);
    }
    tile_regs_commit();
    pack_and_push(0, cb_sdh);
    cb_pop_front(cb_tail, 1);
    cb_wait_front(cb_sdh, 1);
    cb_wait_front(cb_dh_t, block_size);
    tile_regs_acquire();
    reconfig_data_format_srca(cb_dh_t);
    transpose_init(cb_dh_t);
    for (uint32_t k = 0; k < block_size; ++k) {
        transpose_tile(cb_dh_t, k, k);
    }
    reconfig_data_format(cb_sdh, cb_a);
    matmul_init(cb_sdh, cb_a, /*transpose*/ 0);
    for (uint32_t k = 0; k < block_size; ++k) {
        matmul_tiles(cb_sdh, cb_a, 0, k, k);  // accumulates onto the transposed tile in DEST
    }
    tile_regs_commit();
    pack_and_push_block(cb_dh, block_size);
    cb_pop_front(cb_dh_t, block_size);
    cb_pop_front(cb_sdh, 1);
    copy_init(cb_gate);  // the block math starts with copy_tile from cb_gate
}
#endif

void kernel_main() {
    compute_kernel_hw_startup(cb_gate, cb_up, cb_dgate);
    copy_init(cb_gate);
#ifdef TRANSPOSED_GRAD
    const uint32_t start_block = get_arg_val<uint32_t>(0);
#endif

    for (uint32_t block = 0; block < num_blocks_per_core; ++block) {
        cb_wait_front(cb_gate, block_size);
        cb_wait_front(cb_up, block_size);
#ifdef TRANSPOSED_GRAD
        const bool new_col = (block == 0) || ((start_block + block) % Ht == 0);  // column-major blocks
        build_dh_block(new_col, block == 0);
#endif
        cb_wait_front(cb_dh, block_size);

        swiglu_gate_bw_block<
            cb_gate,
            cb_up,
            cb_dh,
            cb_dgate,
            cb_dup,
            cb_sigmoid,
            cb_scratch,
            cb_silu_grad,
            block_size>();

        cb_pop_front(cb_gate, block_size);
        cb_pop_front(cb_up, block_size);
        cb_pop_front(cb_dh, block_size);
    }
#ifdef TRANSPOSED_GRAD
    if (num_blocks_per_core > 0) {
        cb_pop_front(cb_a, block_size);
    }
#endif
}
