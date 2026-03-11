// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// Custom Matmul with Compressed Weights
//
// Uses custom_mm_block init/uninit with a custom _run_ that does
// per-tile format reconfig + variable address increment.
// ct_dim=1 (N=32), kt_dim must be even.
//
// SrcB (in0/activations) auto-advances via HW counters.
// SrcA (in1/weights) address and format set per tile in software loop.

#include "../../../unified_kernels/kernel_utils.hpp"

#if defined(COMPILE_FOR_TRISC)
#include <cstdint>
#include "api/compute/tile_move_copy.h"
#include "api/compute/matmul.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/compute_kernel_api.h"
#include "../../../kernel_includes/tt_metal/include/compute_kernel_api/custom_mm.h"
using namespace ckernel;
#include "../../../kernel_includes/tt_metal/hw/ckernels/blackhole/metal/llk_api/constexpr_args.h"
#include "../../../kernel_includes/tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_custom_mm_compressed_constexpr_compact.h"
#include "../../../kernel_includes/tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_custom_mm_compressed_constexpr_unroll.h"
#include "../../../kernel_includes/tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_custom_mm_compressed_runtime.h"

#ifndef COMPRESSED_MM_IMPL
#define COMPRESSED_MM_IMPL 0
#endif
#elif defined(COMPILE_FOR_NCRISC) || defined(COMPILE_FOR_BRISC)
#include "api/dataflow/dataflow_api.h"
#endif

void kernel_main() {
    constexpr uint32_t cb_in0 = get_named_compile_time_arg_val("cb_in0");
    constexpr uint32_t cb_in1 = get_named_compile_time_arg_val("cb_in1");
    constexpr uint32_t cb_out = get_named_compile_time_arg_val("cb_out");
    constexpr uint32_t num_tiles_k = get_named_compile_time_arg_val("num_tiles_k");
    constexpr uint32_t out_w = get_named_compile_time_arg_val("out_w");

#if defined(COMPILE_FOR_NCRISC)
    constexpr uint32_t cb_in0_num_pages = get_named_compile_time_arg_val("cb_in0_num_pages");
    constexpr uint32_t cb_in1_num_pages = get_named_compile_time_arg_val("cb_in1_num_pages");

    unified_kernels::setup_sharded_buffer(cb_in0, cb_in0_num_pages);
    unified_kernels::setup_sharded_buffer(cb_in1, cb_in1_num_pages);

#elif defined(COMPILE_FOR_BRISC)
    // BRISC: no-op

#elif defined(COMPILE_FOR_TRISC)
    deepseek_compute_kernel_init();

    // Initial HW config (in1→srcA bfp8, in0→srcB bf16)
    reconfig_data_format<false, true>(cb_in1, cb_in0);
    pack_reconfig_data_format<true>(cb_out);

    // Wait for inputs
    cb_wait_front(cb_in0, num_tiles_k);
    cb_wait_front(cb_in1, 1);

    // Get base addresses
    uint32_t addr_in0 = 0;
    uint32_t addr_in1 = 0;
    uint32_t in0_face_r_dim = 0;
    UNPACK(({
        uint32_t in0_id = get_operand_id(cb_in0);
        uint32_t in1_id = get_operand_id(cb_in1);
        addr_in0 = get_local_cb_interface(in0_id).fifo_rd_ptr - 1;
        addr_in1 = get_local_cb_interface(in1_id).fifo_rd_ptr - 1;
        in0_face_r_dim = get_operand_face_r_dim(in0_id);
    }));

    // Reserve output tiles
    cb_reserve_back(cb_out, out_w);

    tile_regs_acquire();

    constexpr uint32_t total_tiles = num_tiles_k * out_w;

#if COMPRESSED_MM_IMPL == 1 || COMPRESSED_MM_IMPL == 2
    // Constexpr paths: use standard custom_mm init (no compressed MOP needed)
    custom_mm_block_init_short<false, true, true>(cb_in0, cb_in1, cb_out, out_w);
    constexpr uint32_t fmt_cta_base = get_named_compile_time_arg_val("fmt_cta_base");
    constexpr uint32_t num_packed = (total_tiles + compressed::TILES_PER_UINT32 - 1) / compressed::TILES_PER_UINT32;
    static constexpr auto fmt_packed = compressed::fill_cta_array<uint32_t, fmt_cta_base, num_packed>();
#if COMPRESSED_MM_IMPL == 2
    compressed::custom_mm_compressed_block_constexpr<num_tiles_k, out_w, num_packed, fmt_packed>(
        addr_in0, addr_in1, in0_face_r_dim, 0);
#else
    compressed::custom_mm_compressed_block_compact<num_tiles_k, out_w, num_packed, fmt_packed>(
        addr_in0, addr_in1, in0_face_r_dim, 0);
#endif
#elif COMPRESSED_MM_IMPL == 0
    // Runtime path: use compressed MOP init (with bfp2 barriers)
    compressed::custom_mm_compressed_block_init_short<true, true>(cb_in0, cb_in1, cb_out, out_w);
    // Runtime loop: read packed pairs from L1 tensor
    constexpr uint32_t fmt_l1_addr = get_named_compile_time_arg_val("fmt_l1_addr");
    compressed::custom_mm_compressed_block_runtime<num_tiles_k, out_w>(
        fmt_l1_addr, addr_in0, addr_in1, in0_face_r_dim, 0);
#else
#error "Invalid COMPRESSED_MM_IMPL: expected 0 (runtime), 1 (constexpr_compact), or 2 (constexpr_unroll)"
#endif

    tile_regs_commit();
    tile_regs_wait();
    for (uint32_t w = 0; w < out_w; w++) {
        pack_tile(w, cb_out, w);
    }
    tile_regs_release();

    custom_mm_block_uninit<true>();

    cb_push_back(cb_out, out_w);
    cb_pop_front(cb_in0, num_tiles_k);

#endif
}
