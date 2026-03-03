// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// Matmul with Compressed Weights kernel
//
// Computes: output[M, N] = A[M, K] @ decompress(B_compressed[K, N])
//
// A (in0): bf16 HEIGHT_SHARDED, [M, K] per core → srcB in matmul
// B (in1): compressed data, WIDTH_SHARDED → srcA in matmul
// assignment: 2-bit packed format indices in L1
// output: bf16 WIDTH_SHARDED, [M, N] per core
//
// NOTE: In matmul LLK, in0→srcB and in1→srcA (swapped from eltwise).
// So compressed weights (in1) need reconfig on srcA, not srcB.
//
// CB in1 constexpr is bfp8 (for valid HW init).
// reconfig_unpack_srca() switches to actual per-tile format at runtime.

#include "../../../unified_kernels/kernel_utils.hpp"

#if defined(COMPILE_FOR_TRISC)
#include <cstdint>
#include "api/compute/tile_move_copy.h"
#include "api/compute/matmul.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/compute_kernel_api.h"
using namespace ckernel;
#include "../../../kernel_includes/tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_unpack_compressed.h"
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
    constexpr uint32_t assign_l1_addr = get_named_compile_time_arg_val("assign_l1_addr");

    deepseek_compute_kernel_init();

    // Initial HW config: in1 (srcA) constexpr is bfp8
    // Note: matmul swaps: in0→srcB, in1→srcA
    // Note: matmul swaps srcA/srcB — in1(weights)→srcA, in0(activations)→srcB
    reconfig_data_format<false, true>(cb_in1, cb_in0);
    pack_reconfig_data_format<true>(cb_out);

    // Init matmul mode
    mm_init_short(cb_in0, cb_in1);

    // Wait for inputs
    cb_wait_front(cb_in0, num_tiles_k);
    cb_wait_front(cb_in1, 1);  // 1 page covering whole compressed shard

    // Read assignment
    volatile uint8_t* assign_ptr = reinterpret_cast<volatile uint8_t*>(assign_l1_addr);

    // Get base addresses from CB interfaces
    // in0 (A, srcB): standard tiled tensor
    // in1 (B compressed, srcA): one big page, walk manually
    uint32_t addr_in0 = 0;
    uint32_t in0_page_size = 0;
    uint32_t addr_in1 = 0;
    bool partial_face_a = false;  // from operandB (in1/weights)
    bool partial_face_b = false;  // from operandA (in0/activations) — note the swap!
    UNPACK(({
        uint32_t in0_id = get_operand_id(cb_in0);
        uint32_t in1_id = get_operand_id(cb_in1);
        addr_in0 = get_local_cb_interface(in0_id).fifo_rd_ptr - 1;
        in0_page_size = get_local_cb_interface(in0_id).fifo_page_size;
        addr_in1 = get_local_cb_interface(in1_id).fifo_rd_ptr - 1;
        partial_face_a = get_operand_partial_face(in1_id);
        partial_face_b = get_operand_partial_face(in0_id);
    }));

    // Reserve output tiles
    cb_reserve_back(cb_out, out_w);

    tile_regs_acquire();

    // B tiles are row-major: (k=0,n=0), (k=0,n=1), ..., (k=1,n=0), ...
    // Iterate K (outer) x N (inner) to walk compressed data linearly.
    uint32_t b_addr = addr_in1;
    uint32_t tile_idx = 0;

    for (uint32_t k = 0; k < num_tiles_k; k++) {
        for (uint32_t w = 0; w < out_w; w++, tile_idx++) {
            uint32_t fmt = compressed::get_tile_format(assign_ptr, tile_idx);

            if (fmt == compressed::FMT_BFP0) {
                continue;
            }

            compressed::reconfig_unpack_srca(fmt);

            uint32_t tile_size = compressed::TILE_SIZES[fmt] >> cb_addr_shift;

            UNPACK((_llk_unpack_AB_matmul_(
                addr_in0,       // base_address_a (in0 → srcB)
                b_addr,         // base_address_b (in1 → srcA)
                k,              // tile_index_a
                0,              // tile_index_b (b_addr already points to tile)
                in0_page_size,  // tile_size_a
                tile_size,      // tile_size_b
                partial_face_a,
                partial_face_b)));
            MATH((llk_math_matmul<MATH_FIDELITY, MM_THROTTLE>(w)));

            b_addr += tile_size;
        }
    }

    tile_regs_commit();
    tile_regs_wait();
    for (uint32_t w = 0; w < out_w; w++) {
        pack_tile(w, cb_out, w);
    }
    tile_regs_release();

    cb_push_back(cb_out, out_w);

    // Pop inputs
    cb_pop_front(cb_in0, num_tiles_k);

#endif
}
