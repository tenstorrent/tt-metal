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
#include "../../../kernel_includes/tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_unpack_compressed.h"
#elif defined(COMPILE_FOR_NCRISC) || defined(COMPILE_FOR_BRISC)
#include "api/dataflow/dataflow_api.h"
#endif

void kernel_main() {
    constexpr uint32_t cb_in0 = get_named_compile_time_arg_val("cb_in0");
    constexpr uint32_t cb_in1 = get_named_compile_time_arg_val("cb_in1");
    constexpr uint32_t cb_out = get_named_compile_time_arg_val("cb_out");
    constexpr uint32_t num_tiles_k = get_named_compile_time_arg_val("num_tiles_k");

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

    // Initial HW config (in1→srcA bfp8, in0→srcB bf16)
    reconfig_data_format<false, true>(cb_in1, cb_in0);
    pack_reconfig_data_format<true>(cb_out);

    // Init custom_mm (ct_dim=1, split_acc, dense_packing)
    constexpr bool split_acc = true;
    constexpr bool dense_packing = true;
    custom_mm_block_init_short<false, split_acc, dense_packing>(cb_in0, cb_in1, cb_out, 1);

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

    volatile uint8_t* assign_ptr = reinterpret_cast<volatile uint8_t*>(assign_l1_addr);

    // Reserve output (ct_dim=1, single tile)
    cb_reserve_back(cb_out, 1);

    tile_regs_acquire();

    // Single call: custom_mm with per-tile format reconfig
    compressed::custom_mm_compressed_block(assign_ptr, addr_in0, addr_in1, in0_face_r_dim, num_tiles_k, 0);

    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_out, 0);
    tile_regs_release();

    custom_mm_block_uninit<dense_packing>();

    cb_push_back(cb_out, 1);
    cb_pop_front(cb_in0, num_tiles_k);

#endif
}
