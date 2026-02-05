// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "kernel_op_api.hpp"

#if defined(COMPILE_FOR_BRISC)
#include "api/dataflow/dataflow_api.h"
#elif defined(COMPILE_FOR_NCRISC)
#include "api/dataflow/dataflow_api.h"
#elif defined(COMPILE_FOR_TRISC)
#include <cstdint>
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/reconfig_data_format.h"
#include "compute_kernel_api.h"
using namespace ckernel;
#endif

namespace deepseek_b1_ops {

// ============================================================================
// EltwiseAdd micro-op
//
// Computes: out[i] = in0[i] + in1[i] for elements in the indexed slice
//
// Used after down_proj in MoE kernel to add fused_add tensor.
//
// Tensor layout:
//   - in0 (down_proj_out): 1x896 per core, WIDTH_SHARDED, 1x32 tiles
//   - in1 (fused_add): 1x7168 per core (replicated), HEIGHT_SHARDED, 1x32 tiles
//   - out: 1x896 per core, WIDTH_SHARDED, 1x32 tiles
//
// CB view: 32x32 tiles (aliasing from 1x32 tensor tiles)
//   - 896 elements = 28 tiles of 1x32, viewed as ~1 tile of 32x32
//   - Last 128 elements are garbage padding (ignored in validation)
//
// CB States:
//   NCRISC: No-op (fused_add CB setup externally via set_buffer_from_tensor)
//   BRISC: No-op
//   TRISC (Compute):
//     - Updates cb_in1 read pointer to indexed offset (no copy needed!)
//     - Waits: cb_in0 (down_proj_out), cb_in1 (fused_add at offset)
//     - add_tiles on 32x32 view
//     - Reserves/Pushes: cb_out
// ============================================================================

#if defined(COMPILE_FOR_TRISC)
// Helper functions to manipulate CB read pointer (from bmm_large_block_zm_fused_bias_activation_gathered.cpp)
FORCE_INLINE uint32_t get_local_cb_rd_ptr(uint32_t cb_id) {
    LocalCBInterface& local_cb = get_local_cb_interface(cb_id);
    return local_cb.fifo_rd_ptr;
}

FORCE_INLINE void update_local_cb_rd_ptr(uint32_t cb_id, uint32_t val) {
    LocalCBInterface& local_cb = get_local_cb_interface(cb_id);
    local_cb.fifo_rd_ptr = val;
}
#endif

struct EltwiseAdd {
    // ========================================================================
    // Compile-time args structs
    // ========================================================================

    // Reader CTArgs (NCRISC) - empty, CB setup done in kernel_main
    struct ReaderCTArgs {};

    // Writer CTArgs (BRISC) - no-op, indexing done in TRISC
    struct WriterCTArgs {};

    // Compute CTArgs (TRISC)
    // cb_in0: down_proj output (32x32 view)
    // cb_in1: fused_add tensor (32x32 view, read pointer updated to offset)
    // cb_out: output (32x32 view)
    // cb_in0_wait: CB to wait on for cb_in0's data (for CB aliasing)
    // cb_in0_wait_tiles: number of tiles to wait for on cb_in0_wait
    // cb_in1_wait_tiles: number of tiles to wait for on cb_in1 (before offset update)
    // sender_index: per-core index (0-7) to compute offset into fused_add
    // slice_size_bytes: size of slice (896 * 2 = 1792 bytes for bfloat16)
    // use_short_init: if true, skip binary_op_init_common and use add_tiles_init_short
    template <
        uint32_t cb_in0_,
        uint32_t cb_in1_,
        uint32_t cb_out_,
        uint32_t num_tiles_,
        uint32_t cb_in0_wait_,
        uint32_t cb_in0_wait_tiles_,
        uint32_t cb_in1_wait_tiles_,
        uint32_t sender_index_,
        uint32_t slice_size_bytes_,
        bool use_short_init_ = false>
    struct ComputeCTArgs {
        static constexpr uint32_t cb_in0 = cb_in0_;
        static constexpr uint32_t cb_in1 = cb_in1_;
        static constexpr uint32_t cb_out = cb_out_;
        static constexpr uint32_t num_tiles = num_tiles_;
        static constexpr uint32_t cb_in0_wait = cb_in0_wait_;
        static constexpr uint32_t cb_in0_wait_tiles = cb_in0_wait_tiles_;
        static constexpr uint32_t cb_in1_wait_tiles = cb_in1_wait_tiles_;
        static constexpr uint32_t sender_index = sender_index_;
        static constexpr uint32_t slice_size_bytes = slice_size_bytes_;
        static constexpr bool use_short_init = use_short_init_;
    };

    // ========================================================================
    // Op - the actual operation, templated on CTArgs and IsActiveCore
    // PopInputs: If true (default), pops input CBs after compute.
    // ========================================================================
    template <typename CTArgs, bool IsActiveCore, bool PopInputs = true>
    class Op {
    public:
        void operator()() {
            if constexpr (IsActiveCore) {
                impl();
            }
        }

    private:
        void impl() {
#if defined(COMPILE_FOR_NCRISC)
            // ================================================================
            // NCRISC: No-op - CB setup done in kernel_main via sharded buffer API
            // ================================================================

#elif defined(COMPILE_FOR_BRISC)
            // ================================================================
            // BRISC: No-op - indexing done in TRISC via CB read pointer update
            // ================================================================

#elif defined(COMPILE_FOR_TRISC)
            // ================================================================
            // TRISC: Element-wise addition with indexed fused_add
            // ================================================================
            constexpr uint32_t num_tiles = CTArgs::num_tiles;

            if constexpr (CTArgs::use_short_init) {
                // Short init - minimal set needed for CB reconfiguration
                UNPACK((llk_unpack_hw_configure<DST_ACCUM_MODE>(CTArgs::cb_in0, CTArgs::cb_in1)));
                MATH((llk_math_hw_configure<DST_ACCUM_MODE>(CTArgs::cb_in0, CTArgs::cb_in1)));
                PACK((llk_pack_hw_configure<DST_ACCUM_MODE>(CTArgs::cb_out)));
                PACK((llk_pack_init(CTArgs::cb_out)));
            } else {
                binary_op_init_common(CTArgs::cb_in0, CTArgs::cb_in1, CTArgs::cb_out);
            }
            // Initialize eltwise binary for addition
            add_tiles_init(CTArgs::cb_in0, CTArgs::cb_in1);

            // Wait for cb_in0 (down_proj output)
            cb_wait_front(CTArgs::cb_in0_wait, CTArgs::cb_in0_wait_tiles);

            // Wait for cb_in1 (all tiles must be ready before we update read pointer)
            cb_wait_front(CTArgs::cb_in1, CTArgs::cb_in1_wait_tiles);

            // Update cb_in1 read pointer to indexed offset (must be in UNPACK context)
            // CB read pointer is in L1_ALIGNMENT (16 byte) units
            constexpr uint32_t offset_aligned = CTArgs::sender_index * CTArgs::slice_size_bytes / L1_ALIGNMENT;
            UNPACK(({
                uint32_t base_rd_ptr = get_local_cb_rd_ptr(CTArgs::cb_in1);
                update_local_cb_rd_ptr(CTArgs::cb_in1, base_rd_ptr + offset_aligned);
            }));

            // Reserve output space
            cb_reserve_back(CTArgs::cb_out, num_tiles);

            // Process tiles - element-wise add
            tile_regs_acquire();
            for (uint32_t i = 0; i < num_tiles; i++) {
                add_tiles(CTArgs::cb_in0, CTArgs::cb_in1, i, i, i);
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t i = 0; i < num_tiles; i++) {
                pack_tile(i, CTArgs::cb_out);
            }
            tile_regs_release();
            cb_push_back(CTArgs::cb_out, num_tiles);

            // Pop inputs if requested
            if constexpr (PopInputs) {
                cb_pop_front(CTArgs::cb_in0_wait, CTArgs::cb_in0_wait_tiles);
                cb_pop_front(CTArgs::cb_in1, CTArgs::cb_in1_wait_tiles);
            }
#endif
        }
    };  // class Op

};  // struct EltwiseAdd

}  // namespace deepseek_b1_ops
