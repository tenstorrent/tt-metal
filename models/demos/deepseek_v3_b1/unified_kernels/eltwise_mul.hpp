// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "kernel_op_api.hpp"

#if defined(COMPILE_FOR_BRISC)
#include "api/dataflow/dataflow_api.h"
#elif defined(COMPILE_FOR_NCRISC)
#include "api/dataflow/dataflow_api.h"
#elif defined(COMPILE_FOR_TRISC)
#include <cstdint>
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/bcast.h"
#include "../kernel_includes/tt_metal/include/compute_kernel_api/eltwise_mul_scalar.h"
#include "../kernel_includes/tt_metal/include/compute_kernel_api/deepseek_compute_kernel_hw_startup.h"
using namespace ckernel;
#endif

namespace deepseek_b1_ops {

// ============================================================================
// EltwiseMul micro-op
//
// Computes: out[i] = in0[i] * in1[i] for all elements
//
// CB States:
//   NCRISC: No-op (inputs already in CBs from previous operations)
//   BRISC: No-op (no DRAM streaming needed, waits for output)
//   TRISC (Compute):
//     - Waits: in0 (num_tiles), in1 (num_tiles)
//     - Reserves/Pushes: out (num_tiles)
//
// The inputs are expected to be the outputs of previous matmul operations.
// For efficiency, 1x256 outputs (8 tiles of 1x32) can be viewed as
// 1 tile of 16x16 for the multiplication.
// ============================================================================
struct EltwiseMul {
    // ========================================================================
    // Compile-time args structs
    // ========================================================================

    // Reader CTArgs (NCRISC) - empty, no setup needed
    struct ReaderCTArgs {};

    // Writer CTArgs (BRISC) - copies scalar from source CB to working CB
    // enable_scalar: when 0, skip scalar copy (used for simple binary multiply without expert scale)
    // num_experts: number of per-expert iterations. BRISC reads num_experts scalars
    //              starting at scalar_index_offset from cb_scalar_src and pushes one
    //              scalar tile per expert into cb_scalar. cb_scalar_src is popped once
    //              after all scalars are read.
    template <
        uint32_t cb_out_,
        uint32_t num_tiles_,
        uint32_t cb_scalar_,
        uint32_t cb_scalar_src_,
        uint32_t scalar_index_offset_,
        uint32_t enable_scalar_ = 1,
        uint32_t num_experts_ = 1>
    struct WriterCTArgs {
        static constexpr uint32_t cb_out = cb_out_;
        static constexpr uint32_t num_tiles = num_tiles_;
        static constexpr uint32_t cb_scalar = cb_scalar_;
        static constexpr uint32_t cb_scalar_src = cb_scalar_src_;
        static constexpr uint32_t scalar_index_offset = scalar_index_offset_;  // offset into scalar source tensor
        static constexpr bool enable_scalar = enable_scalar_ == 1;
        static constexpr uint32_t num_experts = num_experts_;
    };

    // Compute CTArgs (TRISC)
    // cb_in0_wait: CB to wait on for cb_in0's data (for CB aliasing, e.g., wait on 1x32 CB but read from 16x16 CB)
    // cb_in0_wait_tiles: per-expert number of tiles to wait for on cb_in0_wait
    // cb_scalar: CB containing scalar tile (16x16 format, scalar at [0,0])
    // fp32_dest_acc_en: whether to enable FP32 dest accumulation
    // enable_scalar: when 0, skip scalar multiply (simple binary mul: in0 * in1)
    // num_experts: number of per-expert iterations. Each iteration waits for num_tiles
    //              on cb_in0/cb_in1 and pushes num_tiles to cb_out. Init calls run
    //              once outside the loop; tile_regs lifecycle is per-expert.
    template <
        uint32_t cb_in0_,
        uint32_t cb_in1_,
        uint32_t cb_out_,
        uint32_t num_tiles_,
        uint32_t cb_in0_wait_,
        uint32_t cb_in0_wait_tiles_,
        uint32_t cb_in1_wait_,
        uint32_t cb_in1_wait_tiles_,
        uint32_t cb_scalar_,
        uint32_t fp32_dest_acc_en_ = 0,
        uint32_t enable_scalar_ = 1,
        uint32_t num_experts_ = 1>
    struct ComputeCTArgs {
        static constexpr uint32_t cb_in0 = cb_in0_;
        static constexpr uint32_t cb_in1 = cb_in1_;
        static constexpr uint32_t cb_out = cb_out_;
        static constexpr uint32_t num_tiles = num_tiles_;
        static constexpr uint32_t cb_in0_wait = cb_in0_wait_;
        static constexpr uint32_t cb_in0_wait_tiles = cb_in0_wait_tiles_;
        static constexpr uint32_t cb_in1_wait = cb_in1_wait_;
        static constexpr uint32_t cb_in1_wait_tiles = cb_in1_wait_tiles_;
        static constexpr uint32_t cb_scalar = cb_scalar_;
        static constexpr bool fp32_dest_acc_en = fp32_dest_acc_en_ == 1;
        static constexpr bool enable_scalar = enable_scalar_ == 1;
        static constexpr uint32_t num_experts = num_experts_;
    };

    // ========================================================================
    // Op - the actual operation, templated on CTArgs and IsActiveCore
    // PopInputs: If true (default), pops both input CBs after compute.
    //            Set to false if inputs need to be reused.
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
            // NCRISC: No-op - inputs already in CBs from previous ops
            // ================================================================

#elif defined(COMPILE_FOR_BRISC)
            // ================================================================
            // BRISC: Copy num_experts scalars from source CB to working CB
            // (when scalar enabled). Write only 1 element per expert — hardware
            // will broadcast via BroadcastType::SCALAR.
            // ================================================================
            if constexpr (CTArgs::enable_scalar) {
                // Wait for scalar source CB (set up by NCRISC). Source holds all
                // num_experts scalars in a single tile; each expert's value is at
                // scalar_index_offset + e.
                cb_wait_front(CTArgs::cb_scalar_src, 1);

                uint32_t cb_read_addr = get_read_ptr(CTArgs::cb_scalar_src);
                volatile tt_l1_ptr uint16_t* src_ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(cb_read_addr);

                // cb_scalar holds num_experts pages; get_write_ptr advances each push,
                // so recompute the dst pointer per iteration.
                for (uint32_t e = 0; e < CTArgs::num_experts; e++) {
                    cb_reserve_back(CTArgs::cb_scalar, 1);
                    volatile tt_l1_ptr uint16_t* dst_ptr =
                        reinterpret_cast<volatile tt_l1_ptr uint16_t*>(get_write_ptr(CTArgs::cb_scalar));
                    dst_ptr[0] = src_ptr[CTArgs::scalar_index_offset + e];
                    cb_push_back(CTArgs::cb_scalar, 1);
                }

                // Pop scalar source CB once after all experts are read
                // (populated by mcast, must drain for looping).
                cb_pop_front(CTArgs::cb_scalar_src, 1);
            }

#elif defined(COMPILE_FOR_TRISC)
            // ================================================================
            // TRISC: Element-wise multiplication across all num_experts in a single
            // tile_regs cycle. Same shape as the original single-expert loop, just
            // with total_tiles = num_tiles * num_experts held live in DST.
            //
            // Layout: cb_in0 / cb_in1 / cb_out each hold total_tiles tiles produced
            //   by the upstream matmuls. cb_scalar holds num_experts bf16 scalars
            //   (one 1x32 tile per expert, only [0,0] used via SCALAR broadcast).
            // Indexing: for expert e, tile i → dst slot e*num_tiles + i.
            //
            // Assumes total_tiles <= DST capacity (8 in fp32-accum, 16 in bf16).
            // ================================================================
            constexpr uint32_t num_tiles = CTArgs::num_tiles;
            constexpr uint32_t num_experts = CTArgs::num_experts;
            constexpr uint32_t total_tiles = num_tiles * num_experts;
            static_assert(total_tiles <= 8, "total_tiles must fit in DST (fp32-accum capacity = 8)");

            // Wait for all experts' inputs.
            // cb_in0_wait/cb_in1_wait allow waiting on different CBs (for CB aliasing).
            cb_wait_front(CTArgs::cb_in0_wait, CTArgs::cb_in0_wait_tiles * num_experts);
            cb_wait_front(CTArgs::cb_in1_wait, CTArgs::cb_in1_wait_tiles * num_experts);

            // Reserve output space for all experts.
            cb_reserve_back(CTArgs::cb_out, total_tiles);

            if constexpr (CTArgs::enable_scalar) {
                // ---- 3-way multiply: in0 * scalar -> dest, then dest * in1 -> dest ----
                if constexpr (CTArgs::fp32_dest_acc_en != DST_ACCUM_MODE) {
                    deepseek_compute_kernel_hw_startup<CTArgs::fp32_dest_acc_en>(
                        CTArgs::cb_in0, CTArgs::cb_scalar, CTArgs::cb_out);
                } else {
                    reconfig_data_format<false, true>(CTArgs::cb_in0, CTArgs::cb_scalar);
                    pack_reconfig_data_format<true>(CTArgs::cb_out);
                }
                deepseek_mul_tiles_bcast_scalar_init_short(CTArgs::cb_in0, CTArgs::cb_scalar);

                tile_regs_acquire();

                // Step 1: in0[idx] * scalar[e] -> dest[idx] (idx = e*num_tiles + i)
                // Wait one scalar at a time so math can start as soon as BRISC
                // pushes the e-th page (cb_wait_front is cumulative; no pop mid-loop).
                for (uint32_t e = 0; e < num_experts; e++) {
                    cb_wait_front(CTArgs::cb_scalar, e + 1);
                    for (uint32_t i = 0; i < num_tiles; i++) {
                        uint32_t idx = e * num_tiles + i;
                        deepseek_mul_tiles_bcast_scalar<CTArgs::fp32_dest_acc_en>(
                            CTArgs::cb_in0, CTArgs::cb_scalar, idx, e, idx);
                    }
                }
                // Step 2: dest[idx] *= in1[idx] across all experts (init once, pre-loop)
                deepseek_binary_dest_reuse_tiles_init(CTArgs::cb_in1);
                for (uint32_t idx = 0; idx < total_tiles; idx++) {
                    deepseek_binary_dest_reuse_tiles<CTArgs::fp32_dest_acc_en>(CTArgs::cb_in1, idx, idx);
                }
            } else {
                // ---- Simple binary multiply: in0 * in1 -> dest ----
                reconfig_data_format<false, true>(CTArgs::cb_in0, CTArgs::cb_in1);
                pack_reconfig_data_format<true>(CTArgs::cb_out);
                mul_tiles_init(CTArgs::cb_in0, CTArgs::cb_in1);

                tile_regs_acquire();

                for (uint32_t idx = 0; idx < total_tiles; idx++) {
                    mul_tiles(CTArgs::cb_in0, CTArgs::cb_in1, idx, idx, idx);
                }
            }

            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t idx = 0; idx < total_tiles; idx++) {
                pack_tile(idx, CTArgs::cb_out);
            }
            tile_regs_release();
            cb_push_back(CTArgs::cb_out, total_tiles);

            // Pop inputs if requested.
            // Pop from cb_in0_wait (not cb_in0) since that's where tiles were pushed.
            if constexpr (PopInputs) {
                cb_pop_front(CTArgs::cb_in0_wait, CTArgs::cb_in0_wait_tiles * num_experts);
                cb_pop_front(CTArgs::cb_in1_wait, CTArgs::cb_in1_wait_tiles * num_experts);
                if constexpr (CTArgs::enable_scalar) {
                    cb_pop_front(CTArgs::cb_scalar, num_experts);
                }
            }

            // Reset FP32 accum mode if different from DST_ACCUM_MODE
            if constexpr (CTArgs::enable_scalar && CTArgs::fp32_dest_acc_en != DST_ACCUM_MODE) {
                deepseek_compute_kernel_hw_startup<DST_ACCUM_MODE>(CTArgs::cb_in0, CTArgs::cb_scalar, CTArgs::cb_out);
            }
#endif
        }
    };  // class Op

};  // struct EltwiseMul

}  // namespace deepseek_b1_ops
