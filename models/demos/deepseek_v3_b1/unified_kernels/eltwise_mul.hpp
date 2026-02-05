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
#include "compute_kernel_api.h"
#include "compute_kernel_api/bcast.h"
#include "../kernel_includes/tt_metal/include/compute_kernel_api/eltwise_mul_scalar.h"
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

    // Writer CTArgs (BRISC) - waits for output, optionally copies scalar
    template <
        uint32_t cb_out_,
        uint32_t num_tiles_,
        uint32_t enable_scalar_mul_ = 0,
        uint32_t cb_scalar_ = 0,
        uint32_t cb_scalar_src_ = 0,
        uint32_t scalar_index_offset_ = 0>
    struct WriterCTArgs {
        static constexpr uint32_t cb_out = cb_out_;
        static constexpr uint32_t num_tiles = num_tiles_;
        static constexpr uint32_t enable_scalar_mul = enable_scalar_mul_;
        static constexpr uint32_t cb_scalar = cb_scalar_;
        static constexpr uint32_t cb_scalar_src = cb_scalar_src_;
        static constexpr uint32_t scalar_index_offset = scalar_index_offset_;  // offset into scalar source tensor
    };

    // Compute CTArgs (TRISC)
    // cb_in0_wait: CB to wait on for cb_in0's data (for CB aliasing, e.g., wait on 1x32 CB but read from 16x16 CB)
    // cb_in0_wait_tiles: number of tiles to wait for on cb_in0_wait
    // enable_scalar_mul: if true, multiply result by scalar from cb_scalar after element-wise mul
    // cb_scalar: CB containing scalar tile (16x16 format, scalar at [0,0])
    template <
        uint32_t cb_in0_,
        uint32_t cb_in1_,
        uint32_t cb_out_,
        uint32_t num_tiles_,
        uint32_t cb_in0_wait_,
        uint32_t cb_in0_wait_tiles_,
        uint32_t enable_scalar_mul_ = 0,
        uint32_t cb_scalar_ = 0>
    struct ComputeCTArgs {
        static constexpr uint32_t cb_in0 = cb_in0_;
        static constexpr uint32_t cb_in1 = cb_in1_;
        static constexpr uint32_t cb_out = cb_out_;
        static constexpr uint32_t num_tiles = num_tiles_;
        static constexpr uint32_t cb_in0_wait = cb_in0_wait_;
        static constexpr uint32_t cb_in0_wait_tiles = cb_in0_wait_tiles_;
        static constexpr uint32_t enable_scalar_mul = enable_scalar_mul_;
        static constexpr uint32_t cb_scalar = cb_scalar_;
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
            // BRISC: Copy scalar from source CB to working CB if enabled
            // Write only 1 element - hardware will broadcast via BroadcastType::SCALAR
            // ================================================================
            if constexpr (CTArgs::enable_scalar_mul == 1) {
                // Wait for scalar source CB (set up by NCRISC)
                cb_wait_front(CTArgs::cb_scalar_src, 1);

                // Read scalar value from source CB at index_offset (bfloat16 = 2 bytes)
                uint32_t cb_read_addr = get_read_ptr(CTArgs::cb_scalar_src);
                volatile tt_l1_ptr uint16_t* src_ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(cb_read_addr);
                uint16_t scalar_val = src_ptr[CTArgs::scalar_index_offset];

                // Write one value to destination CB (BroadcastType::SCALAR will broadcast)
                cb_reserve_back(CTArgs::cb_scalar, 1);
                uint32_t cb_write_addr = get_write_ptr(CTArgs::cb_scalar);
                volatile tt_l1_ptr uint16_t* dst_ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(cb_write_addr);
                dst_ptr[0] = scalar_val;
                cb_push_back(CTArgs::cb_scalar, 1);
            }

#elif defined(COMPILE_FOR_TRISC)
            // ================================================================
            // TRISC: Element-wise multiplication with optional scalar multiply
            // ================================================================
            constexpr uint32_t num_tiles = CTArgs::num_tiles;

            binary_op_init_common(CTArgs::cb_in0, CTArgs::cb_in1, CTArgs::cb_out);

            // Initialize eltwise binary for multiplication
            mul_tiles_init(CTArgs::cb_in0, CTArgs::cb_in1);

            // Wait for both inputs
            // cb_in0_wait allows waiting on a different CB (for CB aliasing)
            cb_wait_front(CTArgs::cb_in0_wait, CTArgs::cb_in0_wait_tiles);
            cb_wait_front(CTArgs::cb_in1, num_tiles);

            // Reserve output space
            cb_reserve_back(CTArgs::cb_out, num_tiles);

            // Process tiles - element-wise multiply with optional scalar
            tile_regs_acquire();

            if constexpr (CTArgs::enable_scalar_mul == 1) {
                // Step 1: cb_in0 * scalar -> dest (using scalar broadcast)
                cb_wait_front(CTArgs::cb_scalar, 1);
                mul_tiles_bcast_scalar_init_short_fp32(CTArgs::cb_in0, CTArgs::cb_scalar);
                for (uint32_t i = 0; i < num_tiles; i++) {
                    mul_tiles_bcast_scalar_fp32(CTArgs::cb_in0, CTArgs::cb_scalar, i, 0, i);
                }
                // Step 2: dest * cb_in1 -> dest (using binary dest reuse)
                binary_dest_reuse_tiles_init_fp32(CTArgs::cb_in1);
                for (uint32_t i = 0; i < num_tiles; i++) {
                    binary_dest_reuse_tiles_fp32(CTArgs::cb_in1, i, i);
                }
                tile_regs_commit();
                tile_regs_wait();
                for (uint32_t i = 0; i < num_tiles; i++) {
                    pack_tile(i, CTArgs::cb_out);
                }
                tile_regs_release();
                cb_push_back(CTArgs::cb_out, num_tiles);
            } else {
                // Without scalar: cb_in0 * cb_in1 -> dest
                for (uint32_t i = 0; i < num_tiles; i++) {
                    mul_tiles(CTArgs::cb_in0, CTArgs::cb_in1, i, i, i);
                }
                tile_regs_commit();
                tile_regs_wait();
                for (uint32_t i = 0; i < num_tiles; i++) {
                    pack_tile(i, CTArgs::cb_out);
                }
                tile_regs_release();
                cb_push_back(CTArgs::cb_out, num_tiles);
            }

            // Pop inputs if requested
            // Pop from cb_in0_wait (not cb_in0) since that's where tiles were pushed
            if constexpr (PopInputs) {
                cb_pop_front(CTArgs::cb_in0_wait, CTArgs::cb_in0_wait_tiles);
                cb_pop_front(CTArgs::cb_in1, num_tiles);
            }

            // Pop scalar CB if used
            if constexpr (CTArgs::enable_scalar_mul == 1 && PopInputs) {
                cb_pop_front(CTArgs::cb_scalar, 1);
            }
#endif
        }
    };  // class Op

};  // struct EltwiseMul

}  // namespace deepseek_b1_ops
