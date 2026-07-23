// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "kernel_op_api.hpp"
#include "kernel_utils.hpp"

#if defined(COMPILE_FOR_TRISC)
#include "api/compute/eltwise_binary.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/compute_kernel_api.h"
using namespace ckernel;
#endif

namespace deepseek_b1_ops {

// ============================================================================
// EltwiseAddOrCopy micro-op: per-iter elementwise add OR copy, runtime-selected.
//
//   do_add=1: out = in0 + in1  (cb_in0 + cb_in1, both consumed)
//   do_add=0: out = in1        (cb_in1 only, cb_in0 untouched)
//
// Lets the call site keep one CB-flow path while branching on a runtime flag —
// useful when one of the inputs is conditionally produced:
//   - SRAM_DOWN_MERGE: sram_down + shared_down on 112 mcast receiver cores;
//     do_add=1 when n_sram_active>0, else copy shared_down through.
//   - DRAM ELTWISE_ADD (final): down_proj_out + shared_output on 8 gate_proj
//     cores; do_add=1 when n_dram_active>0, else copy shared_output through.
//
// CB layout — base:
//   in0: num_tiles per core, only consumed when do_add=1.
//   in1: num_tiles per core, always consumed.
//   out: num_tiles per core, always produced.
//
// Optional features (DRAM eltwise_add):
//   - Aliased producer for in0: wait/pop on Cb_in0_wait (a different CB ID
//     that shares L1 with cb_in0 at a different tile-shape view, e.g. 1×32
//     producer vs 32×32 consumer). Default is Cb_in0 itself (no alias).
//   - Per-core slice offset on in1: when HasSliceOffset=true, rd_ptr
//     advances by SenderIndex * SliceSizeBytes before compute, restored
//     after. Used when in1 is a replicated tensor and each core reads a
//     different slice. Default HasSliceOffset=false → no offset.
//   - Asymmetric wait counts: cb_in1_wait_tiles (and cb_in0_wait_tiles)
//     default to NumTiles; override when the producer's tile count differs
//     from the compute's NumTiles (e.g., 28 tiles 1×32 from down_proj
//     viewed as ~1 tile 32×32 in compute).
// ============================================================================
struct EltwiseAddOrCopy {
    struct ReaderCTArgs {};
    struct WriterCTArgs {};

    template <
        uint32_t Cb_in0,
        uint32_t Cb_in1,
        uint32_t Cb_out,
        uint32_t NumTiles,
        uint32_t Cb_in0_wait = Cb_in0,          // default: cb_in0 is its own producer
        uint32_t Cb_in0_wait_tiles = NumTiles,  // default: wait count = compute count
        uint32_t Cb_in1_wait_tiles = NumTiles,  // default: wait count = compute count
        bool HasSliceOffset = false,
        uint32_t SenderIndex = 0,     // only read when HasSliceOffset=true
        uint32_t SliceSizeBytes = 0>  // only read when HasSliceOffset=true
    struct ComputeCTArgs {
        static constexpr uint32_t cb_in0 = Cb_in0;
        static constexpr uint32_t cb_in1 = Cb_in1;
        static constexpr uint32_t cb_out = Cb_out;
        static constexpr uint32_t num_tiles = NumTiles;
        static constexpr uint32_t cb_in0_wait = Cb_in0_wait;
        static constexpr uint32_t cb_in0_wait_tiles = Cb_in0_wait_tiles;
        static constexpr uint32_t cb_in1_wait_tiles = Cb_in1_wait_tiles;
        static constexpr bool has_slice_offset = HasSliceOffset;
        static constexpr uint32_t sender_index = SenderIndex;
        static constexpr uint32_t slice_size_bytes = SliceSizeBytes;
    };

    struct ReaderArgs {};
    struct WriterArgs {};

    struct ComputeArgs {
        uint32_t do_add;  // 1 = eltwise_add(in0, in1), 0 = copy(in1)
    };

    using RTArgs = unified_kernels::SelectByRISCV<ReaderArgs, WriterArgs, ComputeArgs>;

    template <typename CTArgs, bool IsActiveCore, bool PopOutput = false>
    class Op {
    public:
        void operator()(const RTArgs& args) {
            if constexpr (IsActiveCore) {
                impl(args);
            }
        }

    private:
        void impl([[maybe_unused]] const RTArgs& args) {
#if defined(COMPILE_FOR_TRISC)
            constexpr uint32_t cb_in0 = CTArgs::cb_in0;
            constexpr uint32_t cb_in1 = CTArgs::cb_in1;
            constexpr uint32_t cb_out = CTArgs::cb_out;
            constexpr uint32_t num_tiles = CTArgs::num_tiles;
            constexpr uint32_t cb_in0_wait = CTArgs::cb_in0_wait;
            constexpr uint32_t cb_in0_wait_tiles = CTArgs::cb_in0_wait_tiles;
            constexpr uint32_t cb_in1_wait_tiles = CTArgs::cb_in1_wait_tiles;

            cb_wait_front(cb_in1, cb_in1_wait_tiles);

            // Optional per-core slice offset on cb_in1 (no-op when has_slice_offset=false).
            uint32_t cb_in1_base_rd_ptr = 0;
            if constexpr (CTArgs::has_slice_offset) {
                constexpr uint32_t offset_aligned = CTArgs::sender_index * CTArgs::slice_size_bytes / L1_ALIGNMENT;
                UNPACK(({
                    cb_in1_base_rd_ptr = unified_kernels::get_local_cb_rd_ptr(cb_in1);
                    unified_kernels::update_local_cb_rd_ptr(cb_in1, cb_in1_base_rd_ptr + offset_aligned);
                }));
            }

            cb_reserve_back(cb_out, num_tiles);

            if (args.do_add) {
                cb_wait_front(cb_in0_wait, cb_in0_wait_tiles);
                reconfig_data_format<false, true>(cb_in0, cb_in1);
                pack_reconfig_data_format<true>(cb_out);
                add_tiles_init(cb_in0, cb_in1);
                tile_regs_acquire();
                for (uint32_t i = 0; i < num_tiles; i++) {
                    add_tiles(cb_in0, cb_in1, i, i, i);
                }
                tile_regs_commit();
                tile_regs_wait();
                for (uint32_t i = 0; i < num_tiles; i++) {
                    pack_tile(i, cb_out);
                }
                tile_regs_release();
                cb_pop_front(cb_in0_wait, cb_in0_wait_tiles);
            } else {
                reconfig_data_format<false, true>(cb_in1, cb_in1);
                pack_reconfig_data_format<true>(cb_out);
                copy_init(cb_in1);
                tile_regs_acquire();
                for (uint32_t i = 0; i < num_tiles; i++) {
                    copy_tile(cb_in1, i, i);
                }
                tile_regs_commit();
                tile_regs_wait();
                for (uint32_t i = 0; i < num_tiles; i++) {
                    pack_tile(i, cb_out);
                }
                tile_regs_release();
            }

            cb_push_back(cb_out, num_tiles);

            // Restore cb_in1 rd_ptr (tensor-backed, no pop). Must come after
            // cb_push_back so the consumer sees the right state.
            if constexpr (CTArgs::has_slice_offset) {
                UNPACK(({ unified_kernels::update_local_cb_rd_ptr(cb_in1, cb_in1_base_rd_ptr); }));
            }

            cb_pop_front(cb_in1, cb_in1_wait_tiles);

            if constexpr (PopOutput) {
                cb_pop_front(cb_out, num_tiles);
            }
#endif
        }
    };
};

}  // namespace deepseek_b1_ops
