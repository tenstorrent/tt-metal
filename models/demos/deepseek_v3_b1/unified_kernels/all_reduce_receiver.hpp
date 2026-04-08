// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "kernel_op_api.hpp"
#include "kernel_utils.hpp"

#if defined(COMPILE_FOR_NCRISC)
#include "api/dataflow/dataflow_api.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"
#include "tt_metal/fabric/hw/inc/edm_fabric/routing_plane_connection_manager.hpp"
#include "tt_metal/fabric/hw/inc/noc_addr.h"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"
#include <cstdint>
#include <utility>
#include "tt_metal/fabric/hw/inc/linear/api.h"
#include "tt_metal/fabric/hw/inc/api_common.h"

using namespace tt::tt_fabric::linear::experimental;
using namespace tt::tt_fabric::common::experimental;

#elif defined(COMPILE_FOR_TRISC)
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary.h"
#endif

namespace deepseek_b1_ops {

// ============================================================================
// CCL All-Reduce Receiver Operations
//
// Receiver core functionality:
// - NCRISC (Reader): Waits for remote data, pushes to compute
// - BRISC: No-op (writer runs on sender core)
// - TRISC (Compute): Performs reduction (local + remote → output)
// ============================================================================
struct AllReduceReceiver {
    // ========================================================================
    // Compile-time args structs
    // ========================================================================

    // Reader CTArgs (NCRISC)
    template <
        uint32_t cbIn1,
        uint32_t cbIn2,
        uint32_t remoteSenderNocX,
        uint32_t remoteSenderNocY,
        uint32_t numStandardTiles,
        uint32_t cbResidual,
        uint32_t hasResidual,
        uint32_t skipLocalPush = 0>  // Skip cb_reserve/push on cb_in2 when fused with gather
    struct ReaderCTArgs {
        static constexpr uint32_t cb_in1 = cbIn1;
        static constexpr uint32_t cb_in2 = cbIn2;
        static constexpr uint32_t remote_sender_noc_x = remoteSenderNocX;
        static constexpr uint32_t remote_sender_noc_y = remoteSenderNocY;
        static constexpr uint32_t num_standard_tiles = numStandardTiles;
        static constexpr uint32_t cb_residual = cbResidual;
        static constexpr bool has_residual = hasResidual;
        static constexpr bool skip_local_push = skipLocalPush;
    };

    // Compute CTArgs (TRISC)
    template <
        uint32_t cbIn0,
        uint32_t cbIn1,
        uint32_t cbOut0,
        uint32_t cbResidual,
        uint32_t hasResidual,
        uint32_t numTiles>
    struct ComputeCTArgs {
        static constexpr uint32_t cb_in0 = cbIn0;
        static constexpr uint32_t cb_in1 = cbIn1;
        static constexpr uint32_t cb_out0 = cbOut0;
        static constexpr uint32_t cb_residual = cbResidual;
        static constexpr bool has_residual = hasResidual;
        static constexpr uint32_t num_tiles = numTiles;
    };

    // ========================================================================
    // Runtime args structs
    // ========================================================================

    // NCRISC reader args
    struct ReaderArgs {
        uint32_t sender_semaphore_addr;
    };

    // BRISC writer args (no-op for receiver)
    struct WriterArgs {};

    // TRISC compute args
    struct ComputeArgs {};

    // Select args type based on current RISC
    using RTArgs = unified_kernels::SelectByRISCV<ReaderArgs, WriterArgs, ComputeArgs>;

    // ========================================================================
    // Op - unified receiver operation
    //
    // ReaderCT: compile-time args for NCRISC reader
    // ComputeCT: compile-time args for TRISC compute
    // ========================================================================
    template <typename CTArgs>
    class Op {
    public:
        void operator()(const RTArgs& args) { impl(args); }

    private:
#if defined(COMPILE_FOR_TRISC)
        template <bool AcquireRegs>
        static FORCE_INLINE void batched_add(uint32_t cb_a, uint32_t cb_b, uint32_t cb_out, uint32_t num_tiles) {
            constexpr uint32_t max_dst_tiles = 4;
            uint32_t num_batches = (num_tiles + max_dst_tiles - 1) / max_dst_tiles;

            // For safety
            MATH((t6_semaphore_wait_on_max<p_stall::STALL_MATH>(semaphore::FPU_SFPU)));

            cb_wait_front(cb_a, num_tiles);
            cb_wait_front(cb_b, num_tiles);
            cb_reserve_back(cb_out, num_tiles);

            if constexpr (AcquireRegs) {
                tile_regs_acquire();
            }
            for (uint32_t batch = 0; batch < num_batches; ++batch) {
                uint32_t start_tile = batch * max_dst_tiles;
                uint32_t batch_size =
                    (start_tile + max_dst_tiles <= num_tiles) ? max_dst_tiles : (num_tiles - start_tile);
                if (batch == num_batches - 1) {
                    tile_regs_wait();
                } else {
                    PACK(t6_semaphore_wait_on_zero<p_stall::STALL_PACK>(semaphore::FPU_SFPU));
                }
                for (uint32_t i = 0; i < batch_size; ++i) {
                    add_tiles(cb_a, cb_b, start_tile + i, start_tile + i, start_tile + i);
                    pack_tile(start_tile + i, cb_out);
                }
                if (batch == num_batches - 1) {
                    tile_regs_commit();
                } else {
                    PACK(t6_semaphore_get<p_stall::PACK>(semaphore::FPU_SFPU));
                    MATH((t6_semaphore_post<p_stall::MATH>(semaphore::FPU_SFPU)));
                }
            }
            tile_regs_release();

            cb_pop_front(cb_a, num_tiles);
            cb_pop_front(cb_b, num_tiles);
            cb_push_back(cb_out, num_tiles);
        }
#endif

        void impl([[maybe_unused]] const RTArgs& args) {
#if defined(COMPILE_FOR_NCRISC)
            // ================================================================
            // NCRISC (Reader) - waits for remote data, pushes to compute
            // ================================================================
            // Push local and residual tiles to compute immediately (they're ready)
            // Skip local push if data is already in CB (e.g., from preceding gather operation)
            if constexpr (CTArgs::has_residual) {
                cb_reserve_back(CTArgs::cb_residual, CTArgs::num_standard_tiles);
                cb_push_back(CTArgs::cb_residual, CTArgs::num_standard_tiles);
            }
            if constexpr (!CTArgs::skip_local_push) {
                cb_reserve_back(CTArgs::cb_in2, CTArgs::num_standard_tiles);
                cb_push_back(CTArgs::cb_in2, CTArgs::num_standard_tiles);
            }
            // Wait for remote sender to signal data has been written to intermediate tensor
            auto local_semaphore_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(args.sender_semaphore_addr);
            noc_semaphore_wait(local_semaphore_ptr, 1);
            noc_semaphore_set(local_semaphore_ptr, 0);

            // Remote data is now ready, push to compute
            cb_reserve_back(CTArgs::cb_in1, CTArgs::num_standard_tiles);
            cb_push_back(CTArgs::cb_in1, CTArgs::num_standard_tiles);

#elif defined(COMPILE_FOR_BRISC)
            // ================================================================
            // BRISC - No-op (writer runs on sender core)
            // ================================================================

#elif defined(COMPILE_FOR_TRISC)
            // ================================================================
            // TRISC (Compute) - performs reduction: local + remote → output
            // ================================================================
            reconfig_data_format<false, true>(CTArgs::cb_in0, CTArgs::cb_in1);
            pack_reconfig_data_format<true>(CTArgs::cb_out0);

            // TODO: Fix this to account for actual dst size
            static_assert(CTArgs::num_tiles <= 8, "num_tiles must be less than or equal to 8");

            if constexpr (CTArgs::has_residual) {
                copy_tile_to_dst_init_short(CTArgs::cb_residual);
                cb_wait_front(CTArgs::cb_residual, CTArgs::num_tiles);
                tile_regs_acquire();
                for (uint32_t i = 0; i < CTArgs::num_tiles; i++) {
                    copy_tile(CTArgs::cb_residual, i, i);
                }
                cb_pop_front(CTArgs::cb_residual, CTArgs::num_tiles);
            }
            constexpr bool acquire_regs = CTArgs::has_residual ? false : true;
            add_tiles_init(CTArgs::cb_in0, CTArgs::cb_in1, CTArgs::has_residual);
            batched_add<acquire_regs>(CTArgs::cb_in0, CTArgs::cb_in1, CTArgs::cb_out0, CTArgs::num_tiles);
#endif
        }
    };

};  // struct AllReduceReceiver

}  // namespace deepseek_b1_ops
