// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

// SRAM matmul with multi-expert compressed weights.
//
// Each core holds a [num_experts, num_tiles] packed-pair fmt table in L1.
// At runtime, expert_idx (read from cb_index, a uint16 HEIGHT_SHARDED tensor)
// selects the row, so addresses are stored explicitly — no per-expert stride calc.
//
// NCRISC: sets up sharded CBs (in0, in1, index).
// BRISC:  no-op.
// TRISC:  reads expert_idx from cb_index, offsets fmt pointer, runs compressed matmul.

#include "kernel_utils.hpp"

#if defined(COMPILE_FOR_NCRISC) || defined(COMPILE_FOR_BRISC)
#include "api/dataflow/dataflow_api.h"
#elif defined(COMPILE_FOR_TRISC)
#include <cstdint>
#include "api/compute/tile_move_copy.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/compute_kernel_api.h"
#include "../kernel_includes/tt_metal/include/compute_kernel_api/custom_mm.h"
#include "../kernel_includes/tt_metal/include/compute_kernel_api/deepseek_compute_kernel_hw_startup.h"
using namespace ckernel;
#include "../kernel_includes/tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_custom_mm_compressed_runtime.h"
#endif

namespace deepseek_b1_ops {

struct MatmulExpertCompressed {
    // ========================================================================
    // Compile-time args structs
    // ========================================================================

    template <
        uint32_t cb_in0_,
        uint32_t cb_in1_,
        uint32_t cb_out_,
        uint32_t cb_index_,
        uint32_t num_tiles_k_,
        uint32_t out_w_,
        uint32_t cb_in0_num_pages_,
        uint32_t fmt_l1_addr_>
    struct ReaderCTArgs {
        static constexpr uint32_t cb_in0 = cb_in0_;
        static constexpr uint32_t cb_in1 = cb_in1_;
        static constexpr uint32_t cb_out = cb_out_;
        static constexpr uint32_t cb_index = cb_index_;
        static constexpr uint32_t num_tiles_k = num_tiles_k_;
        static constexpr uint32_t out_w = out_w_;
        static constexpr uint32_t cb_in0_num_pages = cb_in0_num_pages_;
        static constexpr uint32_t fmt_l1_addr = fmt_l1_addr_;
    };

    template <
        uint32_t cb_in0_,
        uint32_t cb_in1_,
        uint32_t cb_out_,
        uint32_t cb_index_,
        uint32_t num_tiles_k_,
        uint32_t out_w_,
        uint32_t fmt_l1_addr_>
    struct ComputeCTArgs {
        static constexpr uint32_t cb_in0 = cb_in0_;
        static constexpr uint32_t cb_in1 = cb_in1_;
        static constexpr uint32_t cb_out = cb_out_;
        static constexpr uint32_t cb_index = cb_index_;
        static constexpr uint32_t num_tiles_k = num_tiles_k_;
        static constexpr uint32_t out_w = out_w_;
        static constexpr uint32_t fmt_l1_addr = fmt_l1_addr_;
    };

    // ========================================================================
    // Per-RISC operations
    // ========================================================================

#if defined(COMPILE_FOR_NCRISC)
    template <typename CTArgs>
    static void reader() {
        unified_kernels::setup_sharded_buffer(CTArgs::cb_in0, CTArgs::cb_in0_num_pages);
        unified_kernels::setup_sharded_buffer(CTArgs::cb_in1, 1);
        unified_kernels::setup_sharded_buffer(CTArgs::cb_index, 1);
    }

#elif defined(COMPILE_FOR_TRISC)
    template <typename CTArgs>
    static void compute() {
        constexpr uint32_t cb_in0 = CTArgs::cb_in0;
        constexpr uint32_t cb_in1 = CTArgs::cb_in1;
        constexpr uint32_t cb_out = CTArgs::cb_out;
        constexpr uint32_t cb_index = CTArgs::cb_index;
        constexpr uint32_t num_tiles_k = CTArgs::num_tiles_k;
        constexpr uint32_t out_w = CTArgs::out_w;
        constexpr uint32_t fmt_l1_addr_base = CTArgs::fmt_l1_addr;
        constexpr uint32_t total_tiles = num_tiles_k * out_w;

        reconfig_data_format<false, true>(cb_in1, cb_in0);
        pack_reconfig_data_format<true>(cb_out);

        cb_wait_front(cb_in0, num_tiles_k);
        cb_wait_front(cb_in1, 1);

        uint32_t addr_in0 = 0;
        uint32_t addr_in1 = 0;
        uint32_t in0_face_r_dim = 0;

        cb_reserve_back(cb_out, out_w);
        tile_regs_acquire();

        compressed::custom_mm_compressed_block_init_short<true, out_w, true, true>(cb_in0, cb_in1, cb_out);

        // Read expert_idx from index CB, select fmt table row, and derive addr_in1
        // from the selected expert's first tile entry (absolute THCON-shifted address).
        cb_wait_front(cb_index, 1);
        uint32_t fmt_l1_addr = fmt_l1_addr_base;  // safe default for TRISC1/2
        UNPACK(({
            uint32_t in0_id = get_operand_id(cb_in0);
            addr_in0 = get_local_cb_interface(in0_id).fifo_rd_ptr - 1;
            in0_face_r_dim = get_operand_face_r_dim(in0_id);

            uint32_t index_id = get_operand_id(cb_index);
            uint32_t addr_index = get_local_cb_interface(index_id).fifo_rd_ptr << cb_addr_shift;
            volatile tt_l1_ptr uint16_t* index_ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(addr_index);
            uint32_t expert_idx = static_cast<uint32_t>(index_ptr[0]);

            auto* fmt_table = reinterpret_cast<volatile uint32_t(*)[total_tiles]>(fmt_l1_addr_base);
            fmt_l1_addr = reinterpret_cast<uint32_t>(fmt_table[expert_idx]);

            // Derive addr_in1 from the selected expert's first fmt entry.
            union TileInfo {
                uint32_t packed;
                struct {
                    uint8_t fmt;
                    uint32_t addr : 24;
                };
            };
            const volatile TileInfo* first_tile = reinterpret_cast<const volatile TileInfo*>(fmt_l1_addr);
            addr_in1 = first_tile->addr;
        }));

        compressed::custom_mm_compressed_block_runtime<num_tiles_k, out_w>(
            fmt_l1_addr, addr_in0, addr_in1, in0_face_r_dim, 0);

        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t w = 0; w < out_w; w++) {
            pack_tile(w, cb_out, w);
        }
        tile_regs_release();

        custom_mm_block_uninit<true>();

        cb_push_back(cb_out, out_w);
        cb_pop_front(cb_in0, num_tiles_k);
    }
#endif
};

}  // namespace deepseek_b1_ops
