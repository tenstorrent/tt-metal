// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

// SRAM expert matmul with compressed weights.
//
// B data is pre-loaded in L1 (WIDTH_SHARDED or HEIGHT_SHARDED for K-slicing).
// The Op loops over the index array, skips DRAM experts (is_dram=1), and
// performs matmul for each SRAM expert using per-core fmt tables.

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
#ifdef TRISC_PACK
#include "llk_math_eltwise_unary_sfpu_silu.h"
#endif
#endif

namespace deepseek_b1_ops {

// ============================================================================
// MatmulExpertCompressedSRAM
// ============================================================================
struct MatmulExpertCompressedSRAM {
    template <
        uint32_t cb_in0_,
        uint32_t cb_in1_,
        uint32_t cb_out_,
        uint32_t cb_index_,
        uint32_t num_tiles_k_,
        uint32_t out_w_,
        uint32_t cb_in0_num_pages_,
        uint32_t fmt_l1_addr_,
        uint32_t num_active_experts_,
        uint32_t is_dram_l1_addr_,
        uint32_t table_idx_l1_addr_,
        uint32_t index_l1_addr_,
        uint32_t sram_k_per_core_ = 0,
        uint32_t sram_k_offset_ = 0>
    struct ReaderCTArgs {
        static constexpr uint32_t cb_in0 = cb_in0_;
        static constexpr uint32_t cb_in1 = cb_in1_;
        static constexpr uint32_t cb_out = cb_out_;
        static constexpr uint32_t cb_index = cb_index_;
        static constexpr uint32_t num_tiles_k = num_tiles_k_;
        static constexpr uint32_t out_w = out_w_;
        static constexpr uint32_t cb_in0_num_pages = cb_in0_num_pages_;
        static constexpr uint32_t fmt_l1_addr = fmt_l1_addr_;
        static constexpr uint32_t num_active_experts = num_active_experts_;
        static constexpr uint32_t is_dram_l1_addr = is_dram_l1_addr_;
        static constexpr uint32_t table_idx_l1_addr = table_idx_l1_addr_;
        static constexpr uint32_t index_l1_addr = index_l1_addr_;
        static constexpr uint32_t sram_k_per_core = sram_k_per_core_;
        static constexpr uint32_t sram_k_offset = sram_k_offset_;
    };

    template <
        uint32_t cb_in0_,
        uint32_t cb_in1_,
        uint32_t cb_out_,
        uint32_t cb_index_,
        uint32_t num_tiles_k_,
        uint32_t out_w_,
        uint32_t fmt_l1_addr_,
        uint32_t num_active_experts_,
        uint32_t is_dram_l1_addr_,
        uint32_t table_idx_l1_addr_,
        uint32_t index_l1_addr_,
        uint32_t accum_experts_ = 0,
        uint32_t sram_k_per_core_ = 0,
        uint32_t sram_k_offset_ = 0,
        uint32_t cb_out_sram_ = 0>
    struct ComputeCTArgs {
        static constexpr uint32_t cb_in0 = cb_in0_;
        static constexpr uint32_t cb_in1 = cb_in1_;
        static constexpr uint32_t cb_out = cb_out_;
        static constexpr uint32_t cb_index = cb_index_;
        static constexpr uint32_t num_tiles_k = num_tiles_k_;
        static constexpr uint32_t out_w = out_w_;
        static constexpr uint32_t fmt_l1_addr = fmt_l1_addr_;
        static constexpr uint32_t num_active_experts = num_active_experts_;
        static constexpr uint32_t is_dram_l1_addr = is_dram_l1_addr_;
        static constexpr uint32_t table_idx_l1_addr = table_idx_l1_addr_;
        static constexpr uint32_t index_l1_addr = index_l1_addr_;
        static constexpr bool accum_experts = accum_experts_ != 0;
        static constexpr uint32_t sram_k_per_core = sram_k_per_core_;
        static constexpr uint32_t sram_k_offset = sram_k_offset_;
        static constexpr uint32_t cb_out_sram = cb_out_sram_;
    };

    struct WriterCTArgs {};

    template <typename CTArgs, bool IsActiveCore, bool pop_in0 = true, bool pop_in1 = true, bool pop_index = true>
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
            // No-op: sharded buffer setup is done in kernel_main().

#elif defined(COMPILE_FOR_TRISC)
            constexpr uint32_t cb_in0 = CTArgs::cb_in0;
            constexpr uint32_t cb_in1 = CTArgs::cb_in1;
            // When cb_out_sram is set, SRAM writes to a separate output CB.
            constexpr uint32_t cb_out = (CTArgs::cb_out_sram > 0) ? CTArgs::cb_out_sram : CTArgs::cb_out;
            constexpr uint32_t cb_index = CTArgs::cb_index;
            constexpr uint32_t num_tiles_k = CTArgs::num_tiles_k;
            constexpr uint32_t out_w = CTArgs::out_w;
            constexpr uint32_t fmt_l1_addr_base = CTArgs::fmt_l1_addr;
            // k_for_mm: K tiles for the matmul loop (may be < num_tiles_k when K-sliced).
            // Defaults to num_tiles_k when sram_k_per_core is not set by the caller.
            constexpr uint32_t k_for_mm = CTArgs::sram_k_per_core;
            constexpr uint32_t k_offset = CTArgs::sram_k_offset;
            constexpr uint32_t total_tiles = k_for_mm * out_w;
            constexpr uint32_t num_active_experts = CTArgs::num_active_experts;

            cb_wait_front(cb_in1, 1);
            cb_wait_front(cb_index, 1);

            // Count SRAM experts — all TRISCs agree on the loop bound.
            volatile tt_l1_ptr uint16_t* index_ptr =
                reinterpret_cast<volatile tt_l1_ptr uint16_t*>(CTArgs::index_l1_addr);
            const volatile uint8_t* is_dram_arr = reinterpret_cast<const volatile uint8_t*>(CTArgs::is_dram_l1_addr);
            const volatile uint8_t* table_idx_arr =
                reinterpret_cast<const volatile uint8_t*>(CTArgs::table_idx_l1_addr);

            union SramExpertInfo {
                uint32_t packed;
                struct {
                    uint16_t eid;
                    uint16_t act_tile_offset;
                };
            };

            uint32_t num_sram_experts = 0;
            SramExpertInfo sram_expert_info[num_active_experts];
            for (uint32_t exp_i = 0; exp_i < num_active_experts; exp_i++) {
                uint32_t eid = static_cast<uint32_t>(index_ptr[exp_i]);
                if (is_dram_arr[eid]) {
                    continue;
                }
                SramExpertInfo info;
                info.eid = static_cast<uint16_t>(eid);
                if constexpr (CTArgs::accum_experts) {
                    info.act_tile_offset = static_cast<uint16_t>(exp_i * num_tiles_k);
                }
                sram_expert_info[num_sram_experts] = info;
                num_sram_experts++;
            }

            if (num_sram_experts > 0) {
                reconfig_data_format<false, true>(cb_in1, cb_in0);
                pack_reconfig_data_format<true>(cb_out);
                if constexpr (CTArgs::accum_experts) {
                    cb_wait_front(cb_in0, num_tiles_k * num_active_experts);
                } else {
                    cb_wait_front(cb_in0, num_tiles_k);
                }

                uint32_t addr_in0 = 0;
                uint32_t addr_in0_base = 0;
                uint32_t in0_face_r_dim = 0;
                uint32_t in0_tile_size = 0;

                compressed::custom_mm_compressed_block_init_short<true, out_w, true, true>(cb_in0, cb_in1, cb_out);

                UNPACK(({
                    uint32_t in0_id = get_operand_id(cb_in0);
                    addr_in0_base = get_local_cb_interface(in0_id).fifo_rd_ptr - 1;
                    in0_face_r_dim = get_operand_face_r_dim(in0_id);
                    in0_tile_size = get_local_cb_interface(in0_id).fifo_page_size;
                    addr_in0 = addr_in0_base;
                    // Advance activation pointer by k_offset tiles for K-sliced matmul.
                    if constexpr (k_offset > 0) {
                        addr_in0 += k_offset * in0_tile_size;
                    }
                }));

                if constexpr (CTArgs::accum_experts) {
                    // Accumulate all SRAM experts into the same dest registers.
                    // Each expert uses a distinct activation at its index-tensor position.
                    cb_reserve_back(cb_out, out_w);
                    tile_regs_acquire();

                    for (uint32_t i = 0; i < num_sram_experts; i++) {
                        uint32_t table_idx = static_cast<uint32_t>(table_idx_arr[sram_expert_info[i].eid]);
                        uint32_t fmt_l1_addr = 0;
                        uint32_t addr_in1 = 0;
                        uint32_t addr_in0_expert =
                            addr_in0_base + (sram_expert_info[i].act_tile_offset + k_offset) * in0_tile_size;

                        UNPACK(({
                            auto* fmt_table = reinterpret_cast<volatile uint32_t(*)[total_tiles]>(fmt_l1_addr_base);
                            fmt_l1_addr = reinterpret_cast<uint32_t>(fmt_table[table_idx]);

                            union TileInfo {
                                uint32_t packed;
                                struct {
                                    uint8_t fmt;
                                    uint32_t addr : 24;
                                };
                            };
                            const volatile TileInfo* first_tile =
                                reinterpret_cast<const volatile TileInfo*>(fmt_l1_addr);
                            addr_in1 = first_tile->addr;
                        }));

                        if (i < num_sram_experts - 1) {
                            compressed::custom_mm_compressed_block_runtime<k_for_mm, out_w, false>(
                                fmt_l1_addr, addr_in0_expert, addr_in1, in0_face_r_dim, 0);
                        } else {
                            compressed::custom_mm_compressed_block_runtime<k_for_mm, out_w, true>(
                                fmt_l1_addr, addr_in0_expert, addr_in1, in0_face_r_dim, 0);
                        }
                    }

                    tile_regs_commit();
                    tile_regs_wait();
                    for (uint32_t w = 0; w < out_w; w++) {
                        pack_tile(w, cb_out, w);
                    }
                    tile_regs_release();
                    cb_push_back(cb_out, out_w);
                } else {
                    for (uint32_t i = 0; i < num_sram_experts; i++) {
                        uint32_t table_idx = static_cast<uint32_t>(table_idx_arr[sram_expert_info[i].eid]);
                        uint32_t fmt_l1_addr = 0;
                        uint32_t addr_in1 = 0;

                        UNPACK(({
                            auto* fmt_table = reinterpret_cast<volatile uint32_t(*)[total_tiles]>(fmt_l1_addr_base);
                            fmt_l1_addr = reinterpret_cast<uint32_t>(fmt_table[table_idx]);

                            union TileInfo {
                                uint32_t packed;
                                struct {
                                    uint8_t fmt;
                                    uint32_t addr : 24;
                                };
                            };
                            const volatile TileInfo* first_tile =
                                reinterpret_cast<const volatile TileInfo*>(fmt_l1_addr);
                            addr_in1 = first_tile->addr;
                        }));

                        cb_reserve_back(cb_out, out_w);
                        tile_regs_acquire();

                        compressed::custom_mm_compressed_block_runtime<k_for_mm, out_w>(
                            fmt_l1_addr, addr_in0, addr_in1, in0_face_r_dim, 0);

                        tile_regs_commit();
                        tile_regs_wait();
                        for (uint32_t w = 0; w < out_w; w++) {
                            pack_tile(w, cb_out, w);
                        }
                        tile_regs_release();
                        cb_push_back(cb_out, out_w);
                    }
                }

                custom_mm_block_uninit<true>();
            }

            if constexpr (pop_in1) {
                cb_pop_front(cb_in1, 1);
            }
            if constexpr (pop_in0) {
                if constexpr (CTArgs::accum_experts) {
                    cb_pop_front(cb_in0, num_tiles_k * num_active_experts);
                } else {
                    cb_pop_front(cb_in0, num_tiles_k);
                }
            }
            if constexpr (pop_index) {
                cb_pop_front(cb_index, 1);
            }
#endif
        }
    };
};

}  // namespace deepseek_b1_ops
