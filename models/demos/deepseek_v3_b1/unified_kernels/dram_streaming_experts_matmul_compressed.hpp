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
#include "api/compute/tile_move_copy.h"
#include "../kernel_includes/tt_metal/include/compute_kernel_api/custom_mm.h"
#include "../kernel_includes/tt_metal/include/compute_kernel_api/deepseek_compute_kernel_hw_startup.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/pack.h"
#include "../kernel_includes/tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_custom_mm_compressed_common.h"
#include "../kernel_includes/tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_custom_mm_compressed_runtime.h"
#endif

namespace deepseek_b1_ops {

// ============================================================================
// DRAMStreamingExpertsMatmulCompressed micro-op
//
// Multi-expert version of DRAMStreamingMatmulCompressed.
//
// Computes: for each selected expert e:
//   output[e, 1, N] += in0[1, K] @ decompress(in1_e[K, N])
// where in1_e is expert e's compressed weight shard in DRAM.
//
// Expert addressing uses a per-expert byte offset table (cb_expert_offsets)
// instead of the fixed-stride expert_idx * expert_size_bytes used by
// DRAMStreamingExpertsMatmul. This allows experts to have different packed
// sizes (F1: no DRAM padding).
//
// Metadata and format tables are stacked per expert: the L1 buffers hold
//   selected_experts_k * per_core_N * num_subblocks_k  block-size entries and
//   selected_experts_k * subblock_k * num_subblocks_k * per_core_N  tile-info entries.
//
// Multi-core-per-bank mode (cores_per_bank > 1) is NOT supported in this op;
// each core reads its own per_core_N columns from each expert's shard.
// ============================================================================
struct DRAMStreamingExpertsMatmulCompressed {
    // ========================================================================
    // Compile-time args structs
    // ========================================================================

    // Reader CTArgs (NCRISC)
    template <
        uint32_t cb_in1_,
        uint32_t cb_out_,
        uint32_t in1_tensor_addr_,  // buffer_address() of expert-0's data tensor (base)
        uint32_t subblock_k_,
        uint32_t per_core_n_,
        uint32_t out_num_tiles_,
        uint32_t num_subblocks_k_,
        uint32_t bank_id_,
        uint32_t vc_,
        uint32_t meta_l1_addr_,  // L1 addr of stacked block-size metadata (all experts)
        uint32_t cb_in1_size_bytes_,
        uint32_t noc_max_page_size_,
        uint32_t dram_start_offset_,  // within-expert column offset (cores_per_bank partitioning)
        uint32_t pipeline_sem_id_,
        uint32_t next_core_noc_x_,
        uint32_t next_core_noc_y_,
        uint32_t selected_experts_k_,      // number of selected experts to process
        uint32_t expert_offsets_l1_addr_,  // L1 addr of per-expert byte offset table (uint32 x num_experts)
        uint32_t enable_indexing_,         // 1 if expert indices come from cb_index; 0 if sequential
        uint32_t cb_index_,                // CB holding selected expert indices (uint16 array)
        uint32_t index_offset_,            // offset into cb_index (or base expert index when hardcoded)
        uint32_t use_hardcoded_expert_index_ = 0>
    struct ReaderCTArgs {
        static constexpr uint32_t cb_in1 = cb_in1_;
        static constexpr uint32_t cb_out = cb_out_;
        static constexpr uint32_t in1_tensor_addr = in1_tensor_addr_;
        static constexpr uint32_t subblock_k = subblock_k_;
        static constexpr uint32_t per_core_n = per_core_n_;
        static constexpr uint32_t out_num_tiles = out_num_tiles_;
        static constexpr uint32_t num_subblocks_k = num_subblocks_k_;
        static constexpr uint32_t bank_id = bank_id_;
        static constexpr uint32_t vc = vc_;
        static constexpr uint32_t meta_l1_addr = meta_l1_addr_;
        static constexpr uint32_t cb_in1_size_bytes = cb_in1_size_bytes_;
        static constexpr uint32_t noc_max_page_size = noc_max_page_size_;
        static constexpr uint32_t dram_start_offset = dram_start_offset_;
        static constexpr uint32_t pipeline_sem_id = pipeline_sem_id_;
        static constexpr uint32_t next_core_noc_x = next_core_noc_x_;
        static constexpr uint32_t next_core_noc_y = next_core_noc_y_;
        static constexpr uint32_t selected_experts_k = selected_experts_k_;
        static constexpr uint32_t expert_offsets_l1_addr = expert_offsets_l1_addr_;
        static constexpr bool enable_indexing = enable_indexing_ == 1;
        static constexpr uint32_t cb_index = cb_index_;
        static constexpr uint32_t index_offset = index_offset_;
        static constexpr bool use_hardcoded_expert_index = use_hardcoded_expert_index_ == 1;
    };

    // Writer CTArgs (BRISC) — no-op
    struct WriterCTArgs {};

    // Compute CTArgs (TRISC) — identical to DRAMStreamingMatmulCompressed except
    // num_subblocks_k covers all experts: total = selected_experts_k * per_expert_subblocks_k
    template <
        uint32_t cb_in0_,
        uint32_t cb_in1_,
        uint32_t cb_out_,
        uint32_t subblock_k_,
        uint32_t per_core_n_,
        uint32_t num_subblocks_k_per_expert_,
        uint32_t selected_experts_k_,
        uint32_t fmt_l1_addr_>
    struct ComputeCTArgs {
        static constexpr uint32_t cb_in0 = cb_in0_;
        static constexpr uint32_t cb_in1 = cb_in1_;
        static constexpr uint32_t cb_out = cb_out_;
        static constexpr uint32_t subblock_k = subblock_k_;
        static constexpr uint32_t per_core_n = per_core_n_;
        static constexpr uint32_t num_subblocks_k_per_expert = num_subblocks_k_per_expert_;
        static constexpr uint32_t selected_experts_k = selected_experts_k_;
        static constexpr uint32_t fmt_l1_addr = fmt_l1_addr_;
    };

    // ========================================================================
    // Op
    // ========================================================================
    template <typename CTArgs, bool IsActiveCore, bool PopIn0 = true>
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
            // NCRISC: For each selected expert, stream its compressed weight
            // tiles from DRAM using variable-size subblocks.
            //
            // Expert addressing:
            //   expert_base_offset = expert_offsets_l1_addr[expert_idx]  (F1: no padding)
            //   DRAM read address  = in1_tensor_addr + expert_base_offset + dram_start_offset
            //
            // The offset table is a pre-populated L1 tensor holding uint32 byte
            // offsets for each expert, indexed by expert_idx.
            // ================================================================
            constexpr uint32_t per_expert_iterations = CTArgs::num_subblocks_k * CTArgs::per_core_n;
            constexpr uint32_t dram_bank_id = CTArgs::bank_id;
            constexpr uint32_t vc = CTArgs::vc;
            constexpr uint32_t max_page_size = CTArgs::noc_max_page_size;

            // --- Read selected expert indices ---
            uint32_t expert_indices[CTArgs::selected_experts_k];
            if constexpr (CTArgs::enable_indexing) {
                cb_wait_front(CTArgs::cb_index, 1);
                for (uint32_t e = 0; e < CTArgs::selected_experts_k; e++) {
                    if constexpr (CTArgs::use_hardcoded_expert_index) {
                        expert_indices[e] = CTArgs::index_offset + e;
                    } else {
                        volatile tt_l1_ptr uint16_t* index_ptr =
                            reinterpret_cast<volatile tt_l1_ptr uint16_t*>(get_read_ptr(CTArgs::cb_index));
                        expert_indices[e] = static_cast<uint32_t>(index_ptr[CTArgs::index_offset + e]);
                    }
                }
            } else {
                for (uint32_t e = 0; e < CTArgs::selected_experts_k; e++) {
                    expert_indices[e] = e;
                }
            }

            // --- Load per-expert byte offsets from pre-populated L1 table ---
            volatile tt_l1_ptr uint32_t* offset_table_ptr =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(CTArgs::expert_offsets_l1_addr);

            uint32_t expert_base_offsets[CTArgs::selected_experts_k];
            for (uint32_t e = 0; e < CTArgs::selected_experts_k; e++) {
                expert_base_offsets[e] = offset_table_ptr[expert_indices[e]];
            }

            // --- Setup metadata pointer (block sizes, stacked across all experts) ---
            const volatile uint32_t* meta_ptr = reinterpret_cast<const volatile uint32_t*>(CTArgs::meta_l1_addr);

            // --- DRAM base address ---
            uint64_t in1_base_addr = get_noc_addr_from_bank_id<true>(dram_bank_id, CTArgs::in1_tensor_addr);

            // Triple-buffering state
            constexpr uint32_t num_buffers = 3;
            constexpr uint32_t extra_blocks_in_flight = 1;
            constexpr uint32_t max_subblock_bytes = CTArgs::cb_in1_size_bytes / num_buffers;

            // Physical base/end addresses of cb_in1 — constant across all experts,
            // used only for manual write-pointer wrap-around detection.
            // Read BEFORE any cb_reserve_back so we get the true physical base.
            uint32_t cb_in1_base = get_write_ptr(CTArgs::cb_in1);
            uint32_t cb_in1_end = cb_in1_base + CTArgs::cb_in1_size_bytes;

            reset_noc_trid_barrier_counter(NOC_CLEAR_OUTSTANDING_REQ_MASK, noc_index);

            uint32_t global_iter = 0;  // indexes into meta_ptr across all experts

            // How many slots to reserve at the start of each expert.  Clamped to
            // per_expert_iterations so we never pre-reserve more slots than there
            // are blocks to fill — preventing dangling reserved slots that would
            // block the next expert from acquiring its pipeline window.
            constexpr uint32_t initial_reserve_blocks = (per_expert_iterations >= extra_blocks_in_flight + 1)
                                                            ? (extra_blocks_in_flight + 1)
                                                            : per_expert_iterations;

            for (uint32_t e = 0; e < CTArgs::selected_experts_k; e++) {
                // Reset triple-buffer state at the start of each expert.
                uint32_t num_free_blocks_in_buffer = num_buffers;
                uint32_t curr_block_trid = 1;
                uint32_t block_trid_to_wait = 1;

                // Acquire the initial pipeline window for this expert.
                // Using get_write_ptr after the reserve gives the correct L1 write
                // address even when the CB has wrapped around from a previous expert.
                cb_reserve_back(CTArgs::cb_in1, CTArgs::subblock_k * initial_reserve_blocks);
                uint32_t l1_write_addr_in1 = get_write_ptr(CTArgs::cb_in1);

                // DRAM read address = base + expert_base_offset + within-expert column offset
                uint32_t dram_read_offset = expert_base_offsets[e] + CTArgs::dram_start_offset;

                for (uint32_t iter_in_expert = 0; iter_in_expert < per_expert_iterations; iter_in_expert++) {
                    uint32_t block_size = meta_ptr[global_iter];
                    uint32_t slot_start = l1_write_addr_in1;

                    noc_async_read_set_trid(curr_block_trid);

                    uint32_t remaining = block_size;
                    while (remaining > 0) {
                        uint32_t chunk = (remaining > max_page_size) ? max_page_size : remaining;
                        noc_async_read_one_packet_set_state<true>(in1_base_addr, chunk, vc);
                        noc_async_read_one_packet_with_state_with_trid(
                            in1_base_addr, dram_read_offset, l1_write_addr_in1, curr_block_trid);
                        dram_read_offset += chunk;
                        l1_write_addr_in1 += chunk;
                        remaining -= chunk;
                    }

                    l1_write_addr_in1 = slot_start + max_subblock_bytes;

                    if (num_free_blocks_in_buffer == num_buffers - extra_blocks_in_flight) {
                        noc_async_read_barrier_with_trid(block_trid_to_wait);
                        cb_push_back(CTArgs::cb_in1, CTArgs::subblock_k);
                        block_trid_to_wait = block_trid_to_wait == num_buffers ? 1 : (block_trid_to_wait + 1);
                        // Only pre-reserve future pipeline slots when there are more
                        // iterations to fill them.  On the last iteration of an expert,
                        // skip the reserve: the trailing flush below will push the final
                        // in-flight block, leaving no dangling reserved slots for the
                        // next expert's initial cb_reserve_back to trip over.
                        if (iter_in_expert < per_expert_iterations - 1) {
                            cb_reserve_back(CTArgs::cb_in1, CTArgs::subblock_k * (extra_blocks_in_flight + 1));
                        }
                    } else {
                        num_free_blocks_in_buffer -= 1;
                    }

                    curr_block_trid = (curr_block_trid == num_buffers) ? 1 : (curr_block_trid + 1);

                    if (l1_write_addr_in1 >= cb_in1_end) {
                        l1_write_addr_in1 = cb_in1_base;
                    }

                    global_iter++;
                }

                // Flush the trailing in-flight block for this expert before
                // moving to the next expert (or finishing).  TRISC must receive
                // all blocks for expert e before it can start expert e+1.
                noc_async_read_barrier_with_trid(block_trid_to_wait);
                cb_push_back(CTArgs::cb_in1, CTArgs::subblock_k);
                block_trid_to_wait = block_trid_to_wait == num_buffers ? 1 : (block_trid_to_wait + 1);
            }

            noc_async_atomic_barrier();

            if constexpr (CTArgs::enable_indexing) {
                cb_pop_front(CTArgs::cb_index, 1);
            }

#elif defined(COMPILE_FOR_TRISC)
            // ================================================================
            // TRISC: Compressed matmul for all selected experts sequentially.
            // Each expert contributes num_subblocks_k_per_expert × per_core_n
            // subblocks to the output.  Outputs are accumulated per N tile
            // across all experts before packing.
            // ================================================================
            constexpr uint32_t num_tiles_k_per_expert = CTArgs::subblock_k * CTArgs::num_subblocks_k_per_expert;
            constexpr uint32_t total_subblocks =
                CTArgs::selected_experts_k * CTArgs::num_subblocks_k_per_expert * CTArgs::per_core_n;
            constexpr bool split_acc = true;
            constexpr bool dense_packing = false;

            reconfig_data_format<false, true>(CTArgs::cb_in1, CTArgs::cb_in0);
            pack_reconfig_data_format<true>(CTArgs::cb_out);
            compressed::custom_mm_compressed_block_init_short<true, 1, split_acc, dense_packing>(
                CTArgs::cb_in0, CTArgs::cb_in1, CTArgs::cb_out);

            cb_wait_front(CTArgs::cb_in0, num_tiles_k_per_expert);

            uint32_t addr_in0 = 0;
            uint32_t in0_face_r_dim = 0;
            uint32_t in0_tile_size = 0;
            UNPACK(({
                uint32_t in0_id = get_operand_id(CTArgs::cb_in0);
                addr_in0 = get_local_cb_interface(in0_id).fifo_rd_ptr - 1;
                in0_face_r_dim = get_operand_face_r_dim(in0_id);
                in0_tile_size = get_local_cb_interface(in0_id).fifo_page_size;
            }));

            const volatile uint32_t* fmt_base_ptr = reinterpret_cast<const volatile uint32_t*>(CTArgs::fmt_l1_addr);
            uint32_t fmt_tile_offset = 0;

            for (uint32_t e = 0; e < CTArgs::selected_experts_k; e++) {
                uint32_t addr_in0_expert = addr_in0;

                for (uint32_t n = 0; n < CTArgs::per_core_n; n++) {
                    cb_reserve_back(CTArgs::cb_out, 1);
                    tile_regs_acquire();

                    uint32_t addr_in0_subblock = addr_in0_expert;

                    for (uint32_t sb_k = 0; sb_k < CTArgs::num_subblocks_k_per_expert; sb_k++) {
                        cb_wait_front(CTArgs::cb_in1, CTArgs::subblock_k);

                        const volatile uint32_t* fmt_ptr = fmt_base_ptr + fmt_tile_offset;
                        uint32_t fmt_addr = reinterpret_cast<uint32_t>(fmt_ptr);

                        bool is_last_subblock = (sb_k == CTArgs::num_subblocks_k_per_expert - 1);
                        if (!is_last_subblock) {
                            compressed::custom_mm_compressed_block_runtime<CTArgs::subblock_k, 1, false>(
                                fmt_addr, addr_in0_subblock, 0, in0_face_r_dim, 0);
                        } else {
                            compressed::custom_mm_compressed_block_runtime<CTArgs::subblock_k, 1, true>(
                                fmt_addr, addr_in0_subblock, 0, in0_face_r_dim, 0);
                        }

                        addr_in0_subblock += CTArgs::subblock_k * in0_tile_size;
                        fmt_tile_offset += CTArgs::subblock_k;
                        cb_pop_front(CTArgs::cb_in1, CTArgs::subblock_k);
                    }

                    tile_regs_commit();
                    tile_regs_wait();
                    pack_tile(0, CTArgs::cb_out, 0);
                    tile_regs_release();
                    cb_push_back(CTArgs::cb_out, 1);
                }
            }

            custom_mm_block_uninit<dense_packing>();

            if constexpr (PopIn0) {
                cb_pop_front(CTArgs::cb_in0, num_tiles_k_per_expert);
            }
#endif
        }
    };  // class Op

};  // struct DRAMStreamingExpertsMatmulCompressed

}  // namespace deepseek_b1_ops
