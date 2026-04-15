// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

// DRAM expert matmul with compressed weights.
//
// B is WIDTH_SHARDED in DRAM, streamed subblock by subblock with triple-buffering.
//
// Per-core meta table layout (meta_l1_addr):
//   For each expert e (0..num_experts-1), meta_stride = 2 + num_subblocks_k * per_core_n:
//     meta[e * meta_stride + 0] = in1_tensor_addr  (bank-relative DRAM buffer addr)
//     meta[e * meta_stride + 1] = dram_col_offset  (byte offset to this core's col start)
//     meta[e * meta_stride + 2..meta_stride-1] = block_sizes[num_subblocks_k * per_core_n]
//
// Per-core fmt table layout (fmt_l1_addr):
//   For each expert e, tiles_per_expert = subblock_k * num_subblocks_k * per_core_n:
//     fmt[e * tiles_per_expert .. (e+1)*tiles_per_expert - 1] = {abs_slot_addr:24|fmt:8}
//   CB slot addresses are identical for all experts (deterministic triple-buffer rotation).
//
// Pipeline protocol (cores_per_bank > 1): same as DRAMStreamingMatmulCompressed.
// cores_per_bank=1 is the degenerate self-signal case.
//
// NOTE: In hybrid mode, cb_in1 in DRAM CTArgs is a SEPARATE CB from the SRAM
// path's cb_in1. The kernel cpp maps them to different CB indices (e.g. CB 4
// for DRAM streaming vs CB 1 for SRAM B data).

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

struct MatmulExpertCompressedDRAM {
    template <
        uint32_t cb_in0_,
        uint32_t cb_in1_,
        uint32_t cb_out_,
        uint32_t cb_index_,
        uint32_t num_tiles_k_,
        uint32_t subblock_k_,
        uint32_t subblock_n_,
        uint32_t num_subblocks_k_,
        uint32_t per_core_n_,
        uint32_t bank_id_,
        uint32_t vc_,
        uint32_t meta_l1_addr_,
        uint32_t cb_in1_size_bytes_,
        uint32_t noc_max_page_size_,
        uint32_t core_in_bank_idx_,
        uint32_t pipeline_sem_id_,
        uint32_t next_core_noc_x_,
        uint32_t next_core_noc_y_,
        uint32_t cores_per_bank_,
        uint32_t num_active_experts_,
        uint32_t table_idx_l1_addr_,
        uint32_t index_l1_addr_,
        uint32_t cb_fmt_,
        uint32_t fmt_dram_addr_,
        uint32_t fmt_per_expert_bytes_,
        uint32_t fmt_per_core_bytes_,
        uint32_t accum_experts_ = 0,
        uint32_t index_offset_ = 0>
    struct ReaderCTArgs {
        static constexpr uint32_t cb_in0 = cb_in0_;
        static constexpr uint32_t cb_in1 = cb_in1_;
        static constexpr uint32_t cb_out = cb_out_;
        static constexpr uint32_t cb_index = cb_index_;
        static constexpr uint32_t num_tiles_k = num_tiles_k_;
        static constexpr uint32_t subblock_k = subblock_k_;
        static constexpr uint32_t subblock_n = subblock_n_;
        static constexpr uint32_t num_subblocks_k = num_subblocks_k_;
        static constexpr uint32_t per_core_n = per_core_n_;
        static constexpr uint32_t bank_id = bank_id_;
        static constexpr uint32_t vc = vc_;
        static constexpr uint32_t meta_l1_addr = meta_l1_addr_;
        static constexpr uint32_t cb_in1_size_bytes = cb_in1_size_bytes_;
        static constexpr uint32_t noc_max_page_size = noc_max_page_size_;
        static constexpr uint32_t core_in_bank_idx = core_in_bank_idx_;
        static constexpr uint32_t pipeline_sem_id = pipeline_sem_id_;
        static constexpr uint32_t next_core_noc_x = next_core_noc_x_;
        static constexpr uint32_t next_core_noc_y = next_core_noc_y_;
        static constexpr uint32_t cores_per_bank = cores_per_bank_;
        static constexpr uint32_t num_active_experts = num_active_experts_;
        static constexpr uint32_t table_idx_l1_addr = table_idx_l1_addr_;
        static constexpr uint32_t index_l1_addr = index_l1_addr_;
        static constexpr uint32_t cb_fmt = cb_fmt_;
        static constexpr uint32_t fmt_dram_addr = fmt_dram_addr_;
        static constexpr uint32_t fmt_per_expert_bytes = fmt_per_expert_bytes_;
        static constexpr uint32_t fmt_per_core_bytes = fmt_per_core_bytes_;
        static constexpr bool accum_experts = accum_experts_ != 0;
        static constexpr uint32_t index_offset = index_offset_;
    };

    template <
        uint32_t cb_in0_,
        uint32_t cb_in1_,
        uint32_t cb_out_,
        uint32_t cb_index_,
        uint32_t num_tiles_k_,
        uint32_t subblock_k_,
        uint32_t subblock_n_,
        uint32_t num_subblocks_k_,
        uint32_t per_core_n_,
        uint32_t fmt_l1_addr_,
        uint32_t num_active_experts_,
        uint32_t table_idx_l1_addr_,
        uint32_t index_l1_addr_,
        uint32_t cb_fmt_,
        uint32_t accum_experts_ = 0,
        uint32_t fuse_silu_ = 0,
        uint32_t index_offset_ = 0>
    struct ComputeCTArgs {
        static constexpr uint32_t cb_in0 = cb_in0_;
        static constexpr uint32_t cb_in1 = cb_in1_;
        static constexpr uint32_t cb_out = cb_out_;
        static constexpr uint32_t cb_index = cb_index_;
        static constexpr uint32_t num_tiles_k = num_tiles_k_;
        static constexpr uint32_t subblock_k = subblock_k_;
        static constexpr uint32_t subblock_n = subblock_n_;
        static constexpr uint32_t num_subblocks_k = num_subblocks_k_;
        static constexpr uint32_t per_core_n = per_core_n_;
        static constexpr uint32_t fmt_l1_addr = fmt_l1_addr_;
        static constexpr uint32_t num_active_experts = num_active_experts_;
        static constexpr uint32_t table_idx_l1_addr = table_idx_l1_addr_;
        static constexpr uint32_t index_l1_addr = index_l1_addr_;
        static constexpr uint32_t cb_fmt = cb_fmt_;
        static constexpr bool accum_experts = accum_experts_ != 0;
        static constexpr bool fuse_silu = fuse_silu_ != 0;
        static constexpr uint32_t index_offset = index_offset_;
    };

    struct WriterCTArgs {};

    template <typename CTArgs, bool IsActiveCore, bool pop_in0 = true, bool pop_index = true>
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
            constexpr uint32_t tiles_per_block = CTArgs::subblock_k * CTArgs::subblock_n;
            constexpr uint32_t num_iterations = CTArgs::num_subblocks_k * (CTArgs::per_core_n / CTArgs::subblock_n);
            constexpr uint32_t meta_stride = 2 + num_iterations;
            constexpr uint32_t max_page_size = CTArgs::noc_max_page_size;
            constexpr uint32_t num_active_experts = CTArgs::num_active_experts;
            constexpr uint32_t num_buffers = 3;
            constexpr uint32_t extra_blocks_in_flight = 1;
            constexpr uint32_t max_subblock_bytes = CTArgs::cb_in1_size_bytes / num_buffers;

            // Read index array from L1 (direct address, no CB API needed).
            volatile tt_l1_ptr uint16_t* index_ptr =
                reinterpret_cast<volatile tt_l1_ptr uint16_t*>(CTArgs::index_l1_addr);
            const volatile uint32_t* meta_base = reinterpret_cast<const volatile uint32_t*>(CTArgs::meta_l1_addr);

            // Pipeline semaphore for cores_per_bank > 1.
            volatile tt_l1_ptr uint32_t* sem_ptr =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(CTArgs::pipeline_sem_id));
            uint64_t next_sem_l1 = get_semaphore(CTArgs::pipeline_sem_id);
            uint64_t next_noc_addr = get_noc_addr(CTArgs::next_core_noc_x, CTArgs::next_core_noc_y, 0);
            uint64_t next_sem_noc_addr = next_noc_addr | next_sem_l1;

            // CB base address — set on first DRAM expert.
            uint32_t cb_in1_base = 0;
            uint32_t cb_in1_end = 0;
            bool first_dram_expert = true;

            for (uint32_t exp_i = 0; exp_i < num_active_experts; exp_i++) {
                uint32_t raw_idx = static_cast<uint32_t>(index_ptr[exp_i + CTArgs::index_offset]);
                if (raw_idx & 0x8000) {
                    continue;  // bit15=1 → SRAM expert, skip
                }
                uint32_t expert_idx = raw_idx;  // bit15=0, raw value is global expert ID
                // Index meta/fmt tables directly by global expert_idx (no table_idx indirection).
                const volatile uint32_t* expert_meta = meta_base + expert_idx * meta_stride;
                uint32_t expert_in1_addr = expert_meta[0];
                uint32_t dram_read_offset = expert_meta[1];
                const volatile uint32_t* block_size_ptr = expert_meta + 2;

                if constexpr (CTArgs::cores_per_bank > 1 && CTArgs::core_in_bank_idx > 0) {
                    noc_semaphore_wait(sem_ptr, 1);
                    noc_semaphore_set(sem_ptr, 0);
                }

                reset_noc_trid_barrier_counter(NOC_CLEAR_OUTSTANDING_REQ_MASK, noc_index);

                // Initial reserve for this expert.
                cb_reserve_back(CTArgs::cb_in1, tiles_per_block * (extra_blocks_in_flight + 1));

                if (first_dram_expert) {
                    cb_in1_base = get_write_ptr(CTArgs::cb_in1);
                    cb_in1_end = cb_in1_base + CTArgs::cb_in1_size_bytes;
                    first_dram_expert = false;
                }

                uint64_t in1_base_addr = get_noc_addr_from_bank_id<true>(CTArgs::bank_id, expert_in1_addr);
                uint32_t l1_write_addr_in1 = get_write_ptr(CTArgs::cb_in1);
                uint32_t num_free_blocks_in_buffer = num_buffers;
                uint32_t curr_block_trid = 1;
                uint32_t block_trid_to_wait = 1;

                // Read this expert's fmt metadata from DRAM into cb_fmt.
                // Shares trid 1 with the first weight block — completed together.
                cb_reserve_back(CTArgs::cb_fmt, 1);
                {
                    uint32_t fmt_write_addr = get_write_ptr(CTArgs::cb_fmt);
                    uint32_t fmt_dram_offset = CTArgs::fmt_dram_addr +
                                               CTArgs::core_in_bank_idx * CTArgs::fmt_per_core_bytes +
                                               expert_idx * CTArgs::fmt_per_expert_bytes;
                    uint64_t fmt_noc_addr = get_noc_addr_from_bank_id<true>(CTArgs::bank_id, fmt_dram_offset);
                    noc_async_read_set_trid(curr_block_trid);
                    noc_async_read_one_packet_set_state<true>(fmt_noc_addr, CTArgs::fmt_per_expert_bytes, CTArgs::vc);
                    noc_async_read_one_packet_with_state_with_trid(fmt_noc_addr, 0, fmt_write_addr, curr_block_trid);
                }
                // cb_fmt pushed below when trid 1 is waited on.
                bool fmt_pushed = false;

                // Triple-buffer streaming loop.
                uint32_t iter = 0;
                while (iter < num_iterations) {
                    uint32_t batch_size = std::min(num_buffers, num_iterations - iter);

                    for (uint32_t b = 0; b < batch_size; b++) {
                        uint32_t block_size = block_size_ptr[iter];
                        uint32_t slot_start = l1_write_addr_in1;

                        noc_async_read_set_trid(curr_block_trid);

                        uint32_t remaining = block_size;
                        while (remaining > 0) {
                            uint32_t chunk = (remaining > max_page_size) ? max_page_size : remaining;
                            noc_async_read_one_packet_set_state<true>(in1_base_addr, chunk, CTArgs::vc);
                            noc_async_read_one_packet_with_state_with_trid(
                                in1_base_addr, dram_read_offset, l1_write_addr_in1, curr_block_trid);
                            dram_read_offset += chunk;
                            l1_write_addr_in1 += chunk;
                            remaining -= chunk;
                        }

                        l1_write_addr_in1 = slot_start + max_subblock_bytes;

                        if constexpr (CTArgs::cores_per_bank > 1) {
                            if (b == batch_size - 1) {
                                constexpr bool is_last_in_ring =
                                    (CTArgs::core_in_bank_idx == CTArgs::cores_per_bank - 1);
                                if constexpr (is_last_in_ring) {
                                    if (iter < num_iterations - 1) {
                                        noc_semaphore_inc(next_sem_noc_addr, 1);
                                    }
                                } else {
                                    noc_semaphore_inc(next_sem_noc_addr, 1);
                                }
                            }
                        }

                        if (num_free_blocks_in_buffer == num_buffers - extra_blocks_in_flight) {
                            noc_async_read_barrier_with_trid(block_trid_to_wait);
                            cb_push_back(CTArgs::cb_in1, tiles_per_block);
                            // Trid 1 wait also completes the fmt DRAM read issued earlier.
                            if (!fmt_pushed) {
                                cb_push_back(CTArgs::cb_fmt, 1);
                                fmt_pushed = true;
                            }
                            block_trid_to_wait = block_trid_to_wait == num_buffers ? 1 : (block_trid_to_wait + 1);
                            cb_reserve_back(CTArgs::cb_in1, tiles_per_block * (extra_blocks_in_flight + 1));
                        } else {
                            num_free_blocks_in_buffer -= 1;
                        }

                        curr_block_trid = (curr_block_trid == num_buffers) ? 1 : (curr_block_trid + 1);

                        if (l1_write_addr_in1 >= cb_in1_end) {
                            l1_write_addr_in1 = cb_in1_base;
                        }

                        iter++;
                    }

                    if constexpr (CTArgs::cores_per_bank > 1) {
                        if (iter < num_iterations) {
                            noc_semaphore_wait(sem_ptr, 1);
                            noc_semaphore_set(sem_ptr, 0);
                        }
                    }
                }

                // Flush remaining in-flight blocks.
                for (uint32_t i = 0; i < extra_blocks_in_flight; ++i) {
                    noc_async_read_barrier_with_trid(block_trid_to_wait);
                    cb_push_back(CTArgs::cb_in1, tiles_per_block);
                    block_trid_to_wait = block_trid_to_wait == num_buffers ? 1 : (block_trid_to_wait + 1);
                }
                noc_async_atomic_barrier();
            }

#elif defined(COMPILE_FOR_TRISC)
            constexpr uint32_t num_tiles_k = CTArgs::subblock_k * CTArgs::num_subblocks_k;
            constexpr uint32_t tiles_per_expert = CTArgs::subblock_k * CTArgs::num_subblocks_k * CTArgs::per_core_n;
            constexpr uint32_t num_active_experts = CTArgs::num_active_experts;

            cb_wait_front(CTArgs::cb_index, 1);

            volatile tt_l1_ptr uint16_t* index_ptr =
                reinterpret_cast<volatile tt_l1_ptr uint16_t*>(CTArgs::index_l1_addr);

            // Count DRAM experts and record activation tile offsets.
            // Index tensor encodes SRAM/DRAM via bit 15: 1=SRAM, 0=DRAM.
            // index_offset is used in TP=false mode, to offset to the correct expert in
            // index tensor.
            uint32_t num_dram_experts = 0;
            uint32_t dram_act_tile_offset[num_active_experts];
            for (uint32_t exp_i = 0; exp_i < num_active_experts; exp_i++) {
                uint32_t raw_idx = static_cast<uint32_t>(index_ptr[exp_i + CTArgs::index_offset]);
                if (raw_idx & 0x8000) {
                    continue;  // bit15=1 → SRAM expert, skip
                }
                if constexpr (CTArgs::accum_experts) {
                    dram_act_tile_offset[num_dram_experts] = exp_i * num_tiles_k;
                }
                num_dram_experts++;
            }

            if (num_dram_experts > 0) {
                constexpr bool split_acc = true;
                constexpr bool dense_packing = false;

                reconfig_data_format<false, true>(CTArgs::cb_in1, CTArgs::cb_in0);
                pack_reconfig_data_format<true>(CTArgs::cb_out);
                compressed::custom_mm_compressed_block_init_short<true, 1, split_acc, dense_packing>(
                    CTArgs::cb_in0, CTArgs::cb_in1, CTArgs::cb_out);

                if constexpr (CTArgs::fuse_silu) {
                    PACK((llk_math_eltwise_unary_sfpu_silu_init<false>()));
                }

                if constexpr (CTArgs::accum_experts) {
                    cb_wait_front(CTArgs::cb_in0, num_tiles_k * num_active_experts);
                } else {
                    cb_wait_front(CTArgs::cb_in0, num_tiles_k);
                }

                uint32_t addr_in0 = 0;
                uint32_t in0_face_r_dim = 0;
                uint32_t in0_tile_size = 0;
                uint32_t in1_operand_id = 0;

                UNPACK(({
                    uint32_t in0_id = get_operand_id(CTArgs::cb_in0);
                    addr_in0 = get_local_cb_interface(in0_id).fifo_rd_ptr - 1;
                    in0_face_r_dim = get_operand_face_r_dim(in0_id);
                    in0_tile_size = get_local_cb_interface(in0_id).fifo_page_size;
                    in1_operand_id = get_operand_id(CTArgs::cb_in1);
                }));

                if constexpr (CTArgs::accum_experts) {
                    // Packer L1 accumulation: each expert packs per_core_n tiles sequentially.
                    // Each expert uses a distinct activation at its index-tensor position.
                    for (uint32_t i = 0; i < num_dram_experts; i++) {
                        cb_reserve_back(CTArgs::cb_out, CTArgs::per_core_n);

                        if (i == 0) {
                            PACK((llk_pack_reconfig_l1_acc(0)));
                        } else if (i == 1) {
                            PACK((llk_pack_reconfig_l1_acc(1)));
                        }

                        uint32_t fmt_l1_addr = 0;
                        const volatile uint32_t* fmt_base_ptr = nullptr;
                        uint32_t fmt_tile_offset = 0;
                        bool fmt_waited = false;
                        uint32_t addr_in0_expert = addr_in0 + dram_act_tile_offset[i] * in0_tile_size;

                        for (uint32_t n = 0; n < CTArgs::per_core_n; n++) {
                            tile_regs_acquire();
                            uint32_t addr_in0_subblock = addr_in0_expert;

                            for (uint32_t sb_k = 0; sb_k < CTArgs::num_subblocks_k; sb_k++) {
                                cb_wait_front(CTArgs::cb_in1, CTArgs::subblock_k);

                                if (!fmt_waited) {
                                    cb_wait_front(CTArgs::cb_fmt, 1);
                                    UNPACK(({
                                        uint32_t fmt_op_id = get_operand_id(CTArgs::cb_fmt);
                                        fmt_l1_addr = get_local_cb_interface(fmt_op_id).fifo_rd_ptr << 4;
                                    }));
                                    fmt_base_ptr = reinterpret_cast<const volatile uint32_t*>(fmt_l1_addr);
                                    fmt_waited = true;
                                }

                                uint32_t slot_base = 0;
                                UNPACK(({ slot_base = get_local_cb_interface(in1_operand_id).fifo_rd_ptr - 1; }));

                                const volatile uint32_t* fmt_ptr = fmt_base_ptr + fmt_tile_offset;
                                uint32_t fmt_addr = reinterpret_cast<uint32_t>(fmt_ptr);
                                constexpr uint32_t zeros_addr = compressed::ZEROS_ADDR_SHIFTED;
                                if (sb_k < CTArgs::num_subblocks_k - 1) {
                                    compressed::custom_mm_compressed_block_runtime_dram<CTArgs::subblock_k, false>(
                                        fmt_addr, addr_in0_subblock, slot_base, zeros_addr, in0_face_r_dim, 0);
                                } else {
                                    compressed::custom_mm_compressed_block_runtime_dram<CTArgs::subblock_k, true>(
                                        fmt_addr, addr_in0_subblock, slot_base, zeros_addr, in0_face_r_dim, 0);
                                }

                                addr_in0_subblock += CTArgs::subblock_k * in0_tile_size;
                                fmt_tile_offset += CTArgs::subblock_k;
                                cb_pop_front(CTArgs::cb_in1, CTArgs::subblock_k);
                            }

                            tile_regs_commit();
                            tile_regs_wait();
                            pack_tile(0, CTArgs::cb_out);
                            tile_regs_release();
                        }

                        cb_pop_front(CTArgs::cb_fmt, 1);
                        cb_push_back(CTArgs::cb_out, CTArgs::per_core_n);

                        // Reset write pointer for next expert's L1_ACC accumulation.
                        if (i < num_dram_experts - 1) {
                            cb_wait_front(CTArgs::cb_out, CTArgs::per_core_n);
                            cb_pop_front(CTArgs::cb_out, CTArgs::per_core_n);
                        }
                    }

                    PACK((llk_pack_reconfig_l1_acc(0)));
                } else if constexpr (CTArgs::subblock_n > 1) {
                    // Batched path: subblock_n columns share one CB slot.
                    // Single slot_base, fmt offsets accumulate across columns.
                    constexpr uint32_t tiles_per_block = CTArgs::subblock_k * CTArgs::subblock_n;
                    constexpr uint32_t num_subblocks_n = CTArgs::per_core_n / CTArgs::subblock_n;

                    for (uint32_t i = 0; i < num_dram_experts; i++) {
                        uint32_t fmt_l1_addr = 0;
                        const volatile uint32_t* fmt_base_ptr = nullptr;
                        uint32_t fmt_tile_offset = 0;
                        bool fmt_waited = false;

                        for (uint32_t ng = 0; ng < num_subblocks_n; ng++) {
                            cb_wait_front(CTArgs::cb_in1, tiles_per_block);

                            if (!fmt_waited) {
                                cb_wait_front(CTArgs::cb_fmt, 1);
                                UNPACK(({
                                    uint32_t fmt_op_id = get_operand_id(CTArgs::cb_fmt);
                                    fmt_l1_addr = get_local_cb_interface(fmt_op_id).fifo_rd_ptr << 4;
                                }));
                                fmt_base_ptr = reinterpret_cast<const volatile uint32_t*>(fmt_l1_addr);
                                fmt_waited = true;
                            }

                            uint32_t slot_base = 0;
                            UNPACK(({ slot_base = get_local_cb_interface(in1_operand_id).fifo_rd_ptr - 1; }));

                            for (uint32_t sn = 0; sn < CTArgs::subblock_n; sn++) {
                                cb_reserve_back(CTArgs::cb_out, 1);
                                tile_regs_acquire();

                                uint32_t addr_in0_subblock = addr_in0;
                                const volatile uint32_t* fmt_ptr = fmt_base_ptr + fmt_tile_offset;
                                uint32_t fmt_addr = reinterpret_cast<uint32_t>(fmt_ptr);
                                constexpr uint32_t zeros_addr = compressed::ZEROS_ADDR_SHIFTED;

                                compressed::custom_mm_compressed_block_runtime_dram<CTArgs::subblock_k, true>(
                                    fmt_addr, addr_in0_subblock, slot_base, zeros_addr, in0_face_r_dim, 0);

                                fmt_tile_offset += CTArgs::subblock_k;

                                tile_regs_commit();
                                if constexpr (CTArgs::fuse_silu) {
                                    TTI_SEMWAIT(
                                        p_stall::STALL_TDMA | p_stall::STALL_CFG,
                                        semaphore::t6_sem(semaphore::MATH_PACK),
                                        p_stall::STALL_ON_ZERO);
                                    PACK(TT_SETC16(
                                        DEST_TARGET_REG_CFG_MATH_Offset_ADDR32,
                                        ckernel::packer::get_packer_dest_offset()));
                                    PACK((llk_math_eltwise_unary_sfpu_silu<false, false, 2>(0, (int)VectorMode::R)));
                                    PACK(TTI_STALLWAIT(p_stall::STALL_PACK, p_stall::WAIT_SFPU));
                                } else {
                                    tile_regs_wait();
                                }
                                pack_tile(0, CTArgs::cb_out, 0);
                                tile_regs_release();
                                cb_push_back(CTArgs::cb_out, 1);
                            }

                            cb_pop_front(CTArgs::cb_in1, tiles_per_block);
                        }

                        cb_pop_front(CTArgs::cb_fmt, 1);
                    }
                } else {
                    // Default path: per-column cb_wait/pop.
                    for (uint32_t i = 0; i < num_dram_experts; i++) {
                        uint32_t fmt_l1_addr = 0;
                        const volatile uint32_t* fmt_base_ptr = nullptr;
                        uint32_t fmt_tile_offset = 0;
                        bool fmt_waited = false;

                        for (uint32_t n = 0; n < CTArgs::per_core_n; n++) {
                            cb_reserve_back(CTArgs::cb_out, 1);
                            tile_regs_acquire();

                            uint32_t addr_in0_subblock = addr_in0;

                            for (uint32_t sb_k = 0; sb_k < CTArgs::num_subblocks_k; sb_k++) {
                                cb_wait_front(CTArgs::cb_in1, CTArgs::subblock_k);

                                if (!fmt_waited) {
                                    cb_wait_front(CTArgs::cb_fmt, 1);
                                    UNPACK(({
                                        uint32_t fmt_op_id = get_operand_id(CTArgs::cb_fmt);
                                        fmt_l1_addr = get_local_cb_interface(fmt_op_id).fifo_rd_ptr << 4;
                                    }));
                                    fmt_base_ptr = reinterpret_cast<const volatile uint32_t*>(fmt_l1_addr);
                                    fmt_waited = true;
                                }

                                uint32_t slot_base = 0;
                                UNPACK(({ slot_base = get_local_cb_interface(in1_operand_id).fifo_rd_ptr - 1; }));

                                const volatile uint32_t* fmt_ptr = fmt_base_ptr + fmt_tile_offset;
                                uint32_t fmt_addr = reinterpret_cast<uint32_t>(fmt_ptr);
                                constexpr uint32_t zeros_addr = compressed::ZEROS_ADDR_SHIFTED;
                                if (sb_k < CTArgs::num_subblocks_k - 1) {
                                    compressed::custom_mm_compressed_block_runtime_dram<CTArgs::subblock_k, false>(
                                        fmt_addr, addr_in0_subblock, slot_base, zeros_addr, in0_face_r_dim, 0);
                                } else {
                                    compressed::custom_mm_compressed_block_runtime_dram<CTArgs::subblock_k, true>(
                                        fmt_addr, addr_in0_subblock, slot_base, zeros_addr, in0_face_r_dim, 0);
                                }

                                addr_in0_subblock += CTArgs::subblock_k * in0_tile_size;
                                fmt_tile_offset += CTArgs::subblock_k;
                                cb_pop_front(CTArgs::cb_in1, CTArgs::subblock_k);
                            }

                            tile_regs_commit();
                            if constexpr (CTArgs::fuse_silu) {
                                TTI_SEMWAIT(
                                    p_stall::STALL_TDMA | p_stall::STALL_CFG,
                                    semaphore::t6_sem(semaphore::MATH_PACK),
                                    p_stall::STALL_ON_ZERO);
                                PACK(TT_SETC16(
                                    DEST_TARGET_REG_CFG_MATH_Offset_ADDR32, ckernel::packer::get_packer_dest_offset()));
                                PACK((llk_math_eltwise_unary_sfpu_silu<false, false, 2>(0, (int)VectorMode::R)));
                                PACK(TTI_STALLWAIT(p_stall::STALL_PACK, p_stall::WAIT_SFPU));
                            } else {
                                tile_regs_wait();
                            }
                            pack_tile(0, CTArgs::cb_out, 0);
                            tile_regs_release();
                            cb_push_back(CTArgs::cb_out, 1);
                        }

                        cb_pop_front(CTArgs::cb_fmt, 1);
                    }
                }

                custom_mm_block_uninit<false>();
            }

            if constexpr (pop_in0) {
                if constexpr (CTArgs::accum_experts) {
                    cb_pop_front(CTArgs::cb_in0, num_tiles_k * num_active_experts);
                } else {
                    cb_pop_front(CTArgs::cb_in0, num_tiles_k);
                }
            }
            if constexpr (pop_index) {
                cb_pop_front(CTArgs::cb_index, 1);
            }
#endif
        }
    };
};

}  // namespace deepseek_b1_ops
