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
#ifdef TRISC_PACK
#include "ckernel_sfpu_exp.h"
#include "llk_math_eltwise_unary_sfpu_silu.h"
#endif
#endif

namespace deepseek_b1_ops {

// ============================================================================
// DRAMStreamingMatmulCompressed micro-op
//
// Computes: output[1,N] = in0[1,K] @ decompress(in1_compressed[K,N])
// with in1 streamed from DRAM in variable-size subblocks.
//
// Multi-core-per-bank support (cores_per_bank > 1):
//   Each core reads its own portion of N columns from the same DRAM bank.
//   Pipelined access: core 0 reads first, signals core 1 after last request
//   is sent, core 1 waits then reads from its offset, signals core 2, etc.
//   No data forwarding between cores — each reads directly from DRAM.
//   BRISC: no-op. TRISC: compressed matmul (unchanged).
//
// When cores_per_bank=1, pipeline_sem_id does not participate in inter-core
// pipelining; the NCRISC path performs a degenerate self-signal/self-wait on
// the semaphore each batch (functionally a no-op, but not compiled away).
// ============================================================================
struct DRAMStreamingMatmulCompressed {
    // ========================================================================
    // Compile-time args structs
    // ========================================================================

    // Reader CTArgs (NCRISC) - DRAM streaming with pipelined bank access
    template <
        uint32_t cb_in1_,
        uint32_t cb_out_,
        uint32_t in1_tensor_addr_,
        uint32_t subblock_k_,
        uint32_t per_core_n_,
        uint32_t out_num_tiles_,
        uint32_t num_subblocks_k_,
        uint32_t bank_id_,
        uint32_t vc_,
        uint32_t meta_l1_addr_,
        uint32_t cb_in1_size_bytes_,
        uint32_t noc_max_page_size_,
        uint32_t dram_start_offset_,
        uint32_t core_in_bank_idx_,
        uint32_t pipeline_sem_id_,
        uint32_t next_core_noc_x_,
        uint32_t next_core_noc_y_,
        uint32_t enable_indexing_ = 0,
        uint32_t cb_index_ = 0,
        uint32_t index_offset_ = 0,
        uint32_t use_hardcoded_expert_index_ = 0,
        // F1b multi-expert routing: L1 address of uint32 expert-byte-offset table (0 = disabled)
        uint32_t expert_offsets_l1_addr_ = 0,
        // F1b: L1 scratch cell where NCRISC writes the resolved expert_idx for TRISC
        uint32_t expert_idx_scratch_l1_addr_ = 0>
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
        static constexpr uint32_t core_in_bank_idx = core_in_bank_idx_;
        static constexpr uint32_t pipeline_sem_id = pipeline_sem_id_;
        static constexpr uint32_t next_core_noc_x = next_core_noc_x_;
        static constexpr uint32_t next_core_noc_y = next_core_noc_y_;
        // Expert indexing (CB wait/pop for pipeline synchronization)
        static constexpr bool enable_indexing = enable_indexing_ == 1;
        static constexpr uint32_t cb_index = cb_index_;
        static constexpr uint32_t index_offset = index_offset_;
        static constexpr bool use_hardcoded_expert_index = use_hardcoded_expert_index_ == 1;
        // F1b multi-expert routing
        static constexpr uint32_t expert_offsets_l1_addr = expert_offsets_l1_addr_;
        static constexpr uint32_t expert_idx_scratch_l1_addr = expert_idx_scratch_l1_addr_;
    };

    // Writer CTArgs (BRISC) - no-op
    struct WriterCTArgs {};

    // Compute CTArgs (TRISC) - compressed matmul with streaming
    template <
        uint32_t cb_in0_,
        uint32_t cb_in1_,
        uint32_t cb_out_,
        uint32_t subblock_k_,
        uint32_t per_core_n_,
        uint32_t num_subblocks_k_,
        uint32_t fmt_l1_addr_,
        uint32_t fuse_silu_ = 0,
        // F1b multi-expert routing: same scratch cell written by NCRISC; 0 = disabled
        uint32_t expert_idx_scratch_l1_addr_ = 0>
    struct ComputeCTArgs {
        static constexpr uint32_t cb_in0 = cb_in0_;
        static constexpr uint32_t cb_in1 = cb_in1_;
        static constexpr uint32_t cb_out = cb_out_;
        static constexpr uint32_t subblock_k = subblock_k_;
        static constexpr uint32_t per_core_n = per_core_n_;
        static constexpr uint32_t num_subblocks_k = num_subblocks_k_;
        static constexpr uint32_t fmt_l1_addr = fmt_l1_addr_;
        static constexpr bool fuse_silu = fuse_silu_ == 1;
        static constexpr uint32_t expert_idx_scratch_l1_addr = expert_idx_scratch_l1_addr_;
    };

    // ========================================================================
    // Op - the actual operation
    // ========================================================================
    // PopIn0: pop cb_in0 after compute. False to keep in0 for a subsequent projection.
    // PopIndex: pop cb_index after last DMA. Set true on the last consumer (e.g., down_proj).
    // WaitForOutput: after DMA, wait for TRISC to push out_num_tiles to cb_out.
    // CBIn1BaseAddr: when non-zero, force cb_in1 write address to this value at start.
    //   Required when reusing a CB across gate→up→down projections (prevents write-ptr drift).
    template <
        typename CTArgs,
        bool IsActiveCore,
        bool PopIn0 = true,
        bool PopIndex = false,
        bool WaitForOutput = false,
        uint32_t CBIn1BaseAddr = 0>
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
            // NCRISC: Stream compressed in1 from DRAM with pipelined bank access.
            //
            // Pipeline protocol (ring topology):
            //   Cores sharing a DRAM bank take turns reading in chunks of
            //   num_buffers subblocks. After issuing num_buffers reads, a core
            //   signals the next core and waits for its turn again.
            //   Last core wraps back to first core.
            //
            // Each core reads per_core_n columns starting at dram_start_offset.
            // ================================================================
            constexpr uint32_t num_iterations = CTArgs::num_subblocks_k * CTArgs::per_core_n;
            constexpr uint32_t dram_bank_id = CTArgs::bank_id;
            constexpr uint32_t vc = CTArgs::vc;
            constexpr uint32_t max_page_size = CTArgs::noc_max_page_size;

            volatile tt_l1_ptr uint32_t* sem_ptr =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(CTArgs::pipeline_sem_id));
            uint32_t next_sem_l1 = get_semaphore(CTArgs::pipeline_sem_id);
            uint64_t next_noc_addr = get_noc_addr(CTArgs::next_core_noc_x, CTArgs::next_core_noc_y, 0);
            uint64_t next_sem_noc_addr = next_noc_addr | (uint64_t)next_sem_l1;

            // --- Wait for previous core before first batch ---
            if constexpr (CTArgs::core_in_bank_idx > 0) {
                noc_semaphore_wait(sem_ptr, 1);
                noc_semaphore_set(sem_ptr, 0);
            }

            reset_noc_trid_barrier_counter(NOC_CLEAR_OUTSTANDING_REQ_MASK, noc_index);

            // Expert index CB: resolve which expert's data to read from DRAM.
            // F1b path (expert_offsets_l1_addr != 0): read expert_idx from CB, look up
            //   byte offset from L1 table, offset in1_tensor_addr, write expert_idx to
            //   scratch cell for TRISC, offset meta_ptr to the expert's metadata block.
            // Legacy path (expert_offsets_l1_addr == 0): expert offset pre-baked by host;
            //   CB wait is synchronization-only; meta_ptr starts at position 0.
            uint32_t effective_in1_addr = CTArgs::in1_tensor_addr;
            const volatile uint32_t* meta_ptr = reinterpret_cast<const volatile uint32_t*>(CTArgs::meta_l1_addr);

            if constexpr (CTArgs::enable_indexing) {
                cb_wait_front(CTArgs::cb_index, 1);

                if constexpr (CTArgs::expert_offsets_l1_addr != 0) {
                    // Read the routed expert index — matches DRAMStreamingMatmul convention:
                    // index CB holds raw uint16 integer indices (not bf16-encoded floats).
                    // use_hardcoded_expert_index=true means use index_offset directly (testing).
                    uint32_t expert_idx;
                    if constexpr (CTArgs::use_hardcoded_expert_index) {
                        expert_idx = CTArgs::index_offset;
                    } else {
                        volatile tt_l1_ptr uint16_t* index_ptr =
                            reinterpret_cast<volatile tt_l1_ptr uint16_t*>(get_read_ptr(CTArgs::cb_index));
                        expert_idx = static_cast<uint32_t>(index_ptr[CTArgs::index_offset]);
                    }

                    // Look up per-expert byte offset from L1 table (uint32 per expert)
                    const volatile uint32_t* offsets_ptr =
                        reinterpret_cast<const volatile uint32_t*>(CTArgs::expert_offsets_l1_addr);
                    effective_in1_addr = CTArgs::in1_tensor_addr + offsets_ptr[expert_idx];

                    // Offset meta_ptr to this expert's metadata block
                    // Each expert's meta block is per_core_n * num_subblocks_k uint32 entries
                    constexpr uint32_t meta_entries_per_expert = CTArgs::per_core_n * CTArgs::num_subblocks_k;
                    meta_ptr += expert_idx * meta_entries_per_expert;

                    // Write expert_idx to scratch BEFORE first cb_push_back so TRISC sees
                    // it after its cb_wait_front fence (guarantees NCRISC→TRISC ordering).
                    volatile tt_l1_ptr uint32_t* scratch_ptr =
                        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(CTArgs::expert_idx_scratch_l1_addr);
                    *scratch_ptr = expert_idx;
                }
            }

            uint64_t in1_base_addr = get_noc_addr_from_bank_id<true>(dram_bank_id, effective_in1_addr);

            uint32_t l1_write_addr_in1;
            uint32_t dram_read_offset = CTArgs::dram_start_offset;

            // Triple-buffering with transaction IDs
            constexpr uint32_t num_buffers = 3;
            constexpr uint32_t extra_blocks_in_flight = 1;
            uint32_t num_free_blocks_in_buffer = num_buffers;
            uint32_t curr_block_trid = 1;
            uint32_t block_trid_to_wait = 1;
            // Reserve initial space
            cb_reserve_back(CTArgs::cb_in1, CTArgs::subblock_k * (extra_blocks_in_flight + 1));
            l1_write_addr_in1 = get_write_ptr(CTArgs::cb_in1);

            // When reusing a CB across projections, force write addr to the true CB base.
            // get_write_ptr() returns a non-base offset when a previous projection left
            // the fifo_wr_ptr mid-buffer; CBIn1BaseAddr overrides it back to the start.
            if constexpr (CBIn1BaseAddr != 0) {
                l1_write_addr_in1 = CBIn1BaseAddr;
            }

            uint32_t cb_in1_base = l1_write_addr_in1;
            uint32_t cb_in1_end = cb_in1_base + CTArgs::cb_in1_size_bytes;

            constexpr uint32_t max_subblock_bytes = CTArgs::cb_in1_size_bytes / num_buffers;

            uint32_t iter = 0;
            while (iter < num_iterations) {
                uint32_t batch_size = std::min(num_buffers, num_iterations - iter);

                for (uint32_t b = 0; b < batch_size; b++) {
                    uint32_t block_size = meta_ptr[iter];
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

                    // Signal next core after last read in batch, before barrier.
                    // Only meaningful when cores_per_bank > 1 (pipeline_sem_id != 0).
                    // For cores_per_bank=1 (pipeline_sem_id=0, core_in_bank_idx=0) the
                    // signal would target self and the matching wait below is a no-op;
                    // skip both to avoid the self-signal/self-wait race on BH Galaxy.
                    if constexpr (CTArgs::pipeline_sem_id != 0) {
                        if (b == batch_size - 1) {
                            noc_semaphore_inc(next_sem_noc_addr, 1);
                        }
                    }

                    if (num_free_blocks_in_buffer == num_buffers - extra_blocks_in_flight) {
                        noc_async_read_barrier_with_trid(block_trid_to_wait);
                        cb_push_back(CTArgs::cb_in1, CTArgs::subblock_k);
                        block_trid_to_wait = block_trid_to_wait == num_buffers ? 1 : (block_trid_to_wait + 1);
                        cb_reserve_back(CTArgs::cb_in1, CTArgs::subblock_k * (extra_blocks_in_flight + 1));
                    } else {
                        num_free_blocks_in_buffer -= 1;
                    }

                    curr_block_trid = (curr_block_trid == num_buffers) ? 1 : (curr_block_trid + 1);

                    if (l1_write_addr_in1 >= cb_in1_end) {
                        l1_write_addr_in1 = cb_in1_base;
                    }

                    iter++;
                }

                // Wait for our turn again before next batch.
                // Only needed when cores_per_bank > 1 (see signal guard above).
                if constexpr (CTArgs::pipeline_sem_id != 0) {
                    if (iter < num_iterations) {
                        noc_semaphore_wait(sem_ptr, 1);
                        noc_semaphore_set(sem_ptr, 0);
                    }
                }
            }

            // Push remaining blocks
            for (uint32_t i = 0; i < extra_blocks_in_flight; ++i) {
                noc_async_read_barrier_with_trid(block_trid_to_wait);
                cb_push_back(CTArgs::cb_in1, CTArgs::subblock_k);
                block_trid_to_wait = block_trid_to_wait == num_buffers ? 1 : (block_trid_to_wait + 1);
            }
            noc_async_atomic_barrier();

            // Release the expert index CB once all DMA is done (last consumer only).
            if constexpr (PopIndex && CTArgs::enable_indexing) {
                cb_pop_front(CTArgs::cb_index, 1);
            }

            // Optionally stall until TRISC has produced all output tiles.
            // Used by up_proj to ensure gate CB can be safely reused next.
            if constexpr (WaitForOutput) {
                cb_wait_front(CTArgs::cb_out, CTArgs::out_num_tiles);
            }

#elif defined(COMPILE_FOR_TRISC)
            // ================================================================
            // TRISC: Compressed matmul with subblock-K streaming.
            // Identical on all cores — each consumes from its own cb_in1.
            // ================================================================
            constexpr uint32_t num_tiles_k = CTArgs::subblock_k * CTArgs::num_subblocks_k;
            constexpr bool split_acc = true;
            constexpr bool dense_packing = false;

            reconfig_data_format<false, true>(CTArgs::cb_in1, CTArgs::cb_in0);
            pack_reconfig_data_format<true>(CTArgs::cb_out);
            compressed::custom_mm_compressed_block_init_short<true, 1, split_acc, dense_packing>(
                CTArgs::cb_in0, CTArgs::cb_in1, CTArgs::cb_out);

            if constexpr (CTArgs::fuse_silu) {
                PACK((llk_math_eltwise_unary_sfpu_silu_init<true>()));
            }

            cb_wait_front(CTArgs::cb_in0, num_tiles_k);

            // Get in0 base address and tile size
            uint32_t addr_in0 = 0;
            uint32_t in0_face_r_dim = 0;
            uint32_t in0_tile_size = 0;
            UNPACK(({
                uint32_t in0_id = get_operand_id(CTArgs::cb_in0);
                addr_in0 = get_local_cb_interface(in0_id).fifo_rd_ptr - 1;
                in0_face_r_dim = get_operand_face_r_dim(in0_id);
                in0_tile_size = get_local_cb_interface(in0_id).fifo_page_size;
            }));

            // Per-tile metadata pointer
            const volatile uint32_t* fmt_base_ptr = reinterpret_cast<const volatile uint32_t*>(CTArgs::fmt_l1_addr);
            constexpr uint32_t tiles_per_subblock = CTArgs::subblock_k;

            // F1b multi-expert routing: fence on the first in1 subblock (already pushed by
            // NCRISC after writing expert_idx to scratch), then read expert_idx from scratch
            // to offset fmt_base_ptr to the routed expert's format metadata block.
            // This fence provides the NCRISC→TRISC ordering guarantee without a dedicated
            // synchronization primitive (NCRISC writes scratch BEFORE first cb_push_back).
            uint32_t fmt_tile_offset = 0;
            if constexpr (CTArgs::expert_idx_scratch_l1_addr != 0) {
                cb_wait_front(CTArgs::cb_in1, 1);
                const volatile uint32_t* scratch_ptr =
                    reinterpret_cast<const volatile uint32_t*>(CTArgs::expert_idx_scratch_l1_addr);
                uint32_t expert_idx = *scratch_ptr;
                // Each expert's fmt block is per_core_n * num_subblocks_k * subblock_k entries
                fmt_tile_offset = expert_idx * CTArgs::per_core_n * CTArgs::num_subblocks_k * tiles_per_subblock;
            }

            for (uint32_t n = 0; n < CTArgs::per_core_n; n++) {
                cb_reserve_back(CTArgs::cb_out, 1);
                tile_regs_acquire();

                uint32_t addr_in0_subblock = addr_in0;

                for (uint32_t sb_k = 0; sb_k < CTArgs::num_subblocks_k; sb_k++) {
                    cb_wait_front(CTArgs::cb_in1, CTArgs::subblock_k);

                    // Absolute addresses are precomputed in fmt metadata by host
                    // (CB is tensor-backed so L1 address is known at compile time).
                    // addr_in1 is unused when RELATIVE_ADDR=false.
                    const volatile uint32_t* fmt_ptr = fmt_base_ptr + fmt_tile_offset;
                    uint32_t fmt_addr = reinterpret_cast<uint32_t>(fmt_ptr);
                    if (sb_k < CTArgs::num_subblocks_k - 1) {
                        compressed::custom_mm_compressed_block_runtime<CTArgs::subblock_k, 1, false>(
                            fmt_addr, addr_in0_subblock, 0, in0_face_r_dim, 0);
                    } else {
                        compressed::custom_mm_compressed_block_runtime<CTArgs::subblock_k, 1, true>(
                            fmt_addr, addr_in0_subblock, 0, in0_face_r_dim, 0);
                    }

                    addr_in0_subblock += CTArgs::subblock_k * in0_tile_size;
                    fmt_tile_offset += tiles_per_subblock;
                    cb_pop_front(CTArgs::cb_in1, CTArgs::subblock_k);
                }

                tile_regs_commit();
                tile_regs_wait();
                if constexpr (CTArgs::fuse_silu) {
                    PACK((llk_math_eltwise_unary_sfpu_silu<true, false, 8>(0, (int)VectorMode::R)));
                    PACK(TTI_STALLWAIT(p_stall::STALL_PACK, p_stall::WAIT_SFPU));
                }
                pack_tile(0, CTArgs::cb_out, 0);
                tile_regs_release();
                cb_push_back(CTArgs::cb_out, 1);
            }

            custom_mm_block_uninit<dense_packing>();

            if constexpr (PopIn0) {
                cb_pop_front(CTArgs::cb_in0, num_tiles_k);
            }
#endif
        }
    };  // class Op

};  // struct DRAMStreamingMatmulCompressed

}  // namespace deepseek_b1_ops
