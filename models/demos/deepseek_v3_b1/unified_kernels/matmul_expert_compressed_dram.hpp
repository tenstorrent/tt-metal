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
#include "expert_index_encoding.hpp"

#if defined(COMPILE_FOR_NCRISC) || defined(COMPILE_FOR_BRISC)
#include "api/dataflow/dataflow_api.h"
#elif defined(COMPILE_FOR_TRISC)
#include <cstdint>
#include "api/compute/tile_move_copy.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/compute_kernel_api.h"
#include "../kernel_includes/tt_metal/include/compute_kernel_api/custom_mm.h"
#include "../kernel_includes/tt_metal/include/compute_kernel_api/compressed_custom_mm.h"
using namespace ckernel;
#ifdef TRISC_PACK
#include "llk_math_eltwise_unary_sfpu_silu.h"
#endif
#endif

namespace deepseek_b1_ops {

// fmt double-buffer sync protocol — 2 slots, counter sems (0..2).
// NCRISC is the sole producer; UNPACK (sem0) and MATH (sem1) are the consumers.
// NCRISC can run 1 expert ahead of the slowest consumer.
namespace fmt_sync {
FORCE_INLINE void producer_wait_slot(uint32_t sem0, uint32_t sem1) {
    while (unified_kernels::sem_atomic_load(sem0) >= 2 || unified_kernels::sem_atomic_load(sem1) >= 2) {
    }
}
FORCE_INLINE void producer_signal(uint32_t sem0, uint32_t sem1) {
    unified_kernels::sem_atomic_inc(sem0);
    unified_kernels::sem_atomic_inc(sem1);
}
FORCE_INLINE void consumer_wait(uint32_t sem) {
    while (unified_kernels::sem_atomic_load(sem) == 0) {
    }
}
FORCE_INLINE void consumer_release(uint32_t sem) { unified_kernels::sem_atomic_dec(sem); }
}  // namespace fmt_sync

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
        uint32_t expert_offsets_l1_addr_,
        uint32_t block_sizes_l1_addr_,
        uint32_t cb_in1_size_bytes_,
        uint32_t noc_max_page_size_,
        uint32_t core_in_bank_idx_,
        uint32_t pipeline_sem_addr_,
        uint32_t next_core_noc_x_,
        uint32_t next_core_noc_y_,
        uint32_t cores_per_bank_,
        uint32_t num_active_experts_,
        uint32_t index_l1_addr_,
        uint32_t cb_fmt_,
        uint32_t fmt_dram_addr_,
        uint32_t fmt_per_expert_bytes_,
        uint32_t fmt_per_core_bytes_,
        uint32_t fmt_cb_l1_addr_,
        uint32_t fmt_cb_page_size_,
        uint32_t fmt_sem_addr_0_,
        uint32_t fmt_sem_addr_1_,
        uint32_t accum_experts_,
        uint32_t index_offset_,
        // K-parallelism: this core's K-slice index and how many cores share its bank for K.
        // DRAM layout for K-split banks must be K-slice outer so each core reads contiguous.
        // Offset into the bank's expert region is set via expert_offsets[] in op.py —
        // the kernel only needs num_subblocks_k_local and k_slice_idx for compute-side
        // activation/fmt pointer arithmetic.
        uint32_t k_parallel_per_bank_,
        uint32_t k_slice_idx_,
        uint32_t num_subblocks_k_local_,
        // Dedicated global sem for K-reduction (passed as address). PACK on the reducer
        // polls it; NCRISC on senders increments. Separate from pipeline_sem (ring protocol).
        uint32_t partial_sem_addr_>
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
        static constexpr uint32_t expert_offsets_l1_addr = expert_offsets_l1_addr_;
        static constexpr uint32_t block_sizes_l1_addr = block_sizes_l1_addr_;
        static constexpr uint32_t cb_in1_size_bytes = cb_in1_size_bytes_;
        static constexpr uint32_t noc_max_page_size = noc_max_page_size_;
        static constexpr uint32_t core_in_bank_idx = core_in_bank_idx_;
        static constexpr uint32_t pipeline_sem_addr = pipeline_sem_addr_;
        static constexpr uint32_t next_core_noc_x = next_core_noc_x_;
        static constexpr uint32_t next_core_noc_y = next_core_noc_y_;
        static constexpr uint32_t cores_per_bank = cores_per_bank_;
        static constexpr uint32_t num_active_experts = num_active_experts_;
        static constexpr uint32_t index_l1_addr = index_l1_addr_;
        static constexpr uint32_t cb_fmt = cb_fmt_;
        static constexpr uint32_t fmt_dram_addr = fmt_dram_addr_;
        static constexpr uint32_t fmt_per_expert_bytes = fmt_per_expert_bytes_;
        static constexpr uint32_t fmt_per_core_bytes = fmt_per_core_bytes_;
        static constexpr uint32_t fmt_cb_l1_addr = fmt_cb_l1_addr_;
        static constexpr uint32_t fmt_cb_page_size = fmt_cb_page_size_;
        static constexpr uint32_t fmt_sem_addr_0 = fmt_sem_addr_0_;
        static constexpr uint32_t fmt_sem_addr_1 = fmt_sem_addr_1_;
        static constexpr bool accum_experts = accum_experts_ != 0;
        static constexpr uint32_t index_offset = index_offset_;
        static constexpr uint32_t k_parallel_per_bank = k_parallel_per_bank_;
        static constexpr uint32_t k_slice_idx = k_slice_idx_;
        static constexpr uint32_t num_subblocks_k_local =
            num_subblocks_k_local_ != 0 ? num_subblocks_k_local_ : num_subblocks_k_ / k_parallel_per_bank_;
        // Cross-core K-reduction: K is split across k_parallel_per_bank cores in the same bank,
        // which produce partials that are reduced via NOC write + pack-with-L1-acc on the reducer.
        static constexpr bool inner_dim_reduction = (k_parallel_per_bank > 1);
        // Reducer = last K-slice core (holds the final sum); senders are earlier slices.
        static constexpr bool is_reducer = (k_slice_idx + 1 == k_parallel_per_bank);
        // Only meaningful inside `if constexpr (inner_dim_reduction)` scope.
        static constexpr bool is_sender = !is_reducer;
        static constexpr uint32_t partial_sem_addr = partial_sem_addr_;
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
        uint32_t index_l1_addr_,
        uint32_t cb_fmt_,
        uint32_t meta_words_per_block_,
        uint32_t in0_page_size_,
        uint32_t fmt_cb_l1_addr_,
        uint32_t fmt_cb_page_size_,
        uint32_t fmt_sem_addr_0_,
        uint32_t fmt_sem_addr_1_,
        uint32_t accum_experts_,
        uint32_t fuse_silu_,
        uint32_t index_offset_,
        // K-parallelism — compute-side mirror of ReaderCTArgs fields.
        uint32_t k_parallel_per_bank_,
        uint32_t k_slice_idx_,
        uint32_t num_subblocks_k_local_,
        uint32_t partial_sem_addr_,
        // Aliased CB over cb_out with a single-tile view of all expert outputs for
        // the post-reduction silu fast-path (tile shape = [silu_tile_h, tile_w]).
        // silu_tile_h is pre-padded by op.py to a valid face_r_dim ∈ {2,4,8,16}.
        uint32_t cb_out_silu_,
        uint32_t silu_tile_h_>
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
        static constexpr uint32_t index_l1_addr = index_l1_addr_;
        static constexpr uint32_t cb_fmt = cb_fmt_;
        static constexpr uint32_t meta_words_per_block = meta_words_per_block_;
        static constexpr uint32_t in0_page_size = in0_page_size_;
        static constexpr uint32_t fmt_cb_l1_addr = fmt_cb_l1_addr_;
        static constexpr uint32_t fmt_cb_page_size = fmt_cb_page_size_;
        static constexpr uint32_t fmt_sem_addr_0 = fmt_sem_addr_0_;
        static constexpr uint32_t fmt_sem_addr_1 = fmt_sem_addr_1_;
        static constexpr bool accum_experts = accum_experts_ != 0;
        static constexpr bool fuse_silu = fuse_silu_ != 0;
        static constexpr uint32_t index_offset = index_offset_;
        static constexpr uint32_t k_parallel_per_bank = k_parallel_per_bank_;
        static constexpr uint32_t k_slice_idx = k_slice_idx_;
        static constexpr uint32_t num_subblocks_k_local =
            num_subblocks_k_local_ != 0 ? num_subblocks_k_local_ : num_subblocks_k_ / k_parallel_per_bank_;
        static constexpr bool inner_dim_reduction = (k_parallel_per_bank > 1);
        static constexpr bool is_reducer = (k_slice_idx + 1 == k_parallel_per_bank);
        // Only meaningful inside `if constexpr (inner_dim_reduction)` scope.
        static constexpr bool is_sender = !is_reducer;
        static constexpr uint32_t partial_sem_addr = partial_sem_addr_;
        static constexpr uint32_t cb_out_silu = cb_out_silu_;
        static constexpr uint32_t silu_tile_h = silu_tile_h_;
    };

    struct WriterCTArgs {};

    // ResetCBIn1 + CBIn1ResetAddr: when looping or chaining Ops that share cb_in1's
    // L1 region, pass the CB's compile-time L1 base so the triple-buffer wrap window
    // stays anchored at a known address regardless of the runtime wr_ptr. Mirrors
    // DRAMStreamingMatmul's pattern. Defaults preserve the single-invocation behavior.
    template <
        typename CTArgs,
        bool IsActiveCore,
        bool pop_in0 = true,
        bool pop_index = true,
        bool ResetCBIn1 = false,
        uint32_t CBIn1ResetAddr = 0,
        // pop_out: when true, the Op drains its own residual cb_out pushes at the
        // end (cb_wait_front + cb_pop_front). Used for looping / chaining so the
        // next iteration's cb_reserve_back has free space. Default false preserves
        // single-invocation behavior where downstream consumes cb_out.
        bool pop_out = false>
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
            // K-split: each core iterates only its own K-slice. num_subblocks_k_local defaults
            // to num_subblocks_k / k_parallel_per_bank (= full num_subblocks_k when k_parallel=1).
            constexpr uint32_t num_subblocks_k_local = CTArgs::num_subblocks_k_local;
            constexpr uint32_t num_iterations = num_subblocks_k_local * (CTArgs::per_core_n / CTArgs::subblock_n);
            constexpr uint32_t max_page_size = CTArgs::noc_max_page_size;
            constexpr uint32_t num_active_experts = CTArgs::num_active_experts;
            constexpr uint32_t num_buffers = 3;
            constexpr uint32_t extra_blocks_in_flight = 1;
            constexpr uint32_t max_subblock_bytes = CTArgs::cb_in1_size_bytes / num_buffers;
            constexpr uint32_t BLOCK_SIZE_UNIT = 64;

            // Read index array from L1 (direct address, no CB API needed).
            volatile tt_l1_ptr uint16_t* index_ptr =
                reinterpret_cast<volatile tt_l1_ptr uint16_t*>(CTArgs::index_l1_addr);
            const volatile uint32_t* expert_offsets =
                reinterpret_cast<const volatile uint32_t*>(CTArgs::expert_offsets_l1_addr);
            const volatile uint16_t* block_sizes =
                reinterpret_cast<const volatile uint16_t*>(CTArgs::block_sizes_l1_addr);

            // Pipeline semaphore for cores_per_bank > 1 (global sem, addr-based).
            volatile tt_l1_ptr uint32_t* sem_ptr =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(CTArgs::pipeline_sem_addr);
            uint64_t next_sem_noc_addr =
                get_noc_addr(CTArgs::next_core_noc_x, CTArgs::next_core_noc_y, CTArgs::pipeline_sem_addr);

            // Double-buffered fmt slot. Toggles per DRAM expert.
            uint32_t fmt_slot = 0;

            // Triple-buffer state PERSISTS across experts. The in-flight block at the
            // end of expert E is drained naturally by expert E+1's first inner-loop
            // barrier (block_trid_to_wait points at E's tail), which hides the barrier
            // latency behind NOC issue of E+1's ramp reads. Only the LAST expert needs
            // a dedicated flush at the very end. The per-iter cb_reserve_back inside
            // the inner loop keeps TRISC within 1 block of NCRISC at all times.
            // Assumes num_iterations >= 2 so fmt signaling lands inside the inner loop.
            cb_reserve_back(CTArgs::cb_in1, tiles_per_block * (extra_blocks_in_flight + 1));
            // l1_write_addr_in1 MUST start at the CB's current wr_ptr so writes
            // stay coherent with the CB's internal tracking (cb_push_back advances
            // wr_ptr by push_count * page_size regardless of where we wrote). When
            // looping, wr_ptr drifts across iters and we need a stable wrap
            // boundary anchor — ResetCBIn1 + CBIn1ResetAddr give us that, mirroring
            // DRAMStreamingMatmul's pattern.
            uint32_t l1_write_addr_in1 = get_write_ptr(CTArgs::cb_in1);
            uint32_t cb_in1_base;
            if constexpr (ResetCBIn1) {
                cb_in1_base = CBIn1ResetAddr;
            } else {
                cb_in1_base = l1_write_addr_in1;
            }
            uint32_t cb_in1_end = cb_in1_base + CTArgs::cb_in1_size_bytes;
            uint32_t num_free_blocks_in_buffer = num_buffers;
            uint32_t curr_block_trid = 1;
            uint32_t block_trid_to_wait = 1;
            uint32_t num_dram_active = 0;

            reset_noc_trid_barrier_counter(NOC_CLEAR_OUTSTANDING_REQ_MASK, noc_index);

            for (uint32_t exp_i = 0; exp_i < num_active_experts; exp_i++) {
                uint32_t raw_idx = static_cast<uint32_t>(index_ptr[exp_i + CTArgs::index_offset]);
                if (is_sram_expert(raw_idx)) {
                    continue;  // bit15=1 → SRAM expert, skip
                }
                uint32_t expert_idx = raw_idx;  // bit15=0, raw value is global expert ID
                // For K-split, expert_offsets already points at this K-slice's start (set in op.py).
                uint32_t expert_in1_addr = expert_offsets[expert_idx];
                uint32_t dram_read_offset = 0;
                const volatile uint16_t* block_size_ptr = block_sizes + expert_idx * num_iterations;

                if constexpr (CTArgs::cores_per_bank > 1 && CTArgs::core_in_bank_idx > 0) {
                    noc_semaphore_wait(sem_ptr, 1);
                    noc_semaphore_set(sem_ptr, 0);
                }

                // Wait for a free fmt slot (at most 1 expert ahead of each TRISC consumer).
                fmt_sync::producer_wait_slot(CTArgs::fmt_sem_addr_0, CTArgs::fmt_sem_addr_1);

                num_dram_active++;

                uint64_t in1_base_addr = get_noc_addr_from_bank_id<true>(CTArgs::bank_id, expert_in1_addr);

                // Read fmt to double-buffered L1 slot. Shares trid with this expert's
                // first weight block (curr_block_trid) so one barrier drains both.
                uint32_t fmt_write_l1 = CTArgs::fmt_cb_l1_addr + fmt_slot * CTArgs::fmt_cb_page_size;
                {
                    uint32_t fmt_dram_offset = CTArgs::fmt_dram_addr +
                                               CTArgs::core_in_bank_idx * CTArgs::fmt_per_core_bytes +
                                               expert_idx * CTArgs::fmt_per_expert_bytes;
                    uint64_t fmt_noc_addr = get_noc_addr_from_bank_id<true>(CTArgs::bank_id, fmt_dram_offset);
                    noc_async_read_set_trid(curr_block_trid);
                    noc_async_read_one_packet_set_state<true>(fmt_noc_addr, CTArgs::fmt_per_expert_bytes, CTArgs::vc);
                    noc_async_read_one_packet_with_state_with_trid(fmt_noc_addr, 0, fmt_write_l1, curr_block_trid);
                }
                uint32_t fmt_trid_pending = curr_block_trid;
                bool fmt_signaled = false;

                // Triple-buffer streaming loop. Ring protocol operates at batch
                // boundaries — every num_buffers iters, plus the final iter of the
                // expert. At each boundary we inc (pass token forward) and, if more
                // batches remain in this expert, wait (receive token back).
                for (uint32_t iter = 0; iter < num_iterations; iter++) {
                    uint32_t block_size = static_cast<uint32_t>(block_size_ptr[iter]) * BLOCK_SIZE_UNIT;
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

                    // Pass the token NOW (before our old-trid barrier below) so the
                    // next core's reads aren't gated on our barrier completing.
                    bool batch_end = ((iter + 1) % num_buffers == 0) || (iter == (num_iterations - 1));
                    if constexpr (CTArgs::cores_per_bank > 1) {
                        if (batch_end) {
                            noc_semaphore_inc(next_sem_noc_addr, 1);
                        }
                    }

                    if (num_free_blocks_in_buffer == num_buffers - extra_blocks_in_flight) {
                        noc_async_read_barrier_with_trid(block_trid_to_wait);
                        cb_push_back(CTArgs::cb_in1, tiles_per_block);
                        // fmt for this expert shares trid with its first weight block.
                        // When the barrier hits that trid, fmt is in L1 — signal consumers.
                        if (!fmt_signaled && block_trid_to_wait == fmt_trid_pending) {
                            fmt_sync::producer_signal(CTArgs::fmt_sem_addr_0, CTArgs::fmt_sem_addr_1);
                            fmt_signaled = true;
                        }
                        block_trid_to_wait = block_trid_to_wait == num_buffers ? 1 : (block_trid_to_wait + 1);
                        // Re-reserve after push: without the per-expert flush, this is the
                        // only throttle keeping TRISC within 1 block of NCRISC. Missing it
                        // allows NCRISC to wrap into a slot TRISC is still reading (race).
                        cb_reserve_back(CTArgs::cb_in1, tiles_per_block * (extra_blocks_in_flight + 1));
                    } else {
                        num_free_blocks_in_buffer -= 1;
                    }

                    curr_block_trid = (curr_block_trid == num_buffers) ? 1 : (curr_block_trid + 1);

                    if (l1_write_addr_in1 >= cb_in1_end) {
                        l1_write_addr_in1 = cb_in1_base;
                    }

                    // Intra-expert batch pacing: wait for the token to return before
                    // the next batch. Skipped on the expert's final batch — that return
                    // is handled by the end-of-expert wait below (or, on the last DRAM
                    // expert, doesn't matter because no more batches will be issued).
                    if constexpr (CTArgs::cores_per_bank > 1) {
                        if (batch_end && (iter < (num_iterations - 1))) {
                            noc_semaphore_wait(sem_ptr, 1);
                            noc_semaphore_set(sem_ptr, 0);
                        }
                    }
                }

                // No per-expert flush: the tail block stays in flight and is drained
                // by the next expert's first inner-loop barrier (or by the final flush
                // below if this is the last DRAM expert).
                //
                // Safety net for the num_iterations == 1 case: the inner-loop push
                // branch only fires when num_free == num_buffers - extra_blocks_in_flight,
                // which never happens on a single iter (num_free goes 3→2 in the else
                // branch and the loop exits). For num_iterations >= 2, iter 1 always
                // fires the if-branch signal path, so this block is compiled out.
                if constexpr (num_iterations == 1) {
                    noc_async_read_barrier_with_trid(fmt_trid_pending);
                    fmt_sync::producer_signal(CTArgs::fmt_sem_addr_0, CTArgs::fmt_sem_addr_1);
                }
                fmt_slot ^= 1;

                // End-of-expert token return: only core 0 waits. Every other core
                // in the ring is naturally gated by its start-of-expert wait for
                // the next expert, so it can't run ahead or overflow its peer's
                // sem. Core 0 lacks that start-wait, so without this block it
                // would race into the next expert while the tail of the ring is
                // still reading DRAM for this one, causing bank contention.
                if constexpr (CTArgs::cores_per_bank > 1 && CTArgs::core_in_bank_idx == 0) {
                    noc_semaphore_wait(sem_ptr, 1);
                    noc_semaphore_set(sem_ptr, 0);
                }
            }

            // Final flush: drain the in-flight block(s) from the LAST DRAM expert.
            if (num_dram_active > 0) {
                for (uint32_t i = 0; i < extra_blocks_in_flight; ++i) {
                    noc_async_read_barrier_with_trid(block_trid_to_wait);
                    cb_push_back(CTArgs::cb_in1, tiles_per_block);
                    block_trid_to_wait = block_trid_to_wait == num_buffers ? 1 : (block_trid_to_wait + 1);
                }
            }

            // Phase-2 K-reduction. Sender (non-last K-slice) waits for its TRISC to finish
            // packing to cb_out, NOC-writes the whole cb_out to the reducer's cb_out L1,
            // and signals the reducer via pipeline_sem. The reducer's TRISC polls that sem,
            // then pack-with-L1-acc onto cb_out to add its own partial. No new CBs or args.
            if constexpr (CTArgs::inner_dim_reduction) {
                if constexpr (CTArgs::is_sender) {
                    if (num_dram_active > 0) {
                        uint32_t output_tiles = num_dram_active * CTArgs::per_core_n;
                        uint32_t output_bytes = output_tiles * get_tile_size(CTArgs::cb_out);

                        cb_wait_front(CTArgs::cb_out, output_tiles);
                        uint32_t src_addr = get_read_ptr(CTArgs::cb_out);
                        // Reducer's cb_out is allocated at the same L1 offset as sender's cb_out
                        // (both are sharded from the output tensor → identical per-core layout).
                        // Note: there can be a race condition if we run the same kernel back-to-back,
                        // where the sender can issue the next iterations' writes before reducer finishes the
                        // reduction, thus overwriting the l1 data. It should be safe in the models since we
                        // don't run back-to-back same kernel.
                        uint64_t dst_noc = get_noc_addr(CTArgs::next_core_noc_x, CTArgs::next_core_noc_y, src_addr);
                        noc_async_write_one_packet(src_addr, dst_noc, output_bytes);
                        uint64_t sem_noc =
                            get_noc_addr(CTArgs::next_core_noc_x, CTArgs::next_core_noc_y, CTArgs::partial_sem_addr);
                        noc_semaphore_inc(sem_noc, 1);

                        // Pop out from the ncrisc side because it needs to perform wait front. Otherwise there is a
                        // race for unpacker pop vs ncrisc wait.
                        if constexpr (pop_out) {
                            constexpr uint32_t max_push = num_active_experts * CTArgs::per_core_n;
                            cb_wait_front(CTArgs::cb_out, max_push);
                            cb_pop_front(CTArgs::cb_out, max_push);
                        }
                    }
                }
            }

#elif defined(COMPILE_FOR_TRISC)
            constexpr uint32_t num_tiles_k = CTArgs::subblock_k * CTArgs::num_subblocks_k;
            constexpr uint32_t tiles_per_expert = CTArgs::subblock_k * CTArgs::num_subblocks_k * CTArgs::per_core_n;
            constexpr uint32_t num_active_experts = CTArgs::num_active_experts;

            cb_wait_front(CTArgs::cb_index, 1);

            volatile tt_l1_ptr uint16_t* index_ptr =
                reinterpret_cast<volatile tt_l1_ptr uint16_t*>(CTArgs::index_l1_addr);

            // Index tensor encodes SRAM/DRAM via bit 15: 1=SRAM, 0=DRAM.
            // index_offset is used in TP=false mode, to offset to the correct expert.
            constexpr uint32_t meta_words_per_block = CTArgs::meta_words_per_block;
            constexpr uint32_t in0_page_size = CTArgs::in0_page_size;
            constexpr uint32_t tiles_per_block = CTArgs::subblock_k * CTArgs::subblock_n;
            constexpr uint32_t num_subblocks_n = CTArgs::per_core_n / CTArgs::subblock_n;
            // K-split: each core iterates only its K-slice. fmt is packed per-K-slice in op.py
            // so TRISC walks it linearly; activation is shared across K-slice cores, so cb_in0
            // rd_ptr offsets into this core's K-slice via act_k_slice_byte_offset (constexpr).
            constexpr uint32_t num_subblocks_k_local = CTArgs::num_subblocks_k_local;
            constexpr uint32_t act_k_slice_byte_offset =
                CTArgs::k_slice_idx * num_subblocks_k_local * CTArgs::subblock_k * in0_page_size;

            reconfig_data_format<false, true>(CTArgs::cb_in1, CTArgs::cb_in0);
            pack_reconfig_data_format<true>(CTArgs::cb_out);
            compressed_custom_mm_block_init_short<false, true, false>(CTArgs::cb_in0, CTArgs::cb_in1, CTArgs::cb_out);
            if constexpr (CTArgs::accum_experts) {
                cb_wait_front(CTArgs::cb_in0, num_tiles_k * num_active_experts);
            } else {
                cb_wait_front(CTArgs::cb_in0, num_tiles_k);
            }

            uint32_t in0_base = 0, in0_cb_base = 0;
            UNPACK(({ in0_cb_base = unified_kernels::get_cb_rd_ptr(CTArgs::cb_in0); }));
            UNPACK(({ in0_base = in0_cb_base + act_k_slice_byte_offset; }));

            if constexpr (CTArgs::accum_experts) {
                uint32_t num_dram_experts = 0;
                for (uint32_t i = 0; i < num_active_experts; i++) {
                    if (!(is_sram_expert(index_ptr[i + CTArgs::index_offset]))) {
                        num_dram_experts++;
                    }
                }

                uint32_t fmt_slot = 0;
                uint32_t dram_idx = 0;
                for (uint32_t exp_i = 0; exp_i < num_active_experts; exp_i++) {
                    uint32_t raw_idx = static_cast<uint32_t>(index_ptr[exp_i + CTArgs::index_offset]);
                    if (is_sram_expert(raw_idx)) {
                        continue;
                    }

                    cb_reserve_back(CTArgs::cb_out, CTArgs::per_core_n);

                    if (dram_idx == 0) {
                        PACK((llk_pack_reconfig_l1_acc(0)));
                    } else if (dram_idx == 1) {
                        PACK((llk_pack_reconfig_l1_acc(1)));
                    }

                    UNPACK((fmt_sync::consumer_wait(CTArgs::fmt_sem_addr_0)));
                    MATH((fmt_sync::consumer_wait(CTArgs::fmt_sem_addr_1)));
                    const volatile uint32_t* fmt_base_ptr = reinterpret_cast<const volatile uint32_t*>(
                        CTArgs::fmt_cb_l1_addr + fmt_slot * CTArgs::fmt_cb_page_size);
                    uint32_t fmt_meta_offset = 0;
                    uint32_t act_rd_ptr = in0_base + exp_i * num_tiles_k * in0_page_size;

                    for (uint32_t ng = 0; ng < num_subblocks_n; ng++) {
                        tile_regs_acquire();

                        for (uint32_t sb_k = 0; sb_k < CTArgs::num_subblocks_k; sb_k++) {
                            cb_wait_front(CTArgs::cb_in1, tiles_per_block);

                            uint32_t meta_addr = reinterpret_cast<uint32_t>(fmt_base_ptr + fmt_meta_offset);

                            UNPACK(({
                                unified_kernels::override_cb_rd_ptr(
                                    CTArgs::cb_in0, act_rd_ptr + sb_k * CTArgs::subblock_k * in0_page_size);
                            }));

                            if (sb_k < CTArgs::num_subblocks_k - 1) {
                                compressed_custom_mm_block<false>(
                                    CTArgs::cb_in0,
                                    CTArgs::cb_in1,
                                    meta_addr,
                                    0,
                                    CTArgs::subblock_k,
                                    CTArgs::subblock_n);
                            } else {
                                compressed_custom_mm_block<true>(
                                    CTArgs::cb_in0,
                                    CTArgs::cb_in1,
                                    meta_addr,
                                    0,
                                    CTArgs::subblock_k,
                                    CTArgs::subblock_n);
                            }

                            fmt_meta_offset += meta_words_per_block;
                            cb_pop_front(CTArgs::cb_in1, tiles_per_block);
                        }

                        tile_regs_commit();
                        tile_regs_wait();
                        for (uint32_t sn = 0; sn < CTArgs::subblock_n; sn++) {
                            pack_tile(sn, CTArgs::cb_out);
                        }
                        tile_regs_release();
                    }

                    UNPACK((fmt_sync::consumer_release(CTArgs::fmt_sem_addr_0)));
                    MATH((fmt_sync::consumer_release(CTArgs::fmt_sem_addr_1)));
                    fmt_slot ^= 1;
                    cb_push_back(CTArgs::cb_out, CTArgs::per_core_n);

                    if (++dram_idx < num_dram_experts) {
                        cb_wait_front(CTArgs::cb_out, CTArgs::per_core_n);
                        cb_pop_front(CTArgs::cb_out, CTArgs::per_core_n);
                    }
                }

                PACK((llk_pack_reconfig_l1_acc(0)));

                // pop_out: drain the final accumulated per_core_n that was
                // pushed above. Skip if no DRAM experts pushed anything.
                if constexpr (pop_out) {
                    if (num_dram_experts > 0) {
                        cb_wait_front(CTArgs::cb_out, CTArgs::per_core_n);
                        cb_pop_front(CTArgs::cb_out, CTArgs::per_core_n);
                    }
                }
            } else if constexpr (CTArgs::inner_dim_reduction) {
                // K-split path: accumulate ALL experts into dst (one tile per expert since
                // num_subblocks_n==1), then pack once at the end. Reducer waits for sender's
                // NOC write BEFORE the single pack phase, so pack-with-L1-acc reads sender's
                // already-landed data. Sender does the same structure with acc=0.
                static_assert(num_subblocks_n == 1, "K-split requires num_subblocks_n == 1");

                tile_regs_acquire();

                uint32_t fmt_slot = 0;
                uint32_t dst_base = 0;
                for (uint32_t exp_i = 0; exp_i < num_active_experts; exp_i++) {
                    uint32_t raw_idx = static_cast<uint32_t>(index_ptr[exp_i + CTArgs::index_offset]);
                    if (is_sram_expert(raw_idx)) {
                        continue;
                    }

                    UNPACK((fmt_sync::consumer_wait(CTArgs::fmt_sem_addr_0)));
                    MATH((fmt_sync::consumer_wait(CTArgs::fmt_sem_addr_1)));
                    const volatile uint32_t* fmt_base_ptr = reinterpret_cast<const volatile uint32_t*>(
                        CTArgs::fmt_cb_l1_addr + fmt_slot * CTArgs::fmt_cb_page_size);
                    uint32_t fmt_meta_offset = 0;

                    for (uint32_t sb_k = 0; sb_k < num_subblocks_k_local; sb_k++) {
                        cb_wait_front(CTArgs::cb_in1, tiles_per_block);
                        UNPACK(({
                            unified_kernels::override_cb_rd_ptr(
                                CTArgs::cb_in0, in0_base + sb_k * CTArgs::subblock_k * in0_page_size);
                        }));
                        uint32_t meta_addr = reinterpret_cast<uint32_t>(fmt_base_ptr + fmt_meta_offset);
                        if (sb_k < num_subblocks_k_local - 1) {
                            compressed_custom_mm_block<false>(
                                CTArgs::cb_in0,
                                CTArgs::cb_in1,
                                meta_addr,
                                dst_base,
                                CTArgs::subblock_k,
                                CTArgs::subblock_n);
                        } else {
                            compressed_custom_mm_block<true>(
                                CTArgs::cb_in0,
                                CTArgs::cb_in1,
                                meta_addr,
                                dst_base,
                                CTArgs::subblock_k,
                                CTArgs::subblock_n);
                        }
                        fmt_meta_offset += meta_words_per_block;
                        cb_pop_front(CTArgs::cb_in1, tiles_per_block);
                    }

                    UNPACK((fmt_sync::consumer_release(CTArgs::fmt_sem_addr_0)));
                    MATH((fmt_sync::consumer_release(CTArgs::fmt_sem_addr_1)));
                    fmt_slot ^= 1;
                    dst_base += CTArgs::subblock_n;
                }

                // dst_base is now num_active_dram_experts * subblock_n (total packed tiles).
                const uint32_t total_tiles = dst_base;

                tile_regs_commit();
                cb_reserve_back(CTArgs::cb_out, total_tiles);
                tile_regs_wait();

                // Skip pack / silu / sem-reset when zero DRAM experts: no real
                // data in dst, reducer would deadlock on partial_sem (sender
                // didn't inc it), silu would operate on stale L1. tile_regs_release
                // still fires in the else branch to close out the acquire above.
                if (total_tiles > 0) {
                    // Reducer: wait for sender's NOC write then pack-with-L1-acc
                    // to sum sender_partial + reducer_partial in L1 as a RAW sum.
                    if constexpr (CTArgs::is_reducer) {
                        constexpr uint32_t num_senders = CTArgs::k_parallel_per_bank - 1;
                        PACK(({
                            volatile tt_l1_ptr uint32_t* partial_sem_ptr =
                                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(CTArgs::partial_sem_addr);
                            while (*partial_sem_ptr < num_senders) {
                            }
                            unified_kernels::sem_atomic_dec(CTArgs::partial_sem_addr, num_senders);
                        }));
                        PACK((llk_pack_reconfig_l1_acc(1)));
                    }

                    for (uint32_t t = 0; t < total_tiles; t++) {
                        pack_tile(t, CTArgs::cb_out);
                    }
                    tile_regs_release();
                    cb_push_back(CTArgs::cb_out, total_tiles);
                    // Wait here is needed to trisc0 to block until the current code block finishes,
                    // otherwise will perform silu init / unpack too soon.
                    cb_wait_front(CTArgs::cb_out, total_tiles);

                    if constexpr (CTArgs::is_reducer) {
                        PACK((llk_pack_reconfig_l1_acc(0)));
                    }

                    // Post-reduction silu fast-path (reducer only). cb_out_silu is a view
                    // of cb_out's L1 with tile shape [silu_tile_h, tile_w] — op.py pads
                    // silu_tile_h up to the next valid face_r_dim {2,4,8,16}, so one
                    // copy_tile + silu + pack_tile covers all expert outputs in one fat tile.
                    if constexpr (CTArgs::fuse_silu && CTArgs::is_reducer) {
                        // Custom silu: iterations = silu_tile_h/2 (covers both faces
                        // of a partial-face [silu_tile_h, tile_w] tile, each face
                        // holding silu_tile_h * 16 elems / 32 lanes). VectorMode::R
                        // processes by row so the partial-face padding above
                        // silu_tile_h isn't touched. Goes through MATH (same slot
                        // as silu_tile) so the tile_regs commit/wait flow is clean.
                        constexpr uint32_t silu_iterations = CTArgs::silu_tile_h / 2;

                        reconfig_data_format_srca<false, true>(CTArgs::cb_out_silu);
                        pack_reconfig_data_format<true>(CTArgs::cb_out_silu);
                        copy_tile_to_dst_init_short(CTArgs::cb_out_silu);
                        silu_tile_init();

                        cb_reserve_back(CTArgs::cb_out_silu, 1);
                        tile_regs_acquire();
                        copy_tile(CTArgs::cb_out_silu, 0, 0);
                        MATH((llk_math_eltwise_unary_sfpu_silu<true, DST_ACCUM_MODE, silu_iterations>(
                            0, (int)VectorMode::R)));
                        tile_regs_commit();
                        tile_regs_wait();
                        pack_tile(0, CTArgs::cb_out_silu, 0);
                        tile_regs_release();
                        cb_push_back(CTArgs::cb_out_silu, 1);
                        cb_wait_front(CTArgs::cb_out_silu, 1);
                        cb_pop_front(CTArgs::cb_out_silu, 1);
                    }
                } else {
                    tile_regs_release();
                }

                // Pad total pushes to num_active_experts * per_core_n (here
                // per_core_n == subblock_n since K-split requires num_subblocks_n == 1)
                // so CB push count is deterministic across invocations.
                constexpr uint32_t max_push = num_active_experts * CTArgs::per_core_n;
                const uint32_t padding = max_push - total_tiles;
                if (padding > 0) {
                    cb_reserve_back(CTArgs::cb_out, padding);
                    cb_push_back(CTArgs::cb_out, padding);
                }
                // Do not pop here for sender becuase that is handled on ncrisc side to avoid race.
                if constexpr (CTArgs::is_reducer) {
                    if constexpr (pop_out) {
                        constexpr uint32_t max_push = num_active_experts * CTArgs::per_core_n;
                        cb_wait_front(CTArgs::cb_out, max_push);
                        cb_pop_front(CTArgs::cb_out, max_push);
                    }
                }
            } else {
                uint32_t fmt_slot = 0;
                uint32_t num_dram_pushed = 0;

                for (uint32_t exp_i = 0; exp_i < num_active_experts; exp_i++) {
                    uint32_t raw_idx = static_cast<uint32_t>(index_ptr[exp_i + CTArgs::index_offset]);
                    if (is_sram_expert(raw_idx)) {
                        continue;
                    }

                    UNPACK((fmt_sync::consumer_wait(CTArgs::fmt_sem_addr_0)));
                    MATH((fmt_sync::consumer_wait(CTArgs::fmt_sem_addr_1)));
                    const volatile uint32_t* fmt_base_ptr = reinterpret_cast<const volatile uint32_t*>(
                        CTArgs::fmt_cb_l1_addr + fmt_slot * CTArgs::fmt_cb_page_size);
                    uint32_t fmt_meta_offset = 0;

                    for (uint32_t ng = 0; ng < num_subblocks_n; ng++) {
                        cb_reserve_back(CTArgs::cb_out, CTArgs::subblock_n);
                        tile_regs_acquire();

                        for (uint32_t sb_k = 0; sb_k < CTArgs::num_subblocks_k; sb_k++) {
                            cb_wait_front(CTArgs::cb_in1, tiles_per_block);

                            UNPACK(({
                                unified_kernels::override_cb_rd_ptr(
                                    CTArgs::cb_in0, in0_base + sb_k * CTArgs::subblock_k * in0_page_size);
                            }));

                            uint32_t meta_addr = reinterpret_cast<uint32_t>(fmt_base_ptr + fmt_meta_offset);

                            if (sb_k < CTArgs::num_subblocks_k - 1) {
                                compressed_custom_mm_block<false>(
                                    CTArgs::cb_in0,
                                    CTArgs::cb_in1,
                                    meta_addr,
                                    0,
                                    CTArgs::subblock_k,
                                    CTArgs::subblock_n);
                            } else {
                                compressed_custom_mm_block<true>(
                                    CTArgs::cb_in0,
                                    CTArgs::cb_in1,
                                    meta_addr,
                                    0,
                                    CTArgs::subblock_k,
                                    CTArgs::subblock_n);
                            }

                            fmt_meta_offset += meta_words_per_block;
                            cb_pop_front(CTArgs::cb_in1, tiles_per_block);
                        }

                        tile_regs_commit();
                        if constexpr (CTArgs::fuse_silu) {
                            PACK(TTI_SEMWAIT(
                                p_stall::STALL_TDMA | p_stall::STALL_CFG,
                                semaphore::t6_sem(semaphore::MATH_PACK),
                                p_stall::STALL_ON_ZERO));
                            PACK(TT_SETC16(
                                DEST_TARGET_REG_CFG_MATH_Offset_ADDR32, ckernel::packer::get_packer_dest_offset()));
                            for (uint32_t sn = 0; sn < CTArgs::subblock_n; sn++) {
                                PACK((llk_math_eltwise_unary_sfpu_silu<false, false, 2>(sn, (int)VectorMode::R)));
                            }
                            PACK(TTI_STALLWAIT(p_stall::STALL_PACK, p_stall::WAIT_SFPU));
                        } else {
                            tile_regs_wait();
                        }
                        for (uint32_t sn = 0; sn < CTArgs::subblock_n; sn++) {
                            pack_tile(sn, CTArgs::cb_out, 0);
                        }
                        tile_regs_release();
                        cb_push_back(CTArgs::cb_out, CTArgs::subblock_n);
                    }

                    UNPACK((fmt_sync::consumer_release(CTArgs::fmt_sem_addr_0)));
                    MATH((fmt_sync::consumer_release(CTArgs::fmt_sem_addr_1)));
                    fmt_slot ^= 1;
                    num_dram_pushed++;
                }

                // Pad total pushes to num_active_experts * per_core_n so the CB
                // push count is deterministic across invocations (independent of
                // runtime SRAM/DRAM split). The dummy slots carry no real data
                // and are sized to fill the output tensor's per-expert slots.
                const uint32_t padding = (num_active_experts - num_dram_pushed) * CTArgs::per_core_n;
                if (padding > 0) {
                    cb_reserve_back(CTArgs::cb_out, padding);
                    cb_push_back(CTArgs::cb_out, padding);
                }

                if constexpr (pop_out) {
                    constexpr uint32_t max_push = num_active_experts * CTArgs::per_core_n;
                    cb_wait_front(CTArgs::cb_out, max_push);
                    cb_pop_front(CTArgs::cb_out, max_push);
                }
            }

            // Restore cb_in0's UNPACK rd_ptr to its original shard-start value
            // before cb_pop_front. The compute loop overrides rd_ptr to per-expert /
            // per-subblock positions, and cb_pop_front advances from whatever
            // override is in place — leaving it unset would make the next
            // invocation's get_cb_rd_ptr return a stale mid-compute position.
            UNPACK(({ unified_kernels::override_cb_rd_ptr(CTArgs::cb_in0, in0_cb_base); }));
            compressed_custom_mm_block_uninit<false>();

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
