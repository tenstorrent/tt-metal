// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

// SRAM expert matmul with compressed weights.
//
// B data is pre-loaded in L1 (WIDTH_SHARDED or HEIGHT_SHARDED for K-slicing).
// The Op loops over the index array, skips DRAM experts (bit15=0), and
// performs matmul for each SRAM expert (bit15=1) using per-core fmt tables.

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
        uint32_t index_l1_addr_,
        uint32_t sram_base_addrs_l1_addr_,
        uint32_t meta_words_per_expert_,
        uint32_t in0_page_size_,
        uint32_t accum_experts_ = 0,
        uint32_t sram_k_per_core_ = 0,
        uint32_t sram_k_offset_ = 0,
        uint32_t cb_out_sram_ = 0,
        uint32_t compact_in0_ = 0,
        // Mirror of MatmulExpertCompressedDRAM::ComputeCTArgs::enable_indexing.
        // 1 (default) = read index_ptr[exp_i] from L1 (routed MoE path).
        // 0 = synthesize raw_idx = EXPERT_SRAM_FLAG | exp_i — every iteration
        //     treated as a SRAM-flagged expert with slot=exp_i. Used by the
        //     dense-MLP path (no routing): no mcast_index, no scan; op.py
        //     wires index_l1_addr=0 and cb_index=0. Symmetric with the DRAM
        //     kernel where enable_indexing=0 synthesizes raw_idx = exp_i
        //     (DRAM-flagged), so each kernel runs all iters as its native type.
        uint32_t enable_indexing_ = 1,
        // 1 (default) = use compressed_custom_mm path (per-tile format codes
        // read from the fmt table). Required for BSPM (mixed-precision tiles).
        // 0 = use plain custom_mm path (uniform precision from the cb_in1 tile
        // descriptor). Faster when every tile is the same format — the fmt
        // table read and meta_addr arithmetic are skipped at compile time.
        uint32_t use_compression_ = 1>
    struct ComputeCTArgs {
        static constexpr uint32_t cb_in0 = cb_in0_;
        static constexpr uint32_t cb_in1 = cb_in1_;
        static constexpr uint32_t cb_out = cb_out_;
        static constexpr uint32_t cb_index = cb_index_;
        static constexpr uint32_t num_tiles_k = num_tiles_k_;
        static constexpr uint32_t out_w = out_w_;
        static constexpr uint32_t fmt_l1_addr = fmt_l1_addr_;
        static constexpr uint32_t num_active_experts = num_active_experts_;
        static constexpr uint32_t index_l1_addr = index_l1_addr_;
        static constexpr uint32_t sram_base_addrs_l1_addr = sram_base_addrs_l1_addr_;
        static constexpr uint32_t meta_words_per_expert = meta_words_per_expert_;
        static constexpr uint32_t in0_page_size = in0_page_size_;
        static constexpr bool accum_experts = accum_experts_ != 0;
        static constexpr uint32_t sram_k_per_core = sram_k_per_core_;
        static constexpr uint32_t sram_k_offset = sram_k_offset_;
        static constexpr uint32_t cb_out_sram = cb_out_sram_;
        // compact_in0: when true, cb_in0 holds num_sram_experts × num_tiles_k tiles
        // (compact, indexed by sram_idx) instead of num_active_experts × num_tiles_k
        // (expanded, indexed by exp_i). Used by SRAM down_proj where the mcast
        // dst CB only carries the SRAM-flagged TopK experts' GR outputs.
        static constexpr bool compact_in0 = compact_in0_ != 0;
        static constexpr bool enable_indexing = enable_indexing_ != 0;
        static constexpr bool use_compression = use_compression_ != 0;
    };

    struct WriterCTArgs {};

    template <
        typename CTArgs,
        bool IsActiveCore,
        bool pop_in0 = true,
        bool pop_in1 = true,
        bool pop_index = true,
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
            // NOTE: cb_wait_front(cb_index) + TRISC mailbox sync is performed
            // ONCE globally in moe_kernel.cpp right after index_mcast (drives
            // n_sram_active scan), so this kernel can read index_ptr directly.

            // Index tensor encodes SRAM/DRAM via bit 15: 1=SRAM, 0=DRAM.
            // Lower 15 bits hold the compact SRAM slot index (direct fmt table index).
            volatile tt_l1_ptr uint16_t* index_ptr =
                reinterpret_cast<volatile tt_l1_ptr uint16_t*>(CTArgs::index_l1_addr);

            constexpr uint32_t meta_words_per_expert = CTArgs::meta_words_per_expert;
            constexpr uint32_t in0_page_size = CTArgs::in0_page_size;
            const volatile uint32_t* sram_base_addrs =
                reinterpret_cast<const volatile uint32_t*>(CTArgs::sram_base_addrs_l1_addr);
            const volatile uint32_t* fmt_base = reinterpret_cast<const volatile uint32_t*>(fmt_l1_addr_base);

            reconfig_data_format<false, true>(cb_in1, cb_in0);
            pack_reconfig_data_format<true>(cb_out);
            // cb_in0 metadata always advances by the full num_active_experts ×
            // num_tiles_k pages per iter (producer pads if data is compact). This
            // keeps rd/wr ptrs aligned with the CB capacity across iters — any
            // partial advance would drift the wraparound point and corrupt next
            // iter's reads. compact_in0 only changes which OFFSET we read from
            // within that fixed-size window.
            if constexpr (CTArgs::accum_experts) {
                cb_wait_front(cb_in0, num_tiles_k * num_active_experts);
            } else {
                cb_wait_front(cb_in0, num_tiles_k);
            }
            if constexpr (CTArgs::use_compression) {
                compressed_custom_mm_block_init_short<false, true, true>(cb_in0, cb_in1, cb_out);
            } else {
                custom_mm_block_init_short<false, true, true>(cb_in0, cb_in1, cb_out, out_w);
            }

            uint32_t in0_base = 0;
            UNPACK(({ in0_base = unified_kernels::get_cb_rd_ptr(cb_in0); }));

            if constexpr (CTArgs::accum_experts) {
                // When enable_indexing=false (dense MLP), every iter is SRAM-flagged
                // so num_sram_experts is a CT-known constant.
                uint32_t num_sram_experts;
                if constexpr (!CTArgs::enable_indexing) {
                    num_sram_experts = num_active_experts;
                } else {
                    num_sram_experts = 0;
                    for (uint32_t i = 0; i < num_active_experts; i++) {
                        if (is_sram_expert(index_ptr[i])) {
                            num_sram_experts++;
                        }
                    }
                }

                if (num_sram_experts > 0) {
                    cb_reserve_back(cb_out, out_w);
                    tile_regs_acquire();

                    uint32_t sram_idx = 0;
                    for (uint32_t exp_i = 0; exp_i < num_active_experts; exp_i++) {
                        uint32_t raw_idx;
                        if constexpr (CTArgs::enable_indexing) {
                            raw_idx = static_cast<uint32_t>(index_ptr[exp_i]);
                        } else {
                            raw_idx = EXPERT_SRAM_FLAG | exp_i;  // synthesized: slot=exp_i, SRAM-flagged
                        }
                        if (!(is_sram_expert(raw_idx))) {
                            continue;
                        }

                        uint32_t slot = expert_slot(raw_idx);
                        uint32_t expert_base = sram_base_addrs[slot];
                        uint32_t meta_addr = reinterpret_cast<uint32_t>(fmt_base + slot * meta_words_per_expert);
                        // compact_in0=1: cb_in0 metadata is full-size (num_active_experts ×
                        // num_tiles_k pages, padded by the producer) but the actual data is
                        // packed compactly at offsets 0..num_sram_experts-1. So index by
                        // the running sram_idx (compact) instead of exp_i (TopK slot, which
                        // has DRAM gaps).
                        const uint32_t in0_slot_idx = CTArgs::compact_in0 ? sram_idx : exp_i;

                        UNPACK(({
                            unified_kernels::override_cb_rd_ptr(cb_in1, expert_base);
                            unified_kernels::override_cb_rd_ptr(
                                cb_in0, in0_base + (k_offset + in0_slot_idx * num_tiles_k) * in0_page_size);
                        }));

                        if (++sram_idx < num_sram_experts) {
                            if constexpr (CTArgs::use_compression) {
                                compressed_custom_mm_block<false>(cb_in0, cb_in1, meta_addr, 0, k_for_mm, out_w);
                            } else {
                                custom_mm_block<false>(cb_in0, cb_in1, 0, 0, 0, k_for_mm, out_w);
                            }
                        } else {
                            if constexpr (CTArgs::use_compression) {
                                compressed_custom_mm_block<true>(cb_in0, cb_in1, meta_addr, 0, k_for_mm, out_w);
                            } else {
                                custom_mm_block<true>(cb_in0, cb_in1, 0, 0, 0, k_for_mm, out_w);
                            }
                        }
                    }

                    tile_regs_commit();
                    tile_regs_wait();
                    for (uint32_t w = 0; w < out_w; w++) {
                        pack_tile(w, cb_out, w);
                    }
                    tile_regs_release();
                    cb_push_back(cb_out, out_w);

                    if constexpr (pop_out) {
                        cb_wait_front(cb_out, out_w);
                        cb_pop_front(cb_out, out_w);
                    }
                }
            } else {
                if constexpr (k_offset > 0) {
                    UNPACK(({ unified_kernels::override_cb_rd_ptr(cb_in0, in0_base + k_offset * in0_page_size); }));
                }

                uint32_t num_sram_experts_pushed = 0;
                for (uint32_t exp_i = 0; exp_i < num_active_experts; exp_i++) {
                    uint32_t raw_idx;
                    if constexpr (CTArgs::enable_indexing) {
                        raw_idx = static_cast<uint32_t>(index_ptr[exp_i]);
                    } else {
                        raw_idx = EXPERT_SRAM_FLAG | exp_i;  // synthesized: slot=exp_i, SRAM-flagged
                    }
                    if (!(is_sram_expert(raw_idx))) {
                        continue;
                    }

                    uint32_t slot = expert_slot(raw_idx);
                    uint32_t expert_base = sram_base_addrs[slot];
                    uint32_t meta_addr = reinterpret_cast<uint32_t>(fmt_base + slot * meta_words_per_expert);

                    UNPACK(({ unified_kernels::override_cb_rd_ptr(cb_in1, expert_base); }));

                    cb_reserve_back(cb_out, out_w);
                    tile_regs_acquire();

                    if constexpr (CTArgs::use_compression) {
                        compressed_custom_mm_block<true>(cb_in0, cb_in1, meta_addr, 0, k_for_mm, out_w);
                    } else {
                        // split_acc=false at init → finalize MUST be false (custom_mm.h:131:
                        // "finalize ... must be false if split_acc is false"). With
                        // finalize=true and no split partials, the LLK runs replay+ELWADD
                        // against bogus dst contents and corrupts the accumulated K-sum
                        // (manifested as PCC ~0.01 in the unit test).
                        custom_mm_block<true>(cb_in0, cb_in1, 0, 0, 0, k_for_mm, out_w);
                    }

                    tile_regs_commit();
                    tile_regs_wait();
                    for (uint32_t w = 0; w < out_w; w++) {
                        pack_tile(w, cb_out, w);
                    }
                    tile_regs_release();
                    num_sram_experts_pushed++;
                    cb_push_back(cb_out, out_w);
                }

                // Pad total pushes to num_active_experts * out_w so CB push count
                // is deterministic across invocations (independent of SRAM/DRAM split).
                const uint32_t padding = (num_active_experts - num_sram_experts_pushed) * out_w;
                if (padding > 0) {
                    cb_reserve_back(cb_out, padding);
                    cb_push_back(cb_out, padding);
                }

                if constexpr (pop_out) {
                    constexpr uint32_t max_push = num_active_experts * out_w;
                    cb_wait_front(cb_out, max_push);
                    cb_pop_front(cb_out, max_push);
                }
            }

            UNPACK(({ unified_kernels::override_cb_rd_ptr(cb_in0, in0_base); }));
            if constexpr (CTArgs::use_compression) {
                compressed_custom_mm_block_uninit<true>();
            } else {
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
