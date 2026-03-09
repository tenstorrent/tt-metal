// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// Prefill MoE multi-expert reader (dm0/RISCV_0) with FPU combine
//
// Phase 1 — Expert compute reads (same as before):
//   Per expert: read pkt_buf activation tiles → CB0, inter tiles → CB0
//   Barriers between phases, SEM_EXPERT_DONE after each expert
//
// Phase 2 — FPU combine reads (NEW):
//   After all experts complete, read output + expert out_buf tiles for
//   weighted accumulation. Each core handles n_per_core_dn tile columns.
//   For each expert e, for each tile row tr, for each tile d:
//     Read output tile at (tr * n_tiles_dn + core_d_offset + d) → CB3
//     Read expert out_buf tile at same page → CB4
//   After all experts processed, write-back is done by the writer.
//
// Runtime args layout:
//   [0]  pkt_buf_addr
//   [1]  k_tiles_gu
//   [2]  inter_addr
//   [3]  is_leader
//   [4]  num_cores
//   [5]  combine_phys_x  (kept for backward compat, unused in FPU combine mode)
//   [6]  combine_phys_y  (kept for backward compat, unused in FPU combine mode)
//   [7]  num_experts
//   [8]  k_tiles_dn
//   [9]  M_tiles
//   [10..10+2*num_cores-1]  physical coords of all compute cores
//   --- FPU combine args (when ENABLE_FPU_COMBINE is defined) ---
//   [combine_base+0]  output_addr
//   [combine_base+1]  n_tiles_dn (total D columns in tiles)
//   [combine_base+2]  core_d_offset (starting D tile column for this core)
//   [combine_base+3]  n_per_core_dn (D tile columns per core)
//   [combine_base+4]  output_M_tiles
//   [combine_base+5..5+num_experts-1]  out_buf_addr[e] for each expert
//
// Semaphores:
//   SEM_BARRIER (id=0): Cross-core barrier
//   SEM_GO (id=1): Leader -> all cores sync
//   SEM_PKT_READY (id=2): Dispatch -> leader
//   SEM_EXPERT_DONE (id=3): Leader -> combine core (scalar) or self-combine (FPU)

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t pkt_buf_addr = get_arg_val<uint32_t>(0);
    const uint32_t k_tiles_gu = get_arg_val<uint32_t>(1);
    const uint32_t inter_addr = get_arg_val<uint32_t>(2);
    const uint32_t is_leader = get_arg_val<uint32_t>(3);
    const uint32_t num_cores = get_arg_val<uint32_t>(4);
    const uint32_t combine_phys_x = get_arg_val<uint32_t>(5);
    const uint32_t combine_phys_y = get_arg_val<uint32_t>(6);
    const uint32_t num_experts = get_arg_val<uint32_t>(7);
    const uint32_t k_tiles_dn = get_arg_val<uint32_t>(8);
    const uint32_t M_tiles = get_arg_val<uint32_t>(9);

    constexpr auto act_args = TensorAccessorArgs<0>();
    constexpr auto inter_args = TensorAccessorArgs<1>();

    constexpr uint32_t cb_act = 0;
    constexpr uint32_t SEM_BARRIER = 0;
    constexpr uint32_t SEM_GO = 1;
    constexpr uint32_t SEM_PKT_READY = 2;
    constexpr uint32_t SEM_EXPERT_DONE = 3;

    const uint32_t page_bytes = get_local_cb_interface(cb_act).fifo_page_size;
    const auto act_accessor = TensorAccessor(act_args, pkt_buf_addr, page_bytes);
    const auto inter_accessor = TensorAccessor(inter_args, inter_addr, page_bytes);

    const uint32_t expert_page_stride = M_tiles * k_tiles_gu;

    // Pre-dispatch: Leader waits for pkt_buf
    if (is_leader) {
        volatile tt_l1_ptr uint32_t* pkt_sem =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SEM_PKT_READY));
        noc_semaphore_wait(pkt_sem, 1);
        noc_semaphore_set(pkt_sem, 0);
    }

    // ========== Phase 1: Expert compute reads ==========
    for (uint32_t expert = 0; expert < num_experts; ++expert) {
        uint32_t expert_page_offset = expert * expert_page_stride;

        // Start-of-expert sync
        if (is_leader) {
            uint32_t sem_go_l1_addr = get_semaphore(SEM_GO);
            for (uint32_t c = 0; c < num_cores; ++c) {
                uint32_t phys_x = get_arg_val<uint32_t>(10 + c * 2);
                uint32_t phys_y = get_arg_val<uint32_t>(10 + c * 2 + 1);
                uint64_t core_sem_noc = get_noc_addr(phys_x, phys_y, sem_go_l1_addr);
                noc_semaphore_inc(core_sem_noc, 1);
            }
        }
        {
            volatile tt_l1_ptr uint32_t* go_sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SEM_GO));
            noc_semaphore_wait(go_sem, 1);
            noc_semaphore_set(go_sem, 0);
        }

        // Phase A: Read activation tiles
        for (uint32_t m = 0; m < M_tiles; ++m) {
            for (uint32_t k = 0; k < k_tiles_gu; ++k) {
                cb_reserve_back(cb_act, 1);
                uint32_t l1_write_addr = get_write_ptr(cb_act);
                uint32_t page_idx = expert_page_offset + m * k_tiles_gu + k;
                noc_async_read_page(page_idx, act_accessor, l1_write_addr);
                noc_async_read_barrier();
                cb_push_back(cb_act, 1);
            }
        }

        // Barrier A
        if (is_leader) {
            volatile tt_l1_ptr uint32_t* barrier_sem =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SEM_BARRIER));
            noc_semaphore_wait(barrier_sem, num_cores);
            noc_semaphore_set(barrier_sem, 0);

            uint32_t sem_go_l1_addr = get_semaphore(SEM_GO);
            for (uint32_t c = 0; c < num_cores; ++c) {
                uint32_t phys_x = get_arg_val<uint32_t>(10 + c * 2);
                uint32_t phys_y = get_arg_val<uint32_t>(10 + c * 2 + 1);
                uint64_t core_sem_noc = get_noc_addr(phys_x, phys_y, sem_go_l1_addr);
                noc_semaphore_inc(core_sem_noc, 1);
            }
        }
        {
            volatile tt_l1_ptr uint32_t* go_sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SEM_GO));
            noc_semaphore_wait(go_sem, 1);
            noc_semaphore_set(go_sem, 0);
        }

        // Phase B: Read intermediate tiles
        for (uint32_t m = 0; m < M_tiles; ++m) {
            for (uint32_t k = 0; k < k_tiles_dn; ++k) {
                cb_reserve_back(cb_act, 1);
                uint32_t l1_write_addr = get_write_ptr(cb_act);
                uint32_t page_idx = m * k_tiles_dn + k;
                noc_async_read_page(page_idx, inter_accessor, l1_write_addr);
                noc_async_read_barrier();
                cb_push_back(cb_act, 1);
            }
        }

        // Barrier B + SEM_EXPERT_DONE
        if (is_leader) {
            volatile tt_l1_ptr uint32_t* barrier_sem =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SEM_BARRIER));
            noc_semaphore_wait(barrier_sem, num_cores);
            noc_semaphore_set(barrier_sem, 0);

#if !defined(ENABLE_FPU_COMBINE)
            // Scalar combine mode: signal the separate combine core
            uint64_t combine_sem_addr = get_noc_addr(combine_phys_x, combine_phys_y, get_semaphore(SEM_EXPERT_DONE));
            noc_semaphore_inc(combine_sem_addr, 1);
            noc_async_write_barrier();
#endif
        }
    }

    // ========== Phase 2: FPU combine reads ==========
#if defined(ENABLE_FPU_COMBINE)
    {
        constexpr uint32_t cb_combine_out = 3;
        constexpr uint32_t cb_combine_exp = 4;

        // Combine RT args start after physical coords
        const uint32_t combine_base = 10 + 2 * num_cores;

        const uint32_t output_addr = get_arg_val<uint32_t>(combine_base + 0);
        const uint32_t n_tiles_dn = get_arg_val<uint32_t>(combine_base + 1);
        const uint32_t core_d_offset = get_arg_val<uint32_t>(combine_base + 2);
        const uint32_t n_per_core_dn_combine = get_arg_val<uint32_t>(combine_base + 3);
        const uint32_t output_M_tiles = get_arg_val<uint32_t>(combine_base + 4);

        // TensorAccessor for output and out_bufs (same page format as cb_act = BF16 tiles)
        constexpr auto output_args = TensorAccessorArgs<2>();
        const uint32_t combine_page_bytes = get_local_cb_interface(cb_combine_out).fifo_page_size;
        const auto output_accessor = TensorAccessor(output_args, output_addr, combine_page_bytes);

        // Leader broadcasts SEM_GO to start combine on all cores
        if (is_leader) {
            uint32_t sem_go_l1_addr = get_semaphore(SEM_GO);
            for (uint32_t c = 0; c < num_cores; ++c) {
                uint32_t phys_x = get_arg_val<uint32_t>(10 + c * 2);
                uint32_t phys_y = get_arg_val<uint32_t>(10 + c * 2 + 1);
                uint64_t core_sem_noc = get_noc_addr(phys_x, phys_y, sem_go_l1_addr);
                noc_semaphore_inc(core_sem_noc, 1);
            }
        }
        {
            volatile tt_l1_ptr uint32_t* go_sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SEM_GO));
            noc_semaphore_wait(go_sem, 1);
            noc_semaphore_set(go_sem, 0);
        }

        for (uint32_t expert = 0; expert < num_experts; ++expert) {
            uint32_t out_buf_addr = get_arg_val<uint32_t>(combine_base + 5 + expert);
            const auto exp_accessor = TensorAccessor(output_args, out_buf_addr, combine_page_bytes);

            for (uint32_t tr = 0; tr < output_M_tiles; ++tr) {
                for (uint32_t d = 0; d < n_per_core_dn_combine; ++d) {
                    uint32_t page_idx = tr * n_tiles_dn + core_d_offset + d;

                    // Read output tile → CB3
                    cb_reserve_back(cb_combine_out, 1);
                    uint32_t out_l1 = get_write_ptr(cb_combine_out);
                    noc_async_read_page(page_idx, output_accessor, out_l1);
                    noc_async_read_barrier();
                    cb_push_back(cb_combine_out, 1);

                    // Read expert out_buf tile → CB4
                    cb_reserve_back(cb_combine_exp, 1);
                    uint32_t exp_l1 = get_write_ptr(cb_combine_exp);
                    noc_async_read_page(page_idx, exp_accessor, exp_l1);
                    noc_async_read_barrier();
                    cb_push_back(cb_combine_exp, 1);
                }
            }
        }

        // Final barrier: leader waits for all cores to finish combine writes
        if (is_leader) {
            volatile tt_l1_ptr uint32_t* barrier_sem =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SEM_BARRIER));
            noc_semaphore_wait(barrier_sem, num_cores);
            noc_semaphore_set(barrier_sem, 0);
        }
    }
#endif
}
