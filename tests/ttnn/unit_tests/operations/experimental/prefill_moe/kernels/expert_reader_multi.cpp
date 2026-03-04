// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// Prefill MoE multi-expert reader (dm0/RISCV_0)
//
// Dispatch writes pkt_buf once. This kernel loops over num_experts,
// re-reading the same pkt_buf each iteration (same activation for all experts).
//
// Flow per expert:
//   Leader broadcasts SEM_GO -> all cores wait SEM_GO (start-of-expert sync)
//   Phase A: Read activation tiles from pkt_buf -> CB_ACT
//   Barrier A: Leader waits SEM_BARRIER(num_cores), broadcasts SEM_GO
//   Phase B: Read intermediate tiles from inter -> CB_ACT
//   Barrier B: Leader waits SEM_BARRIER(num_cores), signals SEM_EXPERT_DONE
//
// Before expert 0: Leader also waits SEM_PKT_READY from dispatch.
//
// Semaphores:
//   SEM_BARRIER (id=0): Cross-core barrier after DRAM writes
//   SEM_GO (id=1): Leader -> all cores sync signal
//   SEM_PKT_READY (id=2): Dispatch -> compute leader signal (once)
//   SEM_EXPERT_DONE (id=3): Compute leader -> combine core signal (per expert)

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    // Runtime args
    const uint32_t pkt_buf_addr = get_arg_val<uint32_t>(0);
    const uint32_t k_tiles = get_arg_val<uint32_t>(1);
    const uint32_t inter_addr = get_arg_val<uint32_t>(2);
    const uint32_t is_leader = get_arg_val<uint32_t>(3);
    const uint32_t num_cores = get_arg_val<uint32_t>(4);
    const uint32_t combine_phys_x = get_arg_val<uint32_t>(5);
    const uint32_t combine_phys_y = get_arg_val<uint32_t>(6);
    const uint32_t num_experts = get_arg_val<uint32_t>(7);
    // [8..8+2*num_cores-1] = physical coords of all compute cores

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

    // Pre-dispatch: Leader waits for pkt_buf to be written by dispatch core
    if (is_leader) {
        volatile tt_l1_ptr uint32_t* pkt_sem =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SEM_PKT_READY));
        noc_semaphore_wait(pkt_sem, 1);
        noc_semaphore_set(pkt_sem, 0);
    }

    for (uint32_t expert = 0; expert < num_experts; ++expert) {
        // ========== Start-of-expert sync ==========
        // Leader broadcasts SEM_GO so all cores start this expert together.
        // This prevents race conditions with SEM_BARRIER reuse across experts.
        if (is_leader) {
            uint32_t sem_go_l1_addr = get_semaphore(SEM_GO);
            for (uint32_t c = 0; c < num_cores; ++c) {
                uint32_t phys_x = get_arg_val<uint32_t>(8 + c * 2);
                uint32_t phys_y = get_arg_val<uint32_t>(8 + c * 2 + 1);
                uint64_t core_sem_noc = get_noc_addr(phys_x, phys_y, sem_go_l1_addr);
                noc_semaphore_inc(core_sem_noc, 1);
            }
        }

        {
            volatile tt_l1_ptr uint32_t* go_sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SEM_GO));
            noc_semaphore_wait(go_sem, 1);
            noc_semaphore_set(go_sem, 0);
        }

        // ========== Phase A: Read activation tiles ==========
        for (uint32_t k = 0; k < k_tiles; ++k) {
            cb_reserve_back(cb_act, 1);
            uint32_t l1_write_addr = get_write_ptr(cb_act);
            noc_async_read_page(k, act_accessor, l1_write_addr);
            noc_async_read_barrier();
            cb_push_back(cb_act, 1);
        }

        // ========== Barrier A ==========
        if (is_leader) {
            volatile tt_l1_ptr uint32_t* barrier_sem =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SEM_BARRIER));
            noc_semaphore_wait(barrier_sem, num_cores);
            noc_semaphore_set(barrier_sem, 0);

            uint32_t sem_go_l1_addr = get_semaphore(SEM_GO);
            for (uint32_t c = 0; c < num_cores; ++c) {
                uint32_t phys_x = get_arg_val<uint32_t>(8 + c * 2);
                uint32_t phys_y = get_arg_val<uint32_t>(8 + c * 2 + 1);
                uint64_t core_sem_noc = get_noc_addr(phys_x, phys_y, sem_go_l1_addr);
                noc_semaphore_inc(core_sem_noc, 1);
            }
        }

        {
            volatile tt_l1_ptr uint32_t* go_sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SEM_GO));
            noc_semaphore_wait(go_sem, 1);
            noc_semaphore_set(go_sem, 0);
        }

        // ========== Phase B: Read intermediate tiles ==========
        for (uint32_t k = 0; k < k_tiles; ++k) {
            cb_reserve_back(cb_act, 1);
            uint32_t l1_write_addr = get_write_ptr(cb_act);
            noc_async_read_page(k, inter_accessor, l1_write_addr);
            noc_async_read_barrier();
            cb_push_back(cb_act, 1);
        }

        // ========== Barrier B + SEM_EXPERT_DONE ==========
        if (is_leader) {
            volatile tt_l1_ptr uint32_t* barrier_sem =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SEM_BARRIER));
            noc_semaphore_wait(barrier_sem, num_cores);
            noc_semaphore_set(barrier_sem, 0);

            // Signal combine core that this expert's output is done
            uint64_t combine_sem_addr = get_noc_addr(combine_phys_x, combine_phys_y, get_semaphore(SEM_EXPERT_DONE));
            noc_semaphore_inc(combine_sem_addr, 1);
        }
    }
}
