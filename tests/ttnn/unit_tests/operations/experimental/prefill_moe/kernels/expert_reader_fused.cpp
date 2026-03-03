// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// Fused prefill MoE compute: Two-phase activation reader with dispatch/combine signaling (dm0/RISCV_0)
//
// Pre-Phase A: Leader waits for SEM_PKT_READY from dispatch core.
// Phase A: Read activation tiles [1, K_tiles] from interleaved DRAM → CB_ACT
// Barrier A: Leader waits for all cores' SwiGLU writes (SEM_BARRIER), then signals SEM_GO.
//            Non-leader waits for SEM_GO from leader.
// Phase B: Read intermediate tiles [1, K_tiles] from interleaved DRAM → CB_ACT
// Barrier B: Leader waits for all cores' output writes (SEM_BARRIER reused),
//            then signals SEM_EXPERT_DONE on combine core.
//
// Semaphores:
//   SEM_BARRIER (id=0): Leader's copy incremented by all cores after DRAM writes.
//   SEM_GO (id=1): Each core's copy incremented by leader after barrier A is satisfied.
//   SEM_PKT_READY (id=2): Leader's copy incremented by dispatch core after pkt_buf written.
//   SEM_EXPERT_DONE (id=3): Combine core's copy incremented by leader after output writes done.

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    // Runtime args
    const uint32_t act_addr = get_arg_val<uint32_t>(0);
    const uint32_t k_tiles = get_arg_val<uint32_t>(1);
    const uint32_t inter_addr = get_arg_val<uint32_t>(2);
    const uint32_t is_leader = get_arg_val<uint32_t>(3);
    const uint32_t num_cores = get_arg_val<uint32_t>(4);
    const uint32_t combine_phys_x = get_arg_val<uint32_t>(5);
    const uint32_t combine_phys_y = get_arg_val<uint32_t>(6);
    // [7..7+2*num_cores-1] = physical coords of all compute cores (x0,y0, x1,y1, ...)

    // Compile-time args: [TensorAccessorArgs(act), TensorAccessorArgs(inter)]
    constexpr auto act_args = TensorAccessorArgs<0>();
    constexpr auto inter_args = TensorAccessorArgs<1>();

    constexpr uint32_t cb_act = 0;
    constexpr uint32_t SEM_BARRIER = 0;
    constexpr uint32_t SEM_GO = 1;
    constexpr uint32_t SEM_PKT_READY = 2;
    constexpr uint32_t SEM_EXPERT_DONE = 3;

    const uint32_t page_bytes = get_local_cb_interface(cb_act).fifo_page_size;
    const auto act_accessor = TensorAccessor(act_args, act_addr, page_bytes);

    // ========== Pre-Phase A: Wait for dispatch to finish writing pkt_buf ==========
    // Leader waits for SEM_PKT_READY from dispatch, then broadcasts SEM_GO to all
    // compute cores so they all wait for pkt_buf to be fully written before reading.
    if (is_leader) {
        volatile tt_l1_ptr uint32_t* pkt_sem =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SEM_PKT_READY));
        noc_semaphore_wait(pkt_sem, 1);
        noc_semaphore_set(pkt_sem, 0);  // Reset for next expert

        // Broadcast pkt_buf readiness to all compute cores via SEM_GO
        uint32_t sem_go_l1_addr = get_semaphore(SEM_GO);
        for (uint32_t c = 0; c < num_cores; ++c) {
            uint32_t phys_x = get_arg_val<uint32_t>(7 + c * 2);
            uint32_t phys_y = get_arg_val<uint32_t>(7 + c * 2 + 1);
            uint64_t core_sem_noc = get_noc_addr(phys_x, phys_y, sem_go_l1_addr);
            noc_semaphore_inc(core_sem_noc, 1);
        }
    }

    // All cores wait for pkt_buf readiness before reading
    {
        volatile tt_l1_ptr uint32_t* go_sem_pre = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SEM_GO));
        noc_semaphore_wait(go_sem_pre, 1);
        noc_semaphore_set(go_sem_pre, 0);  // Reset for barrier A reuse
    }

    // ========== Phase A: Read activation tiles ==========
    for (uint32_t k = 0; k < k_tiles; ++k) {
        cb_reserve_back(cb_act, 1);
        uint32_t l1_write_addr = get_write_ptr(cb_act);
        noc_async_read_page(k, act_accessor, l1_write_addr);
        noc_async_read_barrier();
        cb_push_back(cb_act, 1);
    }

    // ========== Barrier A: Cross-Core Barrier (same as non-fused version) ==========
    if (is_leader) {
        volatile tt_l1_ptr uint32_t* barrier_sem =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SEM_BARRIER));
        noc_semaphore_wait(barrier_sem, num_cores);
        noc_semaphore_set(barrier_sem, 0);  // Reset for Phase B reuse

        uint32_t sem_go_l1_addr = get_semaphore(SEM_GO);
        for (uint32_t c = 0; c < num_cores; ++c) {
            uint32_t phys_x = get_arg_val<uint32_t>(7 + c * 2);
            uint32_t phys_y = get_arg_val<uint32_t>(7 + c * 2 + 1);
            uint64_t core_sem_noc = get_noc_addr(phys_x, phys_y, sem_go_l1_addr);
            noc_semaphore_inc(core_sem_noc, 1);
        }
    }

    volatile tt_l1_ptr uint32_t* go_sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SEM_GO));
    noc_semaphore_wait(go_sem, 1);
    noc_semaphore_set(go_sem, 0);  // Reset for next expert

    // ========== Phase B: Read intermediate tiles ==========
    const auto inter_accessor = TensorAccessor(inter_args, inter_addr, page_bytes);

    for (uint32_t k = 0; k < k_tiles; ++k) {
        cb_reserve_back(cb_act, 1);
        uint32_t l1_write_addr = get_write_ptr(cb_act);
        noc_async_read_page(k, inter_accessor, l1_write_addr);
        noc_async_read_barrier();
        cb_push_back(cb_act, 1);
    }

    // ========== Barrier B: Wait for all cores' output writes, signal combine ==========
    // dm1 (expert_writer_fused) on each core signals SEM_BARRIER on leader after output write.
    if (is_leader) {
        volatile tt_l1_ptr uint32_t* barrier_sem =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SEM_BARRIER));
        noc_semaphore_wait(barrier_sem, num_cores);
        noc_semaphore_set(barrier_sem, 0);  // Reset for next expert

        // Signal combine core that this expert's output is fully written
        uint64_t combine_sem_addr = get_noc_addr(combine_phys_x, combine_phys_y, get_semaphore(SEM_EXPERT_DONE));
        noc_semaphore_inc(combine_sem_addr, 1);
    }
}
