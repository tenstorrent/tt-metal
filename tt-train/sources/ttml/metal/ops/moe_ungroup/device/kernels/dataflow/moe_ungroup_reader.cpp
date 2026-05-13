// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Reader (NCRISC, NOC_0) for moe_ungroup. Per core:
//   1. Read offsets[E_local + 1] from DRAM into L1.
//   2. Push expert e=0 source tiles to cb_src0 (compute will untilize and
//      hand to BRISC writer via cb_out0).
//   3. Wait for local BRISC to signal "prezero done" (brisc_done_sem).
//      Run the cross-core mcast barrier on NOC_0 (matches moe_group's pattern
//      of putting the mcast in the NCRISC kernel — NOC_0 multicast routing
//      delivers correctly when the source is at the (start_x,start_y) corner).
//      Signal local BRISC via brisc_release_sem.
//   4. For e in 1..E_local-1: push expert e tiles, then handshake again
//      (BRISC has finished writing expert e-1; mcast barrier; release BRISC).
//
// Inactive (padding) steps zero-fill into cb_src0 so compute and the writer
// stay in lockstep across all cores.

#include "api/dataflow/dataflow_api.h"
#include "tt-train/sources/ttml/metal/common/dataflow_utils.hpp"
#include "tt-train/sources/ttml/metal/ops/moe_ungroup/device/kernels/moe_ungroup_utils.hpp"

constexpr uint32_t cb_src0 = tt::CBIndex::c_0;
constexpr uint32_t cb_zero = tt::CBIndex::c_3;  // small scratch for offsets read

constexpr uint32_t h = get_compile_time_arg_val(0);
constexpr uint32_t num_chunks = get_compile_time_arg_val(1);
constexpr uint32_t tiles_per_chunk = get_compile_time_arg_val(2);
constexpr uint32_t e_local = get_compile_time_arg_val(3);
constexpr uint32_t num_total_cores = get_compile_time_arg_val(4);
constexpr uint32_t lead_core_x = get_compile_time_arg_val(5);
constexpr uint32_t lead_core_y = get_compile_time_arg_val(6);
constexpr uint32_t up_sem_id = get_compile_time_arg_val(7);
constexpr uint32_t down_sem_id = get_compile_time_arg_val(8);
constexpr uint32_t brisc_done_sem_id = get_compile_time_arg_val(9);
constexpr uint32_t brisc_release_sem_id = get_compile_time_arg_val(10);
constexpr uint32_t mcast_sx = get_compile_time_arg_val(11);
constexpr uint32_t mcast_sy = get_compile_time_arg_val(12);
constexpr uint32_t mcast_ex = get_compile_time_arg_val(13);
constexpr uint32_t mcast_ey = get_compile_time_arg_val(14);
constexpr uint32_t mcast_num_dests_incl_self = get_compile_time_arg_val(15);
constexpr uint32_t cb_id_ctrl = get_compile_time_arg_val(16);

constexpr auto expert_out_args = TensorAccessorArgs<17>();
constexpr auto offsets_args = TensorAccessorArgs<expert_out_args.next_compile_time_args_offset()>();

constexpr uint32_t TILE_H = 32U;
constexpr uint32_t TILE_W = 32U;
constexpr uint32_t TILE_BYTES = TILE_H * TILE_W * 2U;  // bf16 tile
// ceil(h / TILE_W) — must match the program factory; reader needs to address
// partial-last-tile cases when h isn't a TILE_W multiple.
constexpr uint32_t Wt = round_up(h, TILE_W) / TILE_W;

inline void barrier_fanin_mcast(uint32_t my_core_idx) {
    if (my_core_idx > 0) {
        uint64_t sem_noc = get_noc_addr(lead_core_x, lead_core_y, get_semaphore(up_sem_id));
        noc_semaphore_inc(sem_noc, 1U);
    }
    if (my_core_idx == 0) {
        if (num_total_cores > 1U) {
            volatile tt_l1_ptr uint32_t* up_sem =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(up_sem_id));
            noc_semaphore_wait(up_sem, num_total_cores - 1U);
            noc_semaphore_set(up_sem, 0U);
        }
        uint32_t down_sem_addr = get_semaphore(down_sem_id);
        volatile tt_l1_ptr uint32_t* down_sem_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(down_sem_addr);
        mcast_sender_signal_receivers_loopback(
            down_sem_ptr, down_sem_addr, mcast_sx, mcast_sy, mcast_ex, mcast_ey, mcast_num_dests_incl_self);
    } else {
        volatile tt_l1_ptr uint32_t* down_sem =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(down_sem_id));
        noc_semaphore_wait(down_sem, 1U);
        noc_semaphore_set(down_sem, 0U);
    }
}

// Wait for local BRISC to signal completion of its current phase, run the
// cross-core mcast barrier, then release local BRISC. Reused for prezero
// and inter-expert syncs (semaphores are reset to 0 between calls).
inline void handshake_then_barrier_then_release(uint32_t my_core_idx) {
    volatile tt_l1_ptr uint32_t* brisc_done =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(brisc_done_sem_id));
    do {
        invalidate_l1_cache();
    } while ((*brisc_done) == 0U);

    *brisc_done = 0U;

    barrier_fanin_mcast(my_core_idx);

    volatile tt_l1_ptr uint32_t* brisc_release =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(brisc_release_sem_id));
    *brisc_release = 1U;
}

void kernel_main() {
    const uint32_t expert_out_addr = get_arg_val<uint32_t>(0);
    const uint32_t offsets_addr = get_arg_val<uint32_t>(1);
    const uint32_t my_core_idx = get_arg_val<uint32_t>(2);

    const auto expert_out_addrgen = TensorAccessor(expert_out_args, expert_out_addr, TILE_BYTES);
    const auto offsets_addrgen = TensorAccessor(offsets_args, offsets_addr);

    // L1 layout in cb_zero (sized by host as (e_local+1)*4 + 2*e_local*4):
    //   [offsets_l1 (e_local+1) u32]
    //   [tr_start_per_expert e_local u32]
    //   [my_real_count_per_expert e_local u32]
    // Backing the per-expert caches in L1 instead of NCRISC stack avoids
    // blowing the small RISC stack for large e_local (e.g. 300+).
    cb_reserve_back(cb_zero, 1U);
    uint32_t scratch_l1 = get_write_ptr(cb_zero);
    volatile tt_l1_ptr uint32_t* offsets_l1 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch_l1);
    volatile tt_l1_ptr uint32_t* tr_start_per_expert =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch_l1 + (e_local + 1U) * sizeof(uint32_t));
    volatile tt_l1_ptr uint32_t* my_real_count_per_expert =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch_l1 + (2U * e_local + 1U) * sizeof(uint32_t));
    noc_async_read(get_noc_addr(0, offsets_addrgen), scratch_l1, (e_local + 1U) * sizeof(uint32_t));
    noc_async_read_barrier();
    cb_push_back(cb_zero, 1U);

    // Walk offsets ONCE to compute this core's per-expert work bounds and
    // publish the total block count (steps × num_chunks) to compute via
    // cb_ctrl.
    uint32_t my_total_active_steps = 0U;
    for (uint32_t e = 0; e < e_local; ++e) {
        uint32_t expert_start_tr = offsets_l1[e] / TILE_H;
        uint32_t expert_total_tr = (offsets_l1[e + 1U] - offsets_l1[e]) / TILE_H;
        auto slice = ttml::metal::moe_ungroup::slice_for_core(expert_total_tr, num_total_cores, my_core_idx);
        tr_start_per_expert[e] = expert_start_tr + slice.start;
        my_real_count_per_expert[e] = slice.count;
        my_total_active_steps += slice.count;
    }
    cb_reserve_back(cb_id_ctrl, 1U);
    volatile tt_l1_ptr uint32_t* ctrl_l1 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_id_ctrl));
    ctrl_l1[0] = my_total_active_steps * num_chunks;
    cb_push_back(cb_id_ctrl, 1U);

    for (uint32_t e = 0; e < e_local; ++e) {
        // Handshake BEFORE pushing this expert's tiles. BRISC has just
        // finished its previous phase (prezero for e==0, expert e-1 for
        // e>=1) and is waiting on brisc_release. We do the cross-core mcast
        // barrier on NOC_0, then release BRISC to start consuming expert e
        // from cb_out0. Pushing-then-handshaking would deadlock at large
        // shapes: the reader fills cb_src0, compute fills cb_out0, BRISC is
        // stuck waiting on brisc_release, NCRISC blocks on cb_reserve_back.
        handshake_then_barrier_then_release(my_core_idx);

        uint32_t tr_start = tr_start_per_expert[e];
        uint32_t my_real_count = my_real_count_per_expert[e];

        // Only iterate active steps now — writer/compute use the same per-core
        // count via cb_ctrl, so there's no need to pad with zero-fill.
        for (uint32_t step = 0; step < my_real_count; ++step) {
            uint32_t tr_global = tr_start + step;

            for (uint32_t chunk = 0; chunk < num_chunks; ++chunk) {
                cb_reserve_back(cb_src0, tiles_per_chunk);
                uint32_t dst_l1 = get_write_ptr(cb_src0);
                uint32_t tile_id_base = tr_global * Wt + chunk * tiles_per_chunk;
                for (uint32_t t = 0; t < tiles_per_chunk; ++t) {
                    uint64_t src_noc = get_noc_addr(tile_id_base + t, expert_out_addrgen);
                    noc_async_read(src_noc, dst_l1 + t * TILE_BYTES, TILE_BYTES);
                }
                noc_async_read_barrier();
                cb_push_back(cb_src0, tiles_per_chunk);
            }
        }
    }
}
