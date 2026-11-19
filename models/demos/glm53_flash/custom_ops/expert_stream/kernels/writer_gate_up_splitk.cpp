// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Split-K gate|up writer (BRISC, NOC1): this core's K slice of x, the routing scan, then per active expert i the
// partial tiles go to the expert's owner (bank core rank i % G, slot (i / G) & 1 of its receive CB) with an arrival
// count on the owner's semaphore; for the experts this core owns it builds D_e, waits for the G partials, hands them
// to the compute and writes the output tile to act (the fused arrival count on the down cores as writer_gate_up.cpp,
// slot k = i / G). Runtime args: writer_gate_up.cpp's [x_addr, routing_addr, act_addr, bank_id, group, (base_addr)]
// then [k0, ks, bank core NOC xy x G (rank order)] then the FUSED_SIGNAL tail.
// Defines: SPLITK_KS_MAX, SPLITK_CB_PART, SPLITK_CB_RECV, SPLITK_SEM0 (two semaphores: SEM0, SEM0 + 1).
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "ckernel.h"
#include "ckernel_defs.h"
#include "regions_src.hpp"
#include "routing_scan.hpp"

// profiler zones (tracy -p builds only; each DeviceZoneScopedN needs its own scope)
#if defined(PROFILE_KERNEL) && defined(DeviceZoneScopedN)
#define EX_ZONE(name) DeviceZoneScopedN(name)
#else
#define EX_ZONE(name)
#endif

void kernel_main() {
#ifdef PERSIST_WAIT_SEM
    {
        volatile tt_l1_ptr uint32_t* psem =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(PERSIST_WAIT_SEM));
        EX_ZONE("GU_W_WAIT");
        noc_semaphore_wait_min(psem, 1);
        noc_semaphore_set(psem, 0);
    }
#endif
    const uint32_t x_addr = get_arg_val<uint32_t>(0);
    const uint32_t routing_addr = get_arg_val<uint32_t>(1);
    const uint32_t act_addr = get_arg_val<uint32_t>(2);
    const uint32_t bank_id = get_arg_val<uint32_t>(3);
    const uint32_t group = get_arg_val<uint32_t>(4);
    constexpr uint32_t cb_x = get_compile_time_arg_val(0);
    constexpr uint32_t cb_r = get_compile_time_arg_val(1);
    constexpr uint32_t cb_d = get_compile_time_arg_val(2);
    constexpr uint32_t cb_out = get_compile_time_arg_val(3);
    constexpr uint32_t E = get_compile_time_arg_val(5);
    constexpr uint32_t M = get_compile_time_arg_val(6);
    constexpr uint32_t groups = get_compile_time_arg_val(7);
    constexpr uint32_t i_tiles = get_compile_time_arg_val(8);
    constexpr uint32_t ppb = get_compile_time_arg_val(9);
    static_assert(ppb == 1, "split-K streams one column pair per bank");
    constexpr auto x_args = TensorAccessorArgs<10>();
    constexpr auto routing_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
    constexpr auto act_args = TensorAccessorArgs<routing_args.next_compile_time_args_offset()>();
#ifdef ROUTING_FOLD
    constexpr auto base_args = TensorAccessorArgs<act_args.next_compile_time_args_offset()>();
    constexpr uint32_t ARG_SK = 6;
#else
    constexpr uint32_t ARG_SK = 5;
#endif
    constexpr uint32_t KS_MAX = SPLITK_KS_MAX;
    constexpr uint32_t cb_part = SPLITK_CB_PART;
    constexpr uint32_t cb_recv = SPLITK_CB_RECV;
    constexpr uint32_t SEM0 = SPLITK_SEM0;
    constexpr uint32_t PART_BYTES = 2 * 4096;  // two fp32 tiles
    constexpr uint32_t ARG_XY_SK = ARG_SK + 2;
    constexpr uint32_t ARG_NEXT = ARG_XY_SK + groups;  // the FUSED_SIGNAL tail
    const uint32_t k0 = get_arg_val<uint32_t>(ARG_SK);
    const uint32_t ks = get_arg_val<uint32_t>(ARG_SK + 1);
    constexpr uint32_t E_TILES = (E + 31) / 32;
    constexpr uint32_t tile_page = 2048;
    const auto s_x = TensorAccessor(x_args, x_addr, tile_page);
    const auto s_r = TensorAccessor(routing_args, routing_addr, tile_page);
    const auto s_act = TensorAccessor(act_args, act_addr, tile_page);
    // the receive region: the CB's base (the same address on every bank core: identical CB layouts)
    const uint32_t recv_base = get_write_ptr(cb_recv);
    // routing matrix -> active experts
#ifdef ROUTING_FOLD
    constexpr uint32_t R_TILES = (E + 62) / 32;
    cb_reserve_back(cb_r, R_TILES + 1);
    const uint32_t r_l1 = get_write_ptr(cb_r);
    const uint32_t base_addr = get_arg_val<uint32_t>(5);
    const auto s_b = TensorAccessor(base_args, base_addr, 32);
    const uint32_t b_l1 = r_l1 + R_TILES * tile_page;
    noc_async_read(s_b.get_noc_addr(0), b_l1, 32);
    noc_async_read_barrier();
    const uint32_t col_base = *reinterpret_cast<volatile tt_l1_ptr uint16_t*>(b_l1);
    const uint32_t tb = fold_first_tile(col_base);
    const uint32_t col_off = fold_col_off(col_base);
    const uint32_t n_tiles = fold_tiles<E>(col_base);
#else
    constexpr uint32_t R_TILES = E_TILES;
    cb_reserve_back(cb_r, E_TILES);
    const uint32_t r_l1 = get_write_ptr(cb_r);
    constexpr uint32_t tb = 0;
    constexpr uint32_t col_off = 0;
    constexpr uint32_t n_tiles = E_TILES;
#endif
    for (uint32_t i = 0; i < n_tiles; ++i) {
        noc_async_read(s_r.get_noc_addr(tb + i), r_l1 + i * tile_page, tile_page);
    }
    // this core's K slice of x, resident for the whole op (the slot is KS_MAX tiles, ks valid)
    cb_reserve_back(cb_x, KS_MAX);
    const uint32_t x_l1 = get_write_ptr(cb_x);
    {
        EX_ZONE("GU_W_X");
        for (uint32_t k = 0; k < ks; ++k) {
            noc_async_read(s_x.get_noc_addr(k0 + k), x_l1 + k * tile_page, tile_page);
        }
        noc_async_read_barrier();
    }
    cb_push_back(cb_x, KS_MAX);

    volatile tt_l1_ptr uint16_t* routing = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(r_l1);
    uint16_t active[E];
#if defined(SCAN_WORDS)
    const uint32_t n = scan_active_words<E, M, R_TILES>(routing, active, col_off, n_tiles);
#elif defined(ROUTING_FOLD)
    const uint32_t n = scan_active_fold<E, M>(routing, active, col_off);
#else
    const uint32_t n = scan_active<E, M>(routing, active);
#endif
    ckernel::mailbox_write(ckernel::ThreadId::UnpackThreadId, n);
    ckernel::mailbox_write(ckernel::ThreadId::MathThreadId, n);
    ckernel::mailbox_write(ckernel::ThreadId::PackThreadId, n);
#ifdef FUSED_SIGNAL
    const uint32_t n_down = get_arg_val<uint32_t>(ARG_NEXT);
    const uint32_t sem_base = get_arg_val<uint32_t>(ARG_NEXT + 1);
    const uint32_t sems_per_group = get_arg_val<uint32_t>(ARG_NEXT + 2);
    constexpr uint32_t ARG_XY = ARG_NEXT + 3;
#endif
    // experts in waves of 2G (see compute_gate_up_splitk.cpp): the wave's partials to their owners first, then the
    // two experts this core owns; receive slot (i / G) % 4 (four slots: a sender may be a wave ahead of an owner)
    constexpr uint32_t WAVE = 2 * groups;
    constexpr uint32_t NSLOT = 4;
    for (uint32_t w0 = 0; w0 < n; w0 += WAVE) {
        const uint32_t w1 = (w0 + WAVE < n) ? w0 + WAVE : n;
        for (uint32_t i = w0; i < w1; ++i) {
            EX_ZONE("GU_W_PART");
            const uint32_t owner = i % groups;
            const uint32_t slot = (i / groups) % NSLOT;
            const uint32_t e = active[i];
            if (owner == group) {
                // D_e: zeros, then the token weights on the diagonal (row t = token t)
                cb_reserve_back(cb_d, 1);
                const uint32_t d_l1 = get_write_ptr(cb_d);
                volatile tt_l1_ptr uint32_t* d32 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(d_l1);
                for (uint32_t j = 0; j < tile_page / 4; ++j) {
                    d32[j] = 0;
                }
                volatile tt_l1_ptr uint16_t* d16 = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(d_l1);
                for (uint32_t t = 0; t < M; ++t) {
                    d16[tile_elem(t, t)] = routing[fold_elem(t, e, col_off)];
                }
                cb_push_back(cb_d, 1);
            }
            // this core's partial tiles -> the owner's receive slot, then its arrival count
            cb_wait_front(cb_part, 2);
            const uint32_t oxy = get_arg_val<uint32_t>(ARG_XY_SK + owner);
            const uint64_t dst =
                get_noc_addr(oxy >> 16, oxy & 0xFFFFu, recv_base + (slot * groups + group) * PART_BYTES);
            noc_async_write(get_read_ptr(cb_part), dst, PART_BYTES);
            noc_async_write_barrier();
            noc_semaphore_inc(get_noc_addr(oxy >> 16, oxy & 0xFFFFu, get_semaphore(SEM0 + slot)), 1);
            cb_pop_front(cb_part, 2);
        }
        for (uint32_t i = w0 + group; i < w1; i += groups) {
            EX_ZONE("GU_W_OWNER");
            const uint32_t slot = (i / groups) % NSLOT;
            const uint32_t e = active[i];
            // the owner: the G partials have landed when the slot's count reaches G
            cb_reserve_back(cb_recv, 2 * groups);
            {
                EX_ZONE("GU_W_ARRIVE");
                volatile tt_l1_ptr uint32_t* sem =
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SEM0 + slot));
                noc_semaphore_wait_min(sem, groups);
                noc_semaphore_set(sem, 0);
            }
            cb_push_back(cb_recv, 2 * groups);
            // the output tile -> act[e, col], then the down cores' arrival count (slot k = i / G of group i % G)
            cb_wait_front(cb_out, 1);
#ifdef REGIONS_SRC
            const uint32_t col = regions_act_col(bank_id, 0, i_tiles);
#else
            const uint32_t col = bank_id;
#endif
            noc_async_write(get_read_ptr(cb_out), s_act.get_noc_addr(e * i_tiles + col), tile_page);
            noc_async_write_barrier();
            cb_pop_front(cb_out, 1);
#ifdef FUSED_SIGNAL
            {
                const uint32_t k_mine = i / groups;
                const uint32_t sem_addr = get_semaphore(sem_base + group * sems_per_group + k_mine / 7);
                const uint32_t incr = 1u << (4 * (k_mine % 7));
                for (uint32_t d = 0; d < n_down; ++d) {
                    const uint32_t xy = get_arg_val<uint32_t>(ARG_XY + d);
                    noc_semaphore_inc(get_noc_addr(xy >> 16, xy & 0xFFFFu, sem_addr), incr);
                }
            }
#endif
        }
    }
    noc_async_full_barrier();
}
