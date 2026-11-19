// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Split-K gate|up weight stream (NCRISC, NOC0): the bank's G cores share every active expert's column pair by K --
// this core streams rows [k0, k0 + ks) of the gate and up columns of EVERY active expert (reader_gate_up.cpp streams
// whole columns of every G-th expert). The ring holds slots of KS_MAX tiles; a slot is pushed whole (ks of its tiles
// valid) so the ring arithmetic stays static. Same routing scan, variants and runtime layout as reader_gate_up.cpp
// plus [k0, ks] after the (optional) base page address. Defines: SPLITK_KS_MAX.
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
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
        EX_ZONE("GU_R_WAIT");
        noc_semaphore_wait_min(psem, 1);
        noc_semaphore_set(psem, 0);
    }
#endif
    const uint32_t w_addr = get_arg_val<uint32_t>(0);
    const uint32_t routing_addr = get_arg_val<uint32_t>(1);
    const uint32_t bank_id = get_arg_val<uint32_t>(2);
    const uint32_t vc = get_arg_val<uint32_t>(3);
    constexpr uint32_t cb_w = get_compile_time_arg_val(0);
    constexpr uint32_t cb_r = get_compile_time_arg_val(1);
    constexpr uint32_t kt = get_compile_time_arg_val(2);
    constexpr uint32_t E = get_compile_time_arg_val(3);
    constexpr uint32_t M = get_compile_time_arg_val(4);
    constexpr uint32_t expert_stride = get_compile_time_arg_val(6);
    constexpr uint32_t tile_bytes = get_compile_time_arg_val(7);
    constexpr uint32_t page_bytes = get_compile_time_arg_val(8);
    constexpr uint32_t ppb = get_compile_time_arg_val(9);
    static_assert(ppb == 1, "split-K streams one column pair per bank");
    constexpr auto routing_args = TensorAccessorArgs<10>();
#ifdef ROUTING_FOLD
    constexpr auto base_args = TensorAccessorArgs<routing_args.next_compile_time_args_offset()>();
    constexpr uint32_t ARG_SK = 6;
#else
    constexpr uint32_t ARG_SK = 5;
#endif
    const uint32_t k0 = get_arg_val<uint32_t>(ARG_SK);
    const uint32_t ks = get_arg_val<uint32_t>(ARG_SK + 1);
    constexpr uint32_t KS_MAX = SPLITK_KS_MAX;
    constexpr uint32_t E_TILES = (E + 31) / 32;
    constexpr uint32_t col_bytes = kt * tile_bytes;
    // the stream reads one tile per packet (the plain stream's page_bytes packets of 8 tiles do not divide a K slice)
    constexpr uint32_t slot_bytes = KS_MAX * tile_bytes;
    (void)page_bytes;
#ifdef ROUTING_FOLD
    constexpr uint32_t R_TILES = (E + 62) / 32;
    cb_reserve_back(cb_r, R_TILES + 1);
    const uint32_t r_l1 = get_write_ptr(cb_r);
    const uint32_t base_addr = get_arg_val<uint32_t>(5);
    const auto s_b = TensorAccessor(base_args, base_addr, 32);
    const uint32_t b_l1 = r_l1 + R_TILES * 2048;
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
    const auto s_r = TensorAccessor(routing_args, routing_addr, 2048);
    for (uint32_t i = 0; i < n_tiles; ++i) {
        noc_async_read(s_r.get_noc_addr(tb + i), r_l1 + i * 2048, 2048);
    }
    noc_async_read_barrier();
    uint16_t active[E];
    const volatile tt_l1_ptr uint16_t* routing = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(r_l1);
#if defined(SCAN_WORDS)
    const uint32_t n = scan_active_words<E, M, R_TILES>(routing, active, col_off, n_tiles);
#elif defined(ROUTING_FOLD)
    const uint32_t n = scan_active_fold<E, M>(routing, active, col_off);
#else
    const uint32_t n = scan_active<E, M>(routing, active);
#endif
#ifdef REGIONS_SRC
    const uint64_t w_base = get_noc_addr_from_bank_id<true>(bank_id, 0);
    (void)w_addr;
#else
    const uint64_t w_base = get_noc_addr_from_bank_id<true>(bank_id, w_addr);
#endif
    constexpr uint32_t src_step = tile_bytes;
    constexpr uint32_t PKT_TILES = 8;  // 8 x 1088 B = 8704 B <= the NOC burst
    static_assert(PKT_TILES * tile_bytes <= 8704, "a packet of 8 bfp8 tiles");
    reset_noc_trid_barrier_counter(NOC_CLEAR_OUTSTANDING_REQ_MASK, noc_index);
    constexpr uint32_t num_buffers = 3;
    constexpr uint32_t extra_in_flight = 1;
    cb_reserve_back(cb_w, KS_MAX * (extra_in_flight + 1));
    uint32_t l1_write = get_write_ptr(cb_w);
    const uint32_t cb_base = l1_write;
    const uint32_t cb_end = cb_base + num_buffers * slot_bytes;
    uint32_t free_blocks = num_buffers;
    uint32_t trid = 1;
    uint32_t trid_to_wait = 1;
    uint32_t pushed = 0;
    uint32_t issued = 0;
    for (uint32_t i = 0; i < n; ++i) {  // every active expert: this core's K slice of both columns
#ifdef REGIONS_SRC
        const uint32_t e = active[i];
        const uint32_t gate_addr = get_common_arg_val<uint32_t>(e);
        const uint32_t up_addr = get_common_arg_val<uint32_t>(E + e);
#else
        const uint32_t expert_base = static_cast<uint32_t>(active[i]) * expert_stride;
#endif
        for (uint32_t c = 0; c < 2; ++c) {
            EX_ZONE("GU_R_COL");
#ifdef REGIONS_SRC
            uint32_t src = (c ? up_addr : gate_addr) + k0 * src_step;
#else
            uint32_t src = expert_base + c * col_bytes + k0 * src_step;
#endif
            // the slice is contiguous: packets of PKT_TILES tiles (8 x 1088 B fits a NOC burst), the last one shorter
            noc_async_read_set_trid(trid);
            for (uint32_t t = 0; t < ks; t += PKT_TILES) {
                const uint32_t nt = (ks - t < PKT_TILES) ? ks - t : PKT_TILES;
                noc_async_read_one_packet_set_state<true>(w_base, nt * tile_bytes, vc);
                noc_async_read_one_packet_with_state_with_trid(
                    w_base, src + t * tile_bytes, l1_write + t * tile_bytes, trid);
            }
            l1_write += slot_bytes;
            ++issued;
            if (free_blocks == num_buffers - extra_in_flight) {
                noc_async_read_barrier_with_trid(trid_to_wait);
                cb_push_back(cb_w, KS_MAX);
                ++pushed;
                trid_to_wait = trid_to_wait == num_buffers ? 1 : trid_to_wait + 1;
                cb_reserve_back(cb_w, KS_MAX * (extra_in_flight + 1));
            } else {
                free_blocks -= 1;
            }
            trid = trid == num_buffers ? 1 : trid + 1;
            if (l1_write >= cb_end) {
                l1_write = cb_base;
            }
        }
    }
    while (pushed < issued) {
        EX_ZONE("GU_R_DRAIN");
        noc_async_read_barrier_with_trid(trid_to_wait);
        cb_push_back(cb_w, KS_MAX);
        ++pushed;
        trid_to_wait = trid_to_wait == num_buffers ? 1 : trid_to_wait + 1;
    }
    noc_async_full_barrier();
}
