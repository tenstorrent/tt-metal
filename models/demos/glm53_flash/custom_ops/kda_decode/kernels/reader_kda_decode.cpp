// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Reader of the fused KDA decode block. Work item = (head h, user tile row tb, user part). The item's shared
// inputs -- the head's three conv-history tile blocks + raw block per branch (tap-major tiles: no tilize), the
// taps, the gate blocks and the channel constants -- are read from DRAM by the part-0 core of the (h, tb) slot
// and multicast into the same CB addresses of the other parts (a column of cores); the part-0 core then shifts
// the history in place (tap j+1 -> tap j, raw -> tap 2). Per user the core reads the user's selector tile and
// its recurrent state. On the part-0 core the reader also collects the other parts' output blocks (they write
// into the gather CB and bump a semaphore) for the final norm.

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"

constexpr uint32_t cb_hist = 0, cb_taps = 1, cb_gf = 2, cb_gog = 3, cb_gb = 4, cb_cc = 5, cb_onw = 6, cb_sel = 7;
constexpr uint32_t cb_selT = 8, cb_S = 9, cb_gather = 24;
constexpr uint32_t SEM_READY = 0, SEM_VALID = 1, SEM_ONE = 2, SEM_GATHER = 3;

void kernel_main() {
    constexpr uint32_t Kt = get_compile_time_arg_val(0);
    constexpr uint32_t Vt = get_compile_time_arg_val(1);
    constexpr uint32_t Bt = get_compile_time_arg_val(2);
    constexpr uint32_t B = get_compile_time_arg_val(3);
    constexpr uint32_t H = get_compile_time_arg_val(4);
    constexpr uint32_t Wq_t = get_compile_time_arg_val(5);  // qkv / history row width in tiles (3*H*Kt)
    constexpr uint32_t Wg_t = get_compile_time_arg_val(6);  // gates row width in tiles
    constexpr uint32_t HK_t = get_compile_time_arg_val(7);
    constexpr uint32_t HV_t = get_compile_time_arg_val(8);
    constexpr uint32_t parts = get_compile_time_arg_val(9);
    constexpr uint32_t ONES_PAGE = get_compile_time_arg_val(10);  // selector page of the all-ones tile (identity next)
    constexpr uint32_t conv_taps = 3;

    constexpr auto qkv_a = TensorAccessorArgs<11>();
    constexpr auto hist_a = TensorAccessorArgs<qkv_a.next_compile_time_args_offset()>();
    constexpr auto taps_a = TensorAccessorArgs<hist_a.next_compile_time_args_offset()>();
    constexpr auto gates_a = TensorAccessorArgs<taps_a.next_compile_time_args_offset()>();
    constexpr auto cc_a = TensorAccessorArgs<gates_a.next_compile_time_args_offset()>();
    constexpr auto onw_a = TensorAccessorArgs<cc_a.next_compile_time_args_offset()>();
    constexpr auto sel_a = TensorAccessorArgs<onw_a.next_compile_time_args_offset()>();
    constexpr auto st_a = TensorAccessorArgs<sel_a.next_compile_time_args_offset()>();

    const uint32_t item0 = get_arg_val<uint32_t>(0);
    const uint32_t n_items = get_arg_val<uint32_t>(1);
    const uint32_t stride = get_arg_val<uint32_t>(2);
    const uint32_t qkv_addr = get_arg_val<uint32_t>(3);
    const uint32_t hist_addr = get_arg_val<uint32_t>(4);
    const uint32_t taps_addr = get_arg_val<uint32_t>(5);
    const uint32_t gates_addr = get_arg_val<uint32_t>(6);
    const uint32_t cc_addr = get_arg_val<uint32_t>(7);
    const uint32_t onw_addr = get_arg_val<uint32_t>(8);
    const uint32_t sel_addr = get_arg_val<uint32_t>(9);
    const uint32_t st_addr = get_arg_val<uint32_t>(10);
    const uint32_t p0_x = get_arg_val<uint32_t>(11);  // NoC coords of the slot's part-0 core ...
    const uint32_t p0_y = get_arg_val<uint32_t>(12);
    const uint32_t pe_x = get_arg_val<uint32_t>(13);  // ... and of its last part (same column, larger y)
    const uint32_t pe_y = get_arg_val<uint32_t>(14);

    const uint32_t t32 = get_tile_size(cb_S);
    const uint32_t t16 = get_tile_size(cb_hist);
    const auto qkv_acc = TensorAccessor(qkv_a, qkv_addr, t16);
    const auto hist_acc = TensorAccessor(hist_a, hist_addr, t16);
    const auto taps_acc = TensorAccessor(taps_a, taps_addr, t16);
    const auto gates_acc = TensorAccessor(gates_a, gates_addr, t32);
    const auto cc_acc = TensorAccessor(cc_a, cc_addr, t32);
    const auto onw_acc = TensorAccessor(onw_a, onw_addr, t32);
    const auto sel_acc = TensorAccessor(sel_a, sel_addr, t16);
    const auto st_acc = TensorAccessor(st_a, st_addr, t32);

    Noc noc;
    auto issue = [&](const auto& acc, CircularBuffer& cb, uint32_t tb, uint32_t base, uint32_t n, uint32_t off) {
        for (uint32_t t = 0; t < n; t++) {
            noc.async_read(acc, cb, tb, {.page_id = base + t}, {.offset_bytes = (off + t) * tb});
        }
    };
    auto read_tiles = [&](const auto& acc, uint32_t cb_id, uint32_t tb, uint32_t base, uint32_t n) {
        CircularBuffer cb(cb_id);
        cb.reserve_back(n);
        issue(acc, cb, tb, base, n, 0);
        noc.async_read_barrier();
        cb.push_back(n);
    };

    // once per core: the o_norm weight tiles, the all-ones and identity selector tiles
    read_tiles(onw_acc, cb_onw, t32, 0, Vt);
    read_tiles(sel_acc, cb_sel, t16, ONES_PAGE, 2);

    volatile tt_l1_ptr uint32_t* ready = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SEM_READY));
    volatile tt_l1_ptr uint32_t* valid = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SEM_VALID));
    volatile tt_l1_ptr uint32_t* gath = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SEM_GATHER));

    for (uint32_t it = 0; it < n_items; it++) {
        const uint32_t item = item0 + it * stride;
        const uint32_t h = item / (Bt * parts);
        const uint32_t rem = item % (Bt * parts);
        const uint32_t tb = rem / parts;
        const uint32_t part = rem % parts;
        const uint32_t n_row = (B - tb * 32u < 32u) ? (B - tb * 32u) : 32u;
        const uint32_t up = (n_row + parts - 1) / parts;
        const uint32_t n_active = (n_row + up - 1) / up;  // parts with at least one user
        const uint32_t u0 = part * up;
        const uint32_t u1 = (u0 + up < n_row) ? (u0 + up) : n_row;
        const uint32_t col = h * Kt;  // the head's first tile column inside a branch

        CircularBuffer hist(cb_hist), taps(cb_taps), gf(cb_gf), gog(cb_gog), gb(cb_gb), cc(cb_cc);
        CircularBuffer selT(cb_selT), S(cb_S), gather(cb_gather);
        hist.reserve_back(12 * Kt);
        taps.reserve_back(3 * Kt);
        gf.reserve_back(Kt);
        gog.reserve_back(Vt);
        gb.reserve_back(1);
        cc.reserve_back(Kt);
        // the first user's selector + state and the beta selector go out with the item reads (overlap)
        const bool active = u0 < u1;
        const uint32_t c = h & 31u;
        if (active) {
            selT.reserve_back(2);
            S.reserve_back(Kt * Vt);
            issue(sel_acc, selT, t16, c, 1, 0);   // E_c^T for the beta column
            issue(sel_acc, selT, t16, u0, 1, 1);  // E_{u0}^T
            issue(st_acc, S, t32, ((tb * 32 + u0) * H + h) * Kt * Vt, Kt * Vt, 0);
        }
        if (part == 0) {
            // every per-item input in flight at once: the 3 x (3 history + raw) blocks, taps, gates, constants
            for (uint32_t br = 0; br < 3; br++) {
                for (uint32_t j = 0; j < conv_taps; j++) {
                    issue(hist_acc, hist, t16, (j * Bt + tb) * Wq_t + br * HK_t + col, Kt, (br * 4 + j) * Kt);
                }
                issue(qkv_acc, hist, t16, tb * Wq_t + br * HK_t + col, Kt, (br * 4 + 3) * Kt);
                issue(taps_acc, taps, t16, br * HK_t + col, Kt, br * Kt);
            }
            issue(gates_acc, gf, t32, tb * Wg_t + h * Kt, Kt, 0);
            issue(cc_acc, cc, t32, h * Kt, Kt, 0);
            issue(gates_acc, gb, t32, tb * Wg_t + HK_t + HV_t + (h >> 5), 1, 0);
            issue(gates_acc, gog, t32, tb * Wg_t + HK_t + h * Vt, Vt, 0);
            noc.async_read_barrier();
            if (active) {
                selT.push_back(2);
                S.push_back(Kt * Vt);
            }
            if constexpr (parts > 1) {
                // multicast the item's inputs into the other parts' CBs (same L1 addresses) once they are ready
                noc_semaphore_wait(ready, parts - 1);
                noc_semaphore_set(ready, 0);
                const uint32_t n_dst = parts - 1;
                auto mcast = [&](CircularBuffer& cbo, uint32_t bytes) {
                    auto v = use<CircularBuffer::AddrSelector::WRITE_PTR>(cbo);  // same L1 address on every part
                    noc.async_write_multicast(
                        v,
                        v,
                        bytes,
                        n_dst,
                        {.offset_bytes = 0},
                        {.noc_x_start = p0_x,
                         .noc_y_start = p0_y + 1,
                         .noc_x_end = pe_x,
                         .noc_y_end = pe_y,
                         .offset_bytes = 0});
                };
                mcast(hist, 12 * Kt * t16);
                mcast(taps, 3 * Kt * t16);
                mcast(gf, Kt * t32);
                mcast(gog, Vt * t32);
                mcast(gb, t32);
                mcast(cc, Kt * t32);
                noc.async_write_barrier();  // data landed before the "valid" flag is raised
                noc_semaphore_set_multicast(
                    get_semaphore(SEM_ONE),
                    get_noc_multicast_addr(p0_x, p0_y + 1, pe_x, pe_y, get_semaphore(SEM_VALID)),
                    n_dst);
            }
        } else {
            noc_semaphore_set(valid, 0);
            noc_semaphore_inc(get_noc_addr(p0_x, p0_y, get_semaphore(SEM_READY)), 1);
            noc_semaphore_wait(valid, 1);
            if (active) {
                noc.async_read_barrier();
                selT.push_back(2);
                S.push_back(Kt * Vt);
            }
        }
        hist.push_back(12 * Kt);
        taps.push_back(3 * Kt);
        gf.push_back(Kt);
        gog.push_back(Vt);
        gb.push_back(1);
        cc.push_back(Kt);
        if (part == 0) {
            // history shift: tap j -> tap j-1, raw -> tap 2 (the old tiles stay in L1 until the item is done)
            for (uint32_t br = 0; br < 3; br++) {
                for (uint32_t j = 1; j <= conv_taps; j++) {
                    for (uint32_t t = 0; t < Kt; t++) {
                        noc.async_write(
                            hist,
                            hist_acc,
                            t16,
                            {.offset_bytes = ((br * 4 + j) * Kt + t) * t16},
                            {.page_id = ((j - 1) * Bt + tb) * Wq_t + br * HK_t + col + t});
                    }
                }
            }
        }
        if (!active) {
            noc.async_write_barrier();
            continue;  // an idle part (no users): took part in the multicast only
        }
        if (part == 0 && parts > 1) {
            gather.reserve_back((parts - 1) * Vt);  // the other parts write their output blocks here
        }
        // the remaining users: the selector tile E_r^T and the recurrent state block (b*H + h)
        for (uint32_t r = u0 + 1; r < u1; r++) {
            read_tiles(sel_acc, cb_selT, t16, r, 1);
            read_tiles(st_acc, cb_S, t32, ((tb * 32 + r) * H + h) * Kt * Vt, Kt * Vt);
        }
        noc.async_write_barrier();  // the shift writes (before any reuse of the hist CB)
        if (part == 0 && parts > 1) {
            noc_semaphore_wait(gath, n_active - 1);
            noc_semaphore_set(gath, 0);
            gather.push_back((parts - 1) * Vt);
        }
    }
    // every NoC transaction this kernel issued (the cross-core semaphore increments above included) must be
    // acknowledged before the kernel returns: an atomic acked after the next kernel's noc_local_state_init
    // leaves that kernel's atomic counters unequal forever (its noc_async_atomic_barrier hangs; seen on the 8x4
    // o_proj matmul in0 receiver after kda_decode, 2026-10-20)
    noc_async_full_barrier();
}
