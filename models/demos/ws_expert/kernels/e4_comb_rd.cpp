// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// EP-4 pre-sum (ep4_layer.build_combine), adder reader (BRISC, NOC0: the DRAM reads). This core's range of the group's
// run records (runs pages [a, a + nt), 128 B each: 0 base O row, 1 count, 4 + j gate weight fp32, ...) into the run
// table in L1, then TAB_FLAG = 1 (the compute and the writer wait for it; the writer clears it at its end). Then per
// run, per pair j < count: O row base + j into a CB 0 slot of 4 tiles (the row's H bf16 contiguous; the slot's last
// bytes zero, written once) and the gate weight as bf16 at element 0 of a CB 1 scalar tile (its other elements zero).
// Up to NBUF rows in flight (transaction id 1 + slot), retired in order.
// CT: 0 NBUF, 1 tab (L1), 2 tab flag (L1 word), 3 row bytes, 4 CB 1 bytes, then TensorAccessorArgs(runs), (O).
// E4_OEAST (EP4Cfg.oeast): O row d at page (d >> 2) 8 + 4 + (d & 3) (ws_lane5.oeast_row). E4_OSIDE (EP4Cfg.oside): at
// (d >> 2) 8 + 4 side + (d & 3), side the bit of the pair's local expert (run record word 16 + j) in RT 5-9.
// RT: 0 runs addr, 1 O addr, 2 a, 3 nt, 4 stamps (5 reads wait cycles, 6 CB space wait cycles).
#include "api/dataflow/dataflow_api.h"

FORCE_INLINE uint32_t wall() { return *reinterpret_cast<volatile uint32_t*>(RISCV_DEBUG_REG_WALL_CLOCK_L); }
FORCE_INLINE uint16_t bf16_rne(uint32_t f) { return static_cast<uint16_t>((f + 0x7FFF + ((f >> 16) & 1)) >> 16); }
FORCE_INLINE void zero_l1(uint32_t addr, uint32_t bytes) {
    volatile tt_l1_ptr uint32_t* p = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(addr);
    for (uint32_t i = 0; i < bytes / 4; ++i) {
        p[i] = 0;
    }
}

void kernel_main() {
    constexpr uint32_t NBUF = get_compile_time_arg_val(0);
    constexpr uint32_t tab = get_compile_time_arg_val(1);
    constexpr uint32_t tab_flag = get_compile_time_arg_val(2);
    constexpr uint32_t row_bytes = get_compile_time_arg_val(3);
    constexpr uint32_t w_cb_bytes = get_compile_time_arg_val(4);
    constexpr auto rargs = TensorAccessorArgs<5>();
    constexpr auto oargs = TensorAccessorArgs<rargs.next_compile_time_args_offset()>();
    constexpr uint32_t rows_cb = tt::CBIndex::c_0, w_cb = tt::CBIndex::c_1;
    constexpr uint32_t TPR = 4, tile_bytes = 2048, unit_bytes = TPR * tile_bytes;
    static_assert(row_bytes <= unit_bytes && NBUF >= 1 && NBUF < 16);

    const auto rs = TensorAccessor(rargs, get_arg_val<uint32_t>(0), 128);
    const auto os = TensorAccessor(oargs, get_arg_val<uint32_t>(1), row_bytes);
    const uint32_t a = get_arg_val<uint32_t>(2), nt = get_arg_val<uint32_t>(3);
    volatile tt_l1_ptr uint32_t* stamps = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_arg_val<uint32_t>(4));
#if defined(E4_OSIDE)
    uint32_t smask[5];
    for (uint32_t i = 0; i < 5; ++i) {
        smask[i] = get_arg_val<uint32_t>(5 + i);
    }
#endif
    const uint32_t rows_base = get_write_ptr(rows_cb), w_base = get_write_ptr(w_cb);
    for (uint32_t u = 0; u < NBUF; ++u) {
        zero_l1(rows_base + u * unit_bytes + row_bytes, unit_bytes - row_bytes);
    }
    zero_l1(w_base, w_cb_bytes);

    for (uint32_t i = 0; i < nt; ++i) {
        noc_async_read(rs.get_noc_addr(a + i), tab + i * 128, 128);
    }
    noc_async_read_barrier();
    *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(tab_flag) = 1;

    volatile tt_l1_ptr uint32_t* tb = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(tab);
    uint32_t t_bar = 0, t_res = 0;
    // (i, j) of the next row to issue / to land; issued - landed rows in flight in ring slots
    uint32_t ii = 0, ij = 0, issued = 0, landed = 0;
    uint32_t li = 0, lj = 0;
    auto adv = [&](uint32_t& i, uint32_t& j) {
        ++j;
        while (i < nt && j >= tb[i * 32 + 1]) {
            ++i;
            j = 0;
        }
    };
    auto first = [&](uint32_t& i, uint32_t& j) {
        i = 0;
        j = 0;
        while (i < nt && tb[i * 32 + 1] == 0) {
            ++i;
        }
    };
    first(ii, ij);
    first(li, lj);
    while (li < nt) {
        while (ii < nt && issued - landed < NBUF) {
            const uint32_t s0 = wall();
            cb_reserve_back(rows_cb, (issued - landed + 1) * TPR);
            t_res += wall() - s0;
            const uint32_t slot = issued % NBUF;
            noc_async_read_set_trid(1 + slot);
#if defined(E4_OEAST)
            const uint32_t d = tb[ii * 32] + ij, pg = (d >> 2) * 8 + 4 + (d & 3);
#elif defined(E4_OSIDE)
            const uint32_t d = tb[ii * 32] + ij, le = tb[ii * 32 + 16 + ij];
            const uint32_t pg = (d >> 2) * 8 + ((smask[le >> 5] >> (le & 31)) & 1) * 4 + (d & 3);
#else
            const uint32_t pg = tb[ii * 32] + ij;
#endif
            noc_async_read(os.get_noc_addr(pg), rows_base + slot * unit_bytes, row_bytes);
            noc_async_read_set_trid(0);
            ++issued;
            adv(ii, ij);
        }
        const uint32_t slot = landed % NBUF;
        const uint32_t s0 = wall();
        noc_async_read_barrier_with_trid(1 + slot);
        t_bar += wall() - s0;
        cb_reserve_back(w_cb, 1);
        *reinterpret_cast<volatile tt_l1_ptr uint16_t*>(get_write_ptr(w_cb)) = bf16_rne(tb[li * 32 + 4 + lj]);
        cb_push_back(w_cb, 1);
        cb_push_back(rows_cb, TPR);
        ++landed;
        adv(li, lj);
    }
    stamps[5] = t_bar;
    stamps[6] = t_res;
}
