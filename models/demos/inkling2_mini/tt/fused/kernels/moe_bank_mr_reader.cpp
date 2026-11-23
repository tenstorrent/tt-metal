// SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
// SPDX-License-Identifier: Apache-2.0
//
// Multi-row bank expert step, reader (tt/fused/moe_bank.py, R = `rows` verify rows; bank layout as
// moe_bank_reader.cpp). Every row r routes its own 8 experts; each expert of the union is read once.
// 1. ids / routing weights rows 0..R-1, x rows (gate_up cores, 1x32 tiles: row r's tile t at r * 72 + t).
// 2. The plan (cb_plan, uint32 page, read by compute and writer, never popped):
//      [0] U (union size, first-appearance order over rows 0.., then each row's own order)
//      [1 + r * 8 + e]   union position of row r's expert e
//      [33 + u]          row mask of union expert u
//      [65 + r * 8 + e]  down slot (0..7) of row r's expert e
//      [97 + u]          A output index base of union expert u (rows of u, in row order, follow)
// 3. gate_up cores: each union expert's column (kt_gu tiles) whole into cb_w, one at a time (two in flight).
// 4. down columns through an 8-slot cache (cb_wd, pushed once): before row r, the experts of row r not in a slot
//    are read into slots row r does not use, after compute finished row r - 1 (cb_done); then one token in cb_tok.
// Runtime args: as moe_bank_reader.cpp (no PF / X_MCAST / IDS_FROM_SEM).

#include <cstdint>
#include "api/dataflow/dataflow_api.h"

constexpr uint32_t cb_x = get_named_compile_time_arg_val("cb_x");
constexpr uint32_t cb_w = get_named_compile_time_arg_val("cb_w");
constexpr uint32_t cb_wd = get_named_compile_time_arg_val("cb_wd");
constexpr uint32_t cb_ws16 = get_named_compile_time_arg_val("cb_ws16");
constexpr uint32_t cb_scr = get_named_compile_time_arg_val("cb_scr");
constexpr uint32_t cb_plan = get_named_compile_time_arg_val("cb_plan");
constexpr uint32_t cb_tok = get_named_compile_time_arg_val("cb_tok");
constexpr uint32_t cb_done = get_named_compile_time_arg_val("cb_done");
constexpr uint32_t kt_gu = get_named_compile_time_arg_val("kt_gu");  // 72
constexpr uint32_t kt_dn = get_named_compile_time_arg_val("kt_dn");  // 36
constexpr uint32_t topk = get_named_compile_time_arg_val("topk");    // 8
constexpr uint32_t rows = get_named_compile_time_arg_val("rows");
constexpr uint32_t x_tiles = get_named_compile_time_arg_val("x_tiles");
constexpr uint32_t nb = get_named_compile_time_arg_val("n_banks");
constexpr uint32_t base = get_named_compile_time_arg_val("base");
constexpr uint32_t x_lo = get_named_compile_time_arg_val("x_lo");
constexpr uint32_t dn_slots = get_named_compile_time_arg_val("dn_slots");
constexpr uint32_t F_GU = 1, F_GATE = 2, F_EXTRA = 4, F_DN = 8;
constexpr uint32_t TINY = 64;
constexpr uint32_t FACE_BYTES = 512;
constexpr uint32_t P_POS = 1, P_MASK = 33, P_SLOT = 65, P_OUT = 97;

FORCE_INLINE uint32_t extra_bank(uint32_t e) { return x_lo + e % (nb - x_lo); }
FORCE_INLINE uint32_t gu_slot(uint32_t e, uint32_t b, uint32_t k) {
    const uint32_t before = base * e + (b >= x_lo ? (e + (nb - x_lo) - 1 - (b - x_lo)) / (nb - x_lo) : 0);
    return before + k;
}

void kernel_main() {
    uint32_t a = 0;
    const uint32_t x_addr = get_arg_val<uint32_t>(a++);
    const uint32_t wts_addr = get_arg_val<uint32_t>(a++);
    const uint32_t ids_addr = get_arg_val<uint32_t>(a++);
    const uint32_t gu_addr = get_arg_val<uint32_t>(a++);
    const uint32_t dn_addr = get_arg_val<uint32_t>(a++);
    const uint32_t flags = get_arg_val<uint32_t>(a++);
    const uint32_t bank = get_arg_val<uint32_t>(a++);
    const uint32_t kslot = get_arg_val<uint32_t>(a++);
    const uint32_t col_off = get_arg_val<uint32_t>(a++);
    const uint32_t dslot = get_arg_val<uint32_t>(a++);

    constexpr auto x_args = TensorAccessorArgs<0>();
    constexpr auto wts_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
    constexpr auto ids_args = TensorAccessorArgs<wts_args.next_compile_time_args_offset()>();
    constexpr auto gu_args = TensorAccessorArgs<ids_args.next_compile_time_args_offset()>();
    constexpr auto dn_args = TensorAccessorArgs<gu_args.next_compile_time_args_offset()>();
    const auto xs = TensorAccessor(x_args, x_addr);
    const auto wts = TensorAccessor(wts_args, wts_addr);
    const auto ids = TensorAccessor(ids_args, ids_addr);
    const auto gu = TensorAccessor(gu_args, gu_addr);
    const auto dn = TensorAccessor(dn_args, dn_addr);

    // ids rows (row r: face 0 row r, 8 uint16 = 16 B at byte r * 32) and routing weights rows (bf16, same layout)
    const uint32_t scr = get_write_ptr(cb_scr);  // ids at r * 16, weights at 64 + r * 16
    for (uint32_t r = 0; r < rows; ++r) {
        noc_async_read(ids.get_noc_addr(0) + r * 32, scr + r * 16, 16);
        noc_async_read(wts.get_noc_addr(0) + r * 32, scr + 64 + r * 16, 16);
    }
    if (flags & F_GU) {
        cb_reserve_back(cb_x, rows * x_tiles);
        const uint32_t xl = get_write_ptr(cb_x);
        for (uint32_t r = 0; r < rows; ++r) {
            for (uint32_t t = 0; t < x_tiles; ++t) {
                const uint32_t d = xl + (r * x_tiles + t) * TINY;
                noc_async_read(xs.get_noc_addr(t, r * 32), d, 32);
                noc_async_read(xs.get_noc_addr(t, FACE_BYTES + r * 32), d + 32, 32);
            }
        }
    }
    noc_async_read_barrier();
    if (flags & F_GU) {
        cb_push_back(cb_x, rows * x_tiles);
    }
    volatile tt_l1_ptr uint16_t* id16 = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(scr);

    // plan
    cb_reserve_back(cb_plan, 1);
    volatile tt_l1_ptr uint32_t* plan = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_plan));
    uint32_t uid[rows * topk];
    uint32_t U = 0;
    for (uint32_t r = 0; r < rows; ++r) {
        for (uint32_t e = 0; e < topk; ++e) {
            const uint32_t id = id16[r * 8 + e];
            uint32_t u = 0;
            while (u < U && uid[u] != id) {
                ++u;
            }
            if (u == U) {
                uid[U] = id;
                plan[P_MASK + U] = 0;
                ++U;
            }
            plan[P_POS + r * topk + e] = u;
            plan[P_MASK + u] |= 1u << r;
        }
    }
    plan[0] = U;
    uint32_t out_base = 0;
    for (uint32_t u = 0; u < U; ++u) {
        plan[P_OUT + u] = out_base;
        for (uint32_t r = 0; r < rows; ++r) {
            out_base += (plan[P_MASK + u] >> r) & 1;
        }
    }
    // down slot schedule: slot_of[s] = union position held by slot s (or none)
    constexpr uint32_t NONE = 0xffff;
    uint32_t slot_of[topk];
    uint32_t loads[rows];         // bit s: slot s is (re)loaded before row r
    uint32_t load_u[rows][topk];  // union position loaded into slot s before row r
    for (uint32_t s = 0; s < topk; ++s) {
        slot_of[s] = NONE;
    }
    for (uint32_t r = 0; r < rows; ++r) {
        loads[r] = 0;
        uint32_t keep = 0;  // slots row r uses
        for (uint32_t e = 0; e < topk; ++e) {
            const uint32_t u = plan[P_POS + r * topk + e];
            for (uint32_t s = 0; s < topk; ++s) {
                if (slot_of[s] == u) {
                    keep |= 1u << s;
                    plan[P_SLOT + r * topk + e] = s;
                }
            }
        }
        for (uint32_t e = 0; e < topk; ++e) {
            const uint32_t u = plan[P_POS + r * topk + e];
            bool held = false;
            for (uint32_t s = 0; s < topk; ++s) {
                held |= slot_of[s] == u;
            }
            if (held) {
                continue;
            }
            uint32_t s = 0;
            while (keep & (1u << s)) {
                ++s;
            }
            slot_of[s] = u;
            keep |= 1u << s;
            loads[r] |= 1u << s;
            load_u[r][s] = u;
            plan[P_SLOT + r * topk + e] = s;
        }
    }
    cb_push_back(cb_plan, 1);

    // routing weights per row (gate cores), the 1-row step's tile layout, one page per row
    if (flags & F_GATE) {
        volatile tt_l1_ptr uint16_t* w16 = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(scr + 64);
        cb_reserve_back(cb_ws16, rows);
        for (uint32_t r = 0; r < rows; ++r) {
            volatile tt_l1_ptr uint32_t* ws =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_ws16) + r * 2048);
            for (uint32_t e = 0; e < topk; ++e) {
                const uint32_t v = w16[r * 8 + e];
                const uint32_t pair = v | (v << 16);
                for (uint32_t i = 0; i < 16; ++i) {
                    ws[e * 16 + i] = pair;
                }
            }
        }
        cb_push_back(cb_ws16, rows);
    }

    auto read_down_rows = [&]() {
        const uint32_t tb = get_tile_size(cb_wd);
        cb_reserve_back(cb_wd, topk * kt_dn);
        const uint32_t wd = get_write_ptr(cb_wd);
        for (uint32_t r = 0; r < rows; ++r) {
            if (r > 0) {  // compute is done with row r - 1's slots
                cb_wait_front(cb_done, 1);
                cb_pop_front(cb_done, 1);
            }
            for (uint32_t s = 0; s < topk; ++s) {
                if (loads[r] & (1u << s)) {
                    const uint32_t slot = (uid[load_u[r][s]] * dn_slots + dslot) * kt_dn;
                    noc_async_read(dn.get_noc_addr(slot * nb + bank), wd + s * kt_dn * tb, kt_dn * tb);
                }
            }
            noc_async_read_barrier();
            if (r == 0) {
                cb_push_back(cb_wd, topk * kt_dn);
            }
            cb_reserve_back(cb_tok, 1);
            cb_push_back(cb_tok, 1);
        }
    };

    if (!(flags & F_GU)) {
        if (flags & F_DN) {
            read_down_rows();
        }
        return;
    }

    // gate_up: each union expert's column whole (the pair = gate column then up column, contiguous in this bank);
    // the next column's read is in flight while compute works on this one (cb_w holds two)
    const uint32_t tbw = get_tile_size(cb_w);
    for (uint32_t u = 0; u < U; ++u) {
        const bool active = !(flags & F_EXTRA) || extra_bank(uid[u]) == bank;
        cb_reserve_back(cb_w, kt_gu);
        if (active) {  // not this expert's extra bank: stale tiles, the writer drops the results
            const uint64_t src = gu.get_noc_addr((gu_slot(uid[u], bank, kslot) * 2 * kt_gu + col_off) * nb + bank);
            noc_async_read(src, get_write_ptr(cb_w), kt_gu * tbw);
            noc_async_read_barrier();
        }
        cb_push_back(cb_w, kt_gu);
    }
    if (flags & F_DN) {
        read_down_rows();
    }
}
