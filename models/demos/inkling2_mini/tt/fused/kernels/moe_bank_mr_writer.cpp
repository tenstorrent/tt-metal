// SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
// SPDX-License-Identifier: Apache-2.0
//
// Multi-row bank expert step, writer (moe_bank_mr_reader.cpp has the plan layout; flags as moe_bank_writer.cpp).
//   gate_up cores: gather A's tiles (cb_ga, order u then r) into one 512 B slot per row, that row's 8 experts in
//     its own order (the 1-row step's layout):
//       up core: writes slot r into the gate core's cb_u16 page r, then raises its sem_u once;
//       gate core: writes slot r into its own cb_g32 page r; after sem_u, hands cb_u16 (R pages) to compute, then
//         per row writes h of row r's experts to the collector's cb_hid at (r * 288 + e * 36 + j) * 64 (F_EXTRA:
//         only experts whose extra pair is in this bank) and raises sem_g by the count.
//   down core: after sem_r, hands cb_hid (R x 288 tiles) to compute; per row puts its output 1x32 tile into row r
//     of the zeroed 32x32 output tile; writes page n once, plus row r of the dense twin.
// Runtime args: flags, j, partner x, y, collector x, y, out address, n, ids address, bank (+ dense twin address).

#include <cstdint>
#include "api/dataflow/dataflow_api.h"

constexpr uint32_t cb_ga = get_named_compile_time_arg_val("cb_ga");
constexpr uint32_t cb_g32 = get_named_compile_time_arg_val("cb_g32");
constexpr uint32_t cb_u16 = get_named_compile_time_arg_val("cb_u16");
constexpr uint32_t cb_h16 = get_named_compile_time_arg_val("cb_h16");
constexpr uint32_t cb_hid = get_named_compile_time_arg_val("cb_hid");
constexpr uint32_t cb_o = get_named_compile_time_arg_val("cb_o");
constexpr uint32_t cb_z = get_named_compile_time_arg_val("cb_z");
constexpr uint32_t cb_scr = get_named_compile_time_arg_val("cb_scr");
constexpr uint32_t cb_plan = get_named_compile_time_arg_val("cb_plan");
constexpr uint32_t sem_u_id = get_named_compile_time_arg_val("sem_u");
constexpr uint32_t sem_g_id = get_named_compile_time_arg_val("sem_g");
constexpr uint32_t sem_r_id = get_named_compile_time_arg_val("sem_r");
constexpr uint32_t topk = get_named_compile_time_arg_val("topk");
constexpr uint32_t rows = get_named_compile_time_arg_val("rows");
constexpr uint32_t hid_tiles = get_named_compile_time_arg_val("hid_tiles");  // per row, 36 * 8
constexpr uint32_t nb = get_named_compile_time_arg_val("n_banks");
constexpr uint32_t x_lo = get_named_compile_time_arg_val("x_lo");
constexpr uint32_t F_GU = 1, F_GATE = 2, F_EXTRA = 4, F_DN = 8;
constexpr uint32_t SLOT = 512;   // 8 tiny tiles of 64 B
constexpr uint32_t PAGE = 2048;  // cb_g32 / cb_u16 page
constexpr uint32_t P_POS = 1, P_MASK = 33, P_OUT = 97;

void kernel_main() {
    uint32_t a = 0;
    const uint32_t flags = get_arg_val<uint32_t>(a++);
    const uint32_t j = get_arg_val<uint32_t>(a++);
    const uint32_t px = get_arg_val<uint32_t>(a++);
    const uint32_t py = get_arg_val<uint32_t>(a++);
    const uint32_t cx = get_arg_val<uint32_t>(a++);
    const uint32_t cy = get_arg_val<uint32_t>(a++);
    const uint32_t out_addr = get_arg_val<uint32_t>(a++);
    const uint32_t n = get_arg_val<uint32_t>(a++);
    const uint32_t ids_addr = get_arg_val<uint32_t>(a++);
    const uint32_t bank = get_arg_val<uint32_t>(a++);
    constexpr auto out_args = TensorAccessorArgs<0>();
    constexpr auto ids_args = TensorAccessorArgs<out_args.next_compile_time_args_offset()>();
    const auto out = TensorAccessor(out_args, out_addr);
    const auto ids = TensorAccessor(ids_args, ids_addr);
#ifdef DENSE_OUT  // dense twin, R vectors (tt/fused/common.py dense_matrix): vector r in pages r * P .., P =
                  // dense_pages
    constexpr auto dense_args = TensorAccessorArgs<ids_args.next_compile_time_args_offset()>();
    const auto dense = TensorAccessor(dense_args, get_arg_val<uint32_t>(a++));
#endif
    const uint32_t hid_base = get_write_ptr(cb_hid);
#ifdef X_MCAST  // the collector multicast all rows of x into cb_x: hand them to compute
    if (flags & F_GU) {
        constexpr uint32_t cb_x = get_named_compile_time_arg_val("cb_x");
        constexpr uint32_t x_tiles = get_named_compile_time_arg_val("x_tiles");
        noc_semaphore_wait(
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(get_named_compile_time_arg_val("sem_x"))), 1);
        cb_reserve_back(cb_x, rows * x_tiles);
        cb_push_back(cb_x, rows * x_tiles);
    }
#endif

    const uint32_t z = get_write_ptr(cb_z);
    if (flags & F_DN) {
        volatile tt_l1_ptr uint32_t* zp = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(z);
        for (uint32_t i = 0; i < 2048 / 4; ++i) {
            zp[i] = 0;
        }
    }

    cb_wait_front(cb_plan, 1);
    volatile tt_l1_ptr uint32_t* plan = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_read_ptr(cb_plan));

    // gather row r's 8 A tiles (its own expert order) into dst + r * PAGE; returns after the local copies
    auto gather = [&](uint32_t dst_base, bool remote) {
        cb_wait_front(cb_ga, rows * topk);
        const uint32_t ga = get_read_ptr(cb_ga);
        for (uint32_t r = 0; r < rows; ++r) {
            for (uint32_t e = 0; e < topk; ++e) {
                const uint32_t u = plan[P_POS + r * topk + e];
                const uint32_t m = plan[P_MASK + u];
                uint32_t idx = plan[P_OUT + u];
                for (uint32_t q = 0; q < r; ++q) {
                    idx += (m >> q) & 1;
                }
                const uint32_t dst = dst_base + r * PAGE + e * 64;
                noc_async_write(ga + idx * 64, remote ? get_noc_addr(px, py, dst) : get_noc_addr(dst), 64);
            }
        }
        noc_async_write_barrier();
        cb_pop_front(cb_ga, rows * topk);
    };

    if (flags & F_GATE) {
        cb_reserve_back(cb_g32, rows);
        gather(get_write_ptr(cb_g32), false);
        cb_push_back(cb_g32, rows);
        volatile tt_l1_ptr uint32_t* sem_u = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_u_id));
        noc_semaphore_wait(sem_u, 1);
        cb_reserve_back(cb_u16, rows);
        cb_push_back(cb_u16, rows);

        volatile tt_l1_ptr uint16_t* id16 = nullptr;
        if (flags & F_EXTRA) {
            const uint32_t scr = get_write_ptr(cb_scr) + 128;  // the reader's scratch uses bytes 0..127
            for (uint32_t r = 0; r < rows; ++r) {
                noc_async_read(ids.get_noc_addr(0) + r * 32, scr + r * 16, 16);
            }
            noc_async_read_barrier();
            id16 = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(scr);
        }
        uint32_t sent = 0;
        for (uint32_t r = 0; r < rows; ++r) {
            cb_wait_front(cb_h16, 1);
            for (uint32_t e = 0; e < topk; ++e) {
                if ((flags & F_EXTRA) && x_lo + id16[r * 8 + e] % (nb - x_lo) != bank) {
                    continue;
                }
                noc_async_write(
                    get_read_ptr(cb_h16) + e * 64,
                    get_noc_addr(cx, cy, hid_base + (r * hid_tiles + e * (hid_tiles / topk) + j) * 64),
                    64);
                ++sent;
            }
            noc_async_write_barrier();
            cb_pop_front(cb_h16, 1);
        }
        if (sent) {
            noc_semaphore_inc(get_noc_addr(cx, cy, get_semaphore(sem_g_id)), sent);
        }
    } else if (flags & F_GU) {
        gather(get_write_ptr(cb_u16), true);  // the gate core's cb_u16 has the same address
        noc_semaphore_inc(get_noc_addr(px, py, get_semaphore(sem_u_id)), 1);
    }

    if (flags & F_DN) {
        volatile tt_l1_ptr uint32_t* sem_r = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_r_id));
        noc_semaphore_wait(sem_r, 1);
        cb_reserve_back(cb_hid, rows * hid_tiles);
        cb_push_back(cb_hid, rows * hid_tiles);
        volatile tt_l1_ptr uint32_t* zp = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(z);
        for (uint32_t r = 0; r < rows; ++r) {
            cb_wait_front(cb_o, 1);
            volatile tt_l1_ptr uint32_t* o = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_read_ptr(cb_o));
            for (uint32_t i = 0; i < 8; ++i) {
                zp[r * 8 + i] = o[i];            // cols 0..15 -> face 0 row r
                zp[128 + r * 8 + i] = o[8 + i];  // cols 16..31 -> face 1 row r
            }
#ifdef DENSE_OUT
            const uint32_t el = r * get_named_compile_time_arg_val("dense_pages") * 1024 + 32 * n;
            noc_async_write(get_read_ptr(cb_o), dense.get_noc_addr(el / 1024) + (el % 1024) * 2, 64);
            noc_async_write_barrier();
#endif
            cb_pop_front(cb_o, 1);
        }
        noc_async_write(z, out.get_noc_addr(n), 2048);
        noc_async_write_barrier();
    }
    noc_async_atomic_barrier();
}
