// SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
// SPDX-License-Identifier: Apache-2.0
//
// Fused MoE expert step, writer (BRISC), one of the 72 worker cores.
//   up core j:   sends its 8 up tiles (8 tiny tiles, 512 B) into gate core j's cb_u16, raises its sem_u.
//   gate core j: waits sem_u, hands cb_u16 to compute; sends its 8 hidden tiles (h_e for hidden tile j, 512 B)
//                to the collector's cb_hid at slot j, raises the collector's sem_g.
//   every core:  waits sem_r (the collector multicast the whole hidden [36][8] into cb_hid), hands cb_hid to
//                compute, then writes its output tile (1x32 row -> row 0 of a zeroed 32x32 bf16 tile) to page L.
// Runtime args: is_gate, j, partner x, partner y (NOC coords of the gate core, up cores only), collector x, y,
//   out address, L.

#include <cstdint>
#include "api/dataflow/dataflow_api.h"

constexpr uint32_t cb_g = get_named_compile_time_arg_val("cb_g");
constexpr uint32_t cb_u16 = get_named_compile_time_arg_val("cb_u16");
constexpr uint32_t cb_h16 = get_named_compile_time_arg_val("cb_h16");
constexpr uint32_t cb_hid = get_named_compile_time_arg_val("cb_hid");
constexpr uint32_t cb_o = get_named_compile_time_arg_val("cb_o");
constexpr uint32_t cb_z = get_named_compile_time_arg_val("cb_z");
#ifdef DEBUG_DUMP
constexpr uint32_t cb_x = get_named_compile_time_arg_val("cb_x");
constexpr uint32_t cb_ws16 = get_named_compile_time_arg_val("cb_ws16");
#endif
constexpr uint32_t sem_u_id = get_named_compile_time_arg_val("sem_u");
constexpr uint32_t sem_g_id = get_named_compile_time_arg_val("sem_g");
constexpr uint32_t sem_r_id = get_named_compile_time_arg_val("sem_r");
constexpr uint32_t topk = get_named_compile_time_arg_val("topk");
constexpr uint32_t hid_tiles = get_named_compile_time_arg_val("hid_tiles");  // 36 * 8

constexpr uint32_t SLOT = 512;  // 8 tiny tiles of 64 B

void kernel_main() {
    uint32_t a = 0;
    const uint32_t is_gate = get_arg_val<uint32_t>(a++);
    const uint32_t j = get_arg_val<uint32_t>(a++);
    const uint32_t px = get_arg_val<uint32_t>(a++);
    const uint32_t py = get_arg_val<uint32_t>(a++);
    const uint32_t cx = get_arg_val<uint32_t>(a++);
    const uint32_t cy = get_arg_val<uint32_t>(a++);
    const uint32_t out_addr = get_arg_val<uint32_t>(a++);
    const uint32_t L = get_arg_val<uint32_t>(a++);
    constexpr auto out_args = TensorAccessorArgs<0>();
    const auto out = TensorAccessor(out_args, out_addr);
#ifdef DEBUG_DUMP
    // debug dump: bf16 ROW_MAJOR [256, 256] DRAM, 512 B rows (see moe_expert.py debug_rows)
    constexpr auto dbg_args = TensorAccessorArgs<out_args.next_compile_time_args_offset()>();
    const auto dbg = TensorAccessor(dbg_args, get_arg_val<uint32_t>(8));
    auto dump = [&](uint32_t l1, uint32_t row, uint32_t off, uint32_t bytes) {
        noc_async_write(l1, dbg.get_noc_addr(row, off), bytes);
        noc_async_write_barrier();
    };
#endif

    const uint32_t u16_base = get_write_ptr(cb_u16);
    const uint32_t hid_base = get_write_ptr(cb_hid);

    // zero the output staging tile while compute works
    const uint32_t z = get_write_ptr(cb_z);
    {
        volatile tt_l1_ptr uint32_t* zp = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(z);
        for (uint32_t i = 0; i < 2048 / 4; ++i) {
            zp[i] = 0;
        }
    }

    if (is_gate) {
        volatile tt_l1_ptr uint32_t* sem_u = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_u_id));
        noc_semaphore_wait(sem_u, 1);
        cb_reserve_back(cb_u16, 1);
        cb_push_back(cb_u16, 1);

        cb_wait_front(cb_h16, 1);
#ifdef DEBUG_DUMP
        dump(get_write_ptr(cb_g), j, 0, SLOT);
        dump(u16_base, 36 + j, 0, SLOT);
        dump(get_read_ptr(cb_h16), 72 + j, 0, SLOT);
        dump(get_write_ptr(cb_ws16), 198 + j, 0, SLOT);
#endif
#ifdef STICK_LAYOUT
        // hidden layout [e][kt]: tile (e, j) at slot e * 36 + j
        for (uint32_t e = 0; e < topk; ++e) {
            noc_async_write(
                get_read_ptr(cb_h16) + e * 64, get_noc_addr(cx, cy, hid_base + (e * (hid_tiles / topk) + j) * 64), 64);
        }
#else
        noc_async_write(get_read_ptr(cb_h16), get_noc_addr(cx, cy, hid_base + j * SLOT), SLOT);
#endif
        noc_async_write_barrier();
        noc_semaphore_inc(get_noc_addr(cx, cy, get_semaphore(sem_g_id)), 1);
        cb_pop_front(cb_h16, 1);
    } else {
        cb_wait_front(cb_g, topk);
#ifdef DEBUG_DUMP
        dump(get_read_ptr(cb_g), 144 + j, 0, SLOT);
#endif
        noc_async_write(get_read_ptr(cb_g), get_noc_addr(px, py, u16_base), SLOT);
        noc_async_write_barrier();
        noc_semaphore_inc(get_noc_addr(px, py, get_semaphore(sem_u_id)), 1);
        cb_pop_front(cb_g, topk);
    }

    volatile tt_l1_ptr uint32_t* sem_r = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_r_id));
    noc_semaphore_wait(sem_r, 1);
    cb_reserve_back(cb_hid, hid_tiles);
    cb_push_back(cb_hid, hid_tiles);

    cb_wait_front(cb_o, 1);
#ifdef DEBUG_DUMP
    if (L == 0) {
        for (uint32_t r = 0; r < 36; ++r) {
            dump(hid_base + r * SLOT, 108 + r, 0, SLOT);
        }
        for (uint32_t r = 0; r < 9; ++r) {
            dump(get_write_ptr(cb_x) + r * SLOT, 180 + r, 0, SLOT);
        }
    }
    dump(get_read_ptr(cb_o), 189 + L / 8, (L % 8) * 64, 64);
#endif
    {
        volatile tt_l1_ptr uint32_t* o = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_read_ptr(cb_o));
        volatile tt_l1_ptr uint32_t* zp = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(z);
        for (uint32_t i = 0; i < 8; ++i) {
            zp[i] = o[i];            // cols 0..15 -> face 0 row 0
            zp[128 + i] = o[8 + i];  // cols 16..31 -> face 1 row 0
        }
    }
    noc_async_write(z, out.get_noc_addr(L), 2048);
    noc_async_write_barrier();
    cb_pop_front(cb_o, 1);
    noc_async_atomic_barrier();
}
