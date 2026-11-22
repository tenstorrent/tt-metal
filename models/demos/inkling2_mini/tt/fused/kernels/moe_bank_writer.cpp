// SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
// SPDX-License-Identifier: Apache-2.0
//
// Bank-streamed MoE expert step, writer (BRISC, NOC1), one worker core (tt/fused/moe_bank.py). Flags as the reader.
//   up core (F_GU, not F_GATE; pair j): waits its 8 up tiles (compute's cb_g) and writes them into the gate core's
//     cb_u16, raises its sem_u (as moe_expert_writer.cpp).
//   gate core (F_GATE): waits sem_u, hands cb_u16 to compute, waits the SwiGLU tile (h_e of hidden tile j for the 8
//     experts) and writes h_e into the collector's cb_hid at slot e * 36 + j for every expert it computed (F_EXTRA:
//     only experts whose extra pair is in this core's bank), then raises the collector's sem_g by the number written
//     (the collector waits for all 8 * 36).
//   down core (F_DN): waits sem_r (the collector multicast the whole hidden), hands cb_hid to compute, then writes its
//     output tile (1x32 row -> row 0 of a zeroed 32x32 bf16 tile) to page n.
// Runtime args: flags, j, partner x, y, collector x, y, out address, n, ids address, bank.

#include <cstdint>
#include "api/dataflow/dataflow_api.h"

constexpr uint32_t cb_g = get_named_compile_time_arg_val("cb_g");
constexpr uint32_t cb_u16 = get_named_compile_time_arg_val("cb_u16");
constexpr uint32_t cb_h16 = get_named_compile_time_arg_val("cb_h16");
constexpr uint32_t cb_hid = get_named_compile_time_arg_val("cb_hid");
constexpr uint32_t cb_o = get_named_compile_time_arg_val("cb_o");
constexpr uint32_t cb_z = get_named_compile_time_arg_val("cb_z");
constexpr uint32_t cb_scr = get_named_compile_time_arg_val("cb_scr");
constexpr uint32_t sem_u_id = get_named_compile_time_arg_val("sem_u");
constexpr uint32_t sem_g_id = get_named_compile_time_arg_val("sem_g");
constexpr uint32_t sem_r_id = get_named_compile_time_arg_val("sem_r");
constexpr uint32_t topk = get_named_compile_time_arg_val("topk");
constexpr uint32_t hid_tiles = get_named_compile_time_arg_val("hid_tiles");  // 36 * 8
constexpr uint32_t nb = get_named_compile_time_arg_val("n_banks");
constexpr uint32_t x_lo = get_named_compile_time_arg_val("x_lo");
constexpr uint32_t F_GU = 1, F_GATE = 2, F_EXTRA = 4, F_DN = 8;
constexpr uint32_t SLOT = 512;  // 8 tiny tiles of 64 B

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

    const uint32_t hid_base = get_write_ptr(cb_hid);

    // zero the output staging tile while compute works
    const uint32_t z = get_write_ptr(cb_z);
    if (flags & F_DN) {
        volatile tt_l1_ptr uint32_t* zp = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(z);
        for (uint32_t i = 0; i < 2048 / 4; ++i) {
            zp[i] = 0;
        }
    }

    if (flags & F_GATE) {
        uint32_t send = 0xff;  // experts whose h this core sends
        if (flags & F_EXTRA) {
            const uint32_t scr = get_write_ptr(cb_scr) + 64;  // the reader's scratch uses bytes 0..63
            noc_async_read(ids.get_noc_addr(0), scr, 32);
            noc_async_read_barrier();
            volatile tt_l1_ptr uint16_t* id16 = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(scr);
            send = 0;
            for (uint32_t e = 0; e < topk; ++e) {
                if (x_lo + id16[e] % (nb - x_lo) == bank) {
                    send |= 1u << e;
                }
            }
        }
        volatile tt_l1_ptr uint32_t* sem_u = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_u_id));
        noc_semaphore_wait(sem_u, 1);
        cb_reserve_back(cb_u16, 1);
        cb_push_back(cb_u16, 1);
        cb_wait_front(cb_h16, 1);
        uint32_t sent = 0;
        for (uint32_t e = 0; e < topk; ++e) {
            if (send & (1u << e)) {
                noc_async_write(
                    get_read_ptr(cb_h16) + e * 64,
                    get_noc_addr(cx, cy, hid_base + (e * (hid_tiles / topk) + j) * 64),
                    64);
                ++sent;
            }
        }
        noc_async_write_barrier();
        if (sent) {
            noc_semaphore_inc(get_noc_addr(cx, cy, get_semaphore(sem_g_id)), sent);
        }
        cb_pop_front(cb_h16, 1);
    } else if (flags & F_GU) {
        cb_wait_front(cb_g, topk);
        noc_async_write(get_read_ptr(cb_g), get_noc_addr(px, py, get_write_ptr(cb_u16)), SLOT);
        noc_async_write_barrier();
        noc_semaphore_inc(get_noc_addr(px, py, get_semaphore(sem_u_id)), 1);
        cb_pop_front(cb_g, topk);
    }

    if (flags & F_DN) {
        volatile tt_l1_ptr uint32_t* sem_r = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_r_id));
        noc_semaphore_wait(sem_r, 1);
        cb_reserve_back(cb_hid, hid_tiles);
        cb_push_back(cb_hid, hid_tiles);

        cb_wait_front(cb_o, 1);
        volatile tt_l1_ptr uint32_t* o = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_read_ptr(cb_o));
        volatile tt_l1_ptr uint32_t* zp = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(z);
        for (uint32_t i = 0; i < 8; ++i) {
            zp[i] = o[i];            // cols 0..15 -> face 0 row 0
            zp[128 + i] = o[8 + i];  // cols 16..31 -> face 1 row 0
        }
        noc_async_write(z, out.get_noc_addr(n), 2048);
        noc_async_write_barrier();
        cb_pop_front(cb_o, 1);
    }
    noc_async_atomic_barrier();
}
