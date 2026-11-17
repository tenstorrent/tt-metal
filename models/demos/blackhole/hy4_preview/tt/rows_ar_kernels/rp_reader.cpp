// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
// Rows-phase one-shot reduce, reducer core (NCRISC). Group of NB=4 chips, chip position r owns token block r (T rows).
// Reads chunk p (Q tile columns) of the partial [32 x 6144] TILE tensor: for every block b the T rows b*T.. of both
// faces (H rows read, tiny tile of H rows = face0 rows | face1 rows, 64*H B). Block r goes to CB_OWN, the other blocks
// to CB_PAY and from there to the staging area of the forward / backward communication core (which sends them over the
// fabric). Then waits for the NB-1 chunks of the other chips (they land in the persistent receive buffer) and releases
// compute.
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/debug/kernel_profiler.hpp"

void kernel_main() {
    constexpr uint32_t CB_RECV = get_compile_time_arg_val(0);
    constexpr uint32_t Q = get_compile_time_arg_val(1);
    constexpr uint32_t CB_OWN = get_compile_time_arg_val(2);
    constexpr uint32_t CB_PAY = get_compile_time_arg_val(3);
    constexpr uint32_t TILEB = get_compile_time_arg_val(4);
    constexpr uint32_t T = get_compile_time_arg_val(5);
    constexpr uint32_t H = get_compile_time_arg_val(6);
    constexpr uint32_t CB_STG = get_compile_time_arg_val(7);
    constexpr uint32_t CB_SCR = get_compile_time_arg_val(8);
    constexpr auto in_args = TensorAccessorArgs<9>();
#ifdef RP_FOLD  // B30: the shared-expert output of this mesh row ([T rows] std tile tensor) is added to the OWN block
                // (CB_FLD, wire format of CB_OWN) instead of a separate placement matmul + add
    constexpr uint32_t CB_FLD = 14;
    constexpr auto fold_args = TensorAccessorArgs<in_args.next_compile_time_args_offset()>();
#endif
    constexpr uint32_t NB = 4;
    constexpr uint32_t NR = NB - 1;
#ifdef RP_PK
    constexpr uint32_t NTC =
        1;  // packed mode: the Q tile columns x T token rows of a block are the rows (y = i T + t) of ONE 32x32 tile
#else
    constexpr uint32_t NTC = Q;
#endif
    constexpr uint32_t CHUNK = NTC * TILEB;

    const uint32_t in_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t gsem = get_common_arg_val<uint32_t>(1);
    const uint32_t r = get_arg_val<uint32_t>(0);
    const uint32_t p = get_arg_val<uint32_t>(1);
    const uint32_t nd_f = get_arg_val<uint32_t>(2);
    const uint32_t nd_b = get_arg_val<uint32_t>(3);
    const uint32_t fx = get_arg_val<uint32_t>(4);
    const uint32_t fy = get_arg_val<uint32_t>(5);
    const uint32_t bx = get_arg_val<uint32_t>(6);
    const uint32_t by = get_arg_val<uint32_t>(7);
    const uint32_t sid = get_arg_val<uint32_t>(8);
    const auto in_acc = TensorAccessor(in_args, in_addr, 2048);
#ifdef RP_FOLD
    const auto fold_acc = TensorAccessor(fold_args, get_common_arg_val<uint32_t>(2), 2048);
#endif

    cb_reserve_back(CB_RECV, NR * NTC);
    cb_reserve_back(CB_OWN, NTC);
    cb_reserve_back(CB_PAY, NB * NTC);
    const uint32_t own = get_write_ptr(CB_OWN);
    const uint32_t pay = get_write_ptr(CB_PAY);
    {
        DeviceZoneScopedN("P_READ");
        // DRAM reads must be 64 B aligned: block b is read from the tile row A = (b T) & ~1 (even row = 64 B aligned),
        // H rows: the token row b T sits at tile row (b T - A) <= 1 of the wire tile
#ifdef RP_PK
        constexpr uint32_t RB = (T >= 2) ? 32 * T : 64;  // read piece: T rows of one face (T = 4: 128 B, 64 B aligned)
        const uint32_t scr = get_write_ptr(CB_SCR);      // [b][i][f] RB-byte pieces
        for (uint32_t b = 0; b < NB; ++b) {
            const uint32_t A = (b * T) & ~1u;
            for (uint32_t i = 0; i < Q; ++i) {
                for (uint32_t f = 0; f < 2; ++f) {
                    noc_async_read(
                        in_acc.get_noc_addr(p * Q + i, f * 512 + A * 32), scr + ((b * Q + i) * 2 + f) * RB, RB);
                }
            }
        }
        noc_async_read_barrier();
        for (uint32_t b = 0; b < NB;
             ++b) {  // repack: token row t of tile column i -> row y = i T + t of the block's wire tile
            const uint32_t base = (b == r) ? own : (pay + b * CHUNK);
            const uint32_t sh = (b * T) & 1u;
            for (uint32_t i = 0; i < Q; ++i) {
                for (uint32_t f = 0; f < 2; ++f) {
                    for (uint32_t t = 0; t < T; ++t) {
                        const uint32_t y = i * T + t;
                        noc_async_write(
                            scr + ((b * Q + i) * 2 + f) * RB + (sh + t) * 32,
                            get_noc_addr(base + ((y >> 4) * 2 + f) * 512 + (y & 15) * 32),
                            32);
                    }
                }
            }
        }
        noc_async_write_barrier();
#ifdef RP_FOLD
        {
            cb_reserve_back(CB_FLD, NTC);
            const uint32_t fld = get_write_ptr(CB_FLD);
            const uint32_t fscr =
                scr + NB * Q * 2 * RB;  // scratch piece area after the NB blocks (CB_SCR holds NB + 1 of them)
            for (uint32_t i = 0; i < Q; ++i) {
                for (uint32_t f = 0; f < 2; ++f) {
                    noc_async_read(
                        fold_acc.get_noc_addr(p * Q + i, f * 512),
                        fscr + (i * 2 + f) * RB,
                        RB);  // rows 0..T-1 of the fold tile
                }
            }
            noc_async_read_barrier();
            for (uint32_t i = 0; i < Q; ++i) {
                for (uint32_t f = 0; f < 2; ++f) {
                    for (uint32_t t = 0; t < T; ++t) {
                        const uint32_t y = i * T + t;
                        noc_async_write(
                            fscr + (i * 2 + f) * RB + t * 32,
                            get_noc_addr(fld + ((y >> 4) * 2 + f) * 512 + (y & 15) * 32),
                            32);
                    }
                }
            }
            noc_async_write_barrier();
        }
#endif
#else
        for (uint32_t b = 0; b < NB; ++b) {
            const uint32_t base = (b == r) ? own : (pay + b * CHUNK);
            const uint32_t A = (b * T) & ~1u;
            for (uint32_t i = 0; i < Q; ++i) {
                for (uint32_t f = 0; f < 2; ++f) {
                    noc_async_read(
                        in_acc.get_noc_addr(p * Q + i, f * 512 + A * 32), base + i * TILEB + f * 32 * H, 32 * H);
                }
            }
        }
        noc_async_read_barrier();
#endif
    }
    {
        DeviceZoneScopedN("P_SHIP");
        const uint32_t stg = get_write_ptr(CB_STG);
        for (uint32_t b = 0; b < NB; ++b) {
            if (b == r) {
                continue;
            }
            const uint32_t cx = (b > r) ? fx : bx, cy = (b > r) ? fy : by;
            noc_async_write(pay + b * CHUNK, get_noc_addr(cx, cy, stg + (p * NB + b) * CHUNK), CHUNK);
        }
        noc_async_write_barrier();
        const uint32_t sem_addr = get_semaphore(0);
        if (nd_f) {
            noc_semaphore_inc(get_noc_addr(fx, fy, sem_addr), 1);
        }
        if (nd_b) {
            noc_semaphore_inc(get_noc_addr(bx, by, sem_addr), 1);
        }
        noc_async_atomic_barrier();
    }
    volatile tt_l1_ptr uint32_t* sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(gsem);
    {
        DeviceZoneScopedN("P_WAIT");
        noc_semaphore_wait_min(sem, NR);
    }
    noc_semaphore_set(sem, 0);
    cb_push_back(CB_OWN, NTC);
#ifdef RP_FOLD
    cb_push_back(CB_FLD, NTC);
#endif
    cb_push_back(CB_RECV, NR * NTC);
}
