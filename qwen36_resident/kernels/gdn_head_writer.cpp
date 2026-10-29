// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Resident GDN head core writer (BRISC), one value head h for users u0 .. u0 + nu. Per GDN layer g (copy
// g % copies) and user u: packs row 0 of the gated output tiles into row u of batch x 32 tiles (the stage)
// and stores the user's updated recurrent state back to DRAM. After its last user the head's collector
// (the core of the head's first users) waits for the head's other cores to copy their users' rows into its
// stage, then writes the tiles into every streamer's mixer-output buffer at this head's offset (bumping
// the streamers' head semaphore); another core copies its rows into the collector's stage (the same
// CB address on every head core) and bumps the collector's sem_group.
//
// Compile-time args: 0 Kt, 1 Vt, 2 GDN layers, 3 copies, 4 num_streamers, 5 sem_heads, 6 heads (per chip),
//   7 sem_state (local: state write-backs done), 8 batch, 9 sem_group (collector: other cores' rows in),
//   10 verify (0 / 1: row u writes state slot cur ^ u, see the reader), 11 sem_slot (local: cur + 1)
// Runtime args: 0 state address (DRAM, see the reader), 1 o_in_addr (streamers' buffer), 2 head index,
// then num_streamers x (x, y), then an optional timeline buffer (0 = off; per layer [2] state out,
// [3] output sent), then u0, nu, the head's cores (1: this core is the collector), the collector's x, y

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

constexpr uint32_t Kt = get_compile_time_arg_val(0);
constexpr uint32_t Vt = get_compile_time_arg_val(1);
constexpr uint32_t layers = get_compile_time_arg_val(2);
constexpr uint32_t copies = get_compile_time_arg_val(3);
constexpr uint32_t num_streamers = get_compile_time_arg_val(4);
constexpr uint32_t sem_heads = get_compile_time_arg_val(5);
constexpr uint32_t num_heads = get_compile_time_arg_val(6);
constexpr uint32_t sem_state = get_compile_time_arg_val(7);
constexpr uint32_t batch = get_compile_time_arg_val(8);
constexpr uint32_t sem_group = get_compile_time_arg_val(9);
constexpr bool verify = get_compile_time_arg_val(10) != 0;
constexpr uint32_t sem_slot = get_compile_time_arg_val(11);
constexpr uint32_t st = Kt * Vt;
constexpr uint32_t cb_s_new_out = 24, cb_out = 28, cb_stage = 22;
constexpr uint32_t kBf16Tile = 2048, kF32Tile = 4096, kFaceBytes = 512, kFaceRow = 32;
constexpr uint32_t kRowTile = 64 * batch;

void kernel_main() {
    const uint32_t state = get_arg_val<uint32_t>(0);
    const uint32_t o_in = get_arg_val<uint32_t>(1);
    const uint32_t head = get_arg_val<uint32_t>(2);
    constexpr uint32_t peers_base = 3;
    const uint32_t heads_sem_addr = get_semaphore(sem_heads);
    const InterleavedAddrGenFast<true> state_dram{
        .bank_base_address = state, .page_size = kF32Tile, .data_format = DataFormat::Float32};
    const uint32_t ts_addr = get_arg_val<uint32_t>(peers_base + 2 * num_streamers);
    const uint32_t u0 = get_arg_val<uint32_t>(peers_base + 2 * num_streamers + 1);
    const uint32_t nu = get_arg_val<uint32_t>(peers_base + 2 * num_streamers + 2);
    const uint32_t cores = get_arg_val<uint32_t>(peers_base + 2 * num_streamers + 3);
    const uint32_t cx = get_arg_val<uint32_t>(peers_base + 2 * num_streamers + 4);
    const uint32_t cy = get_arg_val<uint32_t>(peers_base + 2 * num_streamers + 5);
    const bool collector = u0 == 0;
    volatile tt_l1_ptr uint32_t* group_sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_group));
    volatile tt_l1_ptr uint32_t* ts = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ts_addr);
    volatile uint32_t* clk = reinterpret_cast<volatile uint32_t*>(RISCV_DEBUG_REG_WALL_CLOCK_L);
    auto& stage_iface = get_local_cb_interface(cb_stage);
    const uint32_t stage = stage_iface.fifo_limit - stage_iface.fifo_size;
    uint32_t cur = 0;
    if constexpr (verify) {
        volatile tt_l1_ptr uint32_t* slot_sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_slot));
        while (*slot_sem == 0) {
            invalidate_l1_cache();
        }
        cur = *slot_sem - 1;
    }

    for (uint32_t i = 0; i < layers * nu; i++) {
        const uint32_t l = i / nu, u = u0 + i % nu;
        const uint32_t set = l % copies;
        cb_wait_front(cb_out, Vt);
        const uint32_t out = get_read_ptr(cb_out);
        for (uint32_t t = 0; t < Vt; t++) {
            const uint32_t dst = stage + t * kRowTile + u * kFaceRow;
            noc_async_read(get_noc_addr(out + t * kBf16Tile), dst, kFaceRow);
            noc_async_read(get_noc_addr(out + t * kBf16Tile + kFaceBytes), dst + batch * kFaceRow, kFaceRow);
        }
        noc_async_read_barrier();
        cb_pop_front(cb_out, Vt);
        if (u + 1 == u0 + nu && !collector) {
            // users u0 .. u0 + nu are contiguous face rows of every tile
            for (uint32_t t = 0; t < Vt; t++) {
                for (uint32_t f = 0; f < 2; f++) {
                    const uint32_t off = t * kRowTile + (f * batch + u0) * kFaceRow;
                    noc_async_write(stage + off, get_noc_addr(cx, cy, stage + off), nu * kFaceRow);
                }
            }
            noc_async_write_barrier();
            noc_semaphore_inc(get_noc_addr(cx, cy, get_semaphore(sem_group)), 1);
        }
        if (u + 1 == u0 + nu && collector) {
            noc_semaphore_wait_min(group_sem, (cores - 1) * (l + 1));
            for (uint32_t s = 0; s < num_streamers; s++) {
                const uint32_t px = get_arg_val<uint32_t>(peers_base + 2 * s);
                const uint32_t py = get_arg_val<uint32_t>(peers_base + 2 * s + 1);
                noc_async_write(stage, get_noc_addr(px, py, o_in + head * Vt * kRowTile), Vt * kRowTile);
            }
            noc_async_write_barrier();
            for (uint32_t s = 0; s < num_streamers; s++) {
                const uint32_t px = get_arg_val<uint32_t>(peers_base + 2 * s);
                const uint32_t py = get_arg_val<uint32_t>(peers_base + 2 * s + 1);
                noc_semaphore_inc(get_noc_addr(px, py, heads_sem_addr), 1);
            }
        }
        cb_wait_front(cb_s_new_out, st);
        if (ts_addr && u == u0) {
            ts[l * 4 + 2] = *clk;
        }
        const uint32_t first = ((set * num_heads + head) * batch + (verify ? cur ^ u : u)) * st;
        for (uint32_t t = 0; t < st; t++) {
            noc_async_write_page(first + t, state_dram, get_read_ptr(cb_s_new_out) + t * kF32Tile);
        }
        noc_async_write_barrier();
        cb_pop_front(cb_s_new_out, st);
        *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_state)) = i + 1;

        if (ts_addr && u + 1 == u0 + nu) {
            ts[l * 4 + 3] = *clk;
        }
    }
    noc_async_atomic_barrier();
}
