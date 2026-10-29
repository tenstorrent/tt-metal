// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Resident GDN head core writer (BRISC) of the chunked verify (gdn_chunk_compute.cpp), one value head h, the R rows
// of one user. Per GDN layer: stores the committed state as the new base (slot 0) as soon as compute has it, packs
// rows 0..R-1 of the gated output tiles into rows 0..R-1 of batch x 32 tiles (the stage) and writes them into every
// streamer's mixer-output buffer at this head's offset (bumping the streamers' head semaphore), then stores the
// pending block [K^T, Delta, G, Gamma] (slot 1) for the next step's commit.
//
// Compile-time args: 0 Kt, 1 Vt, 2 GDN layers, 3 copies, 4 num_streamers, 5 sem_heads, 6 heads (per chip), 7 unused,
//   8 batch (R), then unused
// Runtime args: as gdn_head_writer.cpp (0 state address, 1 o_in_addr (streamers' buffer), 2 head index, then
//   num_streamers x (x, y), then unused)

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

constexpr uint32_t Kt = get_compile_time_arg_val(0);
constexpr uint32_t Vt = get_compile_time_arg_val(1);
constexpr uint32_t layers = get_compile_time_arg_val(2);
constexpr uint32_t copies = get_compile_time_arg_val(3);
constexpr uint32_t num_streamers = get_compile_time_arg_val(4);
constexpr uint32_t sem_heads = get_compile_time_arg_val(5);
constexpr uint32_t num_heads = get_compile_time_arg_val(6);
constexpr uint32_t batch = get_compile_time_arg_val(8);
constexpr uint32_t st = Kt * Vt;
constexpr uint32_t cb_stage = 9, cb_s_out = 18, cb_out = 30, cb_pend = 31;
constexpr uint32_t kPend = 2 * Kt + 2;
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
    auto& stage_iface = get_local_cb_interface(cb_stage);
    const uint32_t stage = stage_iface.fifo_limit - stage_iface.fifo_size;

    for (uint32_t l = 0; l < layers; l++) {
        const uint32_t base = (((l % copies) * num_heads + head) * batch) * st;
        // the committed state, the next step's base
        cb_wait_front(cb_s_out, st);
        for (uint32_t t = 0; t < st; t++) {
            noc_async_write_page(base + t, state_dram, get_read_ptr(cb_s_out) + t * kF32Tile);
        }
        // the output rows
        cb_wait_front(cb_out, Vt);
        const uint32_t out = get_read_ptr(cb_out);
        for (uint32_t t = 0; t < Vt; t++) {
            noc_async_read(get_noc_addr(out + t * kBf16Tile), stage + t * kRowTile, batch * kFaceRow);
            noc_async_read(
                get_noc_addr(out + t * kBf16Tile + kFaceBytes),
                stage + t * kRowTile + batch * kFaceRow,
                batch * kFaceRow);
        }
        noc_async_read_barrier();
        cb_pop_front(cb_out, Vt);
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
        cb_pop_front(cb_s_out, st);
        // the pending block
        cb_wait_front(cb_pend, kPend);
        for (uint32_t t = 0; t < kPend; t++) {
            noc_async_write_page(base + st + t, state_dram, get_read_ptr(cb_pend) + t * kF32Tile);
        }
        noc_async_write_barrier();
        cb_pop_front(cb_pend, kPend);
    }
    noc_async_atomic_barrier();
}
