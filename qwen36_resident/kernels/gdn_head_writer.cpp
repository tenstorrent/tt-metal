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
//   10 verify (0 / 1: row u writes state slot (cur + u) % batch, see the reader), 11 sem_slot (local: cur + 1)
// Runtime args: 0 state address (DRAM, see the reader), 1 o_in_addr (streamers' buffer), 2 head index,
// then num_streamers x (x, y), then an optional timeline buffer (0 = off; per layer [2] state out,
// [3] output sent), then u0, nu, the head's cores (1: this core is the collector), the collector's x, y, then
// the token state address
// Rows mode (active_rows.hpp): only this core's active users run; its stage rows start zeroed, so inactive users'
// output rows are 0. A non-collector core with no active user sends nothing; the collector waits each layer for the
// cores with one only (counting a core with none would let its count stand in for a busy core's rows) and zeroes the
// idle cores' rows itself. The collector sends the head's output every layer, also when it has no active user.
// Row groups (kBatched, gdn_head_compute.cpp): the writer extracts user k's rows of compute's batched results (q / k
// norms, decay * k, beta) into row 0 of the per-user tiles and spreads its decay over a full fp32 tile (one user
// ahead of compute), gathers each user's o (row 0 of cb_o) into row k of cb_o_b and writes its state back; compute's
// output then holds the n users in rows 0 .. n - 1 of cb_out; the layer's last state write-back follows the output.
// Verify rows (kVRows, see the reader): this core runs group u0's rows (one user; core g of the head = group g, the
// collector group 0's); row u writes slot g * batch + (cur + u) % batch; write-backs are acked eagerly (the reader
// chains row u + 1 from the state once it is stored).

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "active_rows.hpp"

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
// row groups (gdn_head_compute.cpp kBatched)
constexpr bool kBatched = groups > 1;
constexpr bool kVRows = verify && groups > 1;
constexpr uint32_t cb_o = 25, cb_o_b = 39, cb_qn = 13, cb_kn = 14, cb_decay = 15, cb_beta_row0 = 19;
constexpr uint32_t cb_decay_b = 32, cb_betam_b = 33, cb_kdec_b = 35, cb_qn_u = 36, cb_kn_u = 37, cb_kdec_u = 38;
constexpr uint32_t kF32Row = 64, kF32Face = 1024;
// byte offset of row r's first half in a 32 x 32 tile (its second half is one face further)
inline uint32_t row_at(uint32_t r, uint32_t face_bytes, uint32_t row_bytes) {
    return (r / 16) * 2 * face_bytes + (r % 16) * row_bytes;
}

// (kBatched) row k of the n fp32 tiles at cb_src's front (waited for by the caller) -> row 0 of n tiles of cb_dst
void extract_f32(uint32_t cb_src, uint32_t cb_dst, uint32_t n, uint32_t k) {
    cb_reserve_back(cb_dst, n);
    const uint32_t src = get_read_ptr(cb_src) + row_at(k, kF32Face, kF32Row), dst = get_write_ptr(cb_dst);
    for (uint32_t t = 0; t < n; t++) {
        noc_async_read(get_noc_addr(src + t * kF32Tile), dst + t * kF32Tile, kF32Row);
        noc_async_read(get_noc_addr(src + t * kF32Tile + kF32Face), dst + t * kF32Tile + kF32Face, kF32Row);
    }
    noc_async_read_barrier();
    cb_push_back(cb_dst, n);
}

// (kBatched) bits over every element of the fp32 tile at l1_addr
void fill_f32_tile(uint32_t l1_addr, uint32_t bits) {
    volatile tt_l1_ptr uint32_t* p = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1_addr);
    for (uint32_t i = 0; i < 16; i++) {
        p[i] = bits;
    }
    for (uint32_t bytes = 64; bytes < kF32Tile; bytes *= 2) {
        noc_async_read(get_noc_addr(l1_addr), l1_addr + bytes, bytes);
        noc_async_read_barrier();
    }
}
// row groups (active_rows.hpp): the stage and the streamers' o_in hold one block per group (stage block Vt tiles,
// o_in block o_tiles tiles); users are global rows g * batch + u; the state tensor holds groups * batch users
constexpr uint32_t kUsers = batch * groups;

void kernel_main() {
    const uint32_t state = get_arg_val<uint32_t>(0);
    uint32_t o_in = get_arg_val<uint32_t>(1);
    if constexpr (groups > 1) {
        // row groups: the streamers' o memory (it is their activation CB), from streamer 0 at launch
        volatile tt_l1_ptr uint32_t* oaddr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_raddr));
        while (*oaddr == 0) {
            invalidate_l1_cache();
        }
        o_in = *oaddr;
    }
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

    // the stage rows of users r0 .. r0 + nu of every group, zeroed (verify rows: every row of group r0)
    auto zero_rows = [&](uint32_t r0) {
        if constexpr (kVRows) {
            volatile tt_l1_ptr uint32_t* p = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(stage + r0 * Vt * kRowTile);
            for (uint32_t w = 0; w < Vt * kRowTile / 4; w++) {
                p[w] = 0;
            }
            return;
        }
        for (uint32_t t = 0; t < groups * Vt; t++) {
            for (uint32_t f = 0; f < 2; f++) {
                volatile tt_l1_ptr uint32_t* p =
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(stage + t * kRowTile + (f * batch + r0) * kFaceRow);
                for (uint32_t w = 0; w < nu * kFaceRow / 4; w++) {
                    p[w] = 0;
                }
            }
        }
    };
    // collector: the head's other cores it waits for every layer. Rows mode: only those with an active user; a core
    // with none sends nothing (its count would run ahead of the busy cores' rows), its rows are zeroed here instead
    uint32_t busy = cores - 1;
    // this core's users that run this step (see the reader)
    uint32_t act[kUsers], n = 0;
    uint32_t step_mask = 0;
    {
        uint32_t mask = ~0u;
        if constexpr (rows_mode) {
            const uint32_t tok_addr = get_arg_val<uint32_t>(peers_base + 2 * num_streamers + 6);
            noc_async_read(get_noc_addr_from_bank_id<true>(0, tok_addr), stage, kTokHeadBytes);
            noc_async_read_barrier();
            mask = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(stage)[kTokActive];
            step_mask = mask;
            // this core's rows only: another core of the head may already be writing its rows into ours
            zero_rows(u0);
            if (collector) {
                busy = 0;
                for (uint32_t g = 1; g < cores; g++) {  // core g of the head takes users g nu .. (g + 1) nu
                    bool any = false;                   // (in any row group; verify rows: group g's rows)
                    for (uint32_t gr = 0; gr < groups; gr++) {
                        if (kVRows) {
                            any |= gr == g && ((mask >> (gr * batch)) & ((1u << batch) - 1)) != 0;
                        } else {
                            any |= ((mask >> (gr * batch + g * nu)) & ((1u << nu) - 1)) != 0;
                        }
                    }
                    if (any) {
                        busy++;
                    } else {
                        zero_rows(kVRows ? g : g * nu);
                    }
                }
            }
        }
        for (uint32_t gr = kVRows ? u0 : 0; gr < (kVRows ? u0 + 1 : groups); gr++) {
            for (uint32_t u = kVRows ? 0 : u0; u < (kVRows ? nu : u0 + nu); u++) {
                if ((mask >> (gr * batch + u)) & 1) {
                    act[n++] = gr * batch + u;
                }
            }
        }
    }
    // row groups: only the groups with an active row go out (the streamers skip the others' out-projection)
    const ActiveGroups ag(step_mask, batch);
    // after the core's last user of layer l: its rows to the head's collector, or (collector) the head's output
    // into every streamer
    auto layer_out = [&](uint32_t l) {
        if (!collector && kVRows) {
            // group u0's block, every row
            if (ag.has(u0)) {
                const uint32_t off = u0 * Vt * kRowTile;
                noc_async_write(stage + off, get_noc_addr(cx, cy, stage + off), Vt * kRowTile);
            }
            noc_async_write_barrier();
            noc_semaphore_inc(get_noc_addr(cx, cy, get_semaphore(sem_group)), 1);
        } else if (!collector) {
            // users u0 .. u0 + nu are contiguous face rows of every tile (of every group's block)
            for (uint32_t t = 0; t < groups * Vt; t++) {
                for (uint32_t f = 0; ag.has(t / Vt) && f < 2; f++) {
                    const uint32_t off = t * kRowTile + (f * batch + u0) * kFaceRow;
                    noc_async_write(stage + off, get_noc_addr(cx, cy, stage + off), nu * kFaceRow);
                }
            }
            noc_async_write_barrier();
            noc_semaphore_inc(get_noc_addr(cx, cy, get_semaphore(sem_group)), 1);
        } else {
            noc_semaphore_wait_min(group_sem, busy * (l + 1));
            for (uint32_t s = 0; s < num_streamers; s++) {
                const uint32_t px = get_arg_val<uint32_t>(peers_base + 2 * s);
                const uint32_t py = get_arg_val<uint32_t>(peers_base + 2 * s + 1);
                for (uint32_t gr = 0; gr < groups; gr++) {
                    if (!ag.has(gr)) {
                        continue;
                    }
                    noc_async_write(
                        stage + gr * Vt * kRowTile,
                        get_noc_addr(px, py, o_in + (gr * o_tiles + head * Vt) * kRowTile),
                        Vt * kRowTile);
                }
            }
            noc_async_write_barrier();
            for (uint32_t s = 0; s < num_streamers; s++) {
                const uint32_t px = get_arg_val<uint32_t>(peers_base + 2 * s);
                const uint32_t py = get_arg_val<uint32_t>(peers_base + 2 * s + 1);
                noc_semaphore_inc(get_noc_addr(px, py, heads_sem_addr), 1);
            }
        }
    };

    // (kBatched) state write-back i (user r) to DRAM. A write-back is acked lazily (before the next one is issued, so
    // its latency stays off compute's path): sem_state counts the landed ones for the reader, which re-reads a copy's
    // state only after its write-back; eager when the reader's next state may be the one just written (copies x n = 1).
    // (verify rows: sem_state is only the chain's — the next row copies this state from L1 as soon as it is packed,
    // its DRAM write-back (the slot, re-read only by the next step) flushed before the pop and barriered at the end)
    const bool eager = copies * n <= 1 && !kVRows;
    uint32_t landed = 0;
    auto write_state = [&](uint32_t i, uint32_t r) {
        const uint32_t slot = kVRows ? (r / batch) * batch + (cur + r % batch) % batch : r;
        const uint32_t first = ((i / n % copies * num_heads + head) * kUsers + slot) * st;
        cb_wait_front(cb_s_new_out, st);
        if constexpr (kVRows) {
            *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_state)) = i + 1;
        } else if (landed < i) {
            noc_async_write_barrier();
            landed = i;
            *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_state)) = landed;
        }
        for (uint32_t t = 0; t < st; t++) {
            noc_async_write_page(first + t, state_dram, get_read_ptr(cb_s_new_out) + t * kF32Tile);
        }
        noc_async_writes_flushed();
        cb_pop_front(cb_s_new_out, st);
        if (eager) {
            noc_async_write_barrier();
            landed = i + 1;
            *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_state)) = landed;
        }
    };
    // (kBatched) user k's row-0 tiles from compute's batched results, in the order compute frees them
    auto extract = [&](uint32_t k) {
        extract_f32(cb_kdec_b, cb_kdec_u, Kt, k);
        extract_f32(cb_betam_b, cb_beta_row0, 1, k);
        extract_f32(cb_kn, cb_kn_u, Kt, k);
        cb_reserve_back(cb_decay, 1);
        invalidate_l1_cache();  // written by this core's packer
        fill_f32_tile(
            get_write_ptr(cb_decay),
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_read_ptr(cb_decay_b) + row_at(k, kF32Face, kF32Row))[0]);
        cb_push_back(cb_decay, 1);
        extract_f32(cb_qn, cb_qn_u, Kt, k);
    };
    if constexpr (kBatched) {
        // the per-user row-0 tiles: only row 0 is ever written, the other rows stay 0 (they meet zero rows of beta /
        // only row 0 of the results is used, so they must be finite); rows n .. 31 of the gathered o stay finite
        // (compute's row-wise ops read them; nothing reads their results)
        for (uint32_t cb : {cb_qn_u, cb_kn_u, cb_kdec_u, cb_beta_row0, cb_o_b}) {
            auto& f = get_local_cb_interface(cb);
            fill_f32_tile(f.fifo_limit - f.fifo_size, 0u);
            for (uint32_t off = kF32Tile; off < f.fifo_size; off += kF32Tile) {
                noc_async_read(get_noc_addr(f.fifo_limit - f.fifo_size), f.fifo_limit - f.fifo_size + off, kF32Tile);
            }
        }
        noc_async_read_barrier();
    }
    for (uint32_t l = 0; kBatched && l < layers; l++) {
        if (n == 0) {  // rows mode, no active user on this core: the collector still sends the head's output
            if (collector) {
                layer_out(l);
            }
            continue;
        }
        // compute's batched results of the layer (popped after the last user's extraction)
        cb_wait_front(cb_qn, Kt);
        cb_wait_front(cb_kn, Kt);
        cb_wait_front(cb_kdec_b, Kt);
        cb_wait_front(cb_decay_b, 1);
        cb_wait_front(cb_betam_b, 1);
        extract(0);
        cb_reserve_back(cb_o_b, Vt);
        for (uint32_t k = 0; k < n; k++) {
            // the next user's tiles first (compute needs them right after this user's o), then this user's o and state
            if (k + 1 < n) {
                extract(k + 1);
            } else {
                cb_pop_front(cb_qn, Kt);
                cb_pop_front(cb_kn, Kt);
                cb_pop_front(cb_kdec_b, Kt);
                cb_pop_front(cb_decay_b, 1);
                cb_pop_front(cb_betam_b, 1);
            }
            cb_wait_front(cb_o, Vt);
            const uint32_t src = get_read_ptr(cb_o), dst = get_write_ptr(cb_o_b) + row_at(k, kF32Face, kF32Row);
            for (uint32_t t = 0; t < Vt; t++) {
                noc_async_read(get_noc_addr(src + t * kF32Tile), dst + t * kF32Tile, kF32Row);
                noc_async_read(get_noc_addr(src + t * kF32Tile + kF32Face), dst + t * kF32Tile + kF32Face, kF32Row);
            }
            noc_async_read_barrier();
            cb_pop_front(cb_o, Vt);
            if (k + 1 < n) {
                if (ts_addr && k == 0) {
                    ts[l * 4 + 2] = *clk;
                }
                write_state(l * n + k, act[k]);
            }
        }
        cb_push_back(cb_o_b, Vt);
        // the n users' output rows (rows 0 .. n - 1 of cb_out) into their stage rows
        cb_wait_front(cb_out, Vt);
        const uint32_t out = get_read_ptr(cb_out);
        for (uint32_t k = 0; k < n; k++) {
            const uint32_t r = act[k], u = r % batch, gblk = (r / batch) * Vt * kRowTile;
            const uint32_t src = out + row_at(k, kFaceBytes, kFaceRow);
            for (uint32_t t = 0; t < Vt; t++) {
                const uint32_t dst = stage + gblk + t * kRowTile + u * kFaceRow;
                noc_async_read(get_noc_addr(src + t * kBf16Tile), dst, kFaceRow);
                noc_async_read(get_noc_addr(src + t * kBf16Tile + kFaceBytes), dst + batch * kFaceRow, kFaceRow);
            }
        }
        noc_async_read_barrier();
        cb_pop_front(cb_out, Vt);
        layer_out(l);
        if (ts_addr) {
            ts[l * 4 + 3] = *clk;
        }
        write_state(l * n + n - 1, act[n - 1]);  // (the layer's last state after the output)
    }
    if constexpr (kBatched) {
        noc_async_write_barrier();  // the last write-backs have landed before the next launch reads them
    }
    for (uint32_t i = 0; !kBatched && i < layers * (n ? n : 1); i++) {
        const uint32_t l = n ? i / n : i, k = n ? i % n : 0;
        if (n == 0) {  // rows mode, no active user on this core: the collector still sends the head's output
            if (collector) {
                layer_out(l);
            }
            continue;
        }
        const uint32_t r = act[k], u = r % batch, gblk = (r / batch) * Vt * kRowTile;
        const uint32_t set = l % copies;
        cb_wait_front(cb_out, Vt);
        const uint32_t out = get_read_ptr(cb_out);
        for (uint32_t t = 0; t < Vt; t++) {
            const uint32_t dst = stage + gblk + t * kRowTile + u * kFaceRow;
            noc_async_read(get_noc_addr(out + t * kBf16Tile), dst, kFaceRow);
            noc_async_read(get_noc_addr(out + t * kBf16Tile + kFaceBytes), dst + batch * kFaceRow, kFaceRow);
        }
        noc_async_read_barrier();
        cb_pop_front(cb_out, Vt);
        if (k + 1 == n) {
            layer_out(l);
        }
        cb_wait_front(cb_s_new_out, st);
        if (ts_addr && k == 0) {
            ts[l * 4 + 2] = *clk;
        }
        const uint32_t first = ((set * num_heads + head) * kUsers + (verify ? (cur + u) % batch : r)) * st;
        for (uint32_t t = 0; t < st; t++) {
            noc_async_write_page(first + t, state_dram, get_read_ptr(cb_s_new_out) + t * kF32Tile);
        }
        noc_async_write_barrier();
        cb_pop_front(cb_s_new_out, st);
        *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_state)) = i + 1;

        if (ts_addr && k + 1 == n) {
            ts[l * 4 + 3] = *clk;
        }
    }
    noc_async_atomic_barrier();
}
