// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Attention leader reader: reads every user's rope tables from the token state; per attention layer a,
// reads the layer's q norm weight (copy a % copies) and waits for the streamers' projection rows
// [q heads | gates | k | v] (batch x 32 tiles); per user u places head h of u's q and gate into row h of
// cb_qraw / cb_gate, then releases the group heads' partial sums to compute once all of them have
// written theirs. User u + 1's rows are placed before user u's partials are awaited.
//
// Runtime args: 0 row buffer address, 1 children (group heads of the reduction tree), 2 optional
// timeline buffer (0 = off; per layer 8 wall-clock words: [1] rows arrived, [2] rows placed,
// [3] partials arrived), 3 token state address, 4 rope offset in it, 5 q norm weight address (DRAM,
// [copies][Dt] bf16 tiles).
// Rows mode (active_rows.hpp): only the active users run; the active mask goes to compute in cb_lead_active.

#include "api/dataflow/dataflow_api.h"
#include "attn_common.hpp"
#include "active_rows.hpp"
#include "attn_rows.hpp"

using namespace resident_attn;

namespace {

// copy user u's row of a batch x 32 tile into row `row` of a 32x32 bf16 tile (two 16-element face rows)
FORCE_INLINE void place_row(uint32_t src, uint32_t u, uint32_t tile, uint32_t row) {
    const uint32_t off = row_offset(row);
    src += user_offset(u);
    noc_async_read(get_noc_addr(src), tile + off, kFaceRow);
    noc_async_read(get_noc_addr(src + batch * kFaceRow), tile + kFace + off, kFaceRow);
}

// (kGatePass) rows s0 .. s0 + n - 1 of the 32x32 bf16 tile at src -> rows d0 .. d0 + n - 1 of the tile at dst,
// in runs that stay within a face (both face halves of each run); no barrier
FORCE_INLINE void copy_tile_rows(uint32_t src, uint32_t s0, uint32_t dst, uint32_t d0, uint32_t n) {
    for (uint32_t i = 0; i < n;) {
        const uint32_t s = s0 + i, d = d0 + i;
        uint32_t m = n - i;
        if (s < 16 && s + m > 16) {
            m = 16 - s;
        }
        if (d < 16 && d + m > 16) {
            m = 16 - d;
        }
        noc_async_read(get_noc_addr(src + row_offset(s)), dst + row_offset(d), m * kFaceRow);
        noc_async_read(get_noc_addr(src + kFace + row_offset(s)), dst + kFace + row_offset(d), m * kFaceRow);
        i += m;
    }
}

FORCE_INLINE void zero_tiles(uint32_t cb, uint32_t n) {
    auto& iface = get_local_cb_interface(cb);
    volatile tt_l1_ptr uint32_t* p = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(iface.fifo_limit - iface.fifo_size);
    for (uint32_t i = 0; i < n * kTile / 4; i++) {
        p[i] = 0;
    }
}

FORCE_INLINE void push_resident(uint32_t cb, uint32_t n) {
    cb_reserve_back(cb, n);
    cb_push_back(cb, n);
}

}  // namespace

void kernel_main() {
    // the row buffer (block g: row group g; row groups: this core's CB kCbRows, active_rows.hpp)
    const uint32_t rows_base =
        groups > 1 ? get_local_cb_interface(kCbRows).fifo_limit - get_local_cb_interface(kCbRows).fifo_size
                   : get_arg_val<uint32_t>(0);
    const uint32_t children = get_arg_val<uint32_t>(1);
    const uint32_t ts_addr = get_arg_val<uint32_t>(2);
    const uint32_t tok_addr = get_arg_val<uint32_t>(3);
    const uint32_t rope_off = get_arg_val<uint32_t>(4);
    const InterleavedAddrGenFast<true> qw_dram{
        .bank_base_address = get_arg_val<uint32_t>(5), .page_size = kTile, .data_format = DataFormat::Float16_b};
    volatile tt_l1_ptr uint32_t* rows_sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_rows));
    volatile tt_l1_ptr uint32_t* part_sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_part));
    volatile tt_l1_ptr uint32_t* ts = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ts_addr);
    volatile uint32_t* clk = reinterpret_cast<volatile uint32_t*>(RISCV_DEBUG_REG_WALL_CLOCK_L);
    auto mark = [&](uint32_t l, uint32_t i) {
        if (ts_addr) {
            ts[l * 8 + i] = *clk;
        }
    };

    zero_tiles(cb_qraw, 2 * Dt);
    zero_tiles(cb_gate, 2 * Dt);
    push_resident(cb_mean, 1);
    {
        const InterleavedAddrGen<true> tok{
            .bank_base_address = tok_addr, .page_size = rope_off + 3 * batch * groups * kTile};
        cb_reserve_back(cb_rope, kRopeLead);
        noc_async_read(tok.get_noc_addr(0, rope_off), get_write_ptr(cb_rope), 3 * batch * groups * kTile);
        noc_async_read_barrier();
        if constexpr (packed) {
            build_packed_rope(get_write_ptr(cb_rope));
        }
    }
    uint32_t mask = ~0u;
    if constexpr (rows_mode) {
        constexpr uint32_t cb_lead_active = 7;
        cb_reserve_back(cb_lead_active, 1);
        noc_async_read(get_noc_addr_from_bank_id<true>(0, tok_addr), get_write_ptr(cb_lead_active), kTokHeadBytes);
        noc_async_read_barrier();
        mask = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_lead_active))[kTokActive];
        cb_push_back(cb_lead_active, 1);
    }
    const ActiveRows<kUsers> act(pass_mask(mask));  // (verify rows: a bit per pass)
    const uint32_t n_act = act.n;
    if constexpr (kLeadBatched) {
        // the passes' per-row rope tiles: row j * heads + h of pass p from active user p * kQPerPass + j's page
        const uint32_t base = get_write_ptr(cb_rope);
        volatile tt_l1_ptr uint32_t* z = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base + kRopeQPass * kTile);
        for (uint32_t w = 0; w < 3 * kQPasses * kTile / 4; w++) {
            z[w] = 0;
        }
        for (uint32_t a = 0; a < n_act; a++) {
            const uint32_t p = a / kQPerPass, j = a % kQPerPass;
            for (uint32_t kk = 0; kk < 3; kk++) {
                const uint32_t from = base + (3 * act.row[a] + kk) * kTile;
                for (uint32_t h = 0; h < heads; h++) {
                    const uint32_t to = base + (kRopeQPass + 3 * p + kk) * kTile + row_offset(j * heads + h);
                    noc_async_read(get_noc_addr(from), to, kFaceRow);
                    noc_async_read(get_noc_addr(from + kFace), to + kFace, kFaceRow);
                }
            }
        }
        noc_async_read_barrier();
    }
    cb_push_back(cb_rope, kRopeLead);

    constexpr uint32_t gate0 = heads * Dt;
    // pass u: row (r, h) = r * heads + h of q / gate from row u * group_rows + r (one row per pass: row h)
    // (row groups: user u is global row u, its rows in block u / batch of the row buffer, row u % batch)
    auto place = [&](uint32_t ug) {
        cb_reserve_back(cb_qraw, Dt);
        cb_reserve_back(cb_gate, Dt);
        const uint32_t q = get_write_ptr(cb_qraw), g = get_write_ptr(cb_gate);
        const uint32_t u = ug % users, rows = rows_base + (ug / users) * row_tiles * kRowTile;
        for (uint32_t r = 0; r < group_rows; r++) {
            for (uint32_t h = 0; h < heads; h++) {
                for (uint32_t d = 0; d < Dt; d++) {
                    place_row(rows + (h * Dt + d) * kRowTile, u * group_rows + r, q + d * kTile, r * heads + h);
                }
            }
        }
        noc_async_read_barrier();
        cb_push_back(cb_qraw, Dt);
        for (uint32_t r = 0; r < group_rows; r++) {
            for (uint32_t h = 0; h < heads; h++) {
                for (uint32_t d = 0; d < Dt; d++) {
                    place_row(rows + (gate0 + h * Dt + d) * kRowTile, u * group_rows + r, g + d * kTile, r * heads + h);
                }
            }
        }
        noc_async_read_barrier();
        cb_push_back(cb_gate, Dt);
    };
    // kLeadBatched: the q rows of the active users a0 .. a0 + kQPerPass - 1 into rows j * heads + h of one tile set
    auto place_pass = [&](uint32_t a0) {
        cb_reserve_back(cb_qraw, Dt);
        const uint32_t q = get_write_ptr(cb_qraw);
        for (uint32_t j = 0; j < kQPerPass && a0 + j < n_act; j++) {
            const uint32_t ug = act.row[a0 + j];
            const uint32_t rows = rows_base + (ug / users) * row_tiles * kRowTile;
            for (uint32_t h = 0; h < heads; h++) {
                for (uint32_t d = 0; d < Dt; d++) {
                    place_row(rows + (h * Dt + d) * kRowTile, ug % users, q + d * kTile, j * heads + h);
                }
            }
        }
        noc_async_read_barrier();
        cb_push_back(cb_qraw, Dt);
    };
    auto place_gate = [&](uint32_t ug) {
        cb_reserve_back(cb_gate, Dt);
        const uint32_t g = get_write_ptr(cb_gate);
        const uint32_t rows = rows_base + (ug / users) * row_tiles * kRowTile;
        for (uint32_t h = 0; h < heads; h++) {
            for (uint32_t d = 0; d < Dt; d++) {
                place_row(rows + (gate0 + h * Dt + d) * kRowTile, ug % users, g + d * kTile, h);
            }
        }
        noc_async_read_barrier();
        cb_push_back(cb_gate, Dt);
    };
    // kGatePass: the gate rows of the pass's users into rows j * heads + h of one tile set (as place_pass)
    auto place_gate_pass = [&](uint32_t a0) {
        cb_reserve_back(cb_gate, Dt);
        const uint32_t g = get_write_ptr(cb_gate);
        for (uint32_t j = 0; j < kQPerPass && a0 + j < n_act; j++) {
            const uint32_t ug = act.row[a0 + j];
            const uint32_t rows = rows_base + (ug / users) * row_tiles * kRowTile;
            for (uint32_t h = 0; h < heads; h++) {
                for (uint32_t d = 0; d < Dt; d++) {
                    place_row(rows + (gate0 + h * Dt + d) * kRowTile, ug % users, g + d * kTile, j * heads + h);
                }
            }
        }
        noc_async_read_barrier();
        cb_push_back(cb_gate, Dt);
    };
    // kGatePass: active user ka's sigmoid(gate) rows (rows j * heads .. of its pass) into rows 0 .. heads - 1 of cb_gs
    auto extract_gs = [&](uint32_t ka) {
        const uint32_t j = ka % kQPerPass;
        if (j == 0) {
            cb_wait_front(cb_gs_pass, Dt);
        }
        cb_reserve_back(cb_gs, Dt);
        const uint32_t src = get_read_ptr(cb_gs_pass), dst = get_write_ptr(cb_gs);
        for (uint32_t d = 0; d < Dt; d++) {
            copy_tile_rows(src + d * kTile, j * heads, dst + d * kTile, 0, heads);
        }
        noc_async_read_barrier();
        cb_push_back(cb_gs, Dt);
        if (j + 1 == kQPerPass || ka + 1 == n_act) {
            cb_pop_front(cb_gs_pass, Dt);
        }
    };
    auto place_user = [&](uint32_t ka) {
        if constexpr (kLeadBatched) {
            if (ka % kQPerPass == 0) {
                place_pass(ka);
                if constexpr (kGatePass) {
                    place_gate_pass(ka);
                }
            }
            if constexpr (!kGatePass) {
                place_gate(act.row[ka]);
            }
        } else {
            place(act.row[ka]);
        }
    };
    for (uint32_t i = 0; i < layers * n_act; i++) {
        const uint32_t l = i / n_act, k = i % n_act;
        if (k == 0) {
            cb_reserve_back(cb_qw, Dt);
            for (uint32_t d = 0; d < Dt; d++) {
                noc_async_read_page((l % copies) * Dt + d, qw_dram, get_write_ptr(cb_qw) + d * kTile);
            }
            noc_async_read_barrier();
            cb_push_back(cb_qw, Dt);
            noc_semaphore_wait_min(rows_sem, num_streamers * (l + 1));
            mark(l, 1);
            place_user(0);
            mark(l, 2);
        }
        // the next user's rows go in before this user's partials arrive, so compute prepares its q meanwhile
        if (k + 1 < n_act) {
            place_user(k + 1);
        }
        if constexpr (kGatePass) {
            extract_gs(k);
        }

        noc_semaphore_wait_min(part_sem, children * (i + 1));
        if (k + 1 == n_act) {
            mark(l, 3);
        }
        push_resident(cb_parts, fanin * kPart);
    }
}
