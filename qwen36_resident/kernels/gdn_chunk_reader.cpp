// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Resident GDN head core reader (NCRISC) of the chunked verify (gdn_chunk_compute.cpp): one value head h (key head
// kh = h / group), the R = batch rows of one user at p .. p + R - 1 in one pass. Per GDN layer (copy = layer):
//   - the committed state's inputs, ahead of the rows: the base state S (slot 0, 16 fp32 tiles) and the previous
//     step's pending block (slot 1: K^T 4 tiles, delta 4 tiles, the decay-ratio matrix G, the cumulative decays
//     Gamma), from which it builds, for the previous step's accepted count a (token state word kTokPrev), the
//     fill tile Gamma_a and the column tile w (column j = G[a][j] = Gamma_a / Gamma_j for j <= a, else 0); with no
//     previous step (kTokPrev = kNone) Gamma_a = 1 and w = 0, so the committed state is S itself;
//   - once every streamer has written its qkvzab columns: rows 0..R-1 of q_kh, k_kh, v_h, z_h into rows 0..R-1 of
//     32x32 bf16 tiles (rows >= R stay zero), a_h as a row tile (row r = a_r) and a column tile (column r = a_r),
//     b_h as a row tile, the copy's dt_bias_h / neg_exp_A_h as fill tiles, the norm weight.
// Constants once: [ones, Tri (Tri[r][i] = 1 for i <= r, i < R), Tri^T, identity, strict mask (j < r)] bf16 and
// the causal additive mask (0 for j <= r, -10000 after) fp32.
//
// Compile-time args: 0 Kt, 1 Vt, 2 GDN layers, 3 copies, 4 num_streamers, 5 sem_rows, 6 heads (per chip), 7 unused,
//   8 batch (R), 9 verify (1), 10 unused
// Runtime args: as gdn_head_reader.cpp (0 rows_addr, 1 state address, 2 norm weight address, 3 q_tile, 4 k_tile,
//   5 v_tile, 6 z_tile, 7 a_tile, 8 a_elem, 9 b_tile, 10 b_elem, 11 head index, then copies x (dt_bias_h bits,
//   neg_exp_A_h bits), then timeline (unused), u0, nu, the row writes per layer, the token state address)

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

namespace {

constexpr uint32_t Kt = get_compile_time_arg_val(0);
constexpr uint32_t Vt = get_compile_time_arg_val(1);
constexpr uint32_t layers = get_compile_time_arg_val(2);
constexpr uint32_t copies = get_compile_time_arg_val(3);
constexpr uint32_t num_streamers = get_compile_time_arg_val(4);
constexpr uint32_t sem_rows = get_compile_time_arg_val(5);
constexpr uint32_t num_heads = get_compile_time_arg_val(6);
constexpr uint32_t batch = get_compile_time_arg_val(8);
constexpr uint32_t st = Kt * Vt;
constexpr uint32_t kTokPrev = 14;  // token state word: the previous verify step's accepted drafts (kNone: none)
constexpr uint32_t kNone = 0xFFFFFFFFu;
static_assert(batch > 2 && batch <= 16, "a chunk of R rows of one user");

// CBs (see gdn_chunk_compute.cpp)
constexpr uint32_t cb_q_in = 0, cb_k_in = 1, cb_v_in = 2, cb_z_in = 3, cb_norm_w = 4, cb_s_base = 5;
constexpr uint32_t cb_gates = 6;  // [a rows, a columns, b rows, dt_bias fill, neg_exp_A fill], fp32
constexpr uint32_t cb_gamma_a = 11, cb_w = 12, cb_kt_prev = 13, cb_d_prev = 14, cb_consts = 15, cb_maskinc = 16;
constexpr uint32_t cb_meta = 22;
constexpr uint32_t kBf16Tile = 2048, kF32Tile = 4096, kFaceBytes = 512, kFaceRow = 32;
constexpr uint32_t kRowTile = 64 * batch;  // a batch x 32 bf16 tile of the row buffer
constexpr uint32_t kOneF = 0x3f800000u, kOneB = 0x3f80;

// word index of element (r, c) in a 32x32 fp32 tile; halfword index in a bf16 tile
FORCE_INLINE uint32_t at(uint32_t r, uint32_t c) { return ((r / 16) * 2 + c / 16) * 256 + (r % 16) * 16 + c % 16; }

void zero_bytes(uint32_t l1_addr, uint32_t bytes) {
    volatile tt_l1_ptr uint32_t* p = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1_addr);
    for (uint32_t i = 0; i < bytes / 4; i++) {
        p[i] = 0;
    }
}

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

// rows 0..R-1 of n row tiles starting at row tile `first` -> rows 0..R-1 of n 32x32 bf16 tiles of `cb`
void place_rows(uint32_t cb, uint32_t rows, uint32_t first, uint32_t n) {
    cb_reserve_back(cb, n);
    const uint32_t dst = get_write_ptr(cb);
    for (uint32_t t = 0; t < n; t++) {
        const uint32_t src = rows + (first + t) * kRowTile;
        noc_async_read(get_noc_addr(src), dst + t * kBf16Tile, batch * kFaceRow);
        noc_async_read(get_noc_addr(src + batch * kFaceRow), dst + t * kBf16Tile + kFaceBytes, batch * kFaceRow);
    }
    noc_async_read_barrier();
    cb_push_back(cb, n);
}

template <typename AddrGen>
void read_pages(uint32_t l1, const AddrGen& src, uint32_t first, uint32_t n) {
    for (uint32_t t = 0; t < n; t++) {
        noc_async_read_page(first + t, src, l1 + t * kF32Tile);
    }
}

}  // namespace

void kernel_main() {
    const uint32_t rows = get_arg_val<uint32_t>(0);
    const uint32_t state = get_arg_val<uint32_t>(1);
    const uint32_t norm_w = get_arg_val<uint32_t>(2);
    const uint32_t q_tile = get_arg_val<uint32_t>(3);
    const uint32_t k_tile = get_arg_val<uint32_t>(4);
    const uint32_t v_tile = get_arg_val<uint32_t>(5);
    const uint32_t z_tile = get_arg_val<uint32_t>(6);
    const uint32_t a_tile = get_arg_val<uint32_t>(7);
    const uint32_t a_elem = get_arg_val<uint32_t>(8);
    const uint32_t b_tile = get_arg_val<uint32_t>(9);
    const uint32_t b_elem = get_arg_val<uint32_t>(10);
    const uint32_t head = get_arg_val<uint32_t>(11);
    constexpr uint32_t gates_base = 12;
    const uint32_t row_writes = get_arg_val<uint32_t>(gates_base + 2 * copies + 3);
    const uint32_t tok_addr = get_arg_val<uint32_t>(gates_base + 2 * copies + 4);
    const InterleavedAddrGenFast<true> state_dram{
        .bank_base_address = state, .page_size = kF32Tile, .data_format = DataFormat::Float32};
    const InterleavedAddrGenFast<true> norm_dram{
        .bank_base_address = norm_w, .page_size = kBf16Tile, .data_format = DataFormat::Float16_b};

    // the input tiles' rows >= R are never written: zero them once (they must be finite, and zero keeps them out of
    // every product)
    for (uint32_t cb : {cb_q_in, cb_k_in, cb_v_in, cb_z_in, cb_gates}) {
        auto& iface = get_local_cb_interface(cb);
        zero_bytes(iface.fifo_limit - iface.fifo_size, iface.fifo_size);
    }
    // constants
    cb_reserve_back(cb_consts, 5);
    {
        volatile tt_l1_ptr uint16_t* c = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(get_write_ptr(cb_consts));
        for (uint32_t r = 0; r < 32; r++) {
            for (uint32_t j = 0; j < 32; j++) {
                const uint32_t i = at(r, j);
                c[i] = kOneB;
                c[1024 + i] = j <= r && j < batch ? kOneB : 0;
                c[2048 + i] = r <= j && r < batch ? kOneB : 0;
                c[3072 + i] = r == j ? kOneB : 0;
                c[4096 + i] = j < r ? kOneB : 0;
            }
        }
    }
    cb_push_back(cb_consts, 5);
    cb_reserve_back(cb_maskinc, 1);
    {
        volatile tt_l1_ptr uint32_t* m = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_maskinc));
        for (uint32_t r = 0; r < 32; r++) {
            for (uint32_t j = 0; j < 32; j++) {
                m[at(r, j)] = j <= r ? 0u : 0xc61c4000u;  // -10000.0f
            }
        }
    }
    cb_push_back(cb_maskinc, 1);

    // the previous step's accepted drafts
    uint32_t prev = kNone;
    {
        cb_reserve_back(cb_meta, 2);
        const uint32_t scratch = get_write_ptr(cb_meta);
        noc_async_read(get_noc_addr_from_bank_id<true>(0, tok_addr), scratch, 64);
        noc_async_read_barrier();
        prev = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch)[kTokPrev];
    }
    const bool committed = prev != kNone && prev < batch;

    volatile tt_l1_ptr uint32_t* rows_sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_rows));
    volatile tt_l1_ptr uint16_t* a_row = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(rows + a_tile * kRowTile);
    volatile tt_l1_ptr uint16_t* b_row = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(rows + b_tile * kRowTile);
    // element c of user u in a row tile
    auto elem = [](uint32_t c, uint32_t u) { return (c / 16) * 16 * batch + u * 16 + c % 16; };
    for (uint32_t l = 0; l < layers; l++) {
        const uint32_t set = l % copies;
        const uint32_t base = ((set * num_heads + head) * batch) * st;  // slot 0: S; slot 1: the pending block
        // ---- the committed state's inputs (independent of the rows)
        cb_reserve_back(cb_s_base, st);
        read_pages(get_write_ptr(cb_s_base), state_dram, base, st);
        cb_reserve_back(cb_kt_prev, Kt);
        cb_reserve_back(cb_d_prev, Vt);
        read_pages(get_write_ptr(cb_kt_prev), state_dram, base + st, Kt);
        read_pages(get_write_ptr(cb_d_prev), state_dram, base + st + Kt, Vt);
        const uint32_t meta = get_write_ptr(cb_meta);  // [G, Gamma] of the previous step (L1 scratch)
        read_pages(meta, state_dram, base + st + Kt + Vt, 2);
        noc_async_read_barrier();
        cb_push_back(cb_s_base, st);
        cb_push_back(cb_kt_prev, Kt);
        cb_push_back(cb_d_prev, Vt);
        {
            volatile tt_l1_ptr uint32_t* g = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(meta);
            volatile tt_l1_ptr uint32_t* gam = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(meta + kF32Tile);
            cb_reserve_back(cb_gamma_a, 1);
            fill_f32_tile(get_write_ptr(cb_gamma_a), committed ? gam[at(prev, 0)] : kOneF);
            cb_push_back(cb_gamma_a, 1);
            cb_reserve_back(cb_w, 1);
            volatile tt_l1_ptr uint32_t* w = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_w));
            for (uint32_t j = 0; j < 32; j++) {
                const uint32_t v = committed ? g[at(prev, j)] : 0u;
                for (uint32_t r = 0; r < 32; r++) {
                    w[at(r, j)] = v;
                }
            }
            cb_push_back(cb_w, 1);
        }
        cb_reserve_back(cb_norm_w, Vt);
        for (uint32_t t = 0; t < Vt; t++) {
            noc_async_read_page(set * Vt + t, norm_dram, get_write_ptr(cb_norm_w) + t * kBf16Tile);
        }
        noc_async_read_barrier();
        cb_push_back(cb_norm_w, Vt);

        // ---- the rows
        noc_semaphore_wait_min(rows_sem, row_writes * (l + 1));
        place_rows(cb_q_in, rows, q_tile, Kt);
        place_rows(cb_k_in, rows, k_tile, Kt);
        // a/b are the only scalars read by the RISC: the row buffer is written by other cores
        invalidate_l1_cache();
        cb_reserve_back(cb_gates, 5);
        {
            const uint32_t g0 = get_write_ptr(cb_gates);
            volatile tt_l1_ptr uint32_t* ar = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(g0);
            volatile tt_l1_ptr uint32_t* ac = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(g0 + kF32Tile);
            volatile tt_l1_ptr uint32_t* br = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(g0 + 2 * kF32Tile);
            for (uint32_t u = 0; u < batch; u++) {
                const uint32_t av = static_cast<uint32_t>(a_row[elem(a_elem, u)]) << 16;
                const uint32_t bv = static_cast<uint32_t>(b_row[elem(b_elem, u)]) << 16;
                for (uint32_t c = 0; c < 32; c++) {
                    ar[at(u, c)] = av;
                    ac[at(c, u)] = av;
                    br[at(u, c)] = bv;
                }
            }
            fill_f32_tile(g0 + 3 * kF32Tile, get_arg_val<uint32_t>(gates_base + 2 * set));
            fill_f32_tile(g0 + 4 * kF32Tile, get_arg_val<uint32_t>(gates_base + 2 * set + 1));
        }
        cb_push_back(cb_gates, 5);
        place_rows(cb_v_in, rows, v_tile, Vt);
        place_rows(cb_z_in, rows, z_tile, Vt);
    }
}
