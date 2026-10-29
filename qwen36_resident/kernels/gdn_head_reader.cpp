// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Resident GDN head core reader (NCRISC), one value head h (key head kh = h / group). Per GDN layer
// g (copy c = g % copies) and user u, once every streamer has written its qkvzab columns into this core's
// row buffer (batch x 32 tiles, row u for user u, conv already applied):
//   - q_kh, k_kh, v_h, z_h are placed in row 0 of 32x32 bf16 tiles (rows 1..31 stay zero),
//   - a_h / b_h and the copy's dt_bias_h / neg_exp_A_h are broadcast into full fp32 tiles,
//   - the copy's norm weight and this head's recurrent state are read from DRAM (ahead of the rows;
//     the writer stores the updated state back, and a state is read again only in the next step).
// The ones and row-0 mask constants are built once.
//
// Compile-time args: 0 Kt, 1 Vt, 2 GDN layers, 3 copies, 4 num_streamers, 5 sem_rows, 6 heads (per chip),
//   7 sem_state (count of state write-backs done by the writer, one per layer and user; a copy's state is
//   re-read within a step only after its previous write-back), 8 batch, 9 verify (0 / 1), 10 sem_slot (local:
//   the state slot + 1, from the reader to the writer)
// Verify (one core per head, two rows of one user at p and p + 1): the state tensor holds two slots per
// (copy, head) instead of users; row 0 reads and updates slot `cur` (token state word kTokSlot), row 1
// starts from row 0's updated state (copied from L1 once the writer has stored it) and writes it to slot
// cur ^ 1, so the next step continues from either row by its slot.
// Runtime args: 0 rows_addr, 1 state address (DRAM, [copies][heads][batch][Kt * Vt] fp32 tiles), 2 norm weight
//   address (DRAM, [copies][Vt] bf16 row-0 tiles), 3 q_tile, 4 k_tile, 5 v_tile, 6 z_tile, 7 a_tile,
//   8 a_elem, 9 b_tile, 10 b_elem, 11 head index, then copies x (dt_bias_h bits, neg_exp_A_h bits), then an
//   optional timeline buffer (0 = off): per layer 4 words, the reader writes [0] rows arrived, [1] inputs
//   pushed; then this core's users u0 .. u0 + nu and the row writes it gets per layer (one per streamer run
//   that covers it), then the token state address

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
constexpr uint32_t sem_state = get_compile_time_arg_val(7);
constexpr uint32_t batch = get_compile_time_arg_val(8);
constexpr bool verify = get_compile_time_arg_val(9) != 0;
constexpr uint32_t sem_slot = get_compile_time_arg_val(10);
constexpr uint32_t st = Kt * Vt;
constexpr uint32_t kTokSlot = 4;  // token state word: the state slot row 0 continues from (verify)

constexpr uint32_t cb_q_in = 0, cb_k_in = 1, cb_v_in = 2, cb_z_in = 3, cb_norm_w = 4, cb_s_in = 5;
constexpr uint32_t cb_a_full = 6, cb_b_full = 7, cb_ones = 8, cb_row_mask = 9, cb_s_mm = 17, cb_s_new_out = 24;
constexpr uint32_t cb_neg_a_full = 30, cb_dt_bias_full = 31;
constexpr uint32_t kBf16Tile = 2048, kF32Tile = 4096, kFaceBytes = 512, kFaceRow = 32;
constexpr uint32_t kRowTile = 64 * batch;  // a batch x 32 bf16 tile of the row buffer

inline uint32_t row0_offset(uint32_t c) { return c < 16 ? c : 256 + (c - 16); }

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

void zero_bytes(uint32_t l1_addr, uint32_t bytes) {
    volatile tt_l1_ptr uint32_t* p = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1_addr);
    for (uint32_t i = 0; i < bytes / 4; i++) {
        p[i] = 0;
    }
}

// row u of n row tiles starting at row tile `first` -> row 0 of n 32x32 bf16 tiles of `cb`
void place_rows(uint32_t cb, uint32_t rows, uint32_t first, uint32_t n, uint32_t u) {
    cb_reserve_back(cb, n);
    const uint32_t dst = get_write_ptr(cb);
    for (uint32_t t = 0; t < n; t++) {
        const uint32_t src = rows + (first + t) * kRowTile + u * kFaceRow;
        noc_async_read(get_noc_addr(src), dst + t * kBf16Tile, kFaceRow);
        noc_async_read(get_noc_addr(src + batch * kFaceRow), dst + t * kBf16Tile + kFaceBytes, kFaceRow);
    }
    noc_async_read_barrier();
    cb_push_back(cb, n);
}

void push_fill(uint32_t cb, uint32_t bits) {
    cb_reserve_back(cb, 1);
    fill_f32_tile(get_write_ptr(cb), bits);
    cb_push_back(cb, 1);
}

// tiles first .. first + n of an interleaved DRAM tensor
template <typename AddrGen>
void push_pages(uint32_t cb, const AddrGen& src, uint32_t first, uint32_t n, uint32_t tile_bytes) {
    cb_reserve_back(cb, n);
    for (uint32_t t = 0; t < n; t++) {
        noc_async_read_page(first + t, src, get_write_ptr(cb) + t * tile_bytes);
    }
    noc_async_read_barrier();
    cb_push_back(cb, n);
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
    const uint32_t ts_addr = get_arg_val<uint32_t>(gates_base + 2 * copies);
    const uint32_t u0 = get_arg_val<uint32_t>(gates_base + 2 * copies + 1);
    const uint32_t nu = get_arg_val<uint32_t>(gates_base + 2 * copies + 2);
    const uint32_t row_writes = get_arg_val<uint32_t>(gates_base + 2 * copies + 3);
    const uint32_t tok_addr = get_arg_val<uint32_t>(gates_base + 2 * copies + 4);
    const InterleavedAddrGenFast<true> state_dram{
        .bank_base_address = state, .page_size = kF32Tile, .data_format = DataFormat::Float32};
    const InterleavedAddrGenFast<true> norm_dram{
        .bank_base_address = norm_w, .page_size = kBf16Tile, .data_format = DataFormat::Float16_b};
    volatile tt_l1_ptr uint32_t* ts = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ts_addr);
    volatile uint32_t* clk = reinterpret_cast<volatile uint32_t*>(RISCV_DEBUG_REG_WALL_CLOCK_L);

    // Rows 1..31 of the row-0 input tiles are never written again: zero them once (the rank-1 update
    // multiplies them by zero, so they must be finite).
    for (uint32_t cb : {cb_q_in, cb_k_in, cb_v_in, cb_z_in, cb_norm_w}) {
        auto& iface = get_local_cb_interface(cb);
        zero_bytes(iface.fifo_limit - iface.fifo_size, iface.fifo_size);
    }
    push_fill(cb_ones, 0x3f800000u);
    cb_reserve_back(cb_row_mask, 1);
    {
        const uint32_t base = get_write_ptr(cb_row_mask);
        fill_f32_tile(base, 0u);
        volatile tt_l1_ptr uint32_t* p = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base);
        for (uint32_t c = 0; c < 32; c++) {
            p[row0_offset(c)] = 0x3f800000u;
        }
    }
    cb_push_back(cb_row_mask, 1);
    uint32_t cur = 0;
    if constexpr (verify) {
        // the slot word lands in the state CB, unused until the first state read
        const uint32_t scratch = get_write_ptr(cb_s_in);
        noc_async_read(get_noc_addr_from_bank_id<true>(0, tok_addr), scratch, 32);
        noc_async_read_barrier();
        cur = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch)[kTokSlot] & 1;
        *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_slot)) = cur + 1;
    }
    auto& s_new_iface = get_local_cb_interface(cb_s_new_out);
    const uint32_t s_new = s_new_iface.fifo_limit - s_new_iface.fifo_size;

    volatile tt_l1_ptr uint32_t* rows_sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_rows));
    volatile tt_l1_ptr uint32_t* state_sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_state));
    volatile tt_l1_ptr uint16_t* a_row = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(rows + a_tile * kRowTile);
    volatile tt_l1_ptr uint16_t* b_row = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(rows + b_tile * kRowTile);
    // element c of user u in a row tile
    auto elem = [](uint32_t c, uint32_t u) { return (c / 16) * 16 * batch + u * 16 + c % 16; };
    for (uint32_t i = 0; i < layers * nu; i++) {
        const uint32_t l = i / nu, u = u0 + i % nu;
        const uint32_t set = l % copies;
        // the user's state and the norm weight do not depend on the rows: land them first
        if (l >= copies) {
            noc_semaphore_wait_min(state_sem, i - copies * nu + 1);
        }
        cb_reserve_back(cb_s_in, st);
        if (verify && u == 1) {
            // row 0's updated state, still in the writer's CB (compute packs the next one only after this push)
            noc_semaphore_wait_min(state_sem, i);
            cb_reserve_back(cb_s_mm, st);
            noc_async_read(get_noc_addr(s_new), get_write_ptr(cb_s_mm), st * kF32Tile);
            noc_async_read(get_noc_addr(s_new), get_write_ptr(cb_s_in), st * kF32Tile);
            noc_async_read_barrier();
            cb_push_back(cb_s_mm, st);
        } else {
            const uint32_t first = ((set * num_heads + head) * batch + (verify ? cur : u)) * st;
            push_pages(cb_s_mm, state_dram, first, st, kF32Tile);  // then a local copy: one DRAM read per state
            noc_async_read(get_noc_addr(get_read_ptr(cb_s_mm)), get_write_ptr(cb_s_in), st * kF32Tile);
            noc_async_read_barrier();
        }
        cb_push_back(cb_s_in, st);
        push_pages(cb_norm_w, norm_dram, set * Vt, Vt, kBf16Tile);
        if (u == u0) {
            noc_semaphore_wait_min(rows_sem, row_writes * (l + 1));
            if (ts_addr) {
                ts[l * 4 + 0] = *clk;
            }
        }
        place_rows(cb_q_in, rows, q_tile, Kt, u);
        place_rows(cb_k_in, rows, k_tile, Kt, u);
        // a/b are the only scalars read by the RISC: the row buffer is written by other cores
        invalidate_l1_cache();
        const uint32_t a_bits = static_cast<uint32_t>(a_row[elem(a_elem, u)]) << 16;
        const uint32_t b_bits = static_cast<uint32_t>(b_row[elem(b_elem, u)]) << 16;
        push_fill(cb_a_full, a_bits);
        push_fill(cb_dt_bias_full, get_arg_val<uint32_t>(gates_base + 2 * set));
        push_fill(cb_neg_a_full, get_arg_val<uint32_t>(gates_base + 2 * set + 1));
        push_fill(cb_b_full, b_bits);
        place_rows(cb_v_in, rows, v_tile, Vt, u);
        place_rows(cb_z_in, rows, z_tile, Vt, u);
        if (ts_addr && u + 1 == u0 + nu) {
            ts[l * 4 + 1] = *clk;
        }
    }
}
