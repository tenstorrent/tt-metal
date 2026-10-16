// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// The lanes form of gdn_rows_scan's reader: one (value head, lane) item per core, the whole head (VBT = HT = 4) per
// core, and the compute kernel unchanged (its ROWS = the R rows of ONE lane).  The verify tile is lane-major: tile
// row t = lane * R + j is lane u's row j.  Per item this reader feeds the compute exactly what the single form's
// reader feeds it for a 32-row tile whose rows 0..R-1 are the real ones, with three differences:
//   - the state tiles are lane u's ([B, 12, 128, 128] fp32: page (lane * 12 + head) * 16 + kt * 4 + vt);
//   - the FIR history tile is lane u's ([1, B, 32, 2560] bf16: page lane * 80 + column tile), and after the whole-tile
//     shifted-window assembly (slot i, row t = window row t + i, window = [history rows 0..2 | projection rows
//     0..31]) lane u's first rows are PATCHED: slot with shift s = 3 - i, rows t = lane * R + j for j < min(s, R) take
//     lane u's history rows 3 - s + j (the assembly gave them lane u-1's projection rows).  Lane 0 needs no patch;
//   - the one-hot row masks and the replicated gate tiles are those of tile rows lane * R + j, at CB slots j.
// FINITENESS RULE: every row of every assembled slot must hold a finite value.  The compute masks a row's delta by
// MULTIPLICATION (delta = ((v - v_read) * M_row) * beta), and NaN * 0 = NaN reaches every element of S_new through
// k^T delta; the pad rows past B * R and the other lanes' rows are therefore assembled from the projection and the
// history tiles (real rows, or the chain's exact zero pads) and never left as stale L1.  The die test pins it.
// Compile-time args: 0 ROWS (R, rows per lane), 1 VBT (= 4), 2 LANES (B); then TensorAccessorArgs of projected,
// history, tap0..tap3, dtna, norm, state.  Runtime args: 0 projected, 1 history, 2-5 taps, 6 dtna, 7 norm, 8 state
// addresses, 9 items, then (head, lane) pairs.

#include "api/dataflow/dataflow_api.h"
#include "api/tensor/noc_traits.h"
#include "../../kernels/zones.h"
#include "../../kernels/row_mask.h"

namespace {
constexpr uint32_t CB_P = 0, CB_S = 1, CB_T = 2, CB_Z = 3, CB_AB = 4, CB_DTNA = 5, CB_W = 6, CB_STATE = 7;
constexpr uint32_t CB_MASK = 8, CB_SCALER = 9, CB_GALL = 13, CB_BETA = 18, CB_DECAY = 19;
constexpr uint32_t ROWS = get_compile_time_arg_val(0);
constexpr uint32_t VBT = get_compile_time_arg_val(1);
constexpr uint32_t LANES = get_compile_time_arg_val(2);
constexpr uint32_t HEADS = 12, HT = 4, QK_TILES = 2 * HT, NT = QK_TILES + VBT, ST = HT * VBT, HISTORY_ROWS = 3;
constexpr uint32_t K_TILE0 = 16, V_TILE0 = 32, Z_TILE0 = 80, A_TILE = 128, B_TILE = 129;
constexpr uint32_t QKV_TILES = 80;  // column tiles of one lane's history tile row
constexpr uint32_t BF16_TILE = 2048, FP32_TILE = 4096;
constexpr uint32_t FACE_BYTES = fused_rows::FACE_BYTES, HALF_ROW = fused_rows::HALF_ROW_BYTES, FACE_ROWS = 16;
static_assert(VBT == HT, "the lanes form runs the whole head on one core");
static_assert(ROWS >= 1 && LANES >= 1 && LANES * ROWS <= 32, "B lanes x R rows in one tile");

template <typename Acc>
uint32_t read_tiles(uint32_t cb, const Acc& acc, const uint32_t* ids, uint32_t count, uint32_t tile_bytes) {
    cb_reserve_back(cb, count);
    const uint32_t l1 = get_write_ptr(cb);
    for (uint32_t t = 0; t < count; ++t) {
        noc_async_read_page(ids[t], acc, l1 + t * tile_bytes);
    }
    return l1;
}

// Fill one fp32 tile (1024 words) with `value`: RISC word stores, eight per iteration.
inline void fill_tile_words(uint32_t l1, uint32_t value) {
    volatile tt_l1_ptr uint32_t* p = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1);
    for (uint32_t k = 0; k < FP32_TILE / 4; k += 8) {
        p[k] = value;
        p[k + 1] = value;
        p[k + 2] = value;
        p[k + 3] = value;
        p[k + 4] = value;
        p[k + 5] = value;
        p[k + 6] = value;
        p[k + 7] = value;
    }
}

// L1 -> L1 copy on the NoC (asynchronous; a noc_async_read_barrier lands it).
inline void local_copy(uint32_t dst, uint32_t src, uint32_t bytes) { noc_async_read(get_noc_addr(src), dst, bytes); }

// The single form's whole-tile assembly: slot_i (i = 0, 1, 2) of one column tile: rows s..31 (s = 3 - i) <-
// projection rows 0..31-s, rows 0..s-1 <- history rows 3-s..2.  The history tile was read whole into slot 0 (its
// rows 0..2 are already right there); slots 1 and 2 take their history rows from slot 0's rows, which the shift
// below never overwrites.  Tile rows are 32-byte pieces in two faces per 16-row block, so a shift is six
// contiguous copies per slot plus the history pieces.
inline void assemble_slots(uint32_t p_tile, uint32_t s0_tile) {
    for (uint32_t i = 0; i < HISTORY_ROWS; ++i) {
        const uint32_t s = HISTORY_ROWS - i;
        const uint32_t dst_tile = s0_tile + i * BF16_TILE;
        for (uint32_t f = 0; f < 2; ++f) {
            const uint32_t top = f * FACE_BYTES, bottom = (2 + f) * FACE_BYTES;
            // rows s..15 <- projection rows 0..15-s
            local_copy(dst_tile + top + s * HALF_ROW, p_tile + top, (FACE_ROWS - s) * HALF_ROW);
            // rows 16..15+s <- projection rows 16-s..15; rows 16+s..31 <- projection rows 16..31-s
            local_copy(dst_tile + bottom, p_tile + top + (FACE_ROWS - s) * HALF_ROW, s * HALF_ROW);
            local_copy(dst_tile + bottom + s * HALF_ROW, p_tile + bottom, (FACE_ROWS - s) * HALF_ROW);
            if (i > 0) {  // rows 0..s-1 <- history rows 3-s..2 (slot 0's rows 3-s..2)
                local_copy(dst_tile + top, s0_tile + top + (HISTORY_ROWS - s) * HALF_ROW, s * HALF_ROW);
            }
        }
    }
}

// Copy tile row `src_row` of the bf16 tile at `src` into tile row `dst_row` of the tile at `dst` (64 B: two 32 B
// face pieces) with RISC word stores (fused_rows::copy_row_bf16 copies the same row index; the patch moves rows).
inline void copy_row_bf16_between(uint32_t dst, uint32_t dst_row, uint32_t src, uint32_t src_row) {
    volatile tt_l1_ptr uint32_t* d =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dst + fused_rows::bf16_row_offset(dst_row));
    const volatile tt_l1_ptr uint32_t* s =
        reinterpret_cast<const volatile tt_l1_ptr uint32_t*>(src + fused_rows::bf16_row_offset(src_row));
    for (uint32_t k = 0; k < HALF_ROW / 4; ++k) {
        d[k] = s[k];
        d[k + FACE_BYTES / 4] = s[k + FACE_BYTES / 4];
    }
}

// The lane patch, after the assembly has landed: slot i (shift s = 3 - i), rows t = row0 + j for j < min(s, R)
// take lane u's history rows 3 - s + j = i + j, which sit in slot 0's rows 0..2 (never shifted).  RISC word copies
// (16 words per row: two 32-byte face pieces), so no NoC write can race the assembly's.  Slot 0 is patched LAST:
// its patch writes into the very tile the other slots read their history rows from (row0 + j may be 1 or 2 for
// lane 1 at R <= 2), and within slot 0 every write lands at row0 + j >= R > j', past every row j' still to be read.
inline void patch_lane_rows(uint32_t s0_tile, uint32_t row0) {
    for (uint32_t k = 0; k < HISTORY_ROWS; ++k) {
        const uint32_t i = HISTORY_ROWS - 1 - k;  // slots 2, 1, then 0
        const uint32_t s = HISTORY_ROWS - i;
        const uint32_t dst_tile = s0_tile + i * BF16_TILE;
        for (uint32_t j = 0; j < s && j < ROWS; ++j) {
            copy_row_bf16_between(dst_tile, row0 + j, s0_tile, i + j);
        }
    }
}
}  // namespace

void kernel_main() {
    uint32_t arg = 0;
    const uint32_t p_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t h_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t t0_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t t1_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t t2_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t t3_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t dtna_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t w_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t state_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t items = get_arg_val<uint32_t>(arg++);

    constexpr auto p_args = TensorAccessorArgs<3>();
    constexpr auto h_args = TensorAccessorArgs<p_args.next_compile_time_args_offset()>();
    constexpr auto t0_args = TensorAccessorArgs<h_args.next_compile_time_args_offset()>();
    constexpr auto t1_args = TensorAccessorArgs<t0_args.next_compile_time_args_offset()>();
    constexpr auto t2_args = TensorAccessorArgs<t1_args.next_compile_time_args_offset()>();
    constexpr auto t3_args = TensorAccessorArgs<t2_args.next_compile_time_args_offset()>();
    constexpr auto dtna_args = TensorAccessorArgs<t3_args.next_compile_time_args_offset()>();
    constexpr auto w_args = TensorAccessorArgs<dtna_args.next_compile_time_args_offset()>();
    constexpr auto state_args = TensorAccessorArgs<w_args.next_compile_time_args_offset()>();

    const auto p = TensorAccessor(p_args, p_addr);
    const auto h = TensorAccessor(h_args, h_addr);
    const auto t0 = TensorAccessor(t0_args, t0_addr);
    const auto t1 = TensorAccessor(t1_args, t1_addr);
    const auto t2 = TensorAccessor(t2_args, t2_addr);
    const auto t3 = TensorAccessor(t3_args, t3_addr);
    const auto dtna = TensorAccessor(dtna_args, dtna_addr);
    const auto w = TensorAccessor(w_args, w_addr);
    const auto state = TensorAccessor(state_args, state_addr);

    {
        FUSED_ZONE("fz_gsc_lr_setup");
        uint32_t ids[HT] = {0, 1, 2, 3};
        read_tiles(CB_W, w, ids, HT, BF16_TILE);
        noc_async_read_barrier();
        cb_push_back(CB_W, HT);
        // reduce scalers (fp32, whole tiles): 1.0 for sums, 1/128 for means
        cb_reserve_back(CB_SCALER, 2);
        const uint32_t l1 = get_write_ptr(CB_SCALER);
        fused_rows::fill_words(l1, FP32_TILE / 4, 0x3F800000u);
        fused_rows::fill_words(l1 + FP32_TILE, FP32_TILE / 4, 0x3C000000u);
        cb_push_back(CB_SCALER, 2);
    }

    for (uint32_t item = 0; item < items; ++item) {
        FUSED_ZONE("fz_gsc_lr_item");
        const uint32_t head = get_arg_val<uint32_t>(arg++);
        const uint32_t lane = get_arg_val<uint32_t>(arg++);
        const uint32_t qk_head = head / 3;
        const uint32_t row0 = lane * ROWS;  // lane u's first tile row
        const uint32_t history_page0 = lane * QKV_TILES;

        uint32_t ids[NT];
        for (uint32_t c = 0; c < HT; ++c) {
            ids[c] = qk_head * HT + c;
            ids[HT + c] = K_TILE0 + qk_head * HT + c;
        }
        for (uint32_t c = 0; c < VBT; ++c) {
            ids[QK_TILES + c] = V_TILE0 + head * HT + c;
        }
        // The conv inputs go out per output tile t (P[t], the three slots' tile t at CB_S 3t.., the four taps' tile t
        // at CB_T 4t..): the next tile's DRAM reads are in flight while this tile's slots are assembled, and the
        // conv starts after the first group.
        cb_reserve_back(CB_P, NT);
        cb_reserve_back(CB_S, 3 * NT);
        cb_reserve_back(CB_T, 4 * NT);
        const uint32_t p_l1 = get_write_ptr(CB_P);
        const uint32_t s_l1 = get_write_ptr(CB_S);
        const uint32_t t_l1 = get_write_ptr(CB_T);
        auto issue_group = [&](uint32_t t) {
            noc_async_read_page(ids[t], p, p_l1 + t * BF16_TILE);
            // lane u's history tile lands as slot 0
            noc_async_read_page(history_page0 + ids[t], h, s_l1 + (3 * t) * BF16_TILE);
            noc_async_read_page(ids[t], t0, t_l1 + (4 * t + 0) * BF16_TILE);
            noc_async_read_page(ids[t], t1, t_l1 + (4 * t + 1) * BF16_TILE);
            noc_async_read_page(ids[t], t2, t_l1 + (4 * t + 2) * BF16_TILE);
            noc_async_read_page(ids[t], t3, t_l1 + (4 * t + 3) * BF16_TILE);
        };
        issue_group(0);
        noc_async_read_barrier();
        for (uint32_t t = 0; t < NT; ++t) {
            if (t + 1 < NT) {
                issue_group(t + 1);
            }
            assemble_slots(p_l1 + t * BF16_TILE, s_l1 + (3 * t) * BF16_TILE);
            noc_async_read_barrier();  // group t + 1 and tile t's slots landed
            if (lane > 0) {
                patch_lane_rows(s_l1 + (3 * t) * BF16_TILE, row0);
            }
            cb_push_back(CB_P, 1);
            cb_push_back(CB_S, 3);
            cb_push_back(CB_T, 4);
        }

        uint32_t z_ids[HT];
        for (uint32_t c = 0; c < HT; ++c) {
            z_ids[c] = Z_TILE0 + head * HT + c;
        }
        read_tiles(CB_Z, p, z_ids, HT, BF16_TILE);
        // the a and b tiles once: every tile row's gate scalars of every head (the compute's gates take all rows in
        // one pass; this item uses rows row0 .. row0 + R - 1)
        cb_reserve_back(CB_AB, 2);
        const uint32_t ab_l1 = get_write_ptr(CB_AB);
        noc_async_read_page(A_TILE, p, ab_l1);
        noc_async_read_page(B_TILE, p, ab_l1 + BF16_TILE);
        const uint32_t dtna_ids[2] = {head, HEADS + head};
        read_tiles(CB_DTNA, dtna, dtna_ids, 2, FP32_TILE);
        uint32_t state_ids[ST];
        for (uint32_t kt = 0; kt < HT; ++kt) {
            for (uint32_t c = 0; c < VBT; ++c) {
                state_ids[kt * VBT + c] = (lane * HEADS + head) * HT * HT + kt * HT + c;
            }
        }
        read_tiles(CB_STATE, state, state_ids, ST, FP32_TILE);
        noc_async_read_barrier();
        cb_push_back(CB_Z, HT);
        cb_push_back(CB_AB, 2);
        cb_push_back(CB_DTNA, 2);
        cb_push_back(CB_STATE, ST);

        cb_reserve_back(CB_MASK, ROWS);
        {
            const uint32_t l1 = get_write_ptr(CB_MASK);
            for (uint32_t r = 0; r < ROWS; ++r) {
                fused_rows::one_hot_row_bf16(l1 + r * BF16_TILE, row0 + r);
            }
        }
        cb_push_back(CB_MASK, ROWS);
        // The gates: the compute's beta_all / decay_all fp32 tiles (CB_GALL: tile 0 beta, tile 1 decay; element
        // (tile row, head) is that row's scalar) replicated into one full fp32 tile per lane row for the
        // recurrence's element-wise multiplies -- exact 32-bit copies, pushed row by row so row 0 starts first.
        {
            FUSED_ZONE("fz_gsc_lr_gates");
            cb_wait_front(CB_GALL, 2);
            const uint32_t gall = get_read_ptr(CB_GALL);
            cb_reserve_back(CB_BETA, ROWS);
            cb_reserve_back(CB_DECAY, ROWS);
            const uint32_t beta_l1 = get_write_ptr(CB_BETA);
            const uint32_t decay_l1 = get_write_ptr(CB_DECAY);
            for (uint32_t r = 0; r < ROWS; ++r) {
                const uint32_t at = 4 * fused_rows::face_element(row0 + r, head);
                fill_tile_words(beta_l1 + r * FP32_TILE, *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(gall + at));
                fill_tile_words(
                    decay_l1 + r * FP32_TILE, *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(gall + FP32_TILE + at));
                cb_push_back(CB_BETA, 1);
                cb_push_back(CB_DECAY, 1);
            }
            cb_pop_front(CB_GALL, 2);
        }
    }
}
