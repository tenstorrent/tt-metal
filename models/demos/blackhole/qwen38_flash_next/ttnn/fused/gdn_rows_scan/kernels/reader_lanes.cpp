// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// The lanes form of gdn_rows_scan's reader: one (value head, lane) item per core, the whole head
// (block_tiles = head_tiles = 4), and the compute kernel unchanged (rows = the R rows of one lane). The verify tile is
// lane-major (tile row lane * R + j is lane u's row j). It feeds the single form's inputs except that the state tiles
// ([B, 12, 128, 128] fp32) and the FIR history tile ([1, B, 32, 2560] bf16) are lane u's, the masks and gate tiles are
// tile rows lane * R + j, and lane u's first rows are patched with its own history rows after the assembly (lane 0
// needs no patch).
// FINITENESS RULE: every row of every assembled slot must hold a finite value. The compute masks a row's delta by
// multiplication and NaN * 0 = NaN reaches all of S_new through k^T delta, so pad rows and other lanes' rows are
// assembled from real projection / history rows, never left as stale L1.
// Compile-time args: 0 rows (R, rows per lane), 1 block_tiles (= 4, the builder's vbt), 2 lanes (B); then
// TensorAccessorArgs of projected, history, tap0..tap3, dtna, norm, state.  Runtime args: 0 projected, 1 history,
// 2-5 taps, 6 dtna, 7 norm, 8 state addresses, 9 items, then (head, lane) pairs.

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
#include "../../kernels/row_mask.h"

namespace {
constexpr uint32_t cb_p = 0, cb_s = 1, cb_t = 2, cb_z = 3, cb_ab = 4, cb_dtna = 5, cb_w = 6, cb_state = 7;
constexpr uint32_t cb_mask = 8, cb_scaler = 9, cb_gall = 13, cb_beta = 18, cb_decay = 19;
constexpr uint32_t rows = get_compile_time_arg_val(0);
constexpr uint32_t block_tiles = get_compile_time_arg_val(1);
constexpr uint32_t lanes = get_compile_time_arg_val(2);
constexpr uint32_t value_heads = 12, head_tiles = 4, qk_tiles = 2 * head_tiles, conv_tiles = qk_tiles + block_tiles;
constexpr uint32_t block_state_tiles = head_tiles * block_tiles, history_rows = 3;
constexpr uint32_t k_tile0 = 16, v_tile0 = 32, z_tile0 = 80, a_tile = 128, b_tile = 129;
constexpr uint32_t qkv_tiles = 80;  // column tiles of one lane's history tile row
constexpr uint32_t bf16_tile_bytes = 2048, fp32_tile_bytes = 4096;
constexpr uint32_t face_bytes = fused_rows::face_bytes, half_row_bytes = fused_rows::half_row_bytes, face_rows = 16;
static_assert(block_tiles == head_tiles, "the lanes form runs the whole head on one core");
static_assert(rows >= 1 && lanes >= 1 && lanes * rows <= 32, "B lanes x R rows in one tile");

// Reserves `count` pages of `cb` and reads page page_ids[t] of `tensor` into page t; the caller barriers and pushes.
template <typename Accessor>
void read_pages(uint32_t cb_id, const Accessor& tensor, const uint32_t* page_ids, uint32_t count, uint32_t page_bytes) {
    const Noc noc;
    CircularBuffer cb(cb_id);
    cb.reserve_back(count);
    for (uint32_t t = 0; t < count; ++t) {
        noc.async_read(
            tensor, cb, tensor.get_aligned_page_size(), {.page_id = page_ids[t]}, {.offset_bytes = t * page_bytes});
    }
}

// Fill one fp32 tile (1024 words) with `value`: RISC word stores, eight per iteration.
inline void fill_tile_words(uint32_t l1_addr, uint32_t value) {
    CoreLocalMem<volatile uint32_t> words(l1_addr);
    for (uint32_t k = 0; k < fp32_tile_bytes / 4; k += 8) {
        words[k] = value;
        words[k + 1] = value;
        words[k + 2] = value;
        words[k + 3] = value;
        words[k + 4] = value;
        words[k + 5] = value;
        words[k + 6] = value;
        words[k + 7] = value;
    }
}

// L1 -> L1 copy on the NoC (asynchronous; a read barrier lands it).
inline void local_copy(uint32_t dst_addr, uint32_t src_addr, uint32_t bytes) {
    const Noc noc;
    noc.async_read(CoreLocalMem<uint32_t>(src_addr), CoreLocalMem<uint32_t>(dst_addr), bytes, {}, {});
}

// The single form's whole-tile assembly: slot_i (i = 0, 1, 2) of one column tile: rows shift..31 (shift = 3 - i) <-
// projection rows 0..31-shift, rows 0..shift-1 <- history rows 3-shift..2, taken from slot 0's rows 0..2 (never
// overwritten). Tile rows are 32-byte pieces in two faces per 16-row block.
inline void assemble_slots(uint32_t projected_tile, uint32_t slot0_tile) {
    for (uint32_t i = 0; i < history_rows; ++i) {
        const uint32_t shift = history_rows - i;
        const uint32_t dst_tile = slot0_tile + i * bf16_tile_bytes;
        for (uint32_t f = 0; f < 2; ++f) {
            const uint32_t top = f * face_bytes, bottom = (2 + f) * face_bytes;
            // rows shift..15 <- projection rows 0..15-shift
            local_copy(
                dst_tile + top + shift * half_row_bytes, projected_tile + top, (face_rows - shift) * half_row_bytes);
            // rows 16..15+shift <- projection rows 16-shift..15; rows 16+shift..31 <- projection rows 16..31-shift
            local_copy(
                dst_tile + bottom, projected_tile + top + (face_rows - shift) * half_row_bytes, shift * half_row_bytes);
            local_copy(
                dst_tile + bottom + shift * half_row_bytes,
                projected_tile + bottom,
                (face_rows - shift) * half_row_bytes);
            if (i > 0) {  // rows 0..shift-1 <- history rows 3-shift..2 (slot 0's rows 3-shift..2)
                local_copy(
                    dst_tile + top, slot0_tile + top + (history_rows - shift) * half_row_bytes, shift * half_row_bytes);
            }
        }
    }
}

// Copy tile row `src_row` of the bf16 tile at `src_tile` into tile row `dst_row` of the tile at `dst_tile` (64 B: two
// 32 B face pieces) with RISC word stores (fused_rows::copy_row_bf16 copies the same row index; the patch moves rows).
inline void copy_row_bf16_between(uint32_t dst_tile, uint32_t dst_row, uint32_t src_tile, uint32_t src_row) {
    CoreLocalMem<volatile uint32_t> dst(dst_tile + fused_rows::bf16_row_offset(dst_row));
    CoreLocalMem<const volatile uint32_t> src(src_tile + fused_rows::bf16_row_offset(src_row));
    for (uint32_t k = 0; k < half_row_bytes / 4; ++k) {
        dst[k] = src[k];
        dst[k + face_bytes / 4] = src[k + face_bytes / 4];
    }
}

// The lane patch, after the assembly has landed: slot i (shift = 3 - i), rows row0 + j for j < min(shift, R) take lane
// u's history rows i + j from slot 0's rows 0..2. RISC word copies, so no NoC write races the assembly. Slot 0 is
// patched last: the other slots read from it, and its own writes land at rows >= R, past every row still to be read.
inline void patch_lane_rows(uint32_t slot0_tile, uint32_t row0) {
    for (uint32_t k = 0; k < history_rows; ++k) {
        const uint32_t i = history_rows - 1 - k;  // slots 2, 1, then 0
        const uint32_t shift = history_rows - i;
        const uint32_t dst_tile = slot0_tile + i * bf16_tile_bytes;
        for (uint32_t j = 0; j < shift && j < rows; ++j) {
            copy_row_bf16_between(dst_tile, row0 + j, slot0_tile, i + j);
        }
    }
}

// The L1 bases of an item's conv inputs: P, S and T reserved for the whole item.
struct ConvInputs {
    uint32_t projected;  // cb_p: conv_tiles projection tiles
    uint32_t slots;      // cb_s: three slots per tile, tile-major
    uint32_t taps;       // cb_t: four taps per tile, tile-major
};

// Issues (without a barrier) tile `tile`'s conv input reads: the projection tile and the four taps at page `page`, the
// history tile at `history_page` as slot 0.
template <typename Projected, typename History, typename Tap0, typename Tap1, typename Tap2, typename Tap3>
void read_conv_tile(
    uint32_t page,
    uint32_t history_page,
    uint32_t tile,
    const ConvInputs& l1,
    const Projected& projected,
    const History& history,
    const Tap0& tap0,
    const Tap1& tap1,
    const Tap2& tap2,
    const Tap3& tap3) {
    const Noc noc;
    const CoreLocalMem<uint32_t> projected_l1(l1.projected), slots_l1(l1.slots), taps_l1(l1.taps);
    noc.async_read(
        projected,
        projected_l1,
        projected.get_aligned_page_size(),
        {.page_id = page},
        {.offset_bytes = tile * bf16_tile_bytes});
    noc.async_read(
        history,
        slots_l1,
        history.get_aligned_page_size(),
        {.page_id = history_page},
        {.offset_bytes = 3 * tile * bf16_tile_bytes});
    noc.async_read(
        tap0,
        taps_l1,
        tap0.get_aligned_page_size(),
        {.page_id = page},
        {.offset_bytes = (4 * tile + 0) * bf16_tile_bytes});
    noc.async_read(
        tap1,
        taps_l1,
        tap1.get_aligned_page_size(),
        {.page_id = page},
        {.offset_bytes = (4 * tile + 1) * bf16_tile_bytes});
    noc.async_read(
        tap2,
        taps_l1,
        tap2.get_aligned_page_size(),
        {.page_id = page},
        {.offset_bytes = (4 * tile + 2) * bf16_tile_bytes});
    noc.async_read(
        tap3,
        taps_l1,
        tap3.get_aligned_page_size(),
        {.page_id = page},
        {.offset_bytes = (4 * tile + 3) * bf16_tile_bytes});
}
}  // namespace

void kernel_main() {
    uint32_t arg = 0;
    const uint32_t projected_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t history_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t tap0_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t tap1_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t tap2_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t tap3_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t dtna_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t norm_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t state_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t items = get_arg_val<uint32_t>(arg++);

    constexpr auto projected_args = TensorAccessorArgs<3>();
    constexpr auto history_args = TensorAccessorArgs<projected_args.next_compile_time_args_offset()>();
    constexpr auto tap0_args = TensorAccessorArgs<history_args.next_compile_time_args_offset()>();
    constexpr auto tap1_args = TensorAccessorArgs<tap0_args.next_compile_time_args_offset()>();
    constexpr auto tap2_args = TensorAccessorArgs<tap1_args.next_compile_time_args_offset()>();
    constexpr auto tap3_args = TensorAccessorArgs<tap2_args.next_compile_time_args_offset()>();
    constexpr auto dtna_args = TensorAccessorArgs<tap3_args.next_compile_time_args_offset()>();
    constexpr auto norm_args = TensorAccessorArgs<dtna_args.next_compile_time_args_offset()>();
    constexpr auto state_args = TensorAccessorArgs<norm_args.next_compile_time_args_offset()>();

    const auto projected = TensorAccessor(projected_args, projected_addr);
    const auto history = TensorAccessor(history_args, history_addr);
    const auto tap0 = TensorAccessor(tap0_args, tap0_addr);
    const auto tap1 = TensorAccessor(tap1_args, tap1_addr);
    const auto tap2 = TensorAccessor(tap2_args, tap2_addr);
    const auto tap3 = TensorAccessor(tap3_args, tap3_addr);
    const auto dtna = TensorAccessor(dtna_args, dtna_addr);
    const auto norm = TensorAccessor(norm_args, norm_addr);
    const auto state = TensorAccessor(state_args, state_addr);

    const Noc noc;

    {
        const uint32_t norm_pages[head_tiles] = {0, 1, 2, 3};
        read_pages(cb_w, norm, norm_pages, head_tiles, bf16_tile_bytes);
        noc.async_read_barrier();
        CircularBuffer(cb_w).push_back(head_tiles);
        // reduce scalers (fp32, whole tiles): 1.0 for sums, 1/128 for means
        CircularBuffer(cb_scaler).reserve_back(2);
        const uint32_t scaler_l1 = CircularBuffer(cb_scaler).get_write_ptr();
        fused_rows::fill_words(scaler_l1, fp32_tile_bytes / 4, 0x3F800000u);
        fused_rows::fill_words(scaler_l1 + fp32_tile_bytes, fp32_tile_bytes / 4, 0x3C000000u);
        CircularBuffer(cb_scaler).push_back(2);
    }

    for (uint32_t item = 0; item < items; ++item) {
        const uint32_t head = get_arg_val<uint32_t>(arg++);
        const uint32_t lane = get_arg_val<uint32_t>(arg++);
        const uint32_t qk_head = head / 3;
        const uint32_t row0 = lane * rows;  // lane u's first tile row
        const uint32_t history_page0 = lane * qkv_tiles;

        uint32_t conv_pages[conv_tiles];
        for (uint32_t c = 0; c < head_tiles; ++c) {
            conv_pages[c] = qk_head * head_tiles + c;
            conv_pages[head_tiles + c] = k_tile0 + qk_head * head_tiles + c;
        }
        for (uint32_t c = 0; c < block_tiles; ++c) {
            conv_pages[qk_tiles + c] = v_tile0 + head * head_tiles + c;
        }
        // The conv inputs go out per output tile t (cb_s 3t.., cb_t 4t..) so the next tile's DRAM reads overlap this
        // tile's slot assembly. The L1 bases are taken once: the per-tile pushes below move the CBs' write pointers.
        CircularBuffer(cb_p).reserve_back(conv_tiles);
        CircularBuffer(cb_s).reserve_back(3 * conv_tiles);
        CircularBuffer(cb_t).reserve_back(4 * conv_tiles);
        const ConvInputs conv_l1 = {
            .projected = CircularBuffer(cb_p).get_write_ptr(),
            .slots = CircularBuffer(cb_s).get_write_ptr(),
            .taps = CircularBuffer(cb_t).get_write_ptr()};
        read_conv_tile(
            conv_pages[0], history_page0 + conv_pages[0], 0, conv_l1, projected, history, tap0, tap1, tap2, tap3);
        noc.async_read_barrier();
        for (uint32_t t = 0; t < conv_tiles; ++t) {
            if (t + 1 < conv_tiles) {
                read_conv_tile(
                    conv_pages[t + 1],
                    history_page0 + conv_pages[t + 1],
                    t + 1,
                    conv_l1,
                    projected,
                    history,
                    tap0,
                    tap1,
                    tap2,
                    tap3);
            }
            assemble_slots(conv_l1.projected + t * bf16_tile_bytes, conv_l1.slots + (3 * t) * bf16_tile_bytes);
            noc.async_read_barrier();  // group t + 1 and tile t's slots landed
            if (lane > 0) {
                patch_lane_rows(conv_l1.slots + (3 * t) * bf16_tile_bytes, row0);
            }
            CircularBuffer(cb_p).push_back(1);
            CircularBuffer(cb_s).push_back(3);
            CircularBuffer(cb_t).push_back(4);
        }

        uint32_t z_pages[head_tiles];
        for (uint32_t c = 0; c < head_tiles; ++c) {
            z_pages[c] = z_tile0 + head * head_tiles + c;
        }
        read_pages(cb_z, projected, z_pages, head_tiles, bf16_tile_bytes);
        // the a and b tiles once: the compute's gates take every tile row in one pass; this item uses R from row0
        CircularBuffer(cb_ab).reserve_back(2);
        noc.async_read(
            projected,
            CircularBuffer(cb_ab),
            projected.get_aligned_page_size(),
            {.page_id = a_tile},
            {.offset_bytes = 0});
        noc.async_read(
            projected,
            CircularBuffer(cb_ab),
            projected.get_aligned_page_size(),
            {.page_id = b_tile},
            {.offset_bytes = bf16_tile_bytes});
        const uint32_t dtna_pages[2] = {head, value_heads + head};
        read_pages(cb_dtna, dtna, dtna_pages, 2, fp32_tile_bytes);
        uint32_t state_pages[block_state_tiles];
        for (uint32_t kt = 0; kt < head_tiles; ++kt) {
            for (uint32_t c = 0; c < block_tiles; ++c) {
                state_pages[kt * block_tiles + c] =
                    (lane * value_heads + head) * head_tiles * head_tiles + kt * head_tiles + c;
            }
        }
        read_pages(cb_state, state, state_pages, block_state_tiles, fp32_tile_bytes);
        noc.async_read_barrier();
        CircularBuffer(cb_z).push_back(head_tiles);
        CircularBuffer(cb_ab).push_back(2);
        CircularBuffer(cb_dtna).push_back(2);
        CircularBuffer(cb_state).push_back(block_state_tiles);

        CircularBuffer(cb_mask).reserve_back(rows);
        {
            const uint32_t mask_l1 = CircularBuffer(cb_mask).get_write_ptr();
            for (uint32_t r = 0; r < rows; ++r) {
                fused_rows::one_hot_row_bf16(mask_l1 + r * bf16_tile_bytes, row0 + r);
            }
        }
        CircularBuffer(cb_mask).push_back(rows);
        // The gates: each lane row's element of cb_gall (tile 0 beta, tile 1 decay) replicated into one full fp32 tile
        // per row (exact 32-bit copies), pushed row by row so row 0 starts first.
        {
            CircularBuffer(cb_gall).wait_front(2);
            const CoreLocalMem<volatile uint32_t> gates(CircularBuffer(cb_gall).get_read_ptr());
            CircularBuffer(cb_beta).reserve_back(rows);
            CircularBuffer(cb_decay).reserve_back(rows);
            const uint32_t beta_l1 = CircularBuffer(cb_beta).get_write_ptr();
            const uint32_t decay_l1 = CircularBuffer(cb_decay).get_write_ptr();
            for (uint32_t r = 0; r < rows; ++r) {
                const uint32_t element = fused_rows::face_element(row0 + r, head);
                fill_tile_words(beta_l1 + r * fp32_tile_bytes, gates[element]);
                fill_tile_words(decay_l1 + r * fp32_tile_bytes, gates[fp32_tile_bytes / 4 + element]);
                CircularBuffer(cb_beta).push_back(1);
                CircularBuffer(cb_decay).push_back(1);
            }
            CircularBuffer(cb_gall).pop_front(2);
        }
    }
}
