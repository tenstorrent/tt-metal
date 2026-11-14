// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// QWEN36_MAMBA_SPEC GDN block I/O (tt/gdn_block_io.py): every GDN layer's state moves between the B=1 prefill scratch
// and vLLM's state blocks in ONE data-movement program, in place of about 400 to 550 small ttnn ops (slice, reshape,
// pad, to_layout, paged_fill_cache per layer) that are each a few microseconds of launch for a few kilobytes of data.
//
// Block layout (tt/spec_block_state.py, SpecBlockLayout): head h of a block is 16 consecutive fp32 tile pages of a flat
// view of the layer slot's K buffer (two-pool page < POOL1) or V buffer (page - POOL1), in the tile order of one
// [Dk, Dv] = [128, 128] head. The conv history is CONV_PAGES bf16 tile pages of a flat view of V (two-pool page <
// CONV_TOTAL) or K (page - CONV_TOTAL): the KC1 x C carry values in row-major order, 32 per tile row, zero padded.
//
// Modes:
//   WRITE       scratch -> blocks: rec_state [1, NV, 128, 128] fp32 TILE, conv_carry [1, KC1, C] bf16 TILE
//   WRITE_SNAP  in-replay snapshot k -> blocks: _snap_rec [S, NV, 128, 128] fp32 TILE (slot k), _snap_conv_rm
//               [1, KC1 * S, C] bf16 ROW_MAJOR (rows KC1 * k ..)
//   RESTORE     blocks -> scratch (conv_carry tile rows past KC1 are written as zeros)
//   ZERO        zero whole blocks of every K and V buffer (bfp8 [num_blocks, 1, block, head_dim] TILE)
//   WINDOW      the conv rows after token t of row u's verify window (verify_win_cur [B, L, C] bf16 TILE, rows
//               t + 1 .. t + KC1) -> the conv history of a block, in every layer of one group (the boundary conv
//               writes of SpecGdnBlockState.after_verify)
//   CHAIN       block src -> block dst in every layer of one group (head pages; conv history from src's pages, or
//               from row u's window rows after token t), only when row u's accepted count (result buffer row
//               RES_COUNTS, read on the device) is in the entry's count mask (the chained copies of
//               SpecGdnBlockState.run_chain_copies; without the result buffer every entry runs)
//
// Common runtime args, WRITE / WRITE_SNAP / RESTORE:
//   [0] snapshot slot k (WRITE_SNAP; else unused)
//   [1 + g * (NV + 1) + h] two-pool head page of head h of group g's block, [.. + NV] its two-pool conv page
//   [1 + G * (NV + 1) + 5 * l ..] layer l: group, rec address, conv address, K address, V address
// ZERO: [0, KV_BUFS) K / V buffer addresses, [KV_BUFS] block count n <= ZERO_MAX_BLOCKS, then ZERO_MAX_BLOCKS ids.
// WINDOW: [0] entry count n <= WIN_MAX, [1 + 4 e ..] entry e: row u, column t, group, two-pool conv page; then the
// layers as above from [1 + 4 * WIN_MAX] (conv address = the layer's window).
// CHAIN: [0] entry count n <= CH_MAX, [1] result buffer address, [2] 1 = select by the result buffer's counts, then
// CH_MAX entries of CH_WORDS: row u, group, t (NO_T: src holds its conv history), count mask (bit c - 1 = count c),
// src's NV head pages and conv page, dst's; then the layers from [3 + CH_MAX * CH_WORDS] (conv address = window).
//
// Work: core c (logical y * NX + x) runs items c, c + N_CORES, ...; WRITE / RESTORE item i = layer i / (NV + 1), part
// i % (NV + 1) (a head, or NV = the conv history); ZERO item i = (block, buffer, ZERO_CHUNK-page chunk); WINDOW item
// i = (entry i / NL, layer i % NL), skipped when the layer is not in the entry's group.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

namespace {

constexpr uint32_t MODE = get_compile_time_arg_val(0);
constexpr uint32_t NL = get_compile_time_arg_val(1);
constexpr uint32_t NV = get_compile_time_arg_val(2);
constexpr uint32_t G = get_compile_time_arg_val(3);
constexpr uint32_t POOL1 = get_compile_time_arg_val(4);
constexpr uint32_t CONV_TOTAL = get_compile_time_arg_val(5);
constexpr uint32_t CONV_PAGES = get_compile_time_arg_val(6);
constexpr uint32_t KC1 = get_compile_time_arg_val(7);  // conv carry rows (kernel size - 1)
constexpr uint32_t CT = get_compile_time_arg_val(8);   // conv carry tiles per row (C / 32)
constexpr uint32_t NX = get_compile_time_arg_val(9);
constexpr uint32_t N_CORES = get_compile_time_arg_val(10);
constexpr uint32_t CB_L1 = get_compile_time_arg_val(11);
constexpr uint32_t CB_BYTES = get_compile_time_arg_val(12);
constexpr uint32_t KV_BUFS = get_compile_time_arg_val(13);
constexpr uint32_t KV_TPB = get_compile_time_arg_val(14);  // K / V tile pages per block
constexpr uint32_t ZERO_CHUNK = get_compile_time_arg_val(15);
constexpr uint32_t ZERO_MAX_BLOCKS = get_compile_time_arg_val(16);
constexpr uint32_t RM_ROW = get_compile_time_arg_val(17);  // bytes of one _snap_conv_rm row page
constexpr uint32_t WIN_MAX = get_compile_time_arg_val(18);
constexpr uint32_t CH_MAX = get_compile_time_arg_val(19);
constexpr uint32_t CTA_ACCESSORS = 20;

constexpr uint32_t WRITE = 0, WRITE_SNAP = 1, RESTORE = 2, ZERO = 3, WINDOW = 4, CHAIN = 5;
constexpr uint32_t RES_COUNTS = 1, NO_T = 0xFFFFFFFFu;
constexpr uint32_t CH_WORDS = 4 + 2 * (NV + 1);
constexpr uint32_t FP32_PAGE = 4096, BF16_PAGE = 2048, HEAD_PAGES = 16, BATCH = 4;
constexpr uint32_t FACE = 512, HALF_TILE = 1024, ROW = 32;  // bf16 tile: face bytes, two faces, 16-value face row

constexpr uint32_t al64(uint32_t x) { return (x + 63u) & ~63u; }

// L1 layout from the 64 B aligned CB base
// KC1 rows of one face of one source tile, read from the 64 B line holding the first one
constexpr uint32_t STG_HALF = al64(KC1 * ROW + ROW);
constexpr bool TILE_SRC = MODE == WRITE || MODE == WINDOW || MODE == CHAIN;
constexpr uint32_t STG_TILE = CT * 2 * STG_HALF;
constexpr uint32_t STG_PAGES = CONV_PAGES * BF16_PAGE;
constexpr uint32_t STG_BYTES = MODE == CHAIN
                                   ? (STG_TILE > STG_PAGES ? STG_TILE : STG_PAGES)
                                   : (TILE_SRC ? STG_TILE : (MODE == WRITE_SNAP ? KC1 * al64(RM_ROW) : STG_PAGES));
constexpr uint32_t OFF_REC = 0;                            // BATCH fp32 pages
constexpr uint32_t OFF_STG = OFF_REC + BATCH * FP32_PAGE;  // conv source rows
constexpr uint32_t OFF_ASM = OFF_STG + al64(STG_BYTES);    // CONV_PAGES block tiles, or BATCH carry tiles
constexpr uint32_t ASM_BYTES = (MODE == RESTORE ? BATCH : CONV_PAGES) * BF16_PAGE;
constexpr uint32_t OFF_CNT = OFF_ASM + ASM_BYTES;  // CHAIN: the 64 B line of the counts row holding row u's count
constexpr uint32_t L1_BYTES = MODE == ZERO ? FP32_PAGE : OFF_CNT + 64;  // ZERO: one zero page

constexpr uint32_t ARG_PAGES = 1;
constexpr uint32_t ARG_LAYERS =
    MODE == WINDOW ? 1 + 4 * WIN_MAX : (MODE == CHAIN ? 3 + CH_MAX * CH_WORDS : ARG_PAGES + G * (NV + 1));

static_assert(MODE <= CHAIN, "mode");
static_assert(MODE == ZERO || KC1 * CT <= CONV_PAGES * 32, "the conv history fits its pages");
static_assert(MODE == ZERO || KC1 <= 16, "carry rows lie in the top two faces of a tile");
static_assert(L1_BYTES + 64 <= CB_BYTES, "CB size");
static_assert(MODE != ZERO || (KV_TPB % ZERO_CHUNK == 0 && ZERO_CHUNK >= 1), "zero chunks");

inline void zero_l1(uint32_t addr, uint32_t bytes) {
    volatile tt_l1_ptr uint32_t* p = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(addr);
    for (uint32_t i = 0; i < bytes / 4; ++i) {
        p[i] = 0;
    }
}

// one 16-value bf16 face row
inline void copy_row(uint32_t dst, uint32_t src) {
    volatile tt_l1_ptr uint32_t* d = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dst);
    volatile tt_l1_ptr uint32_t* s = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(src);
    for (uint32_t i = 0; i < ROW / 4; ++i) {
        d[i] = s[i];
    }
}

// byte offset in a bf16 tile of the first value of tile row rr, columns 0-15 (+ FACE: columns 16-31)
constexpr uint32_t tile_row(uint32_t rr) { return (rr >= 16 ? HALF_TILE : 0) + (rr % 16) * ROW; }

// The KC1 x C carry -> CONV_PAGES history tiles at q0 of ``view``. Tile source (WRITE, WINDOW): tile page0 + t holds
// carry row r as tile row r0 + r (r0 + KC1 <= 16); row-major source (WRITE_SNAP): carry row r is page page0 + r.
template <typename Src, typename View>
inline void write_history(
    const Src& carry, uint32_t page0, uint32_t r0, const View& view, uint32_t q0, uint32_t stg, uint32_t asm_) {
    const uint32_t start = (r0 * ROW) & ~63u, skew = r0 * ROW - start;
    if constexpr (TILE_SRC) {
        const uint32_t len = al64(skew + KC1 * ROW);
        for (uint32_t t = 0; t < CT; ++t) {
            const uint64_t src = carry.get_noc_addr(page0 + t);
            noc_async_read(src + start, stg + t * 2 * STG_HALF, len);
            noc_async_read(src + FACE + start, stg + t * 2 * STG_HALF + STG_HALF, len);
        }
    } else {
        for (uint32_t r = 0; r < KC1; ++r) {
            noc_async_read(carry.get_noc_addr(page0 + r), stg + r * al64(RM_ROW), RM_ROW);
        }
    }
    zero_l1(asm_, CONV_PAGES * BF16_PAGE);
    noc_async_read_barrier();
    // history row q = carry row q / CT, values 32 (q % CT) .. 32 (q % CT) + 31
    for (uint32_t q = 0; q < KC1 * CT; ++q) {
        const uint32_t r = q / CT, t = q % CT;
        uint32_t lo, hi;
        if constexpr (TILE_SRC) {
            lo = stg + t * 2 * STG_HALF + skew + r * ROW;
            hi = lo + STG_HALF;
        } else {
            lo = stg + r * al64(RM_ROW) + t * 2 * ROW;
            hi = lo + ROW;
        }
        const uint32_t dst = asm_ + (q / 32) * BF16_PAGE + tile_row(q % 32);
        copy_row(dst, lo);
        copy_row(dst + FACE, hi);
    }
    for (uint32_t d = 0; d < CONV_PAGES; ++d) {
        noc_async_write(asm_ + d * BF16_PAGE, view.get_noc_addr(q0 + d), BF16_PAGE);
    }
    noc_async_write_barrier();
}

// 16 fp32 pages of one head between two page spaces, BATCH pages per read / write round
template <typename Src, typename Dst>
inline void copy_head(const Src& src, uint32_t s0, const Dst& dst, uint32_t d0, uint32_t buf) {
    for (uint32_t p = 0; p < HEAD_PAGES; p += BATCH) {
        for (uint32_t i = 0; i < BATCH; ++i) {
            noc_async_read(src.get_noc_addr(s0 + p + i), buf + i * FP32_PAGE, FP32_PAGE);
        }
        noc_async_read_barrier();
        for (uint32_t i = 0; i < BATCH; ++i) {
            noc_async_write(buf + i * FP32_PAGE, dst.get_noc_addr(d0 + p + i), FP32_PAGE);
        }
        noc_async_write_barrier();
    }
}

}  // namespace

void kernel_main() {
    const uint32_t core = static_cast<uint32_t>(get_absolute_logical_y()) * NX + get_absolute_logical_x();
    if (core >= N_CORES) {
        return;
    }
    constexpr auto v32_args = TensorAccessorArgs<CTA_ACCESSORS>();
    constexpr auto v16_args = TensorAccessorArgs<v32_args.next_compile_time_args_offset()>();
    constexpr auto rec_args = TensorAccessorArgs<v16_args.next_compile_time_args_offset()>();
    constexpr auto conv_args = TensorAccessorArgs<rec_args.next_compile_time_args_offset()>();
    constexpr auto kv_args = TensorAccessorArgs<conv_args.next_compile_time_args_offset()>();
    const uint32_t base = al64(get_write_ptr(CB_L1));

    if constexpr (MODE == ZERO) {
        const uint32_t n_blocks = get_common_arg_val<uint32_t>(KV_BUFS);
        constexpr uint32_t chunks = KV_TPB / ZERO_CHUNK;
        const uint32_t page = kv_args.get_aligned_page_size();
        zero_l1(base, al64(page));
        const uint32_t items = n_blocks * KV_BUFS * chunks;
        for (uint32_t i = core; i < items; i += N_CORES) {
            const uint32_t ch = i % chunks;
            const uint32_t buf = (i / chunks) % KV_BUFS;
            const uint32_t blk = get_common_arg_val<uint32_t>(KV_BUFS + 1 + i / (chunks * KV_BUFS));
            const auto kv = TensorAccessor(kv_args, get_common_arg_val<uint32_t>(buf));
            const uint32_t p0 = blk * KV_TPB + ch * ZERO_CHUNK;
            for (uint32_t p = 0; p < ZERO_CHUNK; ++p) {
                noc_async_write(base, kv.get_noc_addr(p0 + p), page);
            }
        }
        noc_async_write_barrier();
        return;
    }

    if constexpr (MODE == WINDOW) {
        const uint32_t n = get_common_arg_val<uint32_t>(0);
        for (uint32_t i = core; i < n * NL; i += N_CORES) {
            const uint32_t e = 1 + 4 * (i / NL), la = ARG_LAYERS + 5 * (i % NL);
            const uint32_t g = get_common_arg_val<uint32_t>(e + 2);
            if (get_common_arg_val<uint32_t>(la) != g) {
                continue;
            }
            const uint32_t u = get_common_arg_val<uint32_t>(e), t = get_common_arg_val<uint32_t>(e + 1);
            const uint32_t page = get_common_arg_val<uint32_t>(e + 3);
            const bool in_v = page < CONV_TOTAL;
            const auto view = TensorAccessor(v16_args, get_common_arg_val<uint32_t>(in_v ? la + 4 : la + 3));
            const auto win = TensorAccessor(conv_args, get_common_arg_val<uint32_t>(la + 2));
            write_history(win, u * CT, t + 1, view, in_v ? page : page - CONV_TOTAL, base + OFF_STG, base + OFF_ASM);
        }
        return;
    }

    if constexpr (MODE == CHAIN) {
        const uint32_t n = get_common_arg_val<uint32_t>(0);
        const auto res = TensorAccessor(kv_args, get_common_arg_val<uint32_t>(1));
        const bool select = get_common_arg_val<uint32_t>(2) != 0;
        constexpr uint32_t PARTS = NL * (NV + 1);
        for (uint32_t i = core; i < n * PARTS; i += N_CORES) {
            const uint32_t e = 3 + CH_WORDS * (i / PARTS), l = (i % PARTS) / (NV + 1), part = i % (NV + 1);
            const uint32_t la = ARG_LAYERS + 5 * l;
            if (get_common_arg_val<uint32_t>(la) != get_common_arg_val<uint32_t>(e + 1)) {
                continue;
            }
            const uint32_t u = get_common_arg_val<uint32_t>(e);
            if (select) {
                const uint32_t line = (4 * u) & ~63u;
                noc_async_read(res.get_noc_addr(RES_COUNTS) + line, base + OFF_CNT, 64);
                noc_async_read_barrier();
                const uint32_t c = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base + OFF_CNT)[(4 * u - line) / 4];
                if (c < 1 || c > 32 || !((get_common_arg_val<uint32_t>(e + 3) >> (c - 1)) & 1u)) {
                    continue;
                }
            }
            const uint32_t k_addr = get_common_arg_val<uint32_t>(la + 3), v_addr = get_common_arg_val<uint32_t>(la + 4);
            const uint32_t sp = get_common_arg_val<uint32_t>(e + 4 + part);
            const uint32_t dp = get_common_arg_val<uint32_t>(e + 4 + (NV + 1) + part);
            if (part < NV) {
                const auto src = TensorAccessor(v32_args, sp < POOL1 ? k_addr : v_addr);
                const auto dst = TensorAccessor(v32_args, dp < POOL1 ? k_addr : v_addr);
                copy_head(src, sp < POOL1 ? sp : sp - POOL1, dst, dp < POOL1 ? dp : dp - POOL1, base + OFF_REC);
                continue;
            }
            const auto dst = TensorAccessor(v16_args, dp < CONV_TOTAL ? v_addr : k_addr);
            const uint32_t q0 = dp < CONV_TOTAL ? dp : dp - CONV_TOTAL;
            const uint32_t t = get_common_arg_val<uint32_t>(e + 2);
            if (t != NO_T) {
                const auto win = TensorAccessor(conv_args, get_common_arg_val<uint32_t>(la + 2));
                write_history(win, u * CT, t + 1, dst, q0, base + OFF_STG, base + OFF_ASM);
                continue;
            }
            const auto src = TensorAccessor(v16_args, sp < CONV_TOTAL ? v_addr : k_addr);
            const uint32_t qs = sp < CONV_TOTAL ? sp : sp - CONV_TOTAL;
            for (uint32_t d = 0; d < CONV_PAGES; ++d) {
                noc_async_read(src.get_noc_addr(qs + d), base + OFF_STG + d * BF16_PAGE, BF16_PAGE);
            }
            noc_async_read_barrier();
            for (uint32_t d = 0; d < CONV_PAGES; ++d) {
                noc_async_write(base + OFF_STG + d * BF16_PAGE, dst.get_noc_addr(q0 + d), BF16_PAGE);
            }
            noc_async_write_barrier();
        }
        return;
    }

    const uint32_t snap = get_common_arg_val<uint32_t>(0);
    if constexpr (MODE == RESTORE) {
        zero_l1(base + OFF_ASM, ASM_BYTES);
    }
    for (uint32_t i = core; i < NL * (NV + 1); i += N_CORES) {
        const uint32_t l = i / (NV + 1), part = i % (NV + 1);
        const uint32_t la = ARG_LAYERS + 5 * l;
        const uint32_t g = get_common_arg_val<uint32_t>(la);
        const uint32_t rec_addr = get_common_arg_val<uint32_t>(la + 1);
        const uint32_t conv_addr = get_common_arg_val<uint32_t>(la + 2);
        const uint32_t k_addr = get_common_arg_val<uint32_t>(la + 3);
        const uint32_t v_addr = get_common_arg_val<uint32_t>(la + 4);
        const uint32_t page = get_common_arg_val<uint32_t>(ARG_PAGES + g * (NV + 1) + part);

        if (part < NV) {
            const bool in_k = page < POOL1;
            const auto view = TensorAccessor(v32_args, in_k ? k_addr : v_addr);
            const auto rec = TensorAccessor(rec_args, rec_addr);
            const uint32_t vp = in_k ? page : page - POOL1;
            const uint32_t rp = (MODE == WRITE_SNAP ? snap * NV * HEAD_PAGES : 0) + part * HEAD_PAGES;
            if constexpr (MODE == RESTORE) {
                copy_head(view, vp, rec, rp, base + OFF_REC);
            } else {
                copy_head(rec, rp, view, vp, base + OFF_REC);
            }
            continue;
        }

        const bool in_v = page < CONV_TOTAL;
        const auto view = TensorAccessor(v16_args, in_v ? v_addr : k_addr);
        const uint32_t q0 = in_v ? page : page - CONV_TOTAL;
        const auto carry = TensorAccessor(conv_args, conv_addr);
        const uint32_t stg = base + OFF_STG, asm_ = base + OFF_ASM;
        if constexpr (MODE == RESTORE) {
            for (uint32_t d = 0; d < CONV_PAGES; ++d) {
                noc_async_read(view.get_noc_addr(q0 + d), stg + d * BF16_PAGE, BF16_PAGE);
            }
            noc_async_read_barrier();
            // carry tile t, row r = value r * C + 32 t .. of the history = history row r * CT + t
            for (uint32_t t = 0; t < CT; ++t) {
                const uint32_t slot = t % BATCH;
                if (slot == 0 && t > 0) {
                    noc_async_write_barrier();
                }
                const uint32_t tile = asm_ + slot * BF16_PAGE;
                for (uint32_t r = 0; r < KC1; ++r) {
                    const uint32_t q = r * CT + t;
                    const uint32_t src = stg + (q / 32) * BF16_PAGE + tile_row(q % 32);
                    copy_row(tile + r * ROW, src);
                    copy_row(tile + FACE + r * ROW, src + FACE);
                }
                noc_async_write(tile, carry.get_noc_addr(t), BF16_PAGE);
            }
            noc_async_write_barrier();
            continue;
        }
        write_history(carry, MODE == WRITE_SNAP ? snap * KC1 : 0, 0, view, q0, stg, asm_);
    }
}
