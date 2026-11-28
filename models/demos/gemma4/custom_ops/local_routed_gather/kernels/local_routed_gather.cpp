// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
// LocalRoutedGather: one data-movement kernel on N_CORES cores for a prefill chunk whose hidden rows and top-k
// routing (values, ids) are replicated on every chip of a whole-expert TP mesh. The cores split the chunk's tokens
// (contiguous 32-token units); together they lay this chip's routed (token, slot) pairs out expert-major:
// 1. every core scans the top-k ids of its tokens and, per pair routed to one of this chip's E_LOC experts, records
//    (local expert, its rank among the core's pairs of that expert); it counts its pairs per local expert;
// 2. the cores send their counts to the leader (core 0), which lays out the tile-aligned expert regions (prefix sum
//    of the totals rounded up to 32, in local-expert order), writes `counts` / `offsets` (indexed by GLOBAL expert
//    id, zero for experts not on this chip) and sends every core its first buffer row per local expert (region start
//    + the pairs of that expert on the cores before it);
// 3. every core copies its tokens' hidden rows to their buffer rows (a token with several local pairs is read once and
//    written once per pair, NR tokens per batch) and writes per token the combine's row index (the buffer row of each
//    local slot, SKIP_ROW elsewhere) and its routing weights (the router's value on local slots, zero elsewhere).
// Rows of the buffer outside the regions and the padding rows of a region are never written (the GeGLU op reads
// whole regions' real rows only and zero-fills a partial tile-row). Tokens >= t_real (the chunk's padding up to its
// row bucket T) have no pairs and their ids / values tiles past the tensors' last tile are not read.
// Semaphores are reset after use so the program replays in a trace.
//
// compile-time args: [cb_scratch, E, E_LOC, K, T, H, M, N_CORES, UNITS_MAX, sem_id, leader_noc_x,
//                     leader_noc_y, NR, ri_page, w_page, vec_page, skip_row]
//                    + TensorAccessorArgs(hidden, ids, values, idx, buffer, counts, offsets, row_index, weights)
// runtime args: [hidden, ids, values, idx, buffer, counts, offsets, row_index, weights addresses, me, unit_start,
//                unit_count, t_real, (noc_x, noc_y) x N_CORES on the leader only]
#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/tensor/noc_traits.h"

constexpr uint32_t CB_SCRATCH = get_compile_time_arg_val(0);
constexpr uint32_t E = get_compile_time_arg_val(1);
constexpr uint32_t E_LOC = get_compile_time_arg_val(2);
constexpr uint32_t K = get_compile_time_arg_val(3);
constexpr uint32_t T = get_compile_time_arg_val(4);
constexpr uint32_t H = get_compile_time_arg_val(5);
constexpr uint32_t M = get_compile_time_arg_val(6);
constexpr uint32_t N_CORES = get_compile_time_arg_val(7);
constexpr uint32_t UNITS_MAX = get_compile_time_arg_val(8);
constexpr uint32_t SEM_ID = get_compile_time_arg_val(9);
constexpr uint32_t LEADER_X = get_compile_time_arg_val(10);
constexpr uint32_t LEADER_Y = get_compile_time_arg_val(11);
constexpr uint32_t NR = get_compile_time_arg_val(12);
constexpr uint32_t RI_PAGE = get_compile_time_arg_val(13);
constexpr uint32_t W_PAGE = get_compile_time_arg_val(14);
constexpr uint32_t VEC_PAGE = get_compile_time_arg_val(15);
constexpr uint32_t SKIP_ROW = get_compile_time_arg_val(16);
constexpr auto hidden_args = TensorAccessorArgs<17>();
constexpr auto ids_args = TensorAccessorArgs<hidden_args.next_compile_time_args_offset()>();
constexpr auto values_args = TensorAccessorArgs<ids_args.next_compile_time_args_offset()>();
constexpr auto idx_args = TensorAccessorArgs<values_args.next_compile_time_args_offset()>();
constexpr auto buffer_args = TensorAccessorArgs<idx_args.next_compile_time_args_offset()>();
constexpr auto counts_args = TensorAccessorArgs<buffer_args.next_compile_time_args_offset()>();
constexpr auto offsets_args = TensorAccessorArgs<counts_args.next_compile_time_args_offset()>();
constexpr auto ri_args = TensorAccessorArgs<offsets_args.next_compile_time_args_offset()>();
constexpr auto w_args = TensorAccessorArgs<ri_args.next_compile_time_args_offset()>();

constexpr uint32_t TILE_BYTES = 2048;  // a 32x32 tile of 16-bit elements (the ids / values TILE pages)
constexpr uint32_t ROW_BYTES = H * 2;
constexpr uint32_t NONE = 0xFFFFFFFFu;
constexpr uint32_t ALIGN = 64;  // Blackhole DRAM transfers need 64-byte aligned L1 addresses
constexpr uint32_t up64(uint32_t v) { return (v + ALIGN - 1) & ~(ALIGN - 1); }
constexpr uint32_t CNT_BYTES = up64(E_LOC * 4);  // one core's per-expert word table (counts or bases)

// scratch layout (bytes, every region 64-byte aligned): global id -> local slot (256 x u8) | idx table | the ids
// tiles | the values tiles | per-pair codes ((local expert << 16) | rank, NONE off-chip) | my counts | my bases |
// leader: the cores' counts | the cores' bases | counts page | offsets page | NR row-index pages | NR weight pages
// | NR hidden rows
constexpr uint32_t OFF_LSLOT = 0;
constexpr uint32_t OFF_IDX = OFF_LSLOT + 256;
constexpr uint32_t OFF_IDS = OFF_IDX + up64(E_LOC * 4);
constexpr uint32_t OFF_VALS = OFF_IDS + UNITS_MAX * TILE_BYTES;
constexpr uint32_t OFF_CODE = OFF_VALS + UNITS_MAX * TILE_BYTES;
constexpr uint32_t OFF_CNT = OFF_CODE + up64(UNITS_MAX * 32 * K * 4);
constexpr uint32_t OFF_BASE = OFF_CNT + CNT_BYTES;
constexpr uint32_t OFF_SLOTS = OFF_BASE + CNT_BYTES;
constexpr uint32_t OFF_BASES = OFF_SLOTS + N_CORES * CNT_BYTES;
constexpr uint32_t OFF_VCNT = OFF_BASES + N_CORES * CNT_BYTES;
constexpr uint32_t OFF_VOFS = OFF_VCNT + up64(VEC_PAGE);
constexpr uint32_t OFF_RI = OFF_VOFS + up64(VEC_PAGE);
constexpr uint32_t OFF_W = OFF_RI + NR * up64(RI_PAGE);
constexpr uint32_t OFF_ROWS = OFF_W + NR * up64(W_PAGE);
constexpr uint32_t SCRATCH_BYTES = OFF_ROWS + NR * up64(ROW_BYTES);

static_assert(E <= 256, "the local-slot table is one byte per global id");
static_assert(E_LOC >= 1 && E_LOC <= 64, "the local expert codes keep the slot in 6 bits");
static_assert(K >= 1 && K <= 16, "the slots of a token sit in the first face column of the ids / values tiles");
static_assert(T % 32 == 0 && T <= 65536, "tile-aligned chunk; ranks fit 16 bits");
static_assert(VEC_PAGE >= E * 4 && RI_PAGE >= K * 4 && W_PAGE >= K * 2, "page sizes hold a page's elements");
static_assert(ROW_BYTES % ALIGN == 0, "every staged hidden row starts 64-byte aligned");

FORCE_INLINE uint32_t tile_elem(uint32_t r, uint32_t c) {  // element offset of (row r, col c) inside a 32x32 tile
    return ((r >> 4) * 2 + (c >> 4)) * 256 + (r & 15) * 16 + (c & 15);
}

void kernel_main() {
    const uint32_t hidden_addr = get_arg_val<uint32_t>(0);
    const uint32_t ids_addr = get_arg_val<uint32_t>(1);
    const uint32_t values_addr = get_arg_val<uint32_t>(2);
    const uint32_t idx_addr = get_arg_val<uint32_t>(3);
    const uint32_t buffer_addr = get_arg_val<uint32_t>(4);
    const uint32_t counts_addr = get_arg_val<uint32_t>(5);
    const uint32_t offsets_addr = get_arg_val<uint32_t>(6);
    const uint32_t ri_addr = get_arg_val<uint32_t>(7);
    const uint32_t w_addr = get_arg_val<uint32_t>(8);
    const uint32_t me = get_arg_val<uint32_t>(9);
    const uint32_t unit_start = get_arg_val<uint32_t>(10);
    const uint32_t unit_count = get_arg_val<uint32_t>(11);
    const uint32_t t_real = get_arg_val<uint32_t>(12);  // real tokens of the chunk (<= T)
    constexpr uint32_t COORDS = 13;
    const uint32_t scratch = get_write_ptr(CB_SCRATCH);
    volatile tt_l1_ptr uint32_t* sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SEM_ID));
    const uint32_t sem_addr = get_semaphore(SEM_ID);

    const auto s_hidden = TensorAccessor(hidden_args, hidden_addr, ROW_BYTES);
    const auto s_ids = TensorAccessor(ids_args, ids_addr, TILE_BYTES);
    const auto s_values = TensorAccessor(values_args, values_addr, TILE_BYTES);
    const auto s_idx = TensorAccessor(idx_args, idx_addr, E_LOC * 4);
    const auto s_buffer = TensorAccessor(buffer_args, buffer_addr, ROW_BYTES);
    const auto s_counts = TensorAccessor(counts_args, counts_addr, VEC_PAGE);
    const auto s_offsets = TensorAccessor(offsets_args, offsets_addr, VEC_PAGE);
    const auto s_ri = TensorAccessor(ri_args, ri_addr, RI_PAGE);
    const auto s_w = TensorAccessor(w_args, w_addr, W_PAGE);

    // 1. the local experts, my tokens' ids / values tiles, and the per-pair codes
    noc_async_read(s_idx.get_noc_addr(0), scratch + OFF_IDX, E_LOC * 4);
    for (uint32_t u = 0; u < unit_count && (unit_start + u) * 32 < t_real; ++u) {
        noc_async_read(s_ids.get_noc_addr(unit_start + u), scratch + OFF_IDS + u * TILE_BYTES, TILE_BYTES);
        noc_async_read(s_values.get_noc_addr(unit_start + u), scratch + OFF_VALS + u * TILE_BYTES, TILE_BYTES);
    }
    volatile tt_l1_ptr uint8_t* lslot = reinterpret_cast<volatile tt_l1_ptr uint8_t*>(scratch + OFF_LSLOT);
    for (uint32_t g = 0; g < 256; ++g) {
        lslot[g] = 0xFF;
    }
    noc_async_read_barrier();
    volatile tt_l1_ptr uint32_t* idx = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch + OFF_IDX);
    for (uint32_t e = 0; e < E_LOC; ++e) {
        if (idx[e] < E) {
            lslot[idx[e]] = static_cast<uint8_t>(e);
        }
    }
    volatile tt_l1_ptr uint32_t* cnt = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch + OFF_CNT);
    for (uint32_t e = 0; e < E_LOC; ++e) {
        cnt[e] = 0;
    }
    volatile tt_l1_ptr uint16_t* ids = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(scratch + OFF_IDS);
    volatile tt_l1_ptr uint32_t* code = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch + OFF_CODE);
    for (uint32_t u = 0; u < unit_count; ++u) {
        for (uint32_t r = 0; r < 32; ++r) {
            const uint32_t t = (unit_start + u) * 32 + r;
            const uint32_t p0 = (u * 32 + r) * K;
            for (uint32_t k = 0; k < K; ++k) {
                uint32_t c = NONE;
                if (t < t_real) {
                    const uint32_t g = ids[u * (TILE_BYTES / 2) + tile_elem(r, k)];
                    if (g < E) {
                        const uint32_t e = lslot[g];
                        if (e != 0xFF) {
                            c = (e << 16) | cnt[e];
                            cnt[e] = cnt[e] + 1;
                        }
                    }
                }
                code[p0 + k] = c;
            }
        }
    }

    // 2. counts -> leader; the leader lays out the regions and sends every core its first row per local expert
    volatile tt_l1_ptr uint32_t* slots = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch + OFF_SLOTS);
    volatile tt_l1_ptr uint32_t* bases = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch + OFF_BASES);
    volatile tt_l1_ptr uint32_t* base = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch + OFF_BASE);
    constexpr uint32_t CNT_WORDS = CNT_BYTES / 4;
    if (me != 0) {
        noc_async_write(
            scratch + OFF_CNT, get_noc_addr(LEADER_X, LEADER_Y, scratch + OFF_SLOTS + me * CNT_BYTES), CNT_BYTES);
        noc_async_write_barrier();
        noc_semaphore_inc(get_noc_addr(LEADER_X, LEADER_Y, sem_addr), 1);
        noc_async_atomic_barrier();
        noc_semaphore_wait_min(sem, 1);
        noc_semaphore_set(sem, 0);
    } else {
        for (uint32_t e = 0; e < E_LOC; ++e) {
            slots[e] = cnt[e];
        }
        if constexpr (N_CORES > 1) {
            noc_semaphore_wait_min(sem, N_CORES - 1);
            noc_semaphore_set(sem, 0);
        }
        volatile tt_l1_ptr uint32_t* vcnt = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch + OFF_VCNT);
        volatile tt_l1_ptr uint32_t* vofs = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch + OFF_VOFS);
        for (uint32_t i = 0; i < VEC_PAGE / 4; ++i) {
            vcnt[i] = 0;
            vofs[i] = 0;
        }
        uint32_t run = 0;
        for (uint32_t e = 0; e < E_LOC; ++e) {
            uint32_t acc = run;
            for (uint32_t c = 0; c < N_CORES; ++c) {
                bases[c * CNT_WORDS + e] = acc;
                acc += slots[c * CNT_WORDS + e];
            }
            const uint32_t total = acc - run;
            if (idx[e] < E) {
                vcnt[idx[e]] = total;
                vofs[idx[e]] = run;
            }
            run += (total + 31) & ~31u;
        }
        // run <= T * K + 31 * E_LOC <= M: every pair names a distinct (token, slot)
        noc_async_write(scratch + OFF_VCNT, s_counts.get_noc_addr(0), VEC_PAGE);
        noc_async_write(scratch + OFF_VOFS, s_offsets.get_noc_addr(0), VEC_PAGE);
        for (uint32_t e = 0; e < E_LOC; ++e) {
            base[e] = bases[e];
        }
        for (uint32_t c = 1; c < N_CORES; ++c) {
            const uint32_t nx = get_arg_val<uint32_t>(COORDS + 2 * c), ny = get_arg_val<uint32_t>(COORDS + 2 * c + 1);
            noc_async_write(scratch + OFF_BASES + c * CNT_BYTES, get_noc_addr(nx, ny, scratch + OFF_BASE), CNT_BYTES);
        }
        noc_async_write_barrier();
        for (uint32_t c = 1; c < N_CORES; ++c) {
            const uint32_t nx = get_arg_val<uint32_t>(COORDS + 2 * c), ny = get_arg_val<uint32_t>(COORDS + 2 * c + 1);
            noc_semaphore_inc(get_noc_addr(nx, ny, sem_addr), 1);
        }
        noc_async_atomic_barrier();
    }

    // 3. hidden rows -> their buffer rows; per token the combine's row-index page and weight page
    volatile tt_l1_ptr uint16_t* vals = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(scratch + OFF_VALS);
    const uint32_t t0 = unit_start * 32;
    const uint32_t n_tok = unit_count * 32;
    for (uint32_t j0 = 0; j0 < n_tok; j0 += NR) {
        const uint32_t n = (n_tok - j0 < NR) ? (n_tok - j0) : NR;
        for (uint32_t j = 0; j < n; ++j) {
            const uint32_t i = j0 + j;
            bool any = false;
            for (uint32_t k = 0; k < K; ++k) {
                any = any || code[i * K + k] != NONE;
            }
            if (any) {
                noc_async_read(s_hidden.get_noc_addr(t0 + i), scratch + OFF_ROWS + j * up64(ROW_BYTES), ROW_BYTES);
            }
        }
        noc_async_read_barrier();
        for (uint32_t j = 0; j < n; ++j) {
            const uint32_t i = j0 + j;
            const uint32_t u = i >> 5, r = i & 31;
            volatile tt_l1_ptr uint32_t* ri =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch + OFF_RI + j * up64(RI_PAGE));
            volatile tt_l1_ptr uint16_t* w =
                reinterpret_cast<volatile tt_l1_ptr uint16_t*>(scratch + OFF_W + j * up64(W_PAGE));
            for (uint32_t k = 0; k < K; ++k) {
                const uint32_t c = code[i * K + k];
                if (c != NONE) {
                    const uint32_t row = base[c >> 16] + (c & 0xFFFF);
                    ri[k] = row;
                    w[k] = vals[u * (TILE_BYTES / 2) + tile_elem(r, k)];
                    noc_async_write(scratch + OFF_ROWS + j * up64(ROW_BYTES), s_buffer.get_noc_addr(row), ROW_BYTES);
                } else {
                    ri[k] = SKIP_ROW;
                    w[k] = 0;
                }
            }
            noc_async_write(scratch + OFF_RI + j * up64(RI_PAGE), s_ri.get_noc_addr(t0 + i), RI_PAGE);
            noc_async_write(scratch + OFF_W + j * up64(W_PAGE), s_w.get_noc_addr(t0 + i), W_PAGE);
        }
        noc_async_write_barrier();
    }
}
