// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Split decode attention, reader (NCRISC) (fused_decode/split_attn.py, sda_common.hpp): one core per (kv head g,
// slice s), fused into the decode QKV program. Pushes:
//   CB_POS: the position (for this core's writer);
//   CB_AUX: [scaler (1.0 in row 0 of each face), the window's first-chunk mask (columns before lo % 32 cut), the
//            position's last-chunk mask (columns after p % 32 cut), both (one chunk), the sink-column mask (keeps
//            column 0), the sink tile (slice 0: column 0 of rows 0..7 = heads 8 g .. 8 g + 7's sink / scale), zero];
//   CB_ONES: an all-ones tile in the K / V format (from the op's consts tensor);
//   CB_K / CB_V: the slice's chunks in groups of G ([32 positions, 32 dims] x 2 per chunk); the groups entirely
//                before the position (already in the cache) while QKV runs, as far as the CBs hold them;
//   CB_Q: heads 8 g .. 8 g + 7's Q rows (rows 0..7 of 2 BF16 tiles, rows 8..31 zero), after the QKV writers signal
//         (SDA_WAIT_N increments of program semaphore SDA_WAIT_SEM: Q and the position's K / V rows are written);
//   CB_MX: M[r] = the max over the kv head's active slices of their row max m[r] (column 0 of rows 0..7), once the
//          slices' writers have all-gathered their m into CB_MG (SDA_M_SEM);
//   leader (slice 0): CB_FOLD and, once the active slices' scaled states are in its pack tiles (SDA_P_SEM), CB_OPACK
//          (this kernel zeroes the unused slots of the last pack tile).
//
// common runtime args: [q_addr, k_addr, v_addr, pos_addr, pt_addr, sink_addr, consts_addr, core index table
// (SDA_NIDX words: one byte per grid core, row-major over a grid SDA_GX wide, 0xFF = not in the op)]

// Size-optimised: the binaries are written to every core range of the (fused QKV) program at each launch.
#pragma GCC optimize("Os")

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "api/debug/kernel_profiler.hpp"
#include "sda_common.hpp"

using namespace sda;

namespace {
inline uint32_t core_index(uint32_t base) {
    const uint32_t c = static_cast<uint32_t>(get_absolute_logical_y()) * SDA_GX + get_absolute_logical_x();
    return (get_common_arg_val<uint32_t>(base + c / 4) >> (8 * (c % 4))) & 0xFF;
}
inline uint32_t f32_key(uint32_t x) { return (x & 0x80000000u) ? ~x : (x | 0x80000000u); }
inline uint32_t f32_unkey(uint32_t k) { return (k & 0x80000000u) ? (k & 0x7FFFFFFFu) : ~k; }
}  // namespace

void kernel_main() {
    const uint32_t q_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t k_addr = get_common_arg_val<uint32_t>(1);
    const uint32_t v_addr = get_common_arg_val<uint32_t>(2);
    const uint32_t pos_addr = get_common_arg_val<uint32_t>(3);
    const uint32_t pt_addr = get_common_arg_val<uint32_t>(4);
    const uint32_t sink_addr = get_common_arg_val<uint32_t>(5);
    const uint32_t consts_addr = get_common_arg_val<uint32_t>(6);

    constexpr uint32_t nkv = get_compile_time_arg_val(0);     // kv heads of the cache
    constexpr uint32_t block = get_compile_time_arg_val(1);   // positions per page-table block
    constexpr uint32_t window = get_compile_time_arg_val(2);  // 0: full attention
    constexpr uint32_t kv_tile_bytes = get_compile_time_arg_val(3);
    constexpr uint32_t pt_page_bytes = get_compile_time_arg_val(4);
    constexpr uint32_t S = get_compile_time_arg_val(5);  // slices per kv head
    constexpr uint32_t min_chunks = get_compile_time_arg_val(6);
    constexpr uint32_t cap_groups = get_compile_time_arg_val(7);  // groups CB_K / CB_V hold
    constexpr auto q_args = TensorAccessorArgs<8>();
    constexpr auto k_args = TensorAccessorArgs<q_args.next_compile_time_args_offset()>();
    constexpr auto v_args = TensorAccessorArgs<k_args.next_compile_time_args_offset()>();
    constexpr auto pos_args = TensorAccessorArgs<v_args.next_compile_time_args_offset()>();
    constexpr auto pt_args = TensorAccessorArgs<pos_args.next_compile_time_args_offset()>();
    constexpr auto sink_args = TensorAccessorArgs<pt_args.next_compile_time_args_offset()>();
    constexpr auto consts_args = TensorAccessorArgs<sink_args.next_compile_time_args_offset()>();

    constexpr uint32_t tile = 2048;  // BF16 tile
    constexpr uint32_t tile32 = 4096;
    constexpr uint32_t tpb = block / 32;
    const auto s_q = TensorAccessor(q_args, q_addr, tile);
    const auto s_k = TensorAccessor(k_args, k_addr, kv_tile_bytes);
    const auto s_v = TensorAccessor(v_args, v_addr, kv_tile_bytes);
    const auto s_pos = TensorAccessor(pos_args, pos_addr, 64);
    const auto s_pt = TensorAccessor(pt_args, pt_addr, pt_page_bytes);
    const auto s_sink = TensorAccessor(sink_args, sink_addr, tile);
    const auto s_consts = TensorAccessor(consts_args, consts_addr, kv_tile_bytes);

    const uint32_t ci = core_index(7);
    const uint32_t g = ci / S, s = ci % S;

    // the position (64-byte DRAM read), also to this core's writer
    cb_reserve_back(CB_SCR, 1);
    const uint32_t scr = get_write_ptr(CB_SCR);  // [pos (64 B) | page-table entries]
    noc_async_read(s_pos.get_noc_addr(0), scr, 64);
    // the Q tiles' rows 8..31 stay zero (the 8 heads land in rows 0..7 after QKV)
    cb_reserve_back(CB_Q, 2);
    const uint32_t ql1 = get_write_ptr(CB_Q);
    const uint64_t zeros = get_noc_addr(MEM_ZEROS_BASE);
    for (uint32_t z = 0; z < 2 * tile; z += MEM_ZEROS_SIZE) {
        noc_async_read(zeros, ql1 + z, MEM_ZEROS_SIZE);
    }
    noc_async_read_barrier();
    invalidate_l1_cache();
    const int32_t p = reinterpret_cast<volatile tt_l1_ptr int32_t*>(scr)[0];
    {
        cb_reserve_back(CB_POS, 1);
        reinterpret_cast<volatile tt_l1_ptr int32_t*>(get_write_ptr(CB_POS))[0] = p;
        cb_push_back(CB_POS, 1);
    }
    const Split sp = split(p, window, S, min_chunks, s);
    if (sp.j1 == sp.j0) {
        return;  // idle slice (the QKV writers' increments may still land here: harmless)
    }
    const bool leader = s == 0;

    // only this slice's page-table entries [e0, e1), read from the 64-byte boundary at or below e0
    const uint32_t first_pos = sp.c0 + 32 * sp.j0, last_pos = sp.c0 + 32 * sp.j1 - 1;
    const uint32_t e0 = first_pos / block, e1 = last_pos / block + 1;
    const uint32_t a0 = (e0 * 4) & ~63u;
    const uint32_t a1 = (e1 * 4 + 63) & ~63u;
    noc_async_read(s_pt.get_noc_addr(0) + a0, scr + 64, (a1 < pt_page_bytes ? a1 : pt_page_bytes) - a0);
    const uint32_t pt_first = a0 / 4;
    const uint32_t nc = sp.j1 - sp.j0;
    const uint32_t n_groups = (nc + G - 1) / G;

    // aux tiles (BF16): zero fill (scaler, the zero tile), the sink rows; the ones tile
    cb_reserve_back(CB_AUX, AUX_TILES);
    const uint32_t aux = get_write_ptr(CB_AUX);
    for (uint32_t z = 0; z < tile; z += MEM_ZEROS_SIZE) {
        noc_async_read(zeros, aux + AUX_SCALER * tile + z, MEM_ZEROS_SIZE);
        noc_async_read(zeros, aux + AUX_ZERO * tile + z, MEM_ZEROS_SIZE);
    }
    if (leader) {
        // heads 8 g .. 8 g + 7 = rows 8 (g % 4) .. of the sink tensor's tile g / 4 -> rows 0..7
        for (uint32_t f = 0; f < 2; ++f) {
            noc_async_read(s_sink.get_noc_addr(g / 4) + rows8(g % 4, f), aux + AUX_SINK * tile + rows8(0, f), 256);
        }
    }
    cb_reserve_back(CB_ONES, 1);
    noc_async_read(s_consts.get_noc_addr(0), get_write_ptr(CB_ONES), kv_tile_bytes);
    noc_async_read_barrier();
    // masks: rows 0..7 (the heads) of a mask tile, BF16 -1e30 at the cut columns (outside [cut, keep))
    auto mask = [&](uint32_t t, uint32_t cut, uint32_t keep) {
        uint32_t w[2][8];
        for (uint32_t h = 0; h < 2; ++h) {
            for (uint32_t i = 0; i < 8; ++i) {
                const uint32_t c0 = h * 16 + 2 * i, c1 = c0 + 1;
                const uint32_t lo = (c0 < cut || c0 >= keep) ? 0xF14Au : 0u;
                const uint32_t hi = (c1 < cut || c1 >= keep) ? 0xF14Au : 0u;
                w[h][i] = lo | (hi << 16);
            }
        }
        for (uint32_t f = 0; f < 2; ++f) {
            volatile tt_l1_ptr uint32_t* m = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(aux + t * tile + f * 512);
            for (uint32_t row = 0; row < 8; ++row) {
                for (uint32_t i = 0; i < 8; ++i) {
                    m[row * 8 + i] = w[f][i];
                }
            }
        }
    };
    const uint32_t cut = sp.lo & 31u, keep = (static_cast<uint32_t>(p) & 31u) + 1;
    if (sp.j0 == 0 && cut) {
        mask(AUX_FIRST, cut, 32);
    }
    if (sp.j1 == sp.n && keep < 32) {
        mask(AUX_LAST, 0, keep);
    }
    if (nc == 1 && sp.j0 == 0 && sp.j1 == sp.n) {
        mask(AUX_SINGLE, cut, keep);
    }
    if (leader) {
        mask(AUX_SINKCOL, 0, 1);
    }
    volatile tt_l1_ptr uint16_t* sc = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(aux + AUX_SCALER * tile);
    for (uint32_t f = 0; f < 4; ++f) {
        for (uint32_t c = 0; c < 16; ++c) {
            sc[f * 256 + c] = 0x3F80;
        }
    }
    invalidate_l1_cache();
    cb_push_back(CB_AUX, AUX_TILES);
    cb_push_back(CB_ONES, 1);

    // K / V groups pipelined (up to cap_groups in flight, NOC transaction id 1 + slot), pushed in order as each lands;
    // Q as soon as the QKV writers have signalled (the group holding the position waits for it too)
    volatile tt_l1_ptr int32_t* pt = reinterpret_cast<volatile tt_l1_ptr int32_t*>(scr + 64);
    const uint32_t k_base = get_write_ptr(CB_K), v_base = get_write_ptr(CB_V);  // fresh CBs: slot 0
    constexpr uint32_t group_pages = 2 * G;
    auto issue_group = [&](uint32_t gi) {
        const uint32_t slot = gi % cap_groups;
        noc_async_read_set_trid(1 + slot);
        const uint32_t kl1 = k_base + slot * group_pages * kv_tile_bytes,
                       vl1 = v_base + slot * group_pages * kv_tile_bytes;
        const uint32_t j0 = gi * G, j1 = j0 + G < nc ? j0 + G : nc;
        for (uint32_t j = j0; j < j1; ++j) {
            const uint32_t pp = sp.c0 + (sp.j0 + j) * 32;
            const uint32_t blk = static_cast<uint32_t>(pt[pp / block - pt_first]);
            const uint32_t t0 = ((blk * nkv + g) * tpb + (pp % block) / 32) * 2;
            for (uint32_t d = 0; d < 2; ++d) {
                noc_async_read(s_k.get_noc_addr(t0 + d), kl1 + (2 * (j - j0) + d) * kv_tile_bytes, kv_tile_bytes);
                noc_async_read(s_v.get_noc_addr(t0 + d), vl1 + (2 * (j - j0) + d) * kv_tile_bytes, kv_tile_bytes);
            }
        }
        noc_async_read_set_trid(0);
    };
    // a group is old when its last chunk ends before the position (written by this step's QKV)
    auto old_group = [&](uint32_t gi) {
        const uint32_t j_last = (gi + 1) * G < nc ? (gi + 1) * G - 1 : nc - 1;
        return sp.c0 + (sp.j0 + j_last) * 32 + 31 < static_cast<uint32_t>(p);
    };
    volatile tt_l1_ptr uint32_t* qkv_sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SDA_WAIT_SEM));
    auto kv_ready = [&]() {
        invalidate_l1_cache();
        return *qkv_sem >= SDA_WAIT_N;
    };
    bool q_done = false;
    auto read_q = [&]() {
        // heads 8 g .. 8 g + 7: rows 8 (g % 4) .. of tile row g / 4 of the [heads, 64] Q tensor -> rows 0..7
        noc_async_read_set_trid(15);
        for (uint32_t d = 0; d < 2; ++d) {
            const uint64_t src = s_q.get_noc_addr(2 * (g / 4) + d);
            for (uint32_t f = 0; f < 2; ++f) {
                noc_async_read(src + rows8(g % 4, f), ql1 + d * tile + rows8(0, f), 256);
            }
        }
        noc_async_read_set_trid(0);
        noc_async_read_barrier_with_trid(15);
        cb_push_back(CB_Q, 2);
        q_done = true;
    };
    uint32_t pushed = 0;
    uint32_t issued = 0;
    while (pushed < n_groups) {
        while (issued < n_groups && issued - pushed < cap_groups && (q_done || old_group(issued)) &&
               cb_pages_reservable_at_back(CB_K, (issued - pushed + 1) * group_pages) &&
               cb_pages_reservable_at_back(CB_V, (issued - pushed + 1) * group_pages)) {  // V frees later than K
            issue_group(issued++);
        }
        if (!q_done && kv_ready()) {
            read_q();
            continue;
        }
        if (issued > pushed) {
            noc_async_read_barrier_with_trid(1 + pushed % cap_groups);
            cb_push_back(CB_K, group_pages);
            cb_push_back(CB_V, group_pages);
            ++pushed;
        } else if (!q_done) {
            while (!kv_ready()) {  // nothing in flight: the next group holds the position
            }
            read_q();
        }
        // else: the CBs are full (compute frees a slot)
    }
    if (!q_done) {
        while (!kv_ready()) {
        }
        read_q();
    }

    // leader: the fold matrix, the unused slots of the last pack tile zeroed (their O / L enter the fold matmul)
    const uint32_t K = (sp.active + 3) / 4;
    uint32_t opack = 0;
    if (leader) {
        cb_reserve_back(CB_FOLD, 1);
        cb_reserve_back(CB_OPACK, 3 * K);
        const uint32_t fold = get_write_ptr(CB_FOLD);
        opack = get_write_ptr(CB_OPACK);
        for (uint32_t z = 0; z < tile32; z += MEM_ZEROS_SIZE) {
            noc_async_read(zeros, fold + z, MEM_ZEROS_SIZE);
        }
        for (uint32_t i = sp.active - 4 * (K - 1); i < 4; ++i) {
            for (uint32_t t = 0; t < 3; ++t) {
                for (uint32_t f = 0; f < 2; ++f) {
                    noc_async_read(zeros, opack + (3 * (K - 1) + t) * tile32 + rows8_32(i, f), 512);
                }
            }
        }
        noc_async_read_barrier();
        // F[r, 8 i + r] = 1 (FP32): rows r < 8 in face 0 (columns < 16) / face 1
        volatile tt_l1_ptr uint32_t* fw = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(fold);
        for (uint32_t r = 0; r < 8; ++r) {
            for (uint32_t i = 0; i < 4; ++i) {
                const uint32_t c = 8 * i + r;
                fw[(c / 16) * 256 + r * 16 + (c % 16)] = 0x3F800000u;
            }
        }
        cb_push_back(CB_FOLD, 1);
    }

    // M = the max over the active slices' row max (their writers put slice t's m[0..7] in slot t of CB_MG)
    {
        DeviceZoneScopedN("sda-rd-m");
        volatile tt_l1_ptr uint32_t* msem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SDA_M_SEM));
        while (true) {
            invalidate_l1_cache();
            if (*msem >= sp.active) {
                break;
            }
        }
        volatile tt_l1_ptr uint32_t* mg = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(CB_MG));
        uint32_t mk[8];
        for (uint32_t r = 0; r < 8; ++r) {
            mk[r] = f32_key(mg[r]);
        }
        for (uint32_t t = 1; t < sp.active; ++t) {
            for (uint32_t r = 0; r < 8; ++r) {
                const uint32_t kk = f32_key(mg[8 * t + r]);
                mk[r] = kk > mk[r] ? kk : mk[r];
            }
        }
        cb_reserve_back(CB_MX, 1);
        volatile tt_l1_ptr uint32_t* mx = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(CB_MX));
        for (uint32_t r = 0; r < 8; ++r) {
            mx[r * 16] = f32_unkey(mk[r]);  // column 0 of row r (face 0)
        }
        cb_push_back(CB_MX, 1);
        noc_semaphore_set(msem, 0);
    }

    if (leader) {
        // the active slices' scaled states are in the pack tiles (their writers count SDA_P_SEM)
        DeviceZoneScopedN("sda-rd-pwait");
        volatile tt_l1_ptr uint32_t* psem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SDA_P_SEM));
        while (true) {
            invalidate_l1_cache();
            if (*psem >= sp.active) {
                break;
            }
        }
        cb_push_back(CB_OPACK, 3 * K);
        noc_semaphore_set(psem, 0);
    }
}
