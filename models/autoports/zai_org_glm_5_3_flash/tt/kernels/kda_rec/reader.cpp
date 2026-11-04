// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// kda_rec reader (BRISC / NoC0): ttnn's reader_fused_recurrent_kda for decode (T = 1) with the items of ONE head per
// core (item = b * H + h for the core's head h, b = 0..B-1), plus the item's gate_out row and, once, the o_norm
// weight row. q / k / g rows go through a scratch tile into column 0 of their CB tiles, v / gate_out rows into row 0,
// beta's scalar is replicated over row 0; the CBs are zeroed once so everything else stays zero. Megakernel: waits
// for the kda_pre writers' hand-off (MK_REC_WAIT_SEM, count = runtime arg) before the first read. Its CB indices are
// used by no other phase, so no rewind at the phase end.
// Compile-time args (PH_CT0 + i): H, Kt, Vt, then the geometry (cols, rows, row tiles, col tiles) of q, k, v, g,
// beta, gate_out (24 values), then the tensor accessors of q, k, v, g, beta, gate_out, state, weight.
// Runtime args (ph_rt0 + i): head, items, q, k, v, g, beta, gate_out, state, weight addresses, hand-off count
// [, REC_CHAIN: the selector page address].
// REC_CHAIN (Stage 17 speculative verify, doc/ttnn/17-mtp-spec-decode/design.md §2.1): the rows are consecutive tokens
// of ONE user — row 0 starts from state row `sel` (word 0 of a [1, 32] uint32 DRAM page, read once before any state
// read), row b >= 1 from row b - 1 = the state this core's writer wrote one row earlier (it raises REC_CHAIN_SEM after
// that write landed; one increment outstanding at most). Writes are unchanged: row b's state -> row b (candidate b).

#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/mk/opt_size.hpp"
#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"
#include "api/core_local_mem.h"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/mk/phase_args.hpp"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/mk/mk_common.hpp"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/kda_rec/kda_rec_common.hpp"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/mk/probe.hpp"

namespace {

constexpr uint32_t FACE_BYTES = 16 * 16 * 4;
constexpr uint32_t FACE_ROW_BYTES = 16 * 4;

struct Geometry {
    uint32_t cols, rows, row_tiles, col_tiles;
};

struct Location {
    uint32_t page, row, col;
};

// Tile page and in-tile (row, col) of flat element index e.
Location locate(const Geometry& gm, uint32_t e) {
    const uint32_t col = e % gm.cols;
    const uint32_t rest = e / gm.cols;
    const uint32_t row = rest % gm.rows;
    const uint32_t outer = rest / gm.rows;
    return {(outer * gm.row_tiles + row / 32) * gm.col_tiles + col / 32, row % 32, col % 32};
}

// Byte offset of the 16-value face row holding (row, col) of a 32x32 fp32 tile.
uint32_t face_row_offset(uint32_t row, uint32_t col) {
    return ((row / 16) * 2 + col / 16) * FACE_BYTES + (row % 16) * FACE_ROW_BYTES;
}

template <uint32_t B>
constexpr Geometry geometry() {
    return {
        get_compile_time_arg_val(B),
        get_compile_time_arg_val(B + 1),
        get_compile_time_arg_val(B + 2),
        get_compile_time_arg_val(B + 3)};
}

}  // namespace

#if defined(REC_CHAIN) && defined(REC_S_FIRST)
#error "REC_CHAIN: the next row's state is written by this launch (no early state read)"
#endif
#if defined(REC_UG)
#define REC_UG_ROWS_OK (REC_UG == 1)
#else
#define REC_UG_ROWS_OK true
#endif

#if defined(REC_V2)
// Stage 14c v2 reader: per item ONE compact fp32 vector tile (kda_rec_common.hpp / kda_rec_v2_sfpu.hpp layout): q, k, g
// at their key-side positions, v / gate_out in the u-layout, beta in every lane of plane V2_BETA; the rows are read
// whole (2 face rows per tile) into scratch and scattered by this RISC; the o_norm weight once, u-layout, into CB_W.
// Same compile-time / runtime arguments as v1 (one head per core, REC_SPREAD 1).
void kernel_main() {
    using namespace kda_rec;
    constexpr uint32_t H = get_compile_time_arg_val(PH_CT0 + 0);
    constexpr uint32_t Kt = get_compile_time_arg_val(PH_CT0 + 1);
    constexpr uint32_t Vt = get_compile_time_arg_val(PH_CT0 + 2);
    constexpr Geometry gq = geometry<PH_CT0 + 3>();
    constexpr Geometry gk = geometry<PH_CT0 + 7>();
    constexpr Geometry gv = geometry<PH_CT0 + 11>();
    constexpr Geometry gg = geometry<PH_CT0 + 15>();
    constexpr Geometry gb = geometry<PH_CT0 + 19>();
    constexpr Geometry ggo = geometry<PH_CT0 + 23>();
    constexpr auto q_a = TensorAccessorArgs<PH_CT0 + 27>();
    constexpr auto k_a = TensorAccessorArgs<q_a.next_compile_time_args_offset()>();
    constexpr auto v_a = TensorAccessorArgs<k_a.next_compile_time_args_offset()>();
    constexpr auto g_a = TensorAccessorArgs<v_a.next_compile_time_args_offset()>();
    constexpr auto b_a = TensorAccessorArgs<g_a.next_compile_time_args_offset()>();
    constexpr auto go_a = TensorAccessorArgs<b_a.next_compile_time_args_offset()>();
    constexpr auto s_a = TensorAccessorArgs<go_a.next_compile_time_args_offset()>();
    constexpr auto w_a = TensorAccessorArgs<s_a.next_compile_time_args_offset()>();
    constexpr uint32_t KV = Kt * Vt;
    static_assert(REC_SPREAD == 1 && Kt == 4 && Vt == 4, "kda_rec v2: one head per core, head dim 128");

    const uint32_t hc = get_arg_val<uint32_t>(ph_rt0 + 0);
    const uint32_t h = hc;
    const uint32_t items = get_arg_val<uint32_t>(ph_rt0 + 1) & 0xFFFF;
    const uint32_t b0 = get_arg_val<uint32_t>(ph_rt0 + 1) >> 16;
    if (h >= H) {
        return;
    }
    const uint32_t tb = get_tile_size(CB_Q);
    const auto q_acc = TensorAccessor(q_a, get_arg_val<uint32_t>(ph_rt0 + 2), tb);
    const auto k_acc = TensorAccessor(k_a, get_arg_val<uint32_t>(ph_rt0 + 3), tb);
    const auto v_acc = TensorAccessor(v_a, get_arg_val<uint32_t>(ph_rt0 + 4), tb);
    const auto g_acc = TensorAccessor(g_a, get_arg_val<uint32_t>(ph_rt0 + 5), tb);
    const auto b_acc = TensorAccessor(b_a, get_arg_val<uint32_t>(ph_rt0 + 6), tb);
    const auto go_acc = TensorAccessor(go_a, get_arg_val<uint32_t>(ph_rt0 + 7), tb);
    const auto s_acc = TensorAccessor(s_a, get_arg_val<uint32_t>(ph_rt0 + 8), tb);
    const auto w_acc = TensorAccessor(w_a, get_arg_val<uint32_t>(ph_rt0 + 9), tb);

    Noc noc;
    CircularBuffer cvec(CB_Q), cs(CB_S0), cw(CB_W), cscr(CB_SCR);
    cscr.reserve_back(1);
    const uint32_t scratch = cscr.get_write_ptr();
    constexpr uint32_t ROW = 128 * 4;  // one 128-value fp32 row in scratch
    // the two face rows of each of the 4 tiles of flat row `row` -> scratch row slot `slot` (contiguous 128 values)
    auto read_row = [&](const auto& acc, const Geometry& gm, uint32_t row, uint32_t row_tiles, uint32_t slot) {
        const Location loc = locate(gm, row * row_tiles * 32);
        const uint32_t src = face_row_offset(loc.row, 0);
        const uint32_t dst = scratch + slot * ROW;
        for (uint32_t j = 0; j < 4; j++) {
            noc.async_read(
                acc,
                CoreLocalMem<uint32_t>(dst + j * 128),
                FACE_ROW_BYTES,
                {.page_id = loc.page + j, .offset_bytes = src},
                {});
            noc.async_read(
                acc,
                CoreLocalMem<uint32_t>(dst + j * 128 + FACE_ROW_BYTES),
                FACE_ROW_BYTES,
                {.page_id = loc.page + j, .offset_bytes = src + FACE_BYTES},
                {});
        }
    };
    volatile tt_l1_ptr uint32_t* sc = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch);

    // the o_norm weight row (host-prepared [32, V] fp32, row 0) -> u-layout, once; a core without rows (user groups >
    // rows) reads nothing (its compute returns at once and never pops CB_W)
    if (items > 0) {
        read_row(w_acc, Geometry{Vt * 32, 32, 1, Vt}, 0, Vt, 0);
        noc.async_read_barrier();
        cw.reserve_back(1);
        volatile tt_l1_ptr uint32_t* w = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cw.get_write_ptr());
        for (uint32_t j = 0; j < 128; j++) {
            w[val_off(0, j)] = sc[j];
        }
        cw.push_back(1);
    }
#if defined(REC_CHAIN)
    static_assert(REC_UG_ROWS_OK, "REC_CHAIN: one user group");
    uint32_t src0 = b0;  // row 0's source state row
    {
        const InterleavedAddrGen<true> sg = {.bank_base_address = get_arg_val<uint32_t>(ph_rt0 + 11), .page_size = 128};
        noc_async_read(sg.get_noc_addr(0), scratch + 7 * ROW, 64);
        noc_async_read_barrier();
        src0 = sc[7 * ROW / 4];
    }
#else
    const uint32_t src0 = b0;
#endif

#if defined(MK_REC_WAIT_SEM)
#if defined(KDA_PREFETCH)
    // Stage 14's prefetch (v2 too): the first item's state tiles (this step's input state, untouched by kda_pre) are
    // requested before the kda_pre hand-off; the first loop iteration skips them (same reserved pages, landed by its
    // barrier)
    if (items > 0) {
        cs.reserve_back(KV);
        for (uint32_t i = 0; i < KV; i++) {
            noc.async_read(s_acc, cs, tb, {.page_id = src0 * H * KV + h * KV + i}, {.offset_bytes = i * tb});
        }
    }
#endif
    mk::wait_handoff(MK_REC_WAIT_SEM, get_arg_val<uint32_t>(ph_rt0 + 10));
#endif
#if defined(REC_ITEM_MARKS)
    MK_MARK();  // kda_rec reader: the kda_pre hand-off passed
#endif

    for (uint32_t b = 0; b < items; b++) {
        const uint32_t row = (b0 + b) * H + h;
        read_row(q_acc, gq, row, Kt, 0);
        read_row(k_acc, gk, row, Kt, 1);
        read_row(g_acc, gg, row, Kt, 2);
        read_row(v_acc, gv, row, Vt, 3);
        read_row(go_acc, ggo, row, Vt, 4);
        const Location bl = locate(gb, row);
        noc.async_read(
            b_acc,
            CoreLocalMem<uint32_t>(scratch + 5 * ROW),
            FACE_ROW_BYTES,
            {.page_id = bl.page, .offset_bytes = face_row_offset(bl.row, bl.col)},
            {});
        noc.async_read_barrier();
#if defined(REC_ITEM_MARKS)
        if (b == 0) {
            MK_MARK();  // kda_rec reader: the first item's kda_pre rows landed
        }
#endif
        cvec.reserve_back(1);
        volatile tt_l1_ptr uint32_t* vec = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cvec.get_write_ptr());
        for (uint32_t r = 0; r < 128; r++) {
            vec[key_off(V2_Q, r)] = sc[r];
            vec[key_off(V2_K, r)] = sc[ROW / 4 + r];
            vec[key_off(V2_G, r)] = sc[2 * ROW / 4 + r];
            vec[val_off(V2_V, r)] = sc[3 * ROW / 4 + r];
            vec[val_off(V2_GO, r)] = sc[4 * ROW / 4 + r];
        }
        const uint32_t beta_bits = sc[5 * ROW / 4 + bl.col % 16];
        for (uint32_t t = 0; t < 32; t++) {
            vec[plane_off(V2_BETA, t)] = beta_bits;
        }
        cvec.push_back(1);

#if defined(KDA_PREFETCH) && defined(MK_REC_WAIT_SEM)
        if (b != 0)
#endif
        {
#if defined(REC_CHAIN)
            const uint32_t srow = b == 0 ? src0 : b0 + b - 1;
            if (b != 0) {
                mk::wait_handoff(REC_CHAIN_SEM, 1);  // row b - 1's state landed (this core's writer)
            }
#else
            const uint32_t srow = b0 + b;
#endif
            cs.reserve_back(KV);
            for (uint32_t i = 0; i < KV; i++) {
                noc.async_read(s_acc, cs, tb, {.page_id = srow * H * KV + h * KV + i}, {.offset_bytes = i * tb});
            }
        }
        noc.async_read_barrier();
        cs.push_back(KV);
    }
}
#else
void kernel_main() {
    using namespace kda_rec;
    constexpr uint32_t H = get_compile_time_arg_val(PH_CT0 + 0);
    constexpr uint32_t Kt = get_compile_time_arg_val(PH_CT0 + 1);
    constexpr uint32_t Vt = get_compile_time_arg_val(PH_CT0 + 2);
    constexpr Geometry gq = geometry<PH_CT0 + 3>();
    constexpr Geometry gk = geometry<PH_CT0 + 7>();
    constexpr Geometry gv = geometry<PH_CT0 + 11>();
    constexpr Geometry gg = geometry<PH_CT0 + 15>();
    constexpr Geometry gb = geometry<PH_CT0 + 19>();
    constexpr Geometry ggo = geometry<PH_CT0 + 23>();
    constexpr auto q_a = TensorAccessorArgs<PH_CT0 + 27>();
    constexpr auto k_a = TensorAccessorArgs<q_a.next_compile_time_args_offset()>();
    constexpr auto v_a = TensorAccessorArgs<k_a.next_compile_time_args_offset()>();
    constexpr auto g_a = TensorAccessorArgs<v_a.next_compile_time_args_offset()>();
    constexpr auto b_a = TensorAccessorArgs<g_a.next_compile_time_args_offset()>();
    constexpr auto go_a = TensorAccessorArgs<b_a.next_compile_time_args_offset()>();
    constexpr auto s_a = TensorAccessorArgs<go_a.next_compile_time_args_offset()>();
    constexpr auto w_a = TensorAccessorArgs<s_a.next_compile_time_args_offset()>();
    constexpr uint32_t KV = Kt * Vt;
    constexpr uint32_t VTC = Vt / REC_SPREAD;  // this core's value-column tiles (REC_SPREAD cores per head)
    static_assert(Vt % REC_SPREAD == 0, "REC_SPREAD divides the value tiles");

    const uint32_t hc = get_arg_val<uint32_t>(ph_rt0 + 0);  // core index in the list
    const uint32_t h = hc / REC_SPREAD, blk = hc % REC_SPREAD;
    // items = (b0 << 16) | n (Stage 14 user groups, REC_UG: this core serves rows b0 .. b0 + n - 1; else b0 = 0)
    const uint32_t items = get_arg_val<uint32_t>(ph_rt0 + 1) & 0xFFFF;
    const uint32_t b0 = get_arg_val<uint32_t>(ph_rt0 + 1) >> 16;
    if (h >= H) {
        return;  // a core past the shard's heads (megakernel owner list): no recurrence work, no hand-off
    }
    const uint32_t tb = get_tile_size(CB_Q);
    const auto q_acc = TensorAccessor(q_a, get_arg_val<uint32_t>(ph_rt0 + 2), tb);
    const auto k_acc = TensorAccessor(k_a, get_arg_val<uint32_t>(ph_rt0 + 3), tb);
    const auto v_acc = TensorAccessor(v_a, get_arg_val<uint32_t>(ph_rt0 + 4), tb);
    const auto g_acc = TensorAccessor(g_a, get_arg_val<uint32_t>(ph_rt0 + 5), tb);
    const auto b_acc = TensorAccessor(b_a, get_arg_val<uint32_t>(ph_rt0 + 6), tb);
    const auto go_acc = TensorAccessor(go_a, get_arg_val<uint32_t>(ph_rt0 + 7), tb);
    const auto s_acc = TensorAccessor(s_a, get_arg_val<uint32_t>(ph_rt0 + 8), tb);
    const auto w_acc = TensorAccessor(w_a, get_arg_val<uint32_t>(ph_rt0 + 9), tb);

    Noc noc;
    CircularBuffer cq(CB_Q), ck(CB_K), cv(CB_V), cg(CB_G), cbeta(CB_BETA), cs(CB_S0), cgo(CB_GO), cw(CB_W);

    // Zero every page of the per-item input CBs once; later reads only write row 0 / column 0.
    auto zero_cb = [&](CircularBuffer& c, uint32_t pages) {
        c.reserve_back(pages);
        noc.async_write_zeros(c, pages * tb);
    };
    zero_cb(cq, REC_DEPTH * Kt);
    zero_cb(ck, REC_DEPTH * Kt);
    zero_cb(cv, REC_DEPTH * VTC);
    zero_cb(cg, REC_DEPTH * Kt);
    zero_cb(cbeta, REC_DEPTH);
    zero_cb(cgo, REC_DEPTH * VTC);
    noc.write_zeros_l1_barrier();

    CircularBuffer cscr(CB_SCR);
    cscr.reserve_back(1);
    const uint32_t scratch = cscr.get_write_ptr();

    // the o_norm weight row tiles (host-prepared [32, V] fp32, row 0 = weight), once; a user group without rows (its
    // compute returns at once) reads none — the reads are flushed only inside the item loop, so they would still be in
    // flight when the kernel ends
    if (items > 0) {
        cw.reserve_back(VTC);
        for (uint32_t j = 0; j < VTC; j++) {
            noc.async_read(w_acc, cw, tb, {.page_id = blk * VTC + j}, {.offset_bytes = j * tb});
        }
    }
#if defined(REC_CHAIN)
    static_assert(REC_UG_ROWS_OK, "REC_CHAIN: one user group");
    uint32_t src0 = b0;  // row 0's source state row
    {
        const InterleavedAddrGen<true> sg = {.bank_base_address = get_arg_val<uint32_t>(ph_rt0 + 11), .page_size = 128};
        constexpr uint32_t SEL_OFF = 7 * 128 * 4;  // past the 7 fp32 scratch rows (as the v2 path's 7 * ROW)
        noc_async_read(sg.get_noc_addr(0), scratch + SEL_OFF, 64);
        noc_async_read_barrier();
        src0 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch + SEL_OFF)[0];
    }
#else
    const uint32_t src0 = b0;
#endif

#if defined(MK_REC_WAIT_SEM)
#if defined(KDA_PREFETCH)
    // Stage 14: the first item's state tiles (this step's input state, untouched by kda_pre) requested before the
    // kda_pre hand-off; the first loop iteration skips them (same reserved pages, landed by its barrier)
    if (items > 0) {
        cs.reserve_back(Kt * VTC);
        const uint32_t item0 = src0 * H + h;
        for (uint32_t mi = 0; mi < Kt; mi++) {
            for (uint32_t ni = 0; ni < VTC; ni++) {
                noc.async_read(
                    s_acc,
                    cs,
                    tb,
                    {.page_id = item0 * KV + mi * Vt + blk * VTC + ni},
                    {.offset_bytes = (mi * VTC + ni) * tb});
            }
        }
    }
#endif
    mk::wait_handoff(MK_REC_WAIT_SEM, get_arg_val<uint32_t>(ph_rt0 + 10));
#endif

    // The two face rows of each of the `tiles` tiles holding flat row `row_index`: tile j's to dst + j * stride and
    // dst + j * stride + half.
    auto read_row = [&](const auto& acc,
                        const Geometry& gm,
                        uint32_t row_index,
                        uint32_t tiles,
                        uint32_t dst,
                        uint32_t stride,
                        uint32_t half,
                        uint32_t row_tiles = 0,
                        uint32_t col_tile0 = 0) {
        // `tiles` tiles of flat row `row_index` (a row of `row_tiles` tiles, default `tiles`), from tile col_tile0 on
        const Location loc = locate(gm, (row_index * (row_tiles ? row_tiles : tiles) + col_tile0) * 32);
        const uint32_t src = face_row_offset(loc.row, 0);
        for (uint32_t j = 0; j < tiles; j++) {
            noc.async_read(
                acc,
                CoreLocalMem<uint32_t>(dst + j * stride),
                FACE_ROW_BYTES,
                {.page_id = loc.page + j, .offset_bytes = src},
                {});
            noc.async_read(
                acc,
                CoreLocalMem<uint32_t>(dst + j * stride + half),
                FACE_ROW_BYTES,
                {.page_id = loc.page + j, .offset_bytes = src + FACE_BYTES},
                {});
        }
    };
    // Scratch vector slot -> column 0 of the CB's Kt reserved tiles.
    auto scatter_columns = [&](uint32_t slot, CircularBuffer& c) {
        volatile tt_l1_ptr uint32_t* src =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch + slot * Kt * 2 * FACE_ROW_BYTES);
        volatile tt_l1_ptr uint32_t* dst = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(c.get_write_ptr());
        for (uint32_t j = 0; j < Kt; j++) {
            for (uint32_t r = 0; r < 32; r++) {
                dst[j * (tb / 4) + (r / 16) * 2 * (FACE_BYTES / 4) + (r % 16) * 16] = src[j * 32 + r];
            }
        }
    };

    // the state tiles of this core's column block: [Kt rows of tiles] x [VTC columns] of the item's [K, V] tiles
    constexpr uint32_t KVC = Kt * VTC;
    auto read_state = [&](uint32_t b, uint32_t item) {
#if defined(KDA_PREFETCH) && defined(MK_REC_WAIT_SEM)
        if (b == 0) {
            return;  // item 0's state was requested before the hand-off (the same reserved pages)
        }
#endif
#if defined(REC_CHAIN)
        // the source state: row `sel` for row 0, row b - 1 (this core's writer, one row earlier) after
        item = (b == 0 ? src0 : b0 + b - 1) * H + h;
        if (b != 0) {
            mk::wait_handoff(REC_CHAIN_SEM, 1);
        }
#endif
        cs.reserve_back(KVC);
        for (uint32_t mi = 0; mi < Kt; mi++) {
            for (uint32_t ni = 0; ni < VTC; ni++) {
                noc.async_read(
                    s_acc,
                    cs,
                    tb,
                    {.page_id = item * KV + mi * Vt + blk * VTC + ni},
                    {.offset_bytes = (mi * VTC + ni) * tb});
            }
        }
    };

    for (uint32_t b = 0; b < items; b++) {
        const uint32_t item = (b0 + b) * H + h;
        const uint32_t row = item;                    // decode: (b * T + t) * H + h with T = 1
        constexpr uint32_t VEC = 2 * FACE_ROW_BYTES;  // one tile row in scratch
#if defined(REC_S_FIRST)
        // KdaRec state_depth 2 with single inputs: the next item's state (the bulk of the item's reads) is issued
        // first, so it streams in while compute runs the previous item; the inputs below wait for its pops
        read_state(b, item);
#endif
        read_row(q_acc, gq, row, Kt, scratch, VEC, FACE_ROW_BYTES);
        read_row(k_acc, gk, row, Kt, scratch + Kt * VEC, VEC, FACE_ROW_BYTES);
        read_row(g_acc, gg, row, Kt, scratch + 2 * Kt * VEC, VEC, FACE_ROW_BYTES);
        cv.reserve_back(VTC);
        read_row(v_acc, gv, row, VTC, cv.get_write_ptr(), tb, FACE_BYTES, Vt, blk * VTC);
        cgo.reserve_back(VTC);
        read_row(go_acc, ggo, row, VTC, cgo.get_write_ptr(), tb, FACE_BYTES, Vt, blk * VTC);

        const Location bl = locate(gb, row);
        cbeta.reserve_back(1);
        noc.async_read(
            b_acc,
            cbeta,
            FACE_ROW_BYTES,
            {.page_id = bl.page, .offset_bytes = face_row_offset(bl.row, bl.col)},
            {.offset_bytes = 0});
        noc.async_read_barrier();
        if (b == 0) {
            cw.push_back(VTC);
        }
        cq.reserve_back(Kt);
        ck.reserve_back(Kt);
        cg.reserve_back(Kt);
        scatter_columns(0, cq);
        scatter_columns(1, ck);
        scatter_columns(2, cg);
        volatile tt_l1_ptr uint32_t* bp = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cbeta.get_write_ptr());
        const uint32_t beta_bits = bp[bl.col % 16];
        for (uint32_t c = 0; c < 16; c++) {
            bp[c] = beta_bits;
            bp[FACE_BYTES / 4 + c] = beta_bits;
        }

        cq.push_back(Kt);
        ck.push_back(Kt);
        cv.push_back(VTC);
        cg.push_back(Kt);
        cbeta.push_back(1);
        cgo.push_back(VTC);

#if !defined(REC_S_FIRST)
        read_state(b, item);
        noc.async_read_barrier();
#endif
        cs.push_back(KVC);
    }
}
#endif  // REC_V2
