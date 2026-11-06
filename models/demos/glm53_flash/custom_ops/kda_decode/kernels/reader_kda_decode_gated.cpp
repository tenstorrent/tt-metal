// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Reader of the fused KDA decode block -- the GATED VARIANT (KdaDecode(gated=True): the speculative commit's
// launches, tt/spec_decode.py under GLM53_SPEC_COMMIT=gated). It is reader_kda_decode_v2.cpp (the knob variant:
// gates="act" / placement="rect") plus a per-user write gate; reader_kda_decode.cpp is the default-path source and
// must stay byte-identical to the shipped kernel.
// Work item = (head h, user tile row tb, user part). The item's shared
// inputs -- the head's three conv-history tile blocks + raw block per branch (tap-major tiles: no tilize), the
// taps, the gate blocks and the channel constants -- are read from DRAM by the part-0 core of the (h, tb) slot
// and multicast into the same CB addresses of the other parts (a column of cores); the part-0 core then shifts
// the history in place (tap j+1 -> tap j, raw -> tap 2). Per user the core reads the user's selector tile and
// its recurrent state. On the part-0 core the reader also collects the other parts' output blocks (they write
// into the gather CB and bump a semaphore) for the final norm.
//
// GATES_ACT (compile-time arg 11): the gates input is already activated (sigmoid fused into the g2 matmul, the
// channel constants folded into its weights): the channel-constant tiles are never read nor multicast.
// Rectangle placement (runtime args 15..19): the slot's parts may span a second column (NoC rectangle
// (rb_x0, rb_y0)..(rb_x1, rb_y1) with n_b cores); the multicasts then go to the part-0 column below the sender
// (parts - 1 - n_b cores, as before) and to that second column. n_b = 0 is the column layout, unchanged.
//
// The write gate (one more accessor block after the recurrent state's, runtime arg 20 = its address): a [Bt, 32]
// int32 ROW_MAJOR tensor, page tb = the 32 users of tile row tb, non-zero = this launch writes that user's state
// and history. The part-0 core reads the page with the item's inputs and gates the history shift PER USER ROW:
// every live user on -> the plain whole-tile shift; every live user off -> no write at all (the history keeps its
// bytes); mixed -> merged tiles built in a scratch region (the pages of cb_onw after the Vt weight tiles, which the
// CB protocol never reserves): the new tile copied by the NoC, the rows of the users that are off patched in from
// the old tile by RISC-V word copies (a tile row is 2 x 32 B: one 16-value run per face of the row's face pair),
// then written to DRAM like the plain shift. The writer gates the recurrent-state tiles the same way (its own copy
// of the gate page lands at +2048 of the scratch region). The compute kernel is the v2 one, untouched: a gated-off
// user's launch is computed and discarded, so its state bytes stay exactly its input.

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"

constexpr uint32_t cb_hist = 0, cb_taps = 1, cb_gf = 2, cb_gog = 3, cb_gb = 4, cb_cc = 5, cb_onw = 6, cb_sel = 7;
constexpr uint32_t cb_selT = 8, cb_S = 9, cb_gather = 24;
constexpr uint32_t SEM_READY = 0, SEM_VALID = 1, SEM_ONE = 2, SEM_GATHER = 3;
constexpr uint32_t GATE_BYTES = 32 * 4;       // one gate page: 32 int32 write gates (op.py GATE_BYTES)
constexpr uint32_t FACE_BYTES = 16 * 16 * 2;  // a 16x16 bf16 face of a 32x32 tile (four faces, row-major)

// row u of a 32x32 bf16 tile: 16 values in face (u / 16) * 2 and 16 in the face next to it, each run at byte offset
// (u % 16) * 32 of its face (8 words per run)
inline void copy_tile_row(uint32_t dst_tile, uint32_t src_tile, uint32_t u) {
    const uint32_t off = ((u >> 4) * 2u) * FACE_BYTES + (u & 15u) * 32u;
    volatile tt_l1_ptr uint32_t* d = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dst_tile + off);
    volatile tt_l1_ptr uint32_t* s = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(src_tile + off);
    for (uint32_t i = 0; i < 8; i++) {
        d[i] = s[i];
        d[FACE_BYTES / 4 + i] = s[FACE_BYTES / 4 + i];
    }
}

void kernel_main() {
    constexpr uint32_t Kt = get_compile_time_arg_val(0);
    constexpr uint32_t Vt = get_compile_time_arg_val(1);
    constexpr uint32_t Bt = get_compile_time_arg_val(2);
    constexpr uint32_t B = get_compile_time_arg_val(3);
    constexpr uint32_t H = get_compile_time_arg_val(4);
    constexpr uint32_t Wq_t = get_compile_time_arg_val(5);  // qkv / history row width in tiles (3*H*Kt)
    constexpr uint32_t Wg_t = get_compile_time_arg_val(6);  // gates row width in tiles
    constexpr uint32_t HK_t = get_compile_time_arg_val(7);
    constexpr uint32_t HV_t = get_compile_time_arg_val(8);
    constexpr uint32_t parts = get_compile_time_arg_val(9);
    constexpr uint32_t ONES_PAGE = get_compile_time_arg_val(10);   // selector page of the all-ones tile (identity next)
    constexpr bool GATES_ACT = get_compile_time_arg_val(11) != 0;  // activated gates: no channel constants
    constexpr uint32_t conv_taps = 3;

    constexpr auto qkv_a = TensorAccessorArgs<12>();
    constexpr auto hist_a = TensorAccessorArgs<qkv_a.next_compile_time_args_offset()>();
    constexpr auto taps_a = TensorAccessorArgs<hist_a.next_compile_time_args_offset()>();
    constexpr auto gates_a = TensorAccessorArgs<taps_a.next_compile_time_args_offset()>();
    constexpr auto cc_a = TensorAccessorArgs<gates_a.next_compile_time_args_offset()>();
    constexpr auto onw_a = TensorAccessorArgs<cc_a.next_compile_time_args_offset()>();
    constexpr auto sel_a = TensorAccessorArgs<onw_a.next_compile_time_args_offset()>();
    constexpr auto st_a = TensorAccessorArgs<sel_a.next_compile_time_args_offset()>();
    constexpr auto gate_a = TensorAccessorArgs<st_a.next_compile_time_args_offset()>();  // the write gate pages

    const uint32_t item0 = get_arg_val<uint32_t>(0);
    const uint32_t n_items = get_arg_val<uint32_t>(1);
    const uint32_t stride = get_arg_val<uint32_t>(2);
    const uint32_t qkv_addr = get_arg_val<uint32_t>(3);
    const uint32_t hist_addr = get_arg_val<uint32_t>(4);
    const uint32_t taps_addr = get_arg_val<uint32_t>(5);
    const uint32_t gates_addr = get_arg_val<uint32_t>(6);
    const uint32_t cc_addr = get_arg_val<uint32_t>(7);
    const uint32_t onw_addr = get_arg_val<uint32_t>(8);
    const uint32_t sel_addr = get_arg_val<uint32_t>(9);
    const uint32_t st_addr = get_arg_val<uint32_t>(10);
    const uint32_t p0_x = get_arg_val<uint32_t>(11);  // NoC coords of the slot's part-0 core ...
    const uint32_t p0_y = get_arg_val<uint32_t>(12);
    const uint32_t pe_x = get_arg_val<uint32_t>(13);  // ... and of the last part in its column (larger y)
    const uint32_t pe_y = get_arg_val<uint32_t>(14);
    const uint32_t rb_x0 = get_arg_val<uint32_t>(15);  // second multicast column (rectangle placement) ...
    const uint32_t rb_y0 = get_arg_val<uint32_t>(16);
    const uint32_t rb_x1 = get_arg_val<uint32_t>(17);
    const uint32_t rb_y1 = get_arg_val<uint32_t>(18);
    const uint32_t n_b = get_arg_val<uint32_t>(19);        // ... and its core count (0: column layout)
    const uint32_t n_a = (parts - 1) - n_b;                // parts below the sender in its own column
    const uint32_t gate_addr = get_arg_val<uint32_t>(20);  // the write gate pages ([Bt, 32] int32)

    const uint32_t t32 = get_tile_size(cb_S);
    const uint32_t t16 = get_tile_size(cb_hist);
    const auto qkv_acc = TensorAccessor(qkv_a, qkv_addr, t16);
    const auto hist_acc = TensorAccessor(hist_a, hist_addr, t16);
    const auto taps_acc = TensorAccessor(taps_a, taps_addr, t16);
    const auto gates_acc = TensorAccessor(gates_a, gates_addr, t32);
    const auto cc_acc = TensorAccessor(cc_a, cc_addr, t32);
    const auto onw_acc = TensorAccessor(onw_a, onw_addr, t32);
    const auto sel_acc = TensorAccessor(sel_a, sel_addr, t16);
    const auto st_acc = TensorAccessor(st_a, st_addr, t32);
    const auto gate_acc = TensorAccessor(gate_a, gate_addr, GATE_BYTES);
    // the scratch region: the pages of cb_onw after its Vt weight tiles (reserved by nobody: the CB sees one reserve /
    // push of Vt pages per kernel run, below) -- the gate page at +0 (the writer lands its own copy at +2048) and the
    // 9 Kt merged history tiles from +t32
    const uint32_t scratch = CircularBuffer(cb_onw).get_write_ptr() + Vt * t32;
    const uint32_t gate_l1 = scratch;
    const uint32_t merge_l1 = scratch + t32;

    Noc noc;
    auto issue = [&](const auto& acc, CircularBuffer& cb, uint32_t tb, uint32_t base, uint32_t n, uint32_t off) {
        for (uint32_t t = 0; t < n; t++) {
            noc.async_read(acc, cb, tb, {.page_id = base + t}, {.offset_bytes = (off + t) * tb});
        }
    };
    auto read_tiles = [&](const auto& acc, uint32_t cb_id, uint32_t tb, uint32_t base, uint32_t n) {
        CircularBuffer cb(cb_id);
        cb.reserve_back(n);
        issue(acc, cb, tb, base, n, 0);
        noc.async_read_barrier();
        cb.push_back(n);
    };

    // once per core: the o_norm weight tiles, the all-ones and identity selector tiles
    read_tiles(onw_acc, cb_onw, t32, 0, Vt);
    read_tiles(sel_acc, cb_sel, t16, ONES_PAGE, 2);

    volatile tt_l1_ptr uint32_t* ready = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SEM_READY));
    volatile tt_l1_ptr uint32_t* valid = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SEM_VALID));
    volatile tt_l1_ptr uint32_t* gath = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SEM_GATHER));

    for (uint32_t it = 0; it < n_items; it++) {
        const uint32_t item = item0 + it * stride;
        const uint32_t h = item / (Bt * parts);
        const uint32_t rem = item % (Bt * parts);
        const uint32_t tb = rem / parts;
        const uint32_t part = rem % parts;
        const uint32_t n_row = (B - tb * 32u < 32u) ? (B - tb * 32u) : 32u;
        const uint32_t up = (n_row + parts - 1) / parts;
        const uint32_t n_active = (n_row + up - 1) / up;  // parts with at least one user
        const uint32_t u0 = part * up;
        const uint32_t u1 = (u0 + up < n_row) ? (u0 + up) : n_row;
        const uint32_t col = h * Kt;  // the head's first tile column inside a branch

        CircularBuffer hist(cb_hist), taps(cb_taps), gf(cb_gf), gog(cb_gog), gb(cb_gb), cc(cb_cc);
        CircularBuffer selT(cb_selT), S(cb_S), gather(cb_gather);
        hist.reserve_back(12 * Kt);
        taps.reserve_back(3 * Kt);
        gf.reserve_back(Kt);
        gog.reserve_back(Vt);
        gb.reserve_back(1);
        if constexpr (!GATES_ACT) {
            cc.reserve_back(Kt);
        }
        // the first user's selector + state and the beta selector go out with the item reads (overlap)
        const bool active = u0 < u1;
        const uint32_t c = h & 31u;
        if (active) {
            selT.reserve_back(2);
            S.reserve_back(Kt * Vt);
            issue(sel_acc, selT, t16, c, 1, 0);   // E_c^T for the beta column
            issue(sel_acc, selT, t16, u0, 1, 1);  // E_{u0}^T
            issue(st_acc, S, t32, ((tb * 32 + u0) * H + h) * Kt * Vt, Kt * Vt, 0);
        }
        if (part == 0) {
            // every per-item input in flight at once: the 3 x (3 history + raw) blocks, taps, gates, constants
            for (uint32_t br = 0; br < 3; br++) {
                for (uint32_t j = 0; j < conv_taps; j++) {
                    issue(hist_acc, hist, t16, (j * Bt + tb) * Wq_t + br * HK_t + col, Kt, (br * 4 + j) * Kt);
                }
                issue(qkv_acc, hist, t16, tb * Wq_t + br * HK_t + col, Kt, (br * 4 + 3) * Kt);
                issue(taps_acc, taps, t16, br * HK_t + col, Kt, br * Kt);
            }
            issue(gates_acc, gf, t32, tb * Wg_t + h * Kt, Kt, 0);
            if constexpr (!GATES_ACT) {
                issue(cc_acc, cc, t32, h * Kt, Kt, 0);
            }
            issue(gates_acc, gb, t32, tb * Wg_t + HK_t + HV_t + (h >> 5), 1, 0);
            issue(gates_acc, gog, t32, tb * Wg_t + HK_t + h * Vt, Vt, 0);
            noc_async_read(gate_acc.get_noc_addr(tb), gate_l1, GATE_BYTES);  // this tile row's write gates
            noc.async_read_barrier();
            if (active) {
                selT.push_back(2);
                S.push_back(Kt * Vt);
            }
            if constexpr (parts > 1) {
                // multicast the item's inputs into the other parts' CBs (same L1 addresses) once they are ready
                noc_semaphore_wait(ready, parts - 1);
                noc_semaphore_set(ready, 0);
                auto mcast = [&](CircularBuffer& cbo, uint32_t bytes) {
                    auto v = use<CircularBuffer::AddrSelector::WRITE_PTR>(cbo);  // same L1 address on every part
                    if (n_a > 0) {
                        noc.async_write_multicast(
                            v,
                            v,
                            bytes,
                            n_a,
                            {.offset_bytes = 0},
                            {.noc_x_start = p0_x,
                             .noc_y_start = p0_y + 1,
                             .noc_x_end = pe_x,
                             .noc_y_end = pe_y,
                             .offset_bytes = 0});
                    }
                    if (n_b > 0) {
                        noc.async_write_multicast(
                            v,
                            v,
                            bytes,
                            n_b,
                            {.offset_bytes = 0},
                            {.noc_x_start = rb_x0,
                             .noc_y_start = rb_y0,
                             .noc_x_end = rb_x1,
                             .noc_y_end = rb_y1,
                             .offset_bytes = 0});
                    }
                };
                mcast(hist, 12 * Kt * t16);
                mcast(taps, 3 * Kt * t16);
                mcast(gf, Kt * t32);
                mcast(gog, Vt * t32);
                mcast(gb, t32);
                if constexpr (!GATES_ACT) {
                    mcast(cc, Kt * t32);
                }
                noc.async_write_barrier();  // data landed before the "valid" flag is raised
                if (n_a > 0) {
                    noc_semaphore_set_multicast(
                        get_semaphore(SEM_ONE),
                        get_noc_multicast_addr(p0_x, p0_y + 1, pe_x, pe_y, get_semaphore(SEM_VALID)),
                        n_a);
                }
                if (n_b > 0) {
                    noc_semaphore_set_multicast(
                        get_semaphore(SEM_ONE),
                        get_noc_multicast_addr(rb_x0, rb_y0, rb_x1, rb_y1, get_semaphore(SEM_VALID)),
                        n_b);
                }
            }
        } else {
            noc_semaphore_set(valid, 0);
            noc_semaphore_inc(get_noc_addr(p0_x, p0_y, get_semaphore(SEM_READY)), 1);
            noc_semaphore_wait(valid, 1);
            if (active) {
                noc.async_read_barrier();
                selT.push_back(2);
                S.push_back(Kt * Vt);
            }
        }
        hist.push_back(12 * Kt);
        taps.push_back(3 * Kt);
        gf.push_back(Kt);
        gog.push_back(Vt);
        gb.push_back(1);
        if constexpr (!GATES_ACT) {
            cc.push_back(Kt);
        }
        if (part == 0) {
            // history shift (tap j -> tap j-1, raw -> tap 2; the old tiles stay in L1 until the item is done), gated
            // per user row: every live user on -> the plain whole-tile shift from the hist CB; every live user off ->
            // no write (the history keeps its bytes); mixed -> merged tiles in the scratch region: the new tile with
            // the rows of the users that are off taken from the old tile (rows >= n_row, the padding, follow the
            // plain shift). The gate page landed with the item reads (barrier above).
            volatile tt_l1_ptr uint32_t* gate = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(gate_l1);
            uint32_t n_on = 0;
            for (uint32_t u = 0; u < n_row; u++) {
                n_on += (gate[u] != 0u) ? 1u : 0u;
            }
            if (n_on == n_row) {
                for (uint32_t br = 0; br < 3; br++) {
                    for (uint32_t j = 1; j <= conv_taps; j++) {
                        for (uint32_t t = 0; t < Kt; t++) {
                            noc.async_write(
                                hist,
                                hist_acc,
                                t16,
                                {.offset_bytes = ((br * 4 + j) * Kt + t) * t16},
                                {.page_id = ((j - 1) * Bt + tb) * Wq_t + br * HK_t + col + t});
                        }
                    }
                }
            } else if (n_on != 0) {
                const uint32_t hist_l1 = hist.get_read_ptr();  // the item's tiles: this core never pops the hist CB
                // 1. the new tiles -> the scratch (NoC copies on this core)
                for (uint32_t br = 0; br < 3; br++) {
                    for (uint32_t j = 1; j <= conv_taps; j++) {
                        for (uint32_t t = 0; t < Kt; t++) {
                            const uint32_t dst = merge_l1 + ((br * 3 + j - 1) * Kt + t) * t16;
                            noc_async_write(
                                hist_l1 + ((br * 4 + j) * Kt + t) * t16,
                                get_noc_addr(my_x[noc_index], my_y[noc_index], dst),
                                t16);
                        }
                    }
                }
                noc_async_write_barrier();  // the copies landed before the row patches below
                // 2. the rows of the users that are off: from the old tile (tap j-1) of the same column
                for (uint32_t u = 0; u < n_row; u++) {
                    if (gate[u] != 0u) {
                        continue;
                    }
                    for (uint32_t br = 0; br < 3; br++) {
                        for (uint32_t j = 1; j <= conv_taps; j++) {
                            for (uint32_t t = 0; t < Kt; t++) {
                                copy_tile_row(
                                    merge_l1 + ((br * 3 + j - 1) * Kt + t) * t16,
                                    hist_l1 + ((br * 4 + j - 1) * Kt + t) * t16,
                                    u);
                            }
                        }
                    }
                }
                // 3. the merged tiles -> DRAM, exactly where the plain shift writes
                for (uint32_t br = 0; br < 3; br++) {
                    for (uint32_t j = 1; j <= conv_taps; j++) {
                        for (uint32_t t = 0; t < Kt; t++) {
                            noc_async_write(
                                merge_l1 + ((br * 3 + j - 1) * Kt + t) * t16,
                                hist_acc.get_noc_addr(((j - 1) * Bt + tb) * Wq_t + br * HK_t + col + t),
                                t16);
                        }
                    }
                }
            }
        }
        if (!active) {
            noc.async_write_barrier();
            continue;  // an idle part (no users): took part in the multicast only
        }
        if (part == 0 && parts > 1) {
            gather.reserve_back((parts - 1) * Vt);  // the other parts write their output blocks here
        }
        // the remaining users: the selector tile E_r^T and the recurrent state block (b*H + h)
        for (uint32_t r = u0 + 1; r < u1; r++) {
            read_tiles(sel_acc, cb_selT, t16, r, 1);
            read_tiles(st_acc, cb_S, t32, ((tb * 32 + r) * H + h) * Kt * Vt, Kt * Vt);
        }
        noc.async_write_barrier();  // the shift writes (before any reuse of the hist CB)
        if (part == 0 && parts > 1) {
            noc_semaphore_wait(gath, n_active - 1);
            noc_semaphore_set(gath, 0);
            gather.push_back((parts - 1) * Vt);
        }
    }
    // every NoC transaction this kernel issued (the cross-core semaphore increments above included) must be
    // acknowledged before the kernel returns: an atomic acked after the next kernel's noc_local_state_init
    // leaves that kernel's atomic counters unequal forever (its noc_async_atomic_barrier hangs; seen on the 8x4
    // o_proj matmul in0 receiver after kda_decode, 2026-10-20)
    noc_async_full_barrier();
}
