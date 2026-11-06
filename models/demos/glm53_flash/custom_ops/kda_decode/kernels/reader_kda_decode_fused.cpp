// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Reader of the fused KDA decode block -- the FUSED T = K + 1 VARIANT (KdaDecode(fused_steps=T): the speculative
// loop of tt/spec_decode.py under GLM53_SPEC_COMMIT=fused). It is reader_kda_decode_v2.cpp (the knob variant:
// gates="act" / placement="rect") with the per-item inputs taken from ONE tile row of R = USERS x STEPS
// candidate-major rows (row j * USERS + u = user u, candidate j); reader_kda_decode.cpp is the default-path source
// and must stay byte-identical to the shipped kernel.
//
// Work item = (head h, tile row 0, user part). The part-0 core of the slot reads the slot's inputs once -- the raw
// candidate block of every branch (3 Kt tiles of the R rows), the head's three live history tiles per branch (into
// the scratch region), the taps, the gate blocks and the channel constants -- and assembles the three conv TAP
// BLOCKS in the history CB (slot t = 0 .. 2 of each branch, exactly where the plain reader keeps history tap t):
// row (j, u) of tap block t is x_{p_u + j + t - 3}: the live history tap (j + t) of user u for j + t < 3, else the
// raw candidate row (j + t - 3, u); rows >= R take the raw tile's (finite) padding rows. The compute kernel then
// runs its conv on the four blocks exactly as on a plain item. The blocks are built with NoC copies on this core of
// contiguous face runs (a run of tile rows inside one 16-row half is 2 x n x 32 B), then multicast with the other
// inputs into the same CB addresses of the other parts (a column of cores, or two with the rectangle placement:
// runtime args 15..19 as in v2). Per user the core reads the user's recurrent state once and one selector tile
// E_r^T per step (r = j * USERS + u). On the part-0 core the reader also collects the other parts' output blocks
// (gather CB + semaphore) for the final norm.
//
// Step counts: dry mode (DRY = 1, the verify) applies STEPS steps to every user and writes NOTHING (no history
// shift, no state); commit mode (DRY = 0) reads the step page (one more accessor block after the recurrent
// state's, runtime arg 20 = its address: [Bt, 32] int32 ROW_MAJOR, word u = n_u = the steps user u takes, 0 =
// untouched), hands the compute a copy as the first selector page, pushes the state and n_u selectors of every
// user with n_u > 0 only, and (part 0) writes the new history: row u of tap i = x_{p_u + n_u + i - 3} = tap block
// (n_u + i) row u for n_u + i < 3, else the raw row (n_u + i - 3, u) -- an untouched user's rows are its own old
// bytes -- merged in the scratch region (the raw tile copied whole for the padding rows, the user rows patched by
// RISC-V word copies) and written exactly where the plain shift writes. The writer gates the recurrent state the
// same way (its copy of the step page lands at +2048 of the scratch region's first page).
//
// GATES_ACT (compile-time arg 11): the gates input is already activated: the channel-constant tiles are never
// read nor multicast. USERS / STEPS / DRY: compile-time args 12 / 13 / 14; the accessor blocks start at 15.

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"

constexpr uint32_t cb_hist = 0, cb_taps = 1, cb_gf = 2, cb_gog = 3, cb_gb = 4, cb_cc = 5, cb_onw = 6, cb_sel = 7;
constexpr uint32_t cb_selT = 8, cb_S = 9, cb_gather = 24;
constexpr uint32_t SEM_READY = 0, SEM_VALID = 1, SEM_ONE = 2, SEM_GATHER = 3;
constexpr uint32_t STEP_BYTES = 32 * 4;       // one step page: 32 int32 per-user step counts (op.py GATE_BYTES)
constexpr uint32_t FACE_BYTES = 16 * 16 * 2;  // a 16x16 bf16 face of a 32x32 tile (four faces, row-major)

// byte offset of row u's first 16-value run inside a 32x32 bf16 tile (face (u / 16) * 2; the second run of the row
// sits FACE_BYTES further, in the next face)
inline uint32_t row_off(uint32_t u) { return ((u >> 4) * 2u) * FACE_BYTES + (u & 15u) * 32u; }

// rows [src_r0, src_r0 + n) of src_tile -> rows [dst_r0, dst_r0 + n) of dst_tile (both L1 tiles of this core) as NoC
// copies on this core of the contiguous face runs: a run of rows inside one 16-row half of both tiles is one piece,
// 2 x (rows x 32 B), one write per face of the pair. Completed by the caller's noc_async_write_barrier().
inline void copy_rows(uint32_t dst_tile, uint32_t dst_r0, uint32_t src_tile, uint32_t src_r0, uint32_t n) {
    while (n > 0) {
        uint32_t piece = n;
        const uint32_t d_room = 16u - (dst_r0 & 15u);
        const uint32_t s_room = 16u - (src_r0 & 15u);
        if (piece > d_room) {
            piece = d_room;
        }
        if (piece > s_room) {
            piece = s_room;
        }
        const uint32_t src = src_tile + row_off(src_r0);
        const uint32_t dst = dst_tile + row_off(dst_r0);
        for (uint32_t f = 0; f < 2; f++) {
            noc_async_write(
                src + f * FACE_BYTES,
                get_noc_addr(my_x[noc_index], my_y[noc_index], dst + f * FACE_BYTES),
                piece * 32u);
        }
        dst_r0 += piece;
        src_r0 += piece;
        n -= piece;
    }
}

// row src_u of src_tile -> row dst_u of dst_tile by RISC-V word copies (2 x 8 words: one 16-value run per face of
// the row's face pair)
inline void copy_tile_row(uint32_t dst_tile, uint32_t dst_u, uint32_t src_tile, uint32_t src_u) {
    volatile tt_l1_ptr uint32_t* d = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dst_tile + row_off(dst_u));
    volatile tt_l1_ptr uint32_t* s = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(src_tile + row_off(src_u));
    for (uint32_t i = 0; i < 8; i++) {
        d[i] = s[i];
        d[FACE_BYTES / 4 + i] = s[FACE_BYTES / 4 + i];
    }
}

void kernel_main() {
    constexpr uint32_t Kt = get_compile_time_arg_val(0);
    constexpr uint32_t Vt = get_compile_time_arg_val(1);
    constexpr uint32_t Bt = get_compile_time_arg_val(2);
    constexpr uint32_t B = get_compile_time_arg_val(3);  // the R = USERS * STEPS candidate rows
    constexpr uint32_t H = get_compile_time_arg_val(4);
    constexpr uint32_t Wq_t = get_compile_time_arg_val(5);  // qkv / history row width in tiles (3*H*Kt)
    constexpr uint32_t Wg_t = get_compile_time_arg_val(6);  // gates row width in tiles
    constexpr uint32_t HK_t = get_compile_time_arg_val(7);
    constexpr uint32_t HV_t = get_compile_time_arg_val(8);
    constexpr uint32_t parts = get_compile_time_arg_val(9);
    constexpr uint32_t ONES_PAGE = get_compile_time_arg_val(10);   // selector page of the all-ones tile (identity next)
    constexpr bool GATES_ACT = get_compile_time_arg_val(11) != 0;  // activated gates: no channel constants
    constexpr uint32_t USERS = get_compile_time_arg_val(12);       // user slots (the state's batch)
    constexpr uint32_t STEPS = get_compile_time_arg_val(13);       // candidates per user: T = K + 1
    constexpr bool DRY = get_compile_time_arg_val(14) != 0;        // verify mode: STEPS per user, nothing written
    constexpr uint32_t conv_taps = 3;
    constexpr uint32_t R = B;
    static_assert(
        Bt == 1 && USERS * STEPS == R && R <= 32 && USERS >= 1 && STEPS >= 1, "one tile row of USERS x STEPS rows");

    constexpr auto qkv_a = TensorAccessorArgs<15>();
    constexpr auto hist_a = TensorAccessorArgs<qkv_a.next_compile_time_args_offset()>();
    constexpr auto taps_a = TensorAccessorArgs<hist_a.next_compile_time_args_offset()>();
    constexpr auto gates_a = TensorAccessorArgs<taps_a.next_compile_time_args_offset()>();
    constexpr auto cc_a = TensorAccessorArgs<gates_a.next_compile_time_args_offset()>();
    constexpr auto onw_a = TensorAccessorArgs<cc_a.next_compile_time_args_offset()>();
    constexpr auto sel_a = TensorAccessorArgs<onw_a.next_compile_time_args_offset()>();
    constexpr auto st_a = TensorAccessorArgs<sel_a.next_compile_time_args_offset()>();
    constexpr auto steps_a = TensorAccessorArgs<st_a.next_compile_time_args_offset()>();  // the step pages

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
    const uint32_t n_b = get_arg_val<uint32_t>(19);         // ... and its core count (0: column layout)
    const uint32_t n_a = (parts - 1) - n_b;                 // parts below the sender in its own column
    const uint32_t steps_addr = get_arg_val<uint32_t>(20);  // the step pages ([Bt, 32] int32)

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
    const auto steps_acc = TensorAccessor(steps_a, steps_addr, STEP_BYTES);
    // the scratch region: the pages of cb_onw after its Vt weight tiles (reserved by nobody: the CB sees one reserve /
    // push of Vt pages per kernel run, below) -- the step page at +0 (the writer lands its own copy at +2048), then
    // 9 Kt bf16 tiles: the live history tiles of the item (the tap-block assembly source), afterwards the merged new
    // history tiles (commit mode)
    const uint32_t scratch = CircularBuffer(cb_onw).get_write_ptr() + Vt * t32;
    const uint32_t page_l1 = scratch;
    const uint32_t orig_l1 = scratch + t32;

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
    volatile tt_l1_ptr uint32_t* page = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(page_l1);

    for (uint32_t it = 0; it < n_items; it++) {
        const uint32_t item = item0 + it * stride;
        const uint32_t h = item / (Bt * parts);
        const uint32_t rem = item % (Bt * parts);
        const uint32_t tb = rem / parts;
        const uint32_t part = rem % parts;
        const uint32_t up = (USERS + parts - 1) / parts;
        const uint32_t n_active = (USERS + up - 1) / up;  // parts with at least one user
        const uint32_t u0 = part * up;
        const uint32_t u1 = (u0 + up < USERS) ? (u0 + up) : USERS;
        const uint32_t col = h * Kt;  // the head's first tile column inside a branch

        // the per-user step counts: STEPS each (dry), or the step page (commit; values above STEPS clamped)
        uint32_t n_steps[32];
        if constexpr (DRY) {
            for (uint32_t uu = 0; uu < USERS; uu++) {
                n_steps[uu] = STEPS;
            }
        } else {
            noc_async_read(steps_acc.get_noc_addr(tb), page_l1, STEP_BYTES);
            noc.async_read_barrier();
            for (uint32_t uu = 0; uu < USERS; uu++) {
                const uint32_t n = page[uu];
                n_steps[uu] = (n < STEPS) ? n : STEPS;
            }
        }
        // the part's first user with a step: its selector and state go out with the item reads (overlap)
        uint32_t ua = u1;
        for (uint32_t uu = u0; uu < u1; uu++) {
            if (n_steps[uu] != 0u) {
                ua = uu;
                break;
            }
        }
        const bool active = u0 < u1;    // the part has users: it takes part in the multicast and the gather
        const bool stepping = ua < u1;  // ... and at least one of them steps

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
        // the first selector pages: [the step page (commit),] E_c^T for the beta column, E_{ua}^T (if ua steps)
        const uint32_t c = h & 31u;
        const uint32_t n_sel0 = (DRY ? 1u : 2u) + (stepping ? 1u : 0u);
        if (active) {
            selT.reserve_back(n_sel0);
            uint32_t p = 0;
            if constexpr (!DRY) {
                noc_async_read(steps_acc.get_noc_addr(tb), selT.get_write_ptr(), STEP_BYTES);  // the compute's copy
                p = 1;
            }
            issue(sel_acc, selT, t16, c, 1, p);  // E_c^T for the beta column
            if (stepping) {
                S.reserve_back(Kt * Vt);
                issue(sel_acc, selT, t16, ua, 1, p + 1);  // E_{0 * USERS + ua}^T
                issue(st_acc, S, t32, ((tb * 32 + ua) * H + h) * Kt * Vt, Kt * Vt, 0);
            }
        }
        const uint32_t hist_l1 = hist.get_write_ptr();  // the item's tile blocks (the CB is reserved whole)
        auto hist_tile = [&](uint32_t br, uint32_t slot, uint32_t t) {
            return hist_l1 + ((br * 4 + slot) * Kt + t) * t16;
        };
        auto orig_tile = [&](uint32_t br, uint32_t j, uint32_t t) { return orig_l1 + ((br * 3 + j) * Kt + t) * t16; };
        if (part == 0) {
            // every per-item input in flight at once: the raw candidate blocks (hist slot 3), the live history tiles
            // (scratch), taps, gates, constants
            for (uint32_t br = 0; br < 3; br++) {
                for (uint32_t j = 0; j < conv_taps; j++) {
                    for (uint32_t t = 0; t < Kt; t++) {
                        noc_async_read(
                            hist_acc.get_noc_addr((j * Bt + tb) * Wq_t + br * HK_t + col + t),
                            orig_tile(br, j, t),
                            t16);
                    }
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
            noc.async_read_barrier();
            // the three tap blocks per branch: row (j, u) of tap t = history tap (j + t) of user u for j + t < 3 (the
            // live tiles, rows 0 .. USERS-1), else raw candidate row (j + t - 3, u) (the raw block shifted down by
            // (3 - t) USERS rows); rows >= R take the raw block's padding rows (finite)
            for (uint32_t br = 0; br < 3; br++) {
                for (uint32_t tp = 0; tp < conv_taps; tp++) {
                    const uint32_t s = conv_taps - tp;  // row shift of the raw block, in candidate blocks
                    for (uint32_t t = 0; t < Kt; t++) {
                        const uint32_t dst = hist_tile(br, tp, t);
                        const uint32_t raw = hist_tile(br, 3, t);
                        if (R > s * USERS) {
                            copy_rows(dst, s * USERS, raw, 0, R - s * USERS);
                        }
                        for (uint32_t jj = 0; jj < s && jj < STEPS; jj++) {
                            copy_rows(dst, jj * USERS, orig_tile(br, tp + jj, t), 0, USERS);
                        }
                        if (R < 32u) {
                            copy_rows(dst, R, raw, R, 32u - R);
                        }
                    }
                }
            }
            noc_async_write_barrier();  // the tap blocks are complete before the push / multicast
            if (active) {
                selT.push_back(n_sel0);
                if (stepping) {
                    S.push_back(Kt * Vt);
                }
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
                selT.push_back(n_sel0);
                if (stepping) {
                    S.push_back(Kt * Vt);
                }
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
        if constexpr (!DRY) {
            if (part == 0) {
                // the new history (commit mode): row u of tap i = x_{p_u + n_u + i - 3} = tap block (n_u + i) row u
                // for n_u + i < 3 (an untouched user: its own old bytes), else raw row (n_u + i - 3, u); merged in the
                // scratch (the live tiles there are dead: the tap blocks hold their rows) and written where the plain
                // shift writes. Every user untouched -> no write at all. The tap blocks stay in L1 until the item
                // is done (this core never pops the hist CB).
                uint32_t any = 0;
                for (uint32_t uu = 0; uu < USERS; uu++) {
                    any |= n_steps[uu];
                }
                if (any != 0u) {
                    // 1. the raw tile of each column -> the 9 merged slots (its rows >= USERS are the padding rows)
                    for (uint32_t br = 0; br < 3; br++) {
                        for (uint32_t i = 0; i < conv_taps; i++) {
                            for (uint32_t t = 0; t < Kt; t++) {
                                noc_async_write(
                                    hist_tile(br, 3, t),
                                    get_noc_addr(my_x[noc_index], my_y[noc_index], orig_tile(br, i, t)),
                                    t16);
                            }
                        }
                    }
                    noc_async_write_barrier();  // the copies landed before the row patches below
                    // 2. the user rows
                    for (uint32_t uu = 0; uu < USERS; uu++) {
                        const uint32_t n = n_steps[uu];
                        for (uint32_t i = 0; i < conv_taps; i++) {
                            const uint32_t q = n + i;  // the sequence index of the row: 0..2 history, 3.. raw
                            const uint32_t slot = (q < conv_taps) ? q : 3u;
                            const uint32_t src_row = (q < conv_taps) ? uu : (q - conv_taps) * USERS + uu;
                            for (uint32_t br = 0; br < 3; br++) {
                                for (uint32_t t = 0; t < Kt; t++) {
                                    copy_tile_row(orig_tile(br, i, t), uu, hist_tile(br, slot, t), src_row);
                                }
                            }
                        }
                    }
                    // 3. the merged tiles -> DRAM, exactly where the plain shift writes
                    for (uint32_t br = 0; br < 3; br++) {
                        for (uint32_t i = 0; i < conv_taps; i++) {
                            for (uint32_t t = 0; t < Kt; t++) {
                                noc_async_write(
                                    orig_tile(br, i, t),
                                    hist_acc.get_noc_addr((i * Bt + tb) * Wq_t + br * HK_t + col + t),
                                    t16);
                            }
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
        // the remaining users and steps: the selector tile E_r^T per step (r = j * USERS + u) and the recurrent
        // state block (u * H + h) once per user, in the order the compute consumes them
        for (uint32_t uu = u0; uu < u1; uu++) {
            const uint32_t n = n_steps[uu];
            if (n == 0u) {
                continue;
            }
            if (uu != ua) {
                read_tiles(sel_acc, cb_selT, t16, uu, 1);
                read_tiles(st_acc, cb_S, t32, ((tb * 32 + uu) * H + h) * Kt * Vt, Kt * Vt);
            }
            for (uint32_t j = 1; j < n; j++) {
                read_tiles(sel_acc, cb_selT, t16, j * USERS + uu, 1);
            }
        }
        noc.async_write_barrier();  // the history writes (before any reuse of the hist CB)
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
