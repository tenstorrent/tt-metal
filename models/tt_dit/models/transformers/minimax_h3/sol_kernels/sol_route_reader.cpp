// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Sol-Attn routing pre-pass (proxy scores -> threshold -> chunk mask / bias), reader (NCRISC).
//
// Work unit (compile-time arg 20 UNITS_PER_SHARD = 0): (head h, Q-chunk tile row it), u = h*NQCT + it: the routing of
// local Q chunks [32*it, 32*it+32) of head h against the pooled K blocks of every shard. UNITS_PER_SHARD = 1: (head h,
// column shard sp), u = h*S + sp: every Q-chunk tile row of head h against shard sp only (see sol_route_compute.cpp).
// Per core, once: the ones-row tile, the Q-chunk selector (qsel), the block->chunk selector (ksel) and, only when the
// bias is emitted, the 32 row-broadcast selectors (erow). They are read after the first unit's pooled K and Q: every
// core needs the same few constant pages, and compute's first phases (the moments) need only the pooled K, so the data
// goes first and the constants land while the moments are summed. Per unit: the head's pooled K in D-major tile order
// (position n*KCT + c holds tile (c, n), so compute can use it as a transposed matmul operand): the whole head [NBG, D]
// (KCT = NBGT) with (head, it) units, only the unit's shard [NBP, D] (KCT = NBPT) with (head, shard) units; and the
// head's pooled Q [NBP, D]; then per (Q-chunk tile row it, column shard sp) of the unit: the constant OR-mask tiles
// of that Q-chunk tile row against shard sp's blocks, and the column constants (full / valid / length-bias rows) of sp.
//
// Partial-moment exchange ((head, shard) units): the head's mean and variance need every shard's pooled K, but each
// core reads only its own shard. Compute reduces its shard's tiles to 8 rows (row 0 of 8 reduced tiles in cb_pl: the
// column sums and sums of squares per dim tile). This kernel copies those rows into its own gather tiles (cb_recv) at
// the rows of its shard (tile sp/4, rows (sp%4)*8 + n: 8 consecutive rows, so the left halves are one 256 B run in
// one face and the right halves one 256 B run in the next), writes the same two runs to the same addresses on the S-1
// other cores of the head, bumps their semaphore 0 and waits for its own to reach S-1; compute then selects and sums
// the rows with the host's selector tiles (cb_shsel, read here with the constants), so no zeroing or assembly is
// needed. Peer cores are found from the unit -> core rule (units [ci*U/N, (ci+1)*U/N)), their NoC coordinates come
// from common runtime args 10..10+N-1 (x | y << 8 per logical core index). Needs UNITS <= NCORES (one unit per core,
// so the gather tiles are written once per launch) and S <= 8.
//
// Mask rows ((head, shard) units): this thread is idle once the exchange is done, so it builds and writes mask rows
// 16..31 of each (it, sp) from a second copy of the mask tiles (cb_mask2, packed by compute alongside cb_mask_out)
// while the writer does rows 0..15; the same word-wise row build as the writer (tile_row_to_words).
//
// Compile-time args: 0 H, 1 S, 2 NBPT, 3 DT, 4 NQCT, 5 NKCLT, 6 NQC, 7 NKC, 8 NKCP, 9 MODE, 10 EMIT_BIAS, 11 TAU,
// 12 EPS2, 13 NEG, 14 INV_NV, 15 TQ, 16 DEBUG, 17 NX, 18 NY, 19 UNITS, 20 UNITS_PER_SHARD, then TensorAccessorArgs of
// q_bar, kc, const_or, colconst, qsel, ksel, erow, ones, mask, shsel.
// Common runtime args: 0..7 the buffer addresses in the same order, 8 the mask address, 9 the shsel address;
// 10..10+NX*NY-1 the NoC coordinates of every logical core ((head, shard) units only). Per-core: none.

#include <cstdint>

#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/endpoints.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"

namespace {
constexpr uint32_t cb_kc = 0, cb_q = 1, cb_ones = 2, cb_qsel = 3, cb_ksel = 4, cb_erow = 5, cb_constor = 6, cb_colc = 7,
                   cb_recv = 23, cb_pl = 24, cb_mask2 = 26, cb_rring = 27, cb_shsel = 29;
constexpr uint32_t PART_T = 8;  // partial-moment tiles per shard: DT sums + DT sums of squares (DT <= 4)
constexpr uint32_t RING_PAGES = 8;

// Row r of ntiles consecutive 32x32 bf16 tiles -> 32*ntiles uint32 words (1 where the element is nonzero) at out_addr;
// two elements per L1 word along the contiguous face rows (the writer's copy of this helper is the reference).
inline void tile_row_to_words(uint32_t tiles_addr, uint32_t r, uint32_t ntiles, uint32_t out_addr) {
    CoreLocalMem<volatile uint32_t> w(out_addr);
    const uint32_t row_bytes = (((r >> 4) << 1) * 256 + (r & 15) * 16) * 2;
    uint32_t k = 0;
    for (uint32_t t = 0; t < ntiles; ++t) {
        for (uint32_t f = 0; f < 2; ++f) {
            CoreLocalMem<volatile uint32_t> e(tiles_addr + t * 2048 + row_bytes + f * 512);
            for (uint32_t i = 0; i < 8; ++i) {
                const uint32_t v = e[i];
                w[k++] = (v & 0xFFFFu) != 0 ? 1u : 0u;
                w[k++] = (v >> 16) != 0 ? 1u : 0u;
            }
        }
    }
}
constexpr uint32_t TILE_BYTES = 2048;
}  // namespace

void kernel_main() {
    constexpr uint32_t S = get_compile_time_arg_val(1);
    constexpr uint32_t NBPT = get_compile_time_arg_val(2);
    constexpr uint32_t DT = get_compile_time_arg_val(3);
    constexpr uint32_t NQCT = get_compile_time_arg_val(4);
    constexpr uint32_t NKCLT = get_compile_time_arg_val(5);
    constexpr uint32_t NBGT = S * NBPT;
    constexpr uint32_t UNITS_PER_SHARD_ = get_compile_time_arg_val(20);
    constexpr uint32_t KCT = UNITS_PER_SHARD_ ? NBPT : NBGT;  // pooled-K column tiles per unit (see the header)
    constexpr uint32_t KC_T = KCT * DT;
    constexpr uint32_t EMIT_BIAS = get_compile_time_arg_val(10);
    constexpr uint32_t DEBUG = get_compile_time_arg_val(16);
    constexpr bool NEED_EROW = EMIT_BIAS == 1 && DEBUG == 0;  // the only consumer of erow is the bias expansion
    constexpr auto qb_args = TensorAccessorArgs<21>();
    constexpr auto kc_args = TensorAccessorArgs<qb_args.next_compile_time_args_offset()>();
    constexpr auto co_args = TensorAccessorArgs<kc_args.next_compile_time_args_offset()>();
    constexpr auto cc_args = TensorAccessorArgs<co_args.next_compile_time_args_offset()>();
    constexpr auto qs_args = TensorAccessorArgs<cc_args.next_compile_time_args_offset()>();
    constexpr auto ks_args = TensorAccessorArgs<qs_args.next_compile_time_args_offset()>();
    constexpr auto er_args = TensorAccessorArgs<ks_args.next_compile_time_args_offset()>();
    constexpr auto on_args = TensorAccessorArgs<er_args.next_compile_time_args_offset()>();
    constexpr auto mk_args = TensorAccessorArgs<on_args.next_compile_time_args_offset()>();
    constexpr auto sl_args = TensorAccessorArgs<mk_args.next_compile_time_args_offset()>();

    constexpr uint32_t NX = get_compile_time_arg_val(17);
    constexpr uint32_t NY = get_compile_time_arg_val(18);
    constexpr uint32_t UNITS = get_compile_time_arg_val(19);
    constexpr uint32_t UNITS_PER_SHARD = get_compile_time_arg_val(20);  // 1: unit = (head, shard); 0: (head, it)
    static_assert(UNITS_PER_SHARD == 0 || get_compile_time_arg_val(9) == 1, "(head, shard) units are MODE 1 only");
    // Unit range of this core from its logical coordinates (no per-core runtime args): units [ci*U/N, (ci+1)*U/N).
    constexpr uint32_t NCORES = NX * NY;
    static_assert(UNITS_PER_SHARD == 0 || UNITS <= NCORES, "the partial-moment exchange needs one unit per core");
    static_assert(UNITS_PER_SHARD == 0 || S <= 8, "gather tiles hold 4 shards x 8 moment rows each");
    constexpr uint32_t GT = (S + 3) / 4;
    const uint32_t ci = static_cast<uint32_t>(get_absolute_logical_y()) * NX + get_absolute_logical_x();
    const uint32_t unit_start = ci * UNITS / NCORES;
    const uint32_t unit_count = (ci + 1) * UNITS / NCORES - unit_start;
    if (unit_count == 0) {
        return;
    }
    // Receive buffer for the peers' partial moments: reserved once, before any peer can write into it; its address is
    // the same on every core of the grid (one CB configuration), which is what the peers write to.
    CircularBuffer cbrecv(cb_recv), cbpl(cb_pl), cbm2(cb_mask2), cbrr(cb_rring);
    uint32_t recv_base = 0, rring_base = 0, ring_used = 0;
    if constexpr (UNITS_PER_SHARD) {
        cbrecv.reserve_back(GT);
        recv_base = cbrecv.get_write_ptr();  // the gather tiles; the peers write their rows into these
        cbrr.reserve_back(RING_PAGES);
        rring_base = cbrr.get_write_ptr();  // RING_PAGES pages of NBPT*32*4 bytes
    }
    const auto qb = TensorAccessor(qb_args, get_common_arg_val<uint32_t>(0));
    const auto kc = TensorAccessor(kc_args, get_common_arg_val<uint32_t>(1));
    const auto co = TensorAccessor(co_args, get_common_arg_val<uint32_t>(2));
    const auto cc = TensorAccessor(cc_args, get_common_arg_val<uint32_t>(3));
    const auto qs = TensorAccessor(qs_args, get_common_arg_val<uint32_t>(4));
    const auto ks = TensorAccessor(ks_args, get_common_arg_val<uint32_t>(5));
    const auto er = TensorAccessor(er_args, get_common_arg_val<uint32_t>(6));
    const auto on = TensorAccessor(on_args, get_common_arg_val<uint32_t>(7));
    const auto mk = TensorAccessor(mk_args, get_common_arg_val<uint32_t>(8));
    const auto sl = TensorAccessor(sl_args, get_common_arg_val<uint32_t>(9));

    Noc noc;
    auto read_pages = [&](auto& acc, uint32_t cb_id, uint32_t first, uint32_t count) {
        CircularBuffer cb(cb_id);
        cb.reserve_back(count);
        for (uint32_t i = 0; i < count; ++i) {
            noc.async_read(acc, cb, TILE_BYTES, {.page_id = first + i}, {.offset_bytes = i * TILE_BYTES});
        }
        noc.async_read_barrier();
        cb.push_back(count);
    };

    CircularBuffer cbkc(cb_kc), cbco(cb_constor), cbcc(cb_colc);
    for (uint32_t u = unit_start; u < unit_start + unit_count; ++u) {
        // Unit decode: one of the two ranges below is a singleton, the loops run the same code either way.
        const uint32_t h = UNITS_PER_SHARD ? u / S : u / NQCT;
        const uint32_t it_lo = UNITS_PER_SHARD ? 0 : u % NQCT;
        const uint32_t it_hi = UNITS_PER_SHARD ? NQCT : it_lo + 1;
        const uint32_t sp_lo = UNITS_PER_SHARD ? u % S : 0;
        const uint32_t sp_hi = UNITS_PER_SHARD ? sp_lo + 1 : S;

        // Pooled K of the head (whole head, or the unit's shard only), D-major tile order.
        cbkc.reserve_back(KC_T);
        {
            const uint32_t cg0 = UNITS_PER_SHARD ? sp_lo * NBPT : 0;
            uint32_t issued = 0;
            for (uint32_t n = 0; n < DT; ++n) {
                for (uint32_t c = 0; c < KCT; ++c) {
                    noc.async_read(
                        kc,
                        cbkc,
                        TILE_BYTES,
                        {.page_id = (h * NBGT + cg0 + c) * DT + n},
                        {.offset_bytes = (n * KCT + c) * TILE_BYTES});
                    if (++issued == 32) {
                        noc.async_read_barrier();
                        issued = 0;
                    }
                }
            }
            noc.async_read_barrier();
        }
        cbkc.push_back(KC_T);

        read_pages(qb, cb_q, h * NBPT * DT, NBPT * DT);

        // Per-core constants, once, after the first unit's data (see the header). Both CBs above hold a whole unit, so
        // the pushes before this point never block on compute, which waits for the constants only after the moments.
        if (u == unit_start) {
            read_pages(on, cb_ones, 0, 1);
            read_pages(qs, cb_qsel, 0, NQCT * NBPT);
            read_pages(ks, cb_ksel, 0, NBPT * NKCLT);
            if constexpr (NEED_EROW) {
                read_pages(er, cb_erow, 0, 32);
            }
            if constexpr (UNITS_PER_SHARD) {
                read_pages(sl, cb_shsel, 0, 8);
            }
        }

        // Partial-moment exchange with the S-1 other cores of the head (see the header).
        if constexpr (UNITS_PER_SHARD) {
            const uint32_t sp = sp_lo;
            const uint32_t sem_addr = get_semaphore(0);
            volatile tt_l1_ptr uint32_t* sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem_addr);
            const UnicastEndpoint remote;
            // This shard's 8 reduced rows -> its rows of the gather tiles (see the header): two 256 B runs.
            const uint32_t gt = recv_base + (sp >> 2) * TILE_BYTES;  // the gather tile of this shard
            const uint32_t run_l = gt + ((sp & 3) >> 1) * 1024 + (sp & 1) * 256;  // left halves (face 0 or 2)
            const uint32_t run_r = run_l + 512;                                     // right halves (face 1 or 3)
            cbpl.wait_front(PART_T);
            {
                const uint32_t pl_base = cbpl.get_read_ptr();
                CoreLocalMem<volatile uint32_t> l(run_l), r(run_r);
                for (uint32_t n = 0; n < PART_T; ++n) {
                    CoreLocalMem<volatile uint32_t> t(pl_base + n * TILE_BYTES);
                    for (uint32_t i = 0; i < 8; ++i) {
                        l[n * 8 + i] = t[i];
                        r[n * 8 + i] = t[128 + i];
                    }
                }
            }
            cbpl.pop_front(PART_T);
            uint32_t peer_xy[S];
            for (uint32_t ps = 0; ps < S; ++ps) {
                if (ps == sp) {
                    peer_xy[ps] = 0;
                    continue;
                }
                const uint32_t pu = h * S + ps;  // the peer's unit; its core is the one whose unit range holds pu
                uint32_t pc = 0;
                while (!((pc * UNITS / NCORES <= pu) && (pu < (pc + 1) * UNITS / NCORES))) {
                    ++pc;
                }
                peer_xy[ps] = get_common_arg_val<uint32_t>(10 + pc);
                const uint32_t px = peer_xy[ps] & 0xFF, py = (peer_xy[ps] >> 8) & 0xFF;
                for (uint32_t run : {run_l, run_r}) {
                    noc.async_write(
                        CoreLocalMem<uint32_t>(run), remote, 256, {}, {.noc_x = px, .noc_y = py, .addr = run});
                }
            }
            noc.async_write_barrier();  // every peer's rows have landed before any peer's semaphore moves
            for (uint32_t ps = 0; ps < S; ++ps) {
                if (ps != sp) {
                    noc_semaphore_inc(get_noc_addr(peer_xy[ps] & 0xFF, (peer_xy[ps] >> 8) & 0xFF, sem_addr), 1);
                }
            }
            noc_async_atomic_barrier();
            noc_semaphore_wait(sem, S - 1);
            noc_semaphore_set(sem, 0);  // quiescent: every peer's increment has landed; ready for the next launch
            cbrecv.push_back(GT);
        }

        for (uint32_t it = it_lo; it < it_hi; ++it) {
            for (uint32_t sp = sp_lo; sp < sp_hi; ++sp) {
                cbco.reserve_back(NBPT);
                for (uint32_t c = 0; c < NBPT; ++c) {
                    noc.async_read(
                        co, cbco, TILE_BYTES, {.page_id = it * NBGT + sp * NBPT + c}, {.offset_bytes = c * TILE_BYTES});
                }
                cbcc.reserve_back(3 * NBPT);
                for (uint32_t kind = 0; kind < 3; ++kind) {
                    for (uint32_t j = 0; j < NBPT; ++j) {
                        noc.async_read(
                            cc,
                            cbcc,
                            TILE_BYTES,
                            {.page_id = (kind * S + sp) * NBPT + j},
                            {.offset_bytes = (kind * NBPT + j) * TILE_BYTES});
                    }
                }
                noc.async_read_barrier();
                cbco.push_back(NBPT);
                cbcc.push_back(3 * NBPT);

                // Mask rows 16..31 of this (it, sp) (see the header); the writer does 0..15.
                if constexpr (UNITS_PER_SHARD) {
                    constexpr uint32_t NBP = NBPT * 32, NQCP = NQCT * 32;
                    cbm2.wait_front(NBPT);  // MASK_T = NBPT with (head, shard) units (MODE 1)
                    const uint32_t mbase = cbm2.get_read_ptr();
                    for (uint32_t r = 16; r < 32; ++r) {
                        if (ring_used != 0 && (ring_used % RING_PAGES) == 0) {
                            noc.async_write_barrier();  // the ring wraps
                        }
                        const uint32_t saddr = rring_base + (ring_used % RING_PAGES) * (NBP * 4);
                        tile_row_to_words(mbase, r, NBPT, saddr);
                        noc.async_write(
                            CoreLocalMem<uint32_t>(saddr),
                            mk,
                            NBP * 4,
                            {},
                            {.page_id = h * NQCP + it * 32 + r, .offset_bytes = sp * NBP * 4});
                        ++ring_used;
                    }
                    noc.async_write_barrier();
                    cbm2.pop_front(NBPT);
                }
            }
        }
    }
}
