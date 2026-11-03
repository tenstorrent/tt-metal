// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Sol-Attn routing pre-pass (proxy scores -> threshold -> chunk mask), reader (NCRISC).
//
// Work unit (compile-time arg 20 UNITS_PER_SHARD = 0): (head h, Q-chunk tile row it), u = h*NQCT + it: the routing of
// local Q chunks [32*it, 32*it+32) of head h against the pooled K blocks of every shard. UNITS_PER_SHARD = 1: (head h,
// column shard sp), u = h*S + sp: every Q-chunk tile row of head h against shard sp only (see sol_route_compute.cpp).
// Per core, once: the ones-row tile, the Q-chunk selector (qsel), the block->chunk selector (ksel) and the shard-row
// selectors (shsel, (head, shard) units). They are read after the first unit's pooled K and Q: every
// core needs the same few constant pages, and compute's first phases (the moments) need only the pooled K, so the data
// goes first and the constants land while the moments are summed. Per unit: the head's pooled K in column-major tile order
// (position n*KCT + c holds tile (c, n), so compute can use it as a transposed matmul operand): the whole head [NBG, D]
// (KCT = NBGT) with (head, it) units, only the unit's shard [NBP, D] (KCT = NBPT) with (head, shard) units; and the
// head's pooled Q [NBP, D]; then per (Q-chunk tile row it, column shard sp) of the unit: the constant OR-mask tiles
// of that Q-chunk tile row against shard sp's blocks, and the full-block column constant of sp.
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
// Compile-time args: 0 H, 1 S, 2 NBPT, 3 DT, 4 NQCT, 5 NKCLT, 6 NQC, 7 NKC, 8 NKCP, 9 MODE, 10 TAU, 11 EPS2, 12 NEG,
// 13 INV_NV, 14 TQ, 15 NX, 16 NY, 17 UNITS, 18 UNITS_PER_SHARD, then TensorAccessorArgs of q_bar, kc, const_or,
// colconst, qsel, ksel, ones, shsel, then Q_SHARED (see below).
// Common runtime args: 0..6 the buffer addresses in the same order, 7 the shsel address; 8..8+NX*NY-1 the NoC
// coordinates of every logical core ((head, shard) units only). Per-core: none. Semaphores ((head, shard) units):
// 0 partial-moment exchange, 1 pooled Q hand-off.

#include <cstdint>

#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/endpoints.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"

namespace {
constexpr uint32_t cb_kc = 0, cb_q = 1, cb_ones = 2, cb_qsel = 3, cb_ksel = 4, cb_constor = 6, cb_colc = 7,
                   cb_recv = 23, cb_pl = 24, cb_shsel = 29;
constexpr uint32_t PART_T = 8;  // partial-moment tiles per shard: DT sums + DT sums of squares (DT <= 4)

constexpr uint32_t TILE_BYTES = 2048;
}  // namespace

void kernel_main() {
    constexpr uint32_t S = get_compile_time_arg_val(1);
    constexpr uint32_t NBPT = get_compile_time_arg_val(2);
    constexpr uint32_t DT = get_compile_time_arg_val(3);
    constexpr uint32_t NQCT = get_compile_time_arg_val(4);
    constexpr uint32_t NKCLT = get_compile_time_arg_val(5);
    constexpr uint32_t NBGT = S * NBPT;
    constexpr uint32_t UNITS_PER_SHARD_ = get_compile_time_arg_val(18);
    constexpr uint32_t KCT = UNITS_PER_SHARD_ ? NBPT : NBGT;  // pooled-K column tiles per unit (see the header)
    constexpr uint32_t KC_T = KCT * DT;
    constexpr uint32_t Q_T = NBPT * DT;
    constexpr auto qb_args = TensorAccessorArgs<19>();
    constexpr auto kc_args = TensorAccessorArgs<qb_args.next_compile_time_args_offset()>();
    constexpr auto co_args = TensorAccessorArgs<kc_args.next_compile_time_args_offset()>();
    constexpr auto cc_args = TensorAccessorArgs<co_args.next_compile_time_args_offset()>();
    constexpr auto qs_args = TensorAccessorArgs<cc_args.next_compile_time_args_offset()>();
    constexpr auto ks_args = TensorAccessorArgs<qs_args.next_compile_time_args_offset()>();
    constexpr auto on_args = TensorAccessorArgs<ks_args.next_compile_time_args_offset()>();
    constexpr auto sl_args = TensorAccessorArgs<on_args.next_compile_time_args_offset()>();
    // Q_SHARED ((head, shard) units): the shard-0 core of each head reads the head's pooled Q from DRAM and ships it to
    // the head's other S-1 cores by NoC unicast (semaphore 1), instead of all S cores reading the same 12 tiles.
    constexpr uint32_t Q_SHARED = get_compile_time_arg_val(sl_args.next_compile_time_args_offset());

    constexpr uint32_t NX = get_compile_time_arg_val(15);
    constexpr uint32_t NY = get_compile_time_arg_val(16);
    constexpr uint32_t UNITS = get_compile_time_arg_val(17);
    constexpr uint32_t UNITS_PER_SHARD = get_compile_time_arg_val(18);  // 1: unit = (head, shard); 0: (head, it)
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
    CircularBuffer cbrecv(cb_recv), cbpl(cb_pl);
    uint32_t recv_base = 0;
    if constexpr (UNITS_PER_SHARD) {
        cbrecv.reserve_back(GT);
        recv_base = cbrecv.get_write_ptr();  // the gather tiles; the peers write their rows into these
    }
    const auto qb = TensorAccessor(qb_args, get_common_arg_val<uint32_t>(0));
    const auto kc = TensorAccessor(kc_args, get_common_arg_val<uint32_t>(1));
    const auto co = TensorAccessor(co_args, get_common_arg_val<uint32_t>(2));
    const auto cc = TensorAccessor(cc_args, get_common_arg_val<uint32_t>(3));
    const auto qs = TensorAccessor(qs_args, get_common_arg_val<uint32_t>(4));
    const auto ks = TensorAccessor(ks_args, get_common_arg_val<uint32_t>(5));
    const auto on = TensorAccessor(on_args, get_common_arg_val<uint32_t>(6));
    const auto sl = TensorAccessor(sl_args, get_common_arg_val<uint32_t>(7));

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

    // Logical core index holding unit pu under the [ci*U/N, (ci+1)*U/N) rule.
    auto core_of_unit = [](uint32_t pu) {
        uint32_t pc = 0;
        while (!((pc * UNITS / NCORES <= pu) && (pu < (pc + 1) * UNITS / NCORES))) {
            ++pc;
        }
        return pc;
    };
    CircularBuffer cbkc(cb_kc), cbq(cb_q), cbco(cb_constor), cbcc(cb_colc);
    for (uint32_t u = unit_start; u < unit_start + unit_count; ++u) {
        // Unit decode: one of the two ranges below is a singleton, the loops run the same code either way.
        const uint32_t h = UNITS_PER_SHARD ? u / S : u / NQCT;
        const uint32_t it_lo = UNITS_PER_SHARD ? 0 : u % NQCT;
        const uint32_t it_hi = UNITS_PER_SHARD ? NQCT : it_lo + 1;
        const uint32_t sp_lo = UNITS_PER_SHARD ? u % S : 0;
        const uint32_t sp_hi = UNITS_PER_SHARD ? sp_lo + 1 : S;

        // Pooled K of the head (whole head, or the unit's shard only), COLUMN-major tile order (tile c*DT + n = column
        // tile c, dims tile n): every column's reads carry their own transaction id and are pushed as soon as they
        // land, so compute's moment sums start on column 0 while the rest is in flight. Q is issued right behind K
        // (its own id) instead of after K's barrier.
        cbq.reserve_back(Q_T);
        const uint32_t q_addr = cbq.get_write_ptr();  // the same on every core (one CB configuration, one unit per core)
        const bool q_from_dram = !(Q_SHARED && UNITS_PER_SHARD) || sp_lo == 0;
        cbkc.reserve_back(KC_T);
        {
            const uint32_t cg0 = UNITS_PER_SHARD ? sp_lo * NBPT : 0;
            for (uint32_t c = 0; c < KCT; ++c) {
                for (uint32_t n = 0; n < DT; ++n) {
                    noc.async_read<NocOptions::TXN_ID>(
                        kc,
                        cbkc,
                        TILE_BYTES,
                        {.page_id = (h * NBGT + cg0 + c) * DT + n},
                        {.offset_bytes = (c * DT + n) * TILE_BYTES},
                        {.trid = 1 + (c % 7)});
                }
            }
        }
        if (q_from_dram) {
            for (uint32_t i = 0; i < Q_T; ++i) {
                noc.async_read<NocOptions::TXN_ID>(
                    qb, cbq, TILE_BYTES, {.page_id = h * NBPT * DT + i}, {.offset_bytes = i * TILE_BYTES}, {.trid = 8});
            }
        }
        for (uint32_t c = 0; c < KCT; ++c) {
            if (c >= 7) {  // ids wrap every 7 columns (never at the 5 s..15 s geometries, KCT = NBPT <= 7)
                noc.async_read_barrier();
            } else {
                noc.async_read_barrier<NocOptions::TXN_ID>({.trid = 1 + c});
            }
            cbkc.push_back(DT);
        }
        if (q_from_dram) {
            noc.async_read_barrier<NocOptions::TXN_ID>({.trid = 8});
            cbq.push_back(Q_T);
            if constexpr (Q_SHARED && UNITS_PER_SHARD) {  // ship the head's Q to its other cores: one 24 KB write per
                const UnicastEndpoint remote;              // peer, no barrier here; the partial-moment exchange below
                for (uint32_t ps = 1; ps < S; ++ps) {      // barriers all writes and then bumps their semaphore 1
                    const uint32_t pxy = get_common_arg_val<uint32_t>(8 + core_of_unit(h * S + ps));
                    noc.async_write(
                        CoreLocalMem<uint32_t>(q_addr), remote, Q_T * TILE_BYTES, {},
                        {.noc_x = pxy & 0xFF, .noc_y = (pxy >> 8) & 0xFF, .addr = q_addr});
                }
            }
        }
        // Receivers take Q after the exchange below (compute needs it only from phase E on, ~25 us into the unit).

        // Per-core constants (ones, qsel, ksel, [shsel]): the same tiles on every core, read by every core itself
        // after its first unit's data (both CBs above hold a whole unit, so nothing before this point blocks on
        // compute). A NoC multicast from an idle core was tried and hung at the 10 s / 15 s geometries, where the
        // blocks exceed one 16 KB burst; per-core reads cost a few us and are safe.
        if (u == unit_start) {
            read_pages(on, cb_ones, 0, 1);
            read_pages(qs, cb_qsel, 0, NQCT * NBPT);
            read_pages(ks, cb_ksel, 0, NBPT * NKCLT);
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
                peer_xy[ps] = get_common_arg_val<uint32_t>(8 + core_of_unit(h * S + ps));  // the peer's core
                const uint32_t px = peer_xy[ps] & 0xFF, py = (peer_xy[ps] >> 8) & 0xFF;
                for (uint32_t run : {run_l, run_r}) {
                    noc.async_write(
                        CoreLocalMem<uint32_t>(run), remote, 256, {}, {.noc_x = px, .noc_y = py, .addr = run});
                }
            }
            noc.async_write_barrier();  // every peer's rows (and the sender's Q tiles) have landed before any semaphore moves
            for (uint32_t ps = 0; ps < S; ++ps) {
                if (ps != sp) {
                    noc_semaphore_inc(get_noc_addr(peer_xy[ps] & 0xFF, (peer_xy[ps] >> 8) & 0xFF, sem_addr), 1);
                    if constexpr (Q_SHARED) {
                        if (sp == 0) {
                            noc_semaphore_inc(get_noc_addr(peer_xy[ps] & 0xFF, (peer_xy[ps] >> 8) & 0xFF, get_semaphore(1)), 1);
                        }
                    }
                }
            }
            noc_async_atomic_barrier();
            noc_semaphore_wait(sem, S - 1);
            noc_semaphore_set(sem, 0);  // quiescent: every peer's increment has landed; ready for the next launch
            cbrecv.push_back(GT);
            if constexpr (Q_SHARED) {
                if (!q_from_dram) {  // Q from the head's shard-0 core (semaphore 1); its tiles landed before the increment
                    volatile tt_l1_ptr uint32_t* sem2 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(1));
                    noc_semaphore_wait(sem2, 1);
                    noc_semaphore_set(sem2, 0);
                    cbq.push_back(Q_T);
                }
            }
        }

        for (uint32_t it = it_lo; it < it_hi; ++it) {
            for (uint32_t sp = sp_lo; sp < sp_hi; ++sp) {
                cbco.reserve_back(NBPT);
                for (uint32_t c = 0; c < NBPT; ++c) {
                    noc.async_read(
                        co, cbco, TILE_BYTES, {.page_id = it * NBGT + sp * NBPT + c}, {.offset_bytes = c * TILE_BYTES});
                }
                cbcc.reserve_back(NBPT);
                for (uint32_t j = 0; j < NBPT; ++j) {
                    noc.async_read(cc, cbcc, TILE_BYTES, {.page_id = sp * NBPT + j}, {.offset_bytes = j * TILE_BYTES});
                }
                noc.async_read_barrier();
                cbco.push_back(NBPT);
                cbcc.push_back(NBPT);
            }
        }
    }
    // The NoC packet tag is sticky per command buffer: leave it at 0 for whatever kernel runs next on this core.
    noc_async_read_set_trid(0, noc.get_noc_id());
}
