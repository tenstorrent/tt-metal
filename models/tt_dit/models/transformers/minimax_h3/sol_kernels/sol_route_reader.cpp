// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Sol-Attn pre-pass, route step, reader kernel.
//
// Feeds the compute kernel. Per unit: the head's pooled K tiles (this shard's only with (head, shard) units), the
// head's pooled Q tiles, then per (Q-chunk tile row, shard) the constant OR-mask tiles and the valid-block row. Once
// per core, after the first unit's data: the ones tile and the selector tiles.
//
// (head, shard) units: a core only has its shard's K, so the head's cores exchange partial moments. Compute hands
// this kernel 8 reduced rows (cb_pl); they are written into the gather tiles (cb_recv) at this shard's slot on every
// core of the head, then semaphore 0 is bumped on each peer. When the semaphore reaches S-1 every shard's rows have
// arrived. With Q_SHARED the shard-0 core reads the pooled Q and sends it to the other cores (semaphore 1).
//
// Compile-time args: 0..18 as sol_route_compute.cpp, then TensorAccessorArgs of q_bar, kc, const_or, colconst, qsel,
// ksel, ones, shsel, then Q_SHARED.
// Common runtime args: 0..7 the buffer addresses in that order, 8.. the NoC coordinates of every logical core
// (x | y << 8). Semaphores: 0 moment exchange, 1 Q hand-off.

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
constexpr uint32_t PART_T = 8;  // DT sum tiles + DT sum-of-squares tiles

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
    constexpr uint32_t KCT = UNITS_PER_SHARD_ ? NBPT : NBGT;  // K column tiles per unit
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
    // Q_SHARED: the shard-0 core reads the head's Q and sends it to the other cores (semaphore 1).
    constexpr uint32_t Q_SHARED = get_compile_time_arg_val(sl_args.next_compile_time_args_offset());

    constexpr uint32_t NX = get_compile_time_arg_val(15);
    constexpr uint32_t NY = get_compile_time_arg_val(16);
    constexpr uint32_t UNITS = get_compile_time_arg_val(17);
    constexpr uint32_t UNITS_PER_SHARD = get_compile_time_arg_val(18);  // 1: unit = (head, shard); 0: (head, it)
    static_assert(UNITS_PER_SHARD == 0 || get_compile_time_arg_val(9) == 1, "(head, shard) units are MODE 1 only");
    // This core's units, [ci*U/N, (ci+1)*U/N), from its logical coordinates.
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
    // Gather tiles for the exchanged moments, reserved once; the same address on every core.
    CircularBuffer cbrecv(cb_recv), cbpl(cb_pl);
    uint32_t recv_base = 0;
    if constexpr (UNITS_PER_SHARD) {
        cbrecv.reserve_back(GT);
        recv_base = cbrecv.get_write_ptr();  // peers write their rows here
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

    // The core that owns unit pu.
    auto core_of_unit = [](uint32_t pu) {
        uint32_t pc = 0;
        while (!((pc * UNITS / NCORES <= pu) && (pu < (pc + 1) * UNITS / NCORES))) {
            ++pc;
        }
        return pc;
    };
    CircularBuffer cbkc(cb_kc), cbq(cb_q), cbco(cb_constor), cbcc(cb_colc);
    for (uint32_t u = unit_start; u < unit_start + unit_count; ++u) {
        // One of the two ranges is a single value; the loops are the same either way.
        const uint32_t h = UNITS_PER_SHARD ? u / S : u / NQCT;
        const uint32_t it_lo = UNITS_PER_SHARD ? 0 : u % NQCT;
        const uint32_t it_hi = UNITS_PER_SHARD ? NQCT : it_lo + 1;
        const uint32_t sp_lo = UNITS_PER_SHARD ? u % S : 0;
        const uint32_t sp_hi = UNITS_PER_SHARD ? sp_lo + 1 : S;

        // Pooled K, column-major (tile c*DT + n). Each column has its own transaction id and is pushed as it lands;
        // Q is read right behind it.
        cbq.reserve_back(Q_T);
        const uint32_t q_addr = cbq.get_write_ptr();  // the same address on every core
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
            if (c >= 7) {  // ids wrap every 7 columns
                noc.async_read_barrier();
            } else {
                noc.async_read_barrier<NocOptions::TXN_ID>({.trid = 1 + c});
            }
            cbkc.push_back(DT);
        }
        if (q_from_dram) {
            noc.async_read_barrier<NocOptions::TXN_ID>({.trid = 8});
            cbq.push_back(Q_T);
            if constexpr (Q_SHARED && UNITS_PER_SHARD) {  // send the head's Q to the other cores; the exchange
                const UnicastEndpoint remote;             // below waits for the writes and bumps semaphore 1
                for (uint32_t ps = 1; ps < S; ++ps) {
                    const uint32_t pxy = get_common_arg_val<uint32_t>(8 + core_of_unit(h * S + ps));
                    noc.async_write(
                        CoreLocalMem<uint32_t>(q_addr), remote, Q_T * TILE_BYTES, {},
                        {.noc_x = pxy & 0xFF, .noc_y = (pxy >> 8) & 0xFF, .addr = q_addr});
                }
            }
        }
        // Receivers push Q after the exchange.

        // Per-core constants, read once.
        if (u == unit_start) {
            read_pages(on, cb_ones, 0, 1);
            read_pages(qs, cb_qsel, 0, NQCT * NBPT);
            read_pages(ks, cb_ksel, 0, NBPT * NKCLT);
            if constexpr (UNITS_PER_SHARD) {
                read_pages(sl, cb_shsel, 0, 8);
            }
        }

        // Partial-moment exchange with the head's other cores.
        if constexpr (UNITS_PER_SHARD) {
            const uint32_t sp = sp_lo;
            const uint32_t sem_addr = get_semaphore(0);
            volatile tt_l1_ptr uint32_t* sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem_addr);
            const UnicastEndpoint remote;
            // This shard's 8 rows -> its slot in the gather tiles: two 256 B runs.
            const uint32_t gt = recv_base + (sp >> 2) * TILE_BYTES;               // gather tile of this shard
            const uint32_t run_l = gt + ((sp & 3) >> 1) * 1024 + (sp & 1) * 256;  // left halves
            const uint32_t run_r = run_l + 512;                                   // right halves
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
            noc.async_write_barrier();  // all rows have landed before any semaphore moves
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
            noc_semaphore_set(sem, 0);  // reset for the next launch
            cbrecv.push_back(GT);
            if constexpr (Q_SHARED) {
                if (!q_from_dram) {  // Q sent by the shard-0 core
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
    // Leave the NoC transaction id at 0 for the next kernel on this core.
    noc_async_read_set_trid(0, noc.get_noc_id());
}
