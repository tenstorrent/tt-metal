// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Activation loader, cross-core row scatter and output writer (BRISC, NOC1) of the per-expert dispatch down op in
// its EXPERT-PARALLEL split (custom_ops/expert_stream/op.py, ExpertDownDispatch(down_split="experts"); decode
// roofline lever E3). This core (bank b, index `reader` of the bank's `readers` cores) plays two roles:
//
// * producer: its work units (expert e, column group q) -- dispatch_common.hpp for_each_unit over the bank's
//   active experts by pass count, the same replay as the reader and (through the metadata page) the compute -- get
//   their pass activations act[k] loaded here (seg_tiles tiles per pass, a_slots ahead), and every finished pass
//   h_k[:, q unit_cols ..] (unit_cols BF16 tiles in cb_hs) is scattered ROW BY ROW into the gather slots of the
//   cores owning those columns: row i of pass k (token t, the pass table) lands in slot j = the number of earlier
//   passes holding t (the position the column-split writer gives it with its per-row-tile cursor scan, so the
//   fp32 slot sums run in the same order and the output is bit-identical), tile row t % 32 of output tile row
//   t / 32, as two 32 B NOC writes per (row, tile). Rows past the J slots are dropped as there.
// * owner: its `cols_own` output columns [col0, col0 + cols_own) of every token row: the gather slots of ALL
//   row tiles (RT x J x cols_own tiles, cb_g) stay resident; they are zero-filled over the NOC from a zero tile
//   before the bank's cores are told (semaphore `ready`) that rows may land; when every producer of the bank has
//   signalled its passes (semaphore `arrive` reaches n_pass x incs_per_pass: each producer counts, after a write
//   barrier, the passes it sent to this owner) the row tiles go to the compute one by one with their used slot
//   count (mailbox) and the summed output tiles are written out.
// Both roles run in one non-blocking loop (cb_pages_reservable_at_back / cb_pages_available_at_front) so neither
// starves the other; the only blocking waits are the NOC barriers of this core's own transfers and, after the
// producer role is done, the two semaphores (every producer's signals are unconditional, so no cycle).
//
// The CB addresses are identical on every core of the program (one CB set on one core range), so a producer
// addresses an owner's slots through its own cb_g base. Semaphores are program-initialised to 0 at every launch
// (trace replays included) and left at 0 on exit.
//
// runtime args: [counts_addr, table_addr, act_addr, out_addr, col0, reader, xy_0 .. xy_{readers-1}] (the bank's
// cores' NOC coordinates x << 16 | y in reader order)
// REGIONS_SRC (regions_src.hpp): the output tile column of this core's c-th column is bank + banks * (r * cols + c)
// (from col0 = bank * C + r * cols); `banks` is the compile-time arg after the accessor args.

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "ckernel.h"
#include "ckernel_defs.h"
#include "dispatch_common.hpp"
#include "regions_src.hpp"

void kernel_main() {
    const uint32_t counts_addr = get_arg_val<uint32_t>(0);
    const uint32_t table_addr = get_arg_val<uint32_t>(1);
    const uint32_t act_addr = get_arg_val<uint32_t>(2);
    const uint32_t out_addr = get_arg_val<uint32_t>(3);
    const uint32_t col0 = get_arg_val<uint32_t>(4);
    const uint32_t reader = get_arg_val<uint32_t>(5);
    constexpr uint32_t ARG_XY = 6;

    constexpr uint32_t cb_c = get_compile_time_arg_val(0);
    constexpr uint32_t cb_t = get_compile_time_arg_val(1);
    constexpr uint32_t cb_a = get_compile_time_arg_val(2);
    constexpr uint32_t cb_id = get_compile_time_arg_val(3);
    constexpr uint32_t cb_hs = get_compile_time_arg_val(4);
    constexpr uint32_t cb_g = get_compile_time_arg_val(5);
    constexpr uint32_t cb_out = get_compile_time_arg_val(6);
    constexpr uint32_t cb_m = get_compile_time_arg_val(7);  // metadata page: [n_units, passes per unit]
    constexpr uint32_t cb_j = get_compile_time_arg_val(8);  // slot-index table: 32 uint16 per pass + the j_max row
    constexpr uint32_t cb_z = get_compile_time_arg_val(9);  // one zero tile (the slot fill source)
    constexpr uint32_t seg_tiles = get_compile_time_arg_val(10);
    constexpr uint32_t E = get_compile_time_arg_val(11);
    constexpr uint32_t T = get_compile_time_arg_val(12);
    constexpr uint32_t cols_own = get_compile_time_arg_val(13);   // output columns this core owns (reduces)
    constexpr uint32_t unit_cols = get_compile_time_arg_val(14);  // output columns per work unit (computes)
    constexpr uint32_t UPE = get_compile_time_arg_val(15);        // units per expert
    constexpr uint32_t readers = get_compile_time_arg_val(16);    // cores per bank
    constexpr uint32_t J = get_compile_time_arg_val(17);
    constexpr uint32_t P_MAX = get_compile_time_arg_val(18);
    constexpr uint32_t h_tiles = get_compile_time_arg_val(19);  // output tile columns of the whole row (H / 32)
    constexpr uint32_t table_bytes = get_compile_time_arg_val(20);
    constexpr uint32_t sem_ready = get_compile_time_arg_val(21);
    constexpr uint32_t sem_arrive = get_compile_time_arg_val(22);
    constexpr auto counts_args = TensorAccessorArgs<23>();
    constexpr auto table_args = TensorAccessorArgs<counts_args.next_compile_time_args_offset()>();
    constexpr auto act_args = TensorAccessorArgs<table_args.next_compile_time_args_offset()>();
    constexpr auto out_args = TensorAccessorArgs<act_args.next_compile_time_args_offset()>();
#ifdef REGIONS_SRC
    constexpr uint32_t banks = get_compile_time_arg_val(out_args.next_compile_time_args_offset());
    constexpr uint32_t C = h_tiles / banks;  // output columns per bank
#endif

    constexpr uint32_t E_TILES = (E + 31) / 32;
    constexpr uint32_t RT = T / 32;
    constexpr uint32_t tile_page = 2048;
    constexpr uint32_t slot_tiles = J * cols_own;
    constexpr uint32_t table_elems = table_bytes / 2;
    constexpr uint32_t incs_per_pass = cols_own > unit_cols ? cols_own / unit_cols : 1;
    static_assert(unit_cols % cols_own == 0 || cols_own % unit_cols == 0, "unit and owner columns must nest");
    static_assert(E < 256 && T <= 256 && RT <= 32, "uint8 counters: counts per token, row tiles");
    static_assert(readers <= 32, "for_each_unit's group bound");
    static_assert(
        P_MAX == dispatch_max_passes(T, E), "the op's pass budget (op.py max_passes) is the kernels' formula");

    const auto s_tab = TensorAccessor(table_args, table_addr, table_bytes);
    const auto s_act = TensorAccessor(act_args, act_addr, tile_page);
    const auto s_out = TensorAccessor(out_args, out_addr, tile_page);

    // counts -> passes
    cb_reserve_back(cb_c, E_TILES);
    const uint32_t c_l1 = get_write_ptr(cb_c);
    const auto s_c = TensorAccessor(counts_args, counts_addr, tile_page);
    for (uint32_t i = 0; i < E_TILES; ++i) {
        noc_async_read(s_c.get_noc_addr(i), c_l1 + i * tile_page, tile_page);
    }
    uint16_t n_tok[E], passes[E], base[E];
    noc_async_read_barrier();
    read_counts<E>(reinterpret_cast<volatile tt_l1_ptr uint16_t*>(c_l1), n_tok);
    pass_bases<E>(n_tok, passes, base);
    bool clamped = false;  // the pass budget applied to passes[] / base[] / n_tok[] themselves (dispatch_common.hpp)
    const uint32_t n_pass = clamp_passes<E>(n_tok, passes, base, P_MAX, clamped);

    // the pass table
    cb_reserve_back(cb_t, P_MAX);
    const uint32_t t_l1 = get_write_ptr(cb_t);
    for (uint32_t k = 0; k < n_pass; ++k) {
        noc_async_read(s_tab.get_noc_addr(k), t_l1 + k * table_bytes, table_bytes);
    }
    // the identity tile (BF16 1.0 = 0x3F80 on the diagonal) for the gather sum, and the zero tile
    cb_reserve_back(cb_id, 1);
    const uint32_t id_l1 = get_write_ptr(cb_id);
    volatile tt_l1_ptr uint32_t* id32 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(id_l1);
    for (uint32_t j = 0; j < tile_page / 4; ++j) {
        id32[j] = 0;
    }
    volatile tt_l1_ptr uint16_t* id16 = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(id_l1);
    for (uint32_t t = 0; t < 32; ++t) {
        id16[tile_elem(t, t)] = 0x3F80;
    }
    cb_push_back(cb_id, 1);
    cb_reserve_back(cb_z, 1);
    const uint32_t z_l1 = get_write_ptr(cb_z);
    volatile tt_l1_ptr uint32_t* z32 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(z_l1);
    for (uint32_t j = 0; j < tile_page / 4; ++j) {
        z32[j] = 0;
    }

    // the bank's cores (NOC bases, reader order); owner o reduces columns [o cols_own, (o + 1) cols_own)
    uint64_t core_base[readers];
    for (uint32_t o = 0; o < readers; ++o) {
        const uint32_t xy = get_arg_val<uint32_t>(ARG_XY + o);
        core_base[o] = get_noc_addr(xy >> 16, xy & 0xFFFFu, 0);
    }
    const uint32_t ready_addr = get_semaphore(sem_ready);
    const uint32_t arrive_addr = get_semaphore(sem_arrive);

    // this core's units (the same LPT replay as the reader) and the passes each owner will receive from it
    uint16_t mine_e[E * UPE];
    uint8_t mine_q[E * UPE];
    uint32_t n_units = 0;
    uint32_t incr[readers];
    for (uint32_t o = 0; o < readers; ++o) {
        incr[o] = 0;
    }
    for_each_unit<E, UPE, 32>(passes, readers, reader, [&](uint32_t e, uint32_t q) {
        mine_e[n_units] = static_cast<uint16_t>(e);
        mine_q[n_units] = static_cast<uint8_t>(q);
        ++n_units;
        const uint32_t o_lo = (q * unit_cols) / cols_own;
        const uint32_t o_hi = ((q + 1) * unit_cols - 1) / cols_own;
        for (uint32_t o = o_lo; o <= o_hi; ++o) {
            incr[o] += passes[e];
        }
    });
    // the metadata page for the compute: [n_units, passes per unit]
    cb_reserve_back(cb_m, 1);
    volatile tt_l1_ptr uint32_t* meta = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_m));
    meta[0] = n_units;
    for (uint32_t u = 0; u < n_units; ++u) {
        meta[1 + u] = passes[mine_e[u]];
    }
    cb_push_back(cb_m, 1);

    // the slot-index table: row k = the slot of each row of pass k (earlier passes holding the token), row P_MAX =
    // the used slot count per output row tile (min(count, J) over its tokens), both pure functions of the table
    noc_async_read_barrier();
    const volatile tt_l1_ptr uint16_t* tab = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(t_l1);
    cb_reserve_back(cb_j, 1);
    volatile tt_l1_ptr uint16_t* jtab = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(get_write_ptr(cb_j));
    volatile tt_l1_ptr uint16_t* j_max = jtab + P_MAX * 32;
    {
        uint8_t cnt[T];
        for (uint32_t t = 0; t < T; ++t) {
            cnt[t] = 0;
        }
        for (uint32_t k = 0; k < n_pass; ++k) {
            const volatile tt_l1_ptr uint16_t* row = tab + k * table_elems;
            const uint32_t nr = row[0];
            for (uint32_t i = 0; i < nr; ++i) {
                const uint32_t t = row[2 + i];
                jtab[k * 32 + i] = cnt[t];
                if (cnt[t] != 255) {
                    ++cnt[t];
                }
            }
        }
        for (uint32_t r = 0; r < RT; ++r) {
            j_max[r] = 0;
        }
        for (uint32_t t = 0; t < T; ++t) {
            const uint8_t used = cnt[t] > J ? static_cast<uint8_t>(J) : cnt[t];
            if (used > j_max[t >> 5]) {
                j_max[t >> 5] = used;
            }
        }
    }

    // owner role, step 1: every gather slot of every row tile zeroed (NOC copies of the zero tile, addressed
    // through this core's host-supplied coordinates like every other target), then the bank's cores may land rows
    cb_reserve_back(cb_g, RT * slot_tiles);
    const uint32_t g_base = get_write_ptr(cb_g);
    const uint64_t self_base = core_base[reader];
    for (uint32_t s = 0; s < RT * slot_tiles; ++s) {
        noc_async_write(z_l1, self_base + g_base + s * tile_page, tile_page);
    }
    noc_async_write_barrier();
    for (uint32_t o = 0; o < readers; ++o) {
        noc_semaphore_inc(core_base[o] + ready_addr, 1);
    }

    // ---- phase 1: the producer role (activations ahead, finished passes scattered) as one non-blocking loop
    uint32_t load_u = 0, load_p = 0;  // next pass to load: unit index, pass within the unit
    uint32_t send_u = 0, send_p = 0;  // next pass to scatter
    bool ready_seen = false;
    volatile tt_l1_ptr uint32_t* ready_sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ready_addr);
    while (send_u < n_units) {
        if (load_u < n_units && cb_pages_reservable_at_back(cb_a, seg_tiles)) {
            const uint32_t k = base[mine_e[load_u]] + load_p;
            cb_reserve_back(cb_a, seg_tiles);
            const uint32_t a_l1 = get_write_ptr(cb_a);
            for (uint32_t s = 0; s < seg_tiles; ++s) {
                noc_async_read(s_act.get_noc_addr(k * seg_tiles + s), a_l1 + s * tile_page, tile_page);
            }
            noc_async_read_barrier();
            cb_push_back(cb_a, seg_tiles);
            if (++load_p == passes[mine_e[load_u]]) {
                load_p = 0;
                ++load_u;
            }
        }
        if (cb_pages_available_at_front(cb_hs, unit_cols)) {
            if (!ready_seen) {  // every owner of the bank has zeroed its slots
                noc_semaphore_wait(ready_sem, readers);
                ready_seen = true;
            }
            const uint32_t e = mine_e[send_u];
            const uint32_t q = mine_q[send_u];
            const uint32_t k = base[e] + send_p;
            const volatile tt_l1_ptr uint16_t* row = tab + k * table_elems;
            const volatile tt_l1_ptr uint16_t* jrow = jtab + k * 32;
            const uint32_t nr = row[0];
            const uint32_t hs_l1 = get_read_ptr(cb_hs);
            for (uint32_t i = 0; i < nr; ++i) {
                const uint32_t j = jrow[i];
                if (j >= J) {
                    continue;  // the slot cap of the column-split writer
                }
                const uint32_t t = row[2 + i];
                const uint32_t src_row = hs_l1 + tile_row_bytes(i);
                const uint32_t dst_row = g_base + ((t >> 5) * J + j) * cols_own * tile_page + tile_row_bytes(t & 31);
                for (uint32_t c = 0; c < unit_cols; ++c) {
                    const uint32_t cc = q * unit_cols + c;  // column within the bank
                    const uint64_t dst = core_base[cc / cols_own] + dst_row + (cc % cols_own) * tile_page;
                    const uint32_t src = src_row + c * tile_page;
                    noc_async_write(src, dst, 32);
                    noc_async_write(src + 512, dst + 512, 32);
                }
            }
            noc_async_writes_flushed();  // the staging tiles are read out before the packer reuses them
            cb_pop_front(cb_hs, unit_cols);
            if (++send_p == passes[e]) {
                send_p = 0;
                ++send_u;
            }
        }
    }
    // every row this core scattered is acknowledged: count the passes on their owners
    noc_async_write_barrier();
    for (uint32_t o = 0; o < readers; ++o) {
        if (incr[o] != 0) {
            noc_semaphore_inc(core_base[o] + arrive_addr, incr[o]);
        }
    }
    if (!ready_seen) {  // nothing to send: still consume the bank's ready signals before resetting the counter
        noc_semaphore_wait(ready_sem, readers);
    }

    // ---- phase 2: the owner role -- all passes landed, the row tiles to the compute, the output tiles out
    noc_semaphore_wait(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(arrive_addr), n_pass * incs_per_pass);
    for (uint32_t r = 0; r < RT; ++r) {
        const uint32_t jm = j_max[r];
        ckernel::mailbox_write(ckernel::ThreadId::UnpackThreadId, jm);
        ckernel::mailbox_write(ckernel::ThreadId::MathThreadId, jm);
        ckernel::mailbox_write(ckernel::ThreadId::PackThreadId, jm);
        cb_push_back(cb_g, slot_tiles);
        cb_wait_front(cb_out, cols_own);
        const uint32_t o_l1 = get_read_ptr(cb_out);
        for (uint32_t c = 0; c < cols_own; ++c) {
#ifdef REGIONS_SRC
            const uint32_t col = regions_out_col(col0, C, banks, c);
#else
            const uint32_t col = col0 + c;
#endif
            noc_async_write(o_l1 + c * tile_page, s_out.get_noc_addr(r * h_tiles + col), tile_page);
        }
        noc_async_write_barrier();
        cb_pop_front(cb_out, cols_own);
    }
    noc_semaphore_set(ready_sem, 0);
    noc_semaphore_set(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(arrive_addr), 0);
    noc_async_full_barrier();  // exit contract: every NOC read / write / atomic acknowledged before the next kernel
}
