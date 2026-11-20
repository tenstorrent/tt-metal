// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// The hub core of a chip in the HEAD program of the fused decode step (BRISC, NOC1) (fused/model/ends_fused.py
// EndsHead; PLAN.md section 46). Data movement only.
//
//   1. wait (bounded) for the normed head readout x in this core's x buffer: word W_X of the MLP half's transport
//      words, counted by the root chip's sender (X1'); take the count off.
//   2. x to every worker core of the chip (one multicast) and one multicast increment of the units' release word
//      (the MoE hub's two calls; this core is the chip's only multicaster).
//   3. wait (bounded) until every unit has delivered its rows' K best (key, id): word W_GATHER of this core, one
//      count per unit; merge the units' lists into the chip's K best per row and add the chip's first token id.
//   4. hand the chip's list (ROWS x K x 8 bytes) on:
//        mode 0 (not a head)        to this chip's sender core (one op: to the row's head)
//        mode 1 (a head)            wait (bounded) for the row's other chips' lists in the stage-1 landing (word
//                                   W_S1), merge them with this chip's, to this chip's sender core (to the root)
//        mode 2 (the root)          stage 1 as a head; then wait (bounded) for the other heads' lists in the stage-2
//                                   landing (word W_S2), merge, and write the rows' best ids (16 uint32, rows 0..7)
//                                   to the id tensor in DRAM: the next step's gather reads them there.
// A merge keeps, per row, the K entries with the largest keys; of equal keys the lower id is ahead (as
// torch.argmax and torch.topk over the whole vocabulary would put it for K = 1).
// A wait that runs out leaves its code in this core's status word (W_STATUS) and the kernel carries on with what it
// has, so the program ends and no chip downstream waits for longer than its own bound.
//
// Stamps (HY4L_TIMING; this core's wall clock, data movement): 0 start, 1 x arrived, 2 the units released, 3 every
// unit's list in, 4 the chip's list handed on (or the ids written).
//
// Words: the MLP half's transport words on this core (layer_common.hpp; HY4L_WORDS): W_X, W_GATHER, W_S1, W_S2,
// W_STATUS, W_RUNS, W_FED. On the hub core the MLP program uses W_X, W_STATUS, W_RUNS and W_FED only, in the same
// meaning; W_GATHER, W_S1 and W_S2 are this kernel's alone there.
//
// compile-time args: [cb_x, x_bytes, mc_x0, mc_y0, mc_x1, mc_y1, mc_dests, units, K, <the id tensor's accessor args>]
// runtime args:
//   0 x address (this core)      1 the units' slots (this core: units x ROWS x K pairs)
//   2 scratch (this core: 3 lists and one 64-byte page)       3 the chip's first token id       4 mode
//   5 stage-1 landing (this core: n1 slots of one list)   6 n1 (chips of the row that take part)   7 this chip's slot
//   8 stage-2 landing (this core: n2 slots)               9 n2 (rows that take part)              10 this chip's slot
//   11 sender core NoC x | y << 16   12 staging address   13 ready word address   14 ready bit
//   15 the id tensor's address (mode 2)
// Defines: HY4_SYNC (the units' sync words), HY4L_* (the transport's), ROWS.

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "moe_dm.hpp"
#include "layer_common.hpp"

namespace {

constexpr uint32_t ST_UNITS = 0xD1u << 24;  // the units did not all deliver
constexpr uint32_t ST_ROW = 0xD2u << 24;    // the row's other chips' lists did not all land
constexpr uint32_t ST_HEADS = 0xD3u << 24;  // the other heads' lists did not all land
constexpr uint32_t ST_SEND = 0xD4u << 24;   // a multicast or a write could not be issued (what: low bits)

struct Pair {
    uint32_t key;
    uint32_t id;
};

// Insert into a list of K (best first): larger key first, of equal keys the lower id.
template <uint32_t K>
FORCE_INLINE void insert(Pair* list, uint32_t key, uint32_t id) {
    const Pair& last = list[K - 1];
    if (key > last.key || (key == last.key && id < last.id)) {
        uint32_t p = K - 1;
        while (p > 0 && (key > list[p - 1].key || (key == list[p - 1].key && id < list[p - 1].id))) {
            list[p] = list[p - 1];
            --p;
        }
        list[p].key = key;
        list[p].id = id;
    }
}

// dst (ROWS x K) = the K best per row of `n` lists of ROWS x K at `src` (stride list_bytes), skipping list `skip`
// (the chip's own slot of a landing: its list is `own`, merged in first), ids + `add`.
template <uint32_t K>
inline void merge(Pair* dst, const Pair* own, uint32_t src, uint32_t n, uint32_t skip, uint32_t add) {
    constexpr uint32_t list_bytes = ROWS * K * sizeof(Pair);
    for (uint32_t r = 0; r < ROWS; ++r) {
        Pair* list = dst + r * K;
        for (uint32_t k = 0; k < K; ++k) {
            if (own != nullptr) {
                list[k] = own[r * K + k];
            } else {
                list[k].key = 0;
                list[k].id = 0xFFFFFFFFu;
            }
        }
        for (uint32_t i = 0; i < n; ++i) {
            if (i == skip) {
                continue;
            }
            const volatile tt_l1_ptr Pair* in =
                reinterpret_cast<volatile tt_l1_ptr Pair*>(src + i * list_bytes) + r * K;
            for (uint32_t k = 0; k < K; ++k) {
                const uint32_t key = in[k].key;
                const uint32_t id = in[k].id + add;
                insert<K>(list, key, id);
            }
        }
    }
}

// The canary on the landings (PLAN.md section 47): the hub core's own sync words 8..10 (it is no unit): checks made,
// checks that failed, the code of the last failure. What is checked is the order the landings rely on: when a
// count is taken it is EXACTLY what this step must deliver (a larger one is a second x, or a list of the next step
// that landed before this step's lists were merged), and when the kernel ends the stage words are zero again
// (nothing of the next step can have arrived: its x leaves the root only after the ids of this step exist).
constexpr uint32_t CANARY = 8;
FORCE_INLINE void canary(bool ok, uint32_t code) {
    volatile tt_l1_ptr uint32_t* c = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(hy4moe::sem(CANARY));
    c[0] = c[0] + 1;
    if (!ok) {
        c[1] = c[1] + 1;
        c[2] = code;
    }
}
FORCE_INLINE uint32_t now(uint32_t word) {
    invalidate_l1_cache();
    return hy4l::word(word)[0];
}

FORCE_INLINE void take(uint32_t word, uint32_t n) {
    if (n != 0) {
        hy4l::take_word(word, n);
    }
}

}  // namespace

void kernel_main() {
    constexpr uint32_t cb_x = get_compile_time_arg_val(0);
    constexpr uint32_t x_bytes = get_compile_time_arg_val(1);
    constexpr uint32_t mc_x0 = get_compile_time_arg_val(2);
    constexpr uint32_t mc_y0 = get_compile_time_arg_val(3);
    constexpr uint32_t mc_x1 = get_compile_time_arg_val(4);
    constexpr uint32_t mc_y1 = get_compile_time_arg_val(5);
    constexpr uint32_t mc_dests = get_compile_time_arg_val(6);
    constexpr uint32_t units = get_compile_time_arg_val(7);
    constexpr uint32_t K = get_compile_time_arg_val(8);
    constexpr auto ids_args = TensorAccessorArgs<9>();
    constexpr uint32_t list_bytes = ROWS * K * sizeof(Pair);

    const uint32_t x_addr = get_arg_val<uint32_t>(0);
    const uint32_t slots_addr = get_arg_val<uint32_t>(1);
    const uint32_t scratch = get_arg_val<uint32_t>(2);
    const uint32_t id0 = get_arg_val<uint32_t>(3);
    const uint32_t mode = get_arg_val<uint32_t>(4);
    const uint32_t land1 = get_arg_val<uint32_t>(5);
    const uint32_t n1 = get_arg_val<uint32_t>(6);
    const uint32_t idx1 = get_arg_val<uint32_t>(7);
    const uint32_t land2 = get_arg_val<uint32_t>(8);
    const uint32_t n2 = get_arg_val<uint32_t>(9);
    const uint32_t idx2 = get_arg_val<uint32_t>(10);
    const uint32_t sender = get_arg_val<uint32_t>(11);
    const uint32_t staging = get_arg_val<uint32_t>(12);
    const uint32_t ready = get_arg_val<uint32_t>(13);
    const uint32_t ready_bit = get_arg_val<uint32_t>(14);
    const uint32_t ids_addr = get_arg_val<uint32_t>(15);

    Pair* chip = reinterpret_cast<Pair*>(scratch);                  // the chip's list
    Pair* row = reinterpret_cast<Pair*>(scratch + list_bytes);      // the mesh row's
    Pair* all = reinterpret_cast<Pair*>(scratch + 2 * list_bytes);  // every chip's
    const uint32_t ids_l1 = scratch + 3 * list_bytes;               // a 64-byte page

    hy4l::stamp(0);
    volatile tt_l1_ptr uint32_t* runs = hy4l::word(hy4l::W_RUNS);
    runs[0] = runs[0] + 1;
    if (hy4l::wait_word(hy4l::W_X, 1, hy4l::ST_X)) {
        canary(now(hy4l::W_X) == 1, 1);
        hy4l::take_word(hy4l::W_X, 1);
    }
    hy4l::stamp(1);

    // x to every core, then the units' release
    const uint32_t x_l1 = get_write_ptr(cb_x);
    uint32_t failed = 0;
    if (!hy4moe::grid_mcast_write(x_addr, mc_x0, mc_y0, mc_x1, mc_y1, x_l1, x_bytes, mc_dests)) {
        failed = 1;
    } else if (!hy4moe::writes_flushed()) {
        failed = 2;
    } else if (!hy4moe::grid_mcast_inc(mc_x0, mc_y0, mc_x1, mc_y1, hy4moe::sem(hy4moe::SEM_X), 1, mc_dests)) {
        failed = 3;
    } else if (!hy4moe::atomics_flushed()) {
        failed = 4;
    }
    if (failed) {
        hy4l::word(hy4l::W_STATUS)[0] = ST_SEND | failed;
    }

    hy4l::stamp(2);

    // the units' lists (a unit that gave up still counts)
    if (hy4l::wait_word(hy4l::W_GATHER, units, ST_UNITS)) {
        take(hy4l::W_GATHER, units);
    }
    hy4l::stamp(3);
    merge<K>(chip, nullptr, slots_addr, units, 0xFFFFFFFFu, id0);

    const Pair* out = chip;
    if (mode != 0) {
        if (hy4l::wait_word(hy4l::W_S1, n1 - 1, ST_ROW)) {
            canary(now(hy4l::W_S1) == n1 - 1, 2);
            take(hy4l::W_S1, n1 - 1);
        }
        merge<K>(row, chip, land1, n1, idx1, 0);
        out = row;
    }
    if (mode == 2) {
        if (hy4l::wait_word(hy4l::W_S2, n2 - 1, ST_HEADS)) {
            canary(now(hy4l::W_S2) == n2 - 1, 4);
            take(hy4l::W_S2, n2 - 1);
        }
        merge<K>(all, row, land2, n2, idx2, 0);
        volatile tt_l1_ptr uint32_t* ids = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ids_l1);
        for (uint32_t r = 0; r < 16; ++r) {
            ids[r] = r < ROWS ? all[r * K].id : 0;
        }
        const auto id_tensor = TensorAccessor(ids_args, ids_addr, 64);
        if (hy4moe::write_buf_free()) {
            noc_async_write(ids_l1, id_tensor.get_noc_addr(0), 64);
            if (!hy4moe::writes_flushed()) {
                hy4l::word(hy4l::W_STATUS)[0] = ST_SEND | 6;
            }
        } else {
            hy4l::word(hy4l::W_STATUS)[0] = ST_SEND | 5;
        }
    } else {
        // to this chip's sender: the list into the op's staging, then the op's ready bit
        const uint32_t sx = sender & 0xFFFF;
        const uint32_t sy = sender >> 16;
        if (hy4moe::write_buf_free()) {
            noc_async_write(reinterpret_cast<uint32_t>(out), get_noc_addr(sx, sy, staging), list_bytes);
            hy4moe::writes_flushed();
        } else {
            hy4l::word(hy4l::W_STATUS)[0] = ST_SEND | 7;
        }
        // (the bit is added even after a failed write: the sender must not be left waiting for its op)
        noc_semaphore_inc(get_noc_addr(sx, sy, ready), ready_bit);
        noc_async_atomic_barrier();
    }
    // (a bring-up head's own op lands in its own stage-2 slot: there the stage-2 word is not checked)
    if (mode != 0) {
        canary(now(hy4l::W_S1) == 0, 3);
    }
    if (mode == 2) {
        canary(now(hy4l::W_S2) == 0, 5);
    }
    volatile tt_l1_ptr uint32_t* fed = hy4l::word(hy4l::W_FED);
    fed[0] = fed[0] + 1;
    hy4l::stamp(4);
}
