// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Writer of the V4.1 single-query range scoring (rangescore_compute.cpp): per chunk of CHUNK key tile rows of this
// core (chunk c on core c mod NCORES, its slot j = c / NCORES of the core's output row) it collects row 0 of the
// compute's result tiles (32 scores each) into a chunk buffer, sets -inf where the key is not visible to the user
// (t > newest), for the tile rows at or past the visible bound (neither read nor computed) and for the slots past
// the last chunk, and writes the chunk at column j * CHUNK * 32 of output row `core` (bf16 row-major [NCORES,
// PER_ROW * CHUNK * 32]: the split layout, one page a core; the selection merges the rows, tt/v41/indexer.py
// select_split). Two chunk buffers alternate so a write overlaps the next collection. The bound is entry `user` of
// the pin page (uint32 [1, 1, 1, PIN_ROWS]: the user's newest visible row, SENTINEL for none, which makes every
// row -inf). With DYN the newest is word 2 of the pin page (the reader's DYN: the one-live-row capture).
//
// compile_time_args = [cb_out, cb_scratch, CHUNK, TR, NCORES, PIN_ROWS, PER_ROW, DYN, TensorAccessorArgs(out)...,
//                      TensorAccessorArgs(pin)...]
// runtime args      = [out_addr, pin_addr, user, core_index]

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"

void kernel_main() {
    const uint32_t out_addr = get_arg_val<uint32_t>(0);
    const uint32_t pin_addr = get_arg_val<uint32_t>(1);
    const uint32_t user = get_arg_val<uint32_t>(2);
    const uint32_t core = get_arg_val<uint32_t>(3);

    constexpr uint32_t cb_out = get_compile_time_arg_val(0);
    constexpr uint32_t cb_scratch = get_compile_time_arg_val(1);
    constexpr uint32_t CHUNK = get_compile_time_arg_val(2);
    constexpr uint32_t TR = get_compile_time_arg_val(3);
    constexpr uint32_t NCORES = get_compile_time_arg_val(4);
    constexpr uint32_t PIN_ROWS = get_compile_time_arg_val(5);
    constexpr uint32_t PER_ROW = get_compile_time_arg_val(6);  // chunk slots of a core's output row
    constexpr uint32_t DYN = get_compile_time_arg_val(7);      // 1: the newest is word 2 of the pin page
    constexpr auto out_args = TensorAccessorArgs<8>();
    constexpr auto pin_args = TensorAccessorArgs<out_args.next_compile_time_args_offset()>();

    constexpr uint32_t ROW_U32 = 16;             // 32 bf16 scores of a key tile row
    constexpr uint32_t ROW_BYTES = ROW_U32 * 4;  // 64
    constexpr uint32_t CHUNK_BYTES = CHUNK * ROW_BYTES;
    constexpr uint32_t FACE1_U32 = 16 * 16 / 2;  // word offset of face 1 in a bf16 tile
    constexpr uint16_t NEG_INF = 0xFF80;
    constexpr uint32_t NEGINF_PAIR = 0xFF80FF80u;
    constexpr uint32_t SENTINEL = 0xFFFFFFFFu;
    constexpr uint32_t PIN_BYTES = PIN_ROWS * 4;
    constexpr uint32_t PIN_SLOT = (PIN_BYTES + 63) & ~63u;
    constexpr uint32_t NCHUNKS = (TR + CHUNK - 1) / CHUNK;

    const auto out = TensorAccessor(out_args, out_addr);
    const auto pin = TensorAccessor(pin_args, pin_addr);
    DataflowBuffer res(cb_out);
    // [0, PIN_SLOT): pin page (64-byte aligned: a DRAM read) | two chunk buffers (CHUNK_BYTES each)
    DataflowBuffer scratch(cb_scratch);

    const uint32_t base_l1 = (scratch.get_write_ptr() + 63) & ~63u;
    const uint32_t buf_l1[2] = {base_l1 + PIN_SLOT, base_l1 + PIN_SLOT + CHUNK_BYTES};
    noc_async_read(pin.get_noc_addr(0, 0), base_l1, PIN_BYTES);
    noc_async_read_barrier();
    volatile tt_l1_ptr uint32_t* pins = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base_l1);
    const uint32_t newest = DYN ? pins[2] : pins[user];
    const uint32_t visible = newest == SENTINEL ? 0 : newest + 1;
    uint32_t vt = (visible + 31) / 32;
    if (vt > TR) {
        vt = TR;
    }

    uint32_t slot = 0;
    for (uint32_t j = 0; j < PER_ROW; ++j) {
        const uint32_t c = j * NCORES + core;
        const uint32_t t0 = c * CHUNK;
        uint32_t t1 = t0 + CHUNK;
        if (t1 > TR) {
            t1 = TR;
        }
        uint32_t n = 0;  // computed tile rows of this chunk (none past the last chunk or the bound)
        if (c < NCHUNKS && t0 < vt) {
            n = (t1 < vt ? t1 : vt) - t0;
        }
        volatile tt_l1_ptr uint32_t* dst = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(buf_l1[slot]);
        volatile tt_l1_ptr uint16_t* dst16 = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(buf_l1[slot]);
        noc_async_write_barrier();  // the write issued from this buffer two chunks ago has left it
        if (n > 0) {
            res.wait_front(CHUNK);
            volatile tt_l1_ptr uint32_t* t = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(res.get_read_ptr());
            for (uint32_t b = 0; b < n; ++b) {
                for (uint32_t i = 0; i < 8; ++i) {
                    dst[b * ROW_U32 + i] = t[i];
                    dst[b * ROW_U32 + 8 + i] = t[FACE1_U32 + i];
                }
                t += 32 * 32 * 2 / 4;  // the next result tile
            }
            res.pop_front(CHUNK);
            // the last computed tile row may reach past the user's newest key
            const uint32_t g = t0 + n - 1;
            if (g * 32 + 31 > newest) {
                for (uint32_t j = 0; j < 32; ++j) {
                    if (g * 32 + j > newest) {
                        dst16[(n - 1) * 32 + j] = NEG_INF;
                    }
                }
            }
        }
        for (uint32_t i = n * ROW_U32; i < CHUNK * ROW_U32; ++i) {
            dst[i] = NEGINF_PAIR;
        }
        noc_async_write(buf_l1[slot], out.get_noc_addr(core, j * CHUNK_BYTES), CHUNK_BYTES);
        slot ^= 1;
    }
    noc_async_write_barrier();
}
