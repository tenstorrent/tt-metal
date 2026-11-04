// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// DRISC GDDR DMA data check for the Blackhole reproducers: one DRAM tile runs `njobs` DMA read jobs, job j on TX
// stream j, each keeping `depth` reads of `bytes` in flight into its own L1 ring. When a job's oldest read is reported
// landed (outstanding <= depth - 1, FIFO assumed, as the flat expert's DRISC senders did) the slot is checked against
// the pattern the host wrote: word k of block b of job j = j << 30 | b << 14 | k (every `stride`-th word and the last).
// Optionally the slot is then NoC-written to a Tensix core (the senders' forward). Results at `res` (L1):
// [0] mismatching words, [1] blocks checked, [2..6] first mismatch: iteration, job, word, got, expected.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "experimental/gddr_dma.h"

void kernel_main() {
    uint32_t a = 0;
    const uint32_t iters = get_arg_val<uint32_t>(a++);
    const uint32_t gddr = get_arg_val<uint32_t>(a++);    // job j's region at gddr + j * region (bank-local)
    const uint32_t region = get_arg_val<uint32_t>(a++);  // bytes per job
    const uint32_t bytes = get_arg_val<uint32_t>(a++);
    const uint32_t l1 = get_arg_val<uint32_t>(a++);  // job j's ring at l1 + j * depth * bytes
    const uint32_t depth = get_arg_val<uint32_t>(a++);
    const uint32_t njobs = get_arg_val<uint32_t>(a++);
    const uint32_t stride = get_arg_val<uint32_t>(a++);  // words between checked words
    const uint32_t res = get_arg_val<uint32_t>(a++);
    const uint32_t forward = get_arg_val<uint32_t>(a++);
    const uint32_t fwd_xy = get_arg_val<uint32_t>(a++);
    const uint32_t fwd_land = get_arg_val<uint32_t>(a++);

    volatile tt_l1_ptr uint32_t* r = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(res);
    for (uint32_t i = 0; i < 7; ++i) {
        r[i] = 0;
    }
    const uint32_t slots = region / bytes;
    const uint32_t words = bytes / 4;
    uint32_t bad = 0, checked = 0;

    auto check = [&](uint32_t j, uint32_t it) {  // the block read at iteration `it` by job j
        const uint32_t b = it % slots;
        volatile tt_l1_ptr uint32_t* s =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1 + (j * depth + it % depth) * bytes);
        auto word = [&](uint32_t k) {
            const uint32_t want = (j << 30) | (b << 14) | k;
            const uint32_t got = s[k];
            if (got != want) {
                if (bad == 0) {
                    r[2] = it;
                    r[3] = j;
                    r[4] = k;
                    r[5] = got;
                    r[6] = want;
                }
                ++bad;
            }
        };
        for (uint32_t k = 0; k < words; k += stride) {
            word(k);
        }
        word(words - 1);  // (the tail lands last)
        ++checked;
        if (forward) {
            noc_async_write(
                l1 + (j * depth + it % depth) * bytes,
                get_noc_addr(fwd_xy >> 16, fwd_xy & 0xFFFF, fwd_land + j * bytes),
                bytes);
            noc_async_writes_flushed();  // (the slot is the next read's destination)
        }
    };

    for (uint32_t i = 0; i < iters; ++i) {
        for (uint32_t j = 0; j < njobs; ++j) {
            if (i >= depth) {
                experimental::dma_async_read_wait_n(j, depth - 1);
                check(j, i - depth);
            }
            experimental::dma_async_read(
                j, gddr + j * region + (i % slots) * bytes, l1 + (j * depth + i % depth) * bytes, bytes);
        }
    }
    for (uint32_t j = 0; j < njobs; ++j) {
        experimental::dma_async_read_barrier(j);
        for (uint32_t i = (iters > depth ? iters - depth : 0); i < iters; ++i) {
            check(j, i);
        }
    }
    noc_async_write_barrier();
    r[0] = bad;
    r[1] = checked;
}
