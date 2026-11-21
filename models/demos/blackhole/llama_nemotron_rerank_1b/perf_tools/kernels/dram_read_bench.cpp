// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// probe_dram_read.py: one core reads REPS weight-like blocks (KB tile-rows x NB tiles of an interleaved
// [K, N] bf8 tensor, row stride N tiles) into L1. MODE 0: one TensorAccessor read per tile (what
// read_in1_block_sync does). MODE 1: the block's tiles grouped by DRAM bank, one stateful read setup per
// bank and a cheap re-issue per tile.
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    Noc noc;
    constexpr uint32_t KB = get_compile_time_arg_val(0);
    constexpr uint32_t NB = get_compile_time_arg_val(1);
    constexpr uint32_t N_TOT = get_compile_time_arg_val(2);
    constexpr uint32_t K_TOT = get_compile_time_arg_val(3);
    constexpr uint32_t REPS = get_compile_time_arg_val(4);
    constexpr uint32_t MODE = get_compile_time_arg_val(5);
    constexpr uint32_t PAGE = get_compile_time_arg_val(6);
    constexpr auto w_args = TensorAccessorArgs<7>();
    const uint32_t w_addr = get_arg_val<uint32_t>(0);
    const uint32_t n0 = get_arg_val<uint32_t>(1);
    const auto w = TensorAccessor(w_args, w_addr);
    CircularBuffer cb(tt::CBIndex::c_0);
    const uint32_t dst0 = cb.get_write_ptr();
    constexpr uint32_t NBANKS = NUM_DRAM_BANKS;
    uint32_t bank_lo[NBANKS];
    if constexpr (MODE == 1) {
        for (uint32_t b = 0; b < NBANKS; ++b) {
            bank_lo[b] = static_cast<uint32_t>(get_noc_addr_from_bank_id<true>(b, w_addr));
        }
    }
    for (uint32_t rep = 0; rep < REPS; ++rep) {
        const uint32_t k0 = (rep % (K_TOT / KB)) * KB;
        if constexpr (MODE == 0) {
            uint32_t dst = dst0;
            for (uint32_t i = 0; i < KB; ++i) {
                for (uint32_t j = 0; j < NB; ++j) {
                    noc.async_read(w, CoreLocalMem<uint32_t>(dst), PAGE, {.page_id = (k0 + i) * N_TOT + n0 + j}, {});
                    dst += PAGE;
                }
            }
        } else {
            for (uint32_t b = 0; b < NBANKS; ++b) {
                noc_async_read_set_state(get_noc_addr_from_bank_id<true>(b, w_addr));
                for (uint32_t i = 0; i < KB; ++i) {
                    for (uint32_t j = 0; j < NB; ++j) {
                        const uint32_t id = (k0 + i) * N_TOT + n0 + j;
                        if (id % NBANKS != b) {
                            continue;
                        }
                        noc_async_read_with_state(bank_lo[b] + (id / NBANKS) * PAGE, dst0 + (i * NB + j) * PAGE, PAGE);
                    }
                }
            }
        }
        noc.async_read_barrier();
    }
}
