// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
// Spin-reader for the all-gather DRAM aliasing test (test_sol_gather_alias_minimax_h3.py). One core (NCRISC).
//
// Reads every page of buffer M `iters` times (iters = word 0 of this device's cfg row, so one chip of each ring can be
// made to lag the others), compares 64 words of every page with the expected pattern (cfg word 1) and writes the counts
// to this device's out row: 0 bad words, 1 first bad iteration, 2 first bad page, 3 iters, 4 pages, 5 bad pages,
// 6 marker 0xA11A5. A nonzero bad count means M changed under the reader: another chip's all-gather wrote its slice
// into M's DRAM address after the host had freed M and handed the address to the gather output.
//
// Compile-time args: 0 PAGES (2 KB tiles in M), then TensorAccessorArgs of M, cfg, out.
// Common runtime args: 0 M address, 1 cfg address, 2 out address.
#include <cstdint>
#include "api/core_local_mem.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/endpoints.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"

namespace {
constexpr uint32_t cb_m = 0, cb_cfg = 1, cb_out = 2;
constexpr uint32_t PAGE_BYTES = 2048, ROW_BYTES = 128;
constexpr uint32_t NONE = 0xFFFFFFFFu;
}  // namespace

void kernel_main() {
    constexpr uint32_t PAGES = get_compile_time_arg_val(0);
    constexpr auto m_args = TensorAccessorArgs<1>();
    constexpr auto cfg_args = TensorAccessorArgs<m_args.next_compile_time_args_offset()>();
    constexpr auto out_args = TensorAccessorArgs<cfg_args.next_compile_time_args_offset()>();
    const auto m = TensorAccessor(m_args, get_common_arg_val<uint32_t>(0));
    const auto cfg = TensorAccessor(cfg_args, get_common_arg_val<uint32_t>(1));
    const auto out = TensorAccessor(out_args, get_common_arg_val<uint32_t>(2));

    Noc noc;
    CircularBuffer cbm(cb_m), cbc(cb_cfg), cbo(cb_out);

    cbc.reserve_back(1);
    noc.async_read(cfg, cbc, ROW_BYTES, {.page_id = 0}, {.offset_bytes = 0});
    noc.async_read_barrier();
    const uint32_t iters = CoreLocalMem<volatile uint32_t>(cbc.get_write_ptr())[0];
    const uint32_t expect = CoreLocalMem<volatile uint32_t>(cbc.get_write_ptr())[1];

    cbm.reserve_back(1);
    CoreLocalMem<volatile uint32_t> pg(cbm.get_write_ptr());
    uint32_t bad = 0, bad_pages = 0, first_it = NONE, first_pg = NONE;
    for (uint32_t it = 0; it < iters; ++it) {
        for (uint32_t p = 0; p < PAGES; ++p) {
            noc.async_read(m, cbm, PAGE_BYTES, {.page_id = p}, {.offset_bytes = 0});
            noc.async_read_barrier();
            uint32_t b = 0;
            for (uint32_t w = 0; w < 512; w += 8) {
                b += (pg[w] != expect) ? 1u : 0u;
            }
            if (b != 0) {
                bad += b;
                ++bad_pages;
                if (first_it == NONE) {
                    first_it = it;
                    first_pg = p;
                }
            }
        }
    }

    cbo.reserve_back(1);
    CoreLocalMem<volatile uint32_t> om(cbo.get_write_ptr());
    for (uint32_t w = 0; w < 32; ++w) {
        om[w] = 0;
    }
    om[0] = bad;
    om[1] = first_it;
    om[2] = first_pg;
    om[3] = iters;
    om[4] = PAGES;
    om[5] = bad_pages;
    om[6] = 0xA11A5u;
    noc.async_write(cbo, out, ROW_BYTES, {.offset_bytes = 0}, {.page_id = 0});
    noc.async_write_barrier();
}
