// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Reader of the fused row sum of squares (tt-blaze G53F-0209): x [Ht*32, Wt*32] fp32 TILE, DRAM interleaved. This
// core's tile rows (an even split of the Ht tile rows over NCORES cores, row-major core index from the logical
// position: no per-core runtime args, G53F-0206), each row's Wt tiles in order, GROUP reads in flight per barrier.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"

constexpr uint32_t Ht = get_compile_time_arg_val(0);
constexpr uint32_t Wt = get_compile_time_arg_val(1);
constexpr uint32_t NCORES = get_compile_time_arg_val(2);
constexpr uint32_t GRID_X = get_compile_time_arg_val(3);
constexpr uint32_t GROUP = get_compile_time_arg_val(4);
constexpr uint32_t cb_in_id = get_compile_time_arg_val(5);
constexpr uint32_t tile_bytes = get_compile_time_arg_val(6);
constexpr auto in_args = TensorAccessorArgs<7>();
static_assert(Wt % GROUP == 0, "a row's tiles are read in whole groups");

void kernel_main() {
    const uint32_t in_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t core_idx = get_absolute_logical_y() * GRID_X + get_absolute_logical_x();
    const uint32_t base = Ht / NCORES, extra = Ht % NCORES;
    const uint32_t n = base + (core_idx < extra ? 1u : 0u);
    const uint32_t r0 = core_idx * base + (core_idx < extra ? core_idx : extra);

    const auto s_in = TensorAccessor(in_args, in_addr, tile_bytes);
    Noc noc;
    CircularBuffer cb_in(cb_in_id);
    for (uint32_t r = r0; r < r0 + n; ++r) {
        for (uint32_t w0 = 0; w0 < Wt; w0 += GROUP) {
            cb_in.reserve_back(GROUP);
            const uint32_t l1 = get_write_ptr(cb_in_id);
            for (uint32_t g = 0; g < GROUP; ++g) {
                noc.async_read(
                    s_in, CoreLocalMem<uint32_t>(l1 + g * tile_bytes), tile_bytes, {.page_id = r * Wt + w0 + g}, {});
            }
            noc.async_read_barrier();
            cb_in.push_back(GROUP);
        }
    }
}
