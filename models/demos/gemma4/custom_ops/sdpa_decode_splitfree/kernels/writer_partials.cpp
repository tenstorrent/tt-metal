// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
// Phase 1 writer of sdpa_decode_splitfree: the reduce scaler and the zero tile once; per unit of this core the
// masks its chunks take (the causal mask of the chunk holding cur_pos, the window mask of the first live chunk,
// each a one-block CB the compute pops), then the unit's (O, m, l) fp32 tiles to the scratch slot
// (b*G + g) * NSLOTS + u % NSLOTS: pages slot*(2+DHt) + {0: m, 1: l, 2..: O}.
//
// compile-time args: the sf:: prefix (24) + TensorAccessorArgs(cur_pos, scratch)
// runtime args: [core, pos_addr, scratch_addr]
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
#include "ttnn/cpp/ttnn/kernel_lib/l1_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"
#include "splitfree_common.hpp"
#include "splitfree_dataflow.hpp"

void kernel_main() {
    using namespace sf;
    Noc noc;
    const uint32_t core = get_arg_val<uint32_t>(0);
    const uint32_t pos_addr = get_arg_val<uint32_t>(1);
    const uint32_t scratch_addr = get_arg_val<uint32_t>(2);
    constexpr auto pos_args = TensorAccessorArgs<CT_ACCESSORS>();
    constexpr auto scratch_args = TensorAccessorArgs<pos_args.next_compile_time_args_offset()>();
    constexpr uint32_t f32_tile = get_tile_size(CB_OUT_O);
    const auto scratch = TensorAccessor(scratch_args, scratch_addr, f32_tile);

    volatile tt_l1_ptr uint32_t* pos_l1 = read_cur_pos<CB_POS_D, POS_STICK>(noc, pos_args, pos_addr);
    uint32_t pos[B];
    for (uint32_t b = 0; b < B; ++b) {
        pos[b] = pos_l1[b];
    }

    dataflow_kernel_lib::calculate_and_prepare_reduce_scaler<
        CB_SCALE,
        ckernel::PoolType::MAX,
        ckernel::ReduceDim::REDUCE_ROW,
        dataflow_kernel_lib::SUM_AND_MAX_REDUCE_FACTOR>();
    dataflow_kernel_lib::prepare_zero_tile<CB_ZERO>();

    CircularBuffer cb_o(CB_OUT_O);
    CircularBuffer cb_m(CB_OUT_M);
    CircularBuffer cb_l(CB_OUT_L);
    for_each_unit(core, pos, [&](uint32_t b, uint32_t /*g*/, uint32_t bg, uint32_t u, const Row& r) {
        const uint32_t j_lo = chunk_lo(r, u);
        const uint32_t j_hi = chunk_hi(r, u);
        if constexpr (WINDOW > 0) {
            if (r.window_start > 0 && r.first_chunk >= j_lo && r.first_chunk <= j_hi) {
                generate_window_mask<CB_WMASK, SK>(noc, r.window_start);
            }
        }
        if (r.tail_chunk >= j_lo && r.tail_chunk <= j_hi) {
            generate_tail_mask<CB_MASK, SK>(noc, pos[b]);
        }

        const uint32_t page0 = slot_of(bg, u) * TILES_PER_SLOT;
        cb_o.wait_front(DHT);
        const uint32_t o_ptr = cb_o.get_read_ptr();
        for (uint32_t c = 0; c < DHT; ++c) {
            noc.async_write(
                CoreLocalMem<uint32_t>(o_ptr + c * f32_tile), scratch, f32_tile, {}, {.page_id = page0 + 2 + c});
        }
        cb_m.wait_front(1);
        noc.async_write(CoreLocalMem<uint32_t>(cb_m.get_read_ptr()), scratch, f32_tile, {}, {.page_id = page0});
        cb_l.wait_front(1);
        noc.async_write(CoreLocalMem<uint32_t>(cb_l.get_read_ptr()), scratch, f32_tile, {}, {.page_id = page0 + 1});
        noc.async_write_barrier();
        cb_o.pop_front(DHT);
        cb_m.pop_front(1);
        cb_l.pop_front(1);
    });
}
