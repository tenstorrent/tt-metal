// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// main_tail, rope core reader: the -1 scalar tile, the cos/sin tiles, then the hand-off from its norm core.
// Compile-time args: TensorAccessorArgs cos, sin.  Runtime args: 0 cos, 1 sin addresses.

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"
#include "ttnn/kernel/dataflow/generate_bcast_scalar.hpp"
#include "main_tail_cbs.h"
#include "../../kernels/zones.h"

using namespace main_tail;

void kernel_main() {
    const uint32_t cos_addr = get_arg_val<uint32_t>(0);
    const uint32_t sin_addr = get_arg_val<uint32_t>(1);
    constexpr auto cos_args = TensorAccessorArgs<0>();
    constexpr auto sin_args = TensorAccessorArgs<cos_args.next_compile_time_args_offset()>();
    const auto cos = TensorAccessor(cos_args, cos_addr);
    const auto sin = TensorAccessor(sin_args, sin_addr);
    Semaphore<> sem_rope(SEM_ROPE);

    {
        FUSED_ZONE("fz_qs_mt_rr_reads");
        generate_bcast_col_scalar(CircularBuffer(CB_SCALAR), 0xBF800000u);
        cb_reserve_back(CB_COS, ROPE_TILES);
        cb_reserve_back(CB_SIN, ROPE_TILES);
        for (uint32_t t = 0; t < ROPE_TILES; ++t) {
            noc_async_read_page(t, cos, get_write_ptr(CB_COS) + t * TILE_BYTES);
            noc_async_read_page(t, sin, get_write_ptr(CB_SIN) + t * TILE_BYTES);
        }
        noc_async_read_barrier();
        cb_push_back(CB_COS, ROPE_TILES);
        cb_push_back(CB_SIN, ROPE_TILES);
    }

    {
        FUSED_ZONE("fz_qs_mt_rr_handoff");
        cb_reserve_back(CB_IN, ROPE_TILES);
        cb_reserve_back(CB_ROT, ROPE_TILES);
        sem_rope.wait_min(1);
        cb_push_back(CB_IN, ROPE_TILES);
        cb_push_back(CB_ROT, ROPE_TILES);
        sem_rope.set(0);
    }
}
