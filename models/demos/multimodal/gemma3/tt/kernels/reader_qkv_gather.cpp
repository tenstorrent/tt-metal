// SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Gather reader for the gemma3 prefill head split (the C++ Metalium rung on
// NlpCreateHeadsDeviceOperation).
//
// The stock op's interleaved factory sets num_blocks = batch * 1 * seq / TILE_HEIGHT, i.e. it
// parallelises over SEQ TILES ONLY -- 4 work units at S=128, so 4 cores of ~110 light up and the
// profiler tags the op grid=tiny. This kernel instead walks the OUTPUT tile space, which is
// (heads x seq_tiles x head_dim_tiles), so the same split_work_to_cores call hands out 16x more
// work units and fills the grid (GUIDELINES 03 s.5(c), the head_groups idea).
//
// Output tiles are consecutive, so the caller can pair this with the stock
// writer_unary_interleaved_start_id.cpp; only the READ side is a gather.
//
//   out[h, st, dt] <- in[st * in_w_tiles + col0 + h * d_tiles + dt]
//
// col0 selects which of q / k / v this instance produces.

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t num_pages = get_arg_val<uint32_t>(1);
    const uint32_t start_id = get_arg_val<uint32_t>(2);

    constexpr uint32_t s_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t d_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t in_w_tiles = get_compile_time_arg_val(2);
    constexpr uint32_t col0 = get_compile_time_arg_val(3);
    constexpr auto src_args = TensorAccessorArgs<4>();

    constexpr uint32_t cb_id_in0 = 0;
    constexpr uint32_t onepage = 1;

    const uint32_t page_bytes = get_local_cb_interface(cb_id_in0).fifo_page_size;
    const auto s = TensorAccessor(src_args, src_addr);

    Noc noc;
    DataflowBuffer dfb(cb_id_in0);

    const uint32_t end_id = start_id + num_pages;
    for (uint32_t o = start_id; o < end_id; ++o) {
        // Unpack the flat OUTPUT tile id into (head, seq_tile, head_dim_tile).
        const uint32_t dt = o % d_tiles;
        const uint32_t rest = o / d_tiles;
        const uint32_t st = rest % s_tiles;
        const uint32_t h = rest / s_tiles;
        const uint32_t src_page = st * in_w_tiles + col0 + h * d_tiles + dt;

        dfb.reserve_back(onepage);
        noc.async_read(s, dfb, page_bytes, {.page_id = src_page}, {.offset_bytes = 0});
        noc.async_read_barrier();
        dfb.push_back(onepage);
    }
}
