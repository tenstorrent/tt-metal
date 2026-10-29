// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// PLE gate split (QWEN38_PLE_SPLIT=1), products program reader, one core per tile block: N key tiles and N query tiles
// (bf16 TILE [1,1,4,2560]) from tile `first` into CBs 0 / 1.
// Compile-time args: 0 N, then TensorAccessorArgs(key), (query).  Runtime args: key addr, query addr, first tile.
#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "../../kernels/zones.h"

constexpr uint32_t N = get_compile_time_arg_val(0);
constexpr uint32_t BF16_TILE = 2048;

void kernel_main() {
    constexpr auto a_key = TensorAccessorArgs<1>();
    constexpr auto a_query = TensorAccessorArgs<a_key.next_compile_time_args_offset()>();
    const auto key = TensorAccessor(a_key, get_arg_val<uint32_t>(0));
    const auto query = TensorAccessor(a_query, get_arg_val<uint32_t>(1));
    const uint32_t first = get_arg_val<uint32_t>(2);
    Noc noc;
    DataflowBuffer key_dfb(0), query_dfb(1);
    FUSED_ZONE("fz_pl_gp_r");
    key_dfb.reserve_back(N);
    query_dfb.reserve_back(N);
    for (uint32_t t = 0; t < N; ++t) {
        noc.async_read(
            key, key_dfb, BF16_TILE, {.page_id = first + t, .offset_bytes = 0}, {.offset_bytes = t * BF16_TILE});
        noc.async_read(
            query, query_dfb, BF16_TILE, {.page_id = first + t, .offset_bytes = 0}, {.offset_bytes = t * BF16_TILE});
    }
    noc.async_read_barrier();
    key_dfb.push_back(N);
    query_dfb.push_back(N);
}
