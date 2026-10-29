// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// PLE gate split, products program writer, one core per tile block: the N fp32 product tiles (CB 2) -> pages
// first..first+N-1 of the products tensor.  Compile-time args: 0 N, then TensorAccessorArgs(products).
// Runtime args: products addr, first tile.
#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "../../kernels/zones.h"

constexpr uint32_t N = get_compile_time_arg_val(0);
constexpr uint32_t FP32_TILE = 4096;

void kernel_main() {
    constexpr auto a_out = TensorAccessorArgs<1>();
    const auto out = TensorAccessor(a_out, get_arg_val<uint32_t>(0));
    const uint32_t first = get_arg_val<uint32_t>(1);
    Noc noc;
    DataflowBuffer prod(2);
    FUSED_ZONE("fz_pl_gp_w");
    prod.wait_front(N);
    for (uint32_t t = 0; t < N; ++t) {
        noc.async_write(
            prod, out, FP32_TILE, {.offset_bytes = t * FP32_TILE}, {.page_id = first + t, .offset_bytes = 0});
    }
    noc.async_write_barrier();
    prod.pop_front(N);
}
