// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// QWEN38_TOPK_PAR=1, fold core: reads logit tiles NT*j .. NT*j+NT-1 (tile row 0), same source as core B's reader.
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "../../kernels/zones.h"

void kernel_main() {
    const uint32_t logits_addr = get_arg_val<uint32_t>(0);
    const uint32_t tile_base = get_arg_val<uint32_t>(1);
    constexpr uint32_t cb_in0 = get_named_compile_time_arg_val("cb_in0");
    constexpr uint32_t NT = get_named_compile_time_arg_val("NT");
    constexpr auto logits_args = TensorAccessorArgs<0, 0>();
    const auto logits = TensorAccessor(logits_args, logits_addr);
    Noc noc;
    DataflowBuffer in0(cb_in0);
    FUSED_ZONE("fz_rt_f_reads");
    const uint32_t tile_bytes = in0.get_entry_size();
    in0.reserve_back(NT);
    for (uint32_t w = 0; w < NT; ++w) {
        noc.async_read(logits, in0, tile_bytes, {.page_id = tile_base + w}, {.offset_bytes = w * tile_bytes});
    }
    noc.async_read_barrier();
    in0.push_back(NT);
}
