// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/dataflow/recipe_state_transfer.hpp"

void kernel_main() {
    constexpr bool fp32 = get_compile_time_arg_val(0);
    constexpr auto oa = TensorAccessorArgs<1>();
    constexpr auto sa = TensorAccessorArgs<oa.next_compile_time_args_offset()>();
    auto out = TensorAccessor(oa, get_arg_val<uint32_t>(0));
    auto state = TensorAccessor(sa, get_arg_val<uint32_t>(1));
    Noc noc;
    DataflowBuffer cb(16);
    for (uint32_t pass = 0; pass < 2; ++pass) {
        transfer_recipe_state<fp32, 8, 17, 18>(noc, state);
        for (uint32_t row = 0; row < 8; ++row) {
            cb.wait_front(4);
            if (pass == 1) {
                for (uint32_t i = 0; i < 4; ++i) {
                    noc.async_write(cb, out, 2048, {.offset_bytes = i * 2048}, {.page_id = row * 4 + i});
                }
                noc.async_write_barrier();
            }
            cb.pop_front(4);
        }
    }
}
