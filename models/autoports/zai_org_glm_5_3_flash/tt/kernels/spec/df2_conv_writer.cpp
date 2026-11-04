// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// DFlash2 fused grouped dynamic conv -- writer (NCRISC, NoC1): the output tile of column tile c -> page c of the
// [1,1,32,H] bf16 output. Compile-time args: TensorAccessorArgs of out. Runtime args: out address, c0, n.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "df2_conv_common.hpp"

using namespace df2conv;

void kernel_main() {
    constexpr auto out_args = TensorAccessorArgs<0>();
    const auto out = TensorAccessor(out_args, get_arg_val<uint32_t>(0));
    const uint32_t c0 = get_arg_val<uint32_t>(1);
    const uint32_t n = get_arg_val<uint32_t>(2);
    for (uint32_t c = c0; c < c0 + n; ++c) {
        cb_wait_front(CB_OUT, 1);
        noc_async_write_page(c, out, get_read_ptr(CB_OUT));
        noc_async_write_barrier();
        cb_pop_front(CB_OUT, 1);
    }
}
