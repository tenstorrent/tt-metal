// SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
// Hy4 B22 fused front norms, writer: core k writes its TPC normalised tiles to pages [k TPC, (k+1) TPC) of the output.
// CT: 0 cb_out, 1 TPC, then TensorAccessorArgs(out). Common: out addr. RT: k.
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    constexpr uint32_t cb_out = get_compile_time_arg_val(0);
    constexpr uint32_t TPC = get_compile_time_arg_val(1);
    constexpr auto o_args = TensorAccessorArgs<2>();
    const uint32_t k = get_arg_val<uint32_t>(0);
    const auto o_acc = TensorAccessor(o_args, get_common_arg_val<uint32_t>(0), 2048);
    Noc noc;
    CircularBuffer out(cb_out);
    out.wait_front(TPC);
    for (uint32_t i = 0; i < TPC; ++i) {
        noc.async_write(out, o_acc, 2048, {.offset_bytes = i * 2048}, {.page_id = k * TPC + i, .offset_bytes = 0});
    }
    noc.async_write_barrier();
    out.pop_front(TPC);
}
