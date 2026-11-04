// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// QWEN36_MLP_GU_FUSED split-SwiGLU (tt/mlp_swiglu.py), writer: this core's output tiles [start, start + count) from
// CB_OUT to the output tensor (row-major tile index, any layout a TensorAccessor resolves; the model uses L1
// interleaved, the 1D down-projection matmul's in0 layout).

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/circular_buffer.h"

void kernel_main() {
    constexpr uint32_t CB_OUT = get_compile_time_arg_val(0);
    constexpr auto out_args = TensorAccessorArgs<1>();

    const uint32_t out_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t start = get_arg_val<uint32_t>(0);
    const uint32_t count = get_arg_val<uint32_t>(1);
    const auto out = TensorAccessor(out_args, out_addr);
    const uint32_t tile_bytes = out.get_aligned_page_size();

    CircularBuffer cb(CB_OUT);
    for (uint32_t i = 0; i < count; ++i) {
        cb.wait_front(1);
        noc_async_write(get_read_ptr(CB_OUT), out.get_noc_addr(start + i), tile_bytes);
        noc_async_writes_flushed();
        cb.pop_front(1);
    }
    noc_async_write_barrier();
}
