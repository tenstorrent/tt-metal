// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// glu_split reader (NCRISC): per output tile (r, c) of this core's range, the gate tile (r, c) and the up tile
// (r, It + c) of the fused [T, 2 I] gate|up matmul output, into two single-tile CBs. Ends with
// noc_async_full_barrier().

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

namespace {
constexpr uint32_t CB_G = get_compile_time_arg_val(0);
constexpr uint32_t CB_U = get_compile_time_arg_val(1);
constexpr uint32_t IT = get_compile_time_arg_val(2);  // I / 32 output tile columns
constexpr uint32_t TILE_BYTES = get_compile_time_arg_val(3);
constexpr auto in_args = TensorAccessorArgs<4>();
}  // namespace

void kernel_main() {
    const uint32_t start = get_arg_val<uint32_t>(0);  // first output tile (row-major over [Rt, It])
    const uint32_t count = get_arg_val<uint32_t>(1);
    const uint32_t in_addr = get_common_arg_val<uint32_t>(0);
    const auto s_in = TensorAccessor(in_args, in_addr);
    CircularBuffer cbg(CB_G), cbu(CB_U);
    for (uint32_t i = start; i < start + count; ++i) {
        const uint32_t r = i / IT;
        const uint32_t c = i - r * IT;
        cbg.reserve_back(1);
        cbu.reserve_back(1);
        noc_async_read(s_in.get_noc_addr(r * 2 * IT + c), cbg.get_write_ptr(), TILE_BYTES);
        noc_async_read(s_in.get_noc_addr(r * 2 * IT + IT + c), cbu.get_write_ptr(), TILE_BYTES);
        noc_async_read_barrier();
        cbg.push_back(1);
        cbu.push_back(1);
    }
    noc_async_full_barrier();
}
