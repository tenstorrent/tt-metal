// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"

namespace {

constexpr uint32_t cb_auxiliary = 1;
constexpr uint32_t num_tiles = get_compile_time_arg_val(0);
constexpr uint32_t words_per_tile = 3;

template <uint32_t tile = 0>
FORCE_INLINE void prepare_tiles() {
    if constexpr (tile < num_tiles) {
        constexpr uint32_t base = 1 + tile * words_per_tile;
        dataflow_kernel_lib::prepare_reduce_auxiliary_tile<
            cb_auxiliary,
            static_cast<ttnn::kernel_lib::ReduceAuxiliaryTileType>(get_compile_time_arg_val(base)),
            get_compile_time_arg_val(base + 1),
            get_compile_time_arg_val(base + 2)>();
        prepare_tiles<tile + 1>();
    }
}

}  // namespace

void kernel_main() {
    prepare_tiles();
#ifdef REDUCE_STREAM_OUTPUT
    // Drain the one-page compute output CB into the resident output tensor. A bulk reservation in
    // compute would deadlock before the first output; per-tile publication permits every ring wrap.
    DataflowBuffer computed(16), output(17);
    const uint32_t output_tiles = get_arg_val<uint32_t>(0);
    const uint32_t tile_bytes = get_tile_size(16);
    for (uint32_t tile = 0; tile < output_tiles; ++tile) {
        computed.wait_front(1);
        output.reserve_back(1);
        noc_async_write(computed.get_read_ptr(), get_noc_addr(output.get_write_ptr()), tile_bytes);
        noc_async_write_barrier();
        computed.pop_front(1);
        output.push_back(1);
    }
#endif
}
