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

void kernel_main() { prepare_tiles(); }
