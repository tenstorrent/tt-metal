// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
// silu(a)*b writer, block-sharded a / b, interleaved out: local tile j of this core's [shard_h, shard_w]-tile shard is
// global tile (row0 + j / shard_w, col0 + j % shard_w); columns at or past valid_w are shard padding and are skipped.
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    const uint32_t out_addr = get_arg_val<uint32_t>(0);
    const uint32_t row0 = get_arg_val<uint32_t>(1);
    const uint32_t col0 = get_arg_val<uint32_t>(2);
    const uint32_t valid_w = get_arg_val<uint32_t>(3);
    constexpr uint32_t CH = get_compile_time_arg_val(0);
    constexpr uint32_t n_units = get_compile_time_arg_val(1);
    constexpr uint32_t shard_w = get_compile_time_arg_val(2);
    constexpr uint32_t Wt = get_compile_time_arg_val(3);
    constexpr auto o_args = TensorAccessorArgs<4>();
    constexpr uint32_t cb_out = 16;
    const auto so = TensorAccessor(o_args, out_addr);
    const uint32_t to = get_tile_size(cb_out);
    Noc noc;
    CircularBuffer co(cb_out);
    uint32_t r = 0, c = 0;
    for (uint32_t u = 0; u < n_units; ++u) {
        co.wait_front(CH);
        for (uint32_t i = 0; i < CH; ++i) {
            if (c < valid_w) {
                noc.async_write(co, so, to, {.offset_bytes = i * to}, {.page_id = (row0 + r) * Wt + col0 + c});
            }
            if (++c == shard_w) {
                c = 0;
                ++r;
            }
        }
        noc.async_writes_flushed();
        co.pop_front(CH);
    }
    noc.async_write_barrier();
}
