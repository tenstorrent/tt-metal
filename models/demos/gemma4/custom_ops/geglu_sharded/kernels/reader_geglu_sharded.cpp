// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
// Reader for GegluShardedDecode. This core owns `count` output column tiles j0..j0+count-1 of the GLU result (its
// shard of the down-proj activation). For output tile j the up tile is input column tile j and the gate tile is
// input column tile cols + j; input column tile i lives on gate_up shard core i / per_in at local tile i % per_in
// (the width-sharded gate_up matmul output: the same L1 address on every shard core). Both tiles are read over
// the NoC into cb_in as the pair [up, gate] the GeGLU compute takes.
// compile-time args: [cb_in, cols (GLU half in tiles), per_in (input tiles per shard)]
// runtime args: [x_addr, j0, count, n_in_cores, (noc_x, noc_y) x n_in_cores]
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/circular_buffer.h"

void kernel_main() {
    const uint32_t x_addr = get_arg_val<uint32_t>(0);
    const uint32_t j0 = get_arg_val<uint32_t>(1);
    const uint32_t count = get_arg_val<uint32_t>(2);
    constexpr uint32_t cb_in = get_compile_time_arg_val(0);
    constexpr uint32_t cols = get_compile_time_arg_val(1);
    constexpr uint32_t per_in = get_compile_time_arg_val(2);
    constexpr uint32_t tile_bytes = 2048;
    CircularBuffer cbi(cb_in);
    auto src_of = [&](uint32_t i) -> uint64_t {
        const uint32_t core = i / per_in, local = i - core * per_in;
        const uint32_t nx = get_arg_val<uint32_t>(4 + 2 * core), ny = get_arg_val<uint32_t>(5 + 2 * core);
        return get_noc_addr(nx, ny, x_addr + local * tile_bytes);
    };
    // every tile pair of this core's unit in flight at once, one barrier (cb_in holds the whole unit: 2 * count
    // tiles); the compute consumes the pairs from the front as they are pushed
    cbi.reserve_back(2 * count);
    uint32_t dst = get_write_ptr(cb_in);
    for (uint32_t j = j0; j < j0 + count; ++j) {
        noc_async_read(src_of(j), dst, tile_bytes);                      // up
        noc_async_read(src_of(cols + j), dst + tile_bytes, tile_bytes);  // gate
        dst += 2 * tile_bytes;
    }
    noc_async_read_barrier();
    cbi.push_back(2 * count);
}
