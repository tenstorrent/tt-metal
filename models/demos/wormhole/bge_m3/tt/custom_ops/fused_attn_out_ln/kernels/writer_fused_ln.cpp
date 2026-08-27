// SPDX-License-Identifier: Apache-2.0
// Single-core writer: drain cb_out [M_block,N] blocks to the output DRAM tensor.
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
constexpr uint32_t cb_out = tt::CBIndex::c_2;
void kernel_main() {
    uint32_t argidx = 0;
    const uint32_t out_addr = get_arg_val<uint32_t>(argidx++);
    constexpr uint32_t M_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t N_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t M_block = get_compile_time_arg_val(2);
    constexpr uint32_t out_tile_bytes = get_compile_time_arg_val(3);
    constexpr auto out_args = TensorAccessorArgs<4>();
    const auto out_acc = TensorAccessor(out_args, out_addr, out_tile_bytes);
    const uint32_t num_m_blocks = M_tiles / M_block;
    for (uint32_t mb = 0; mb < num_m_blocks; mb++) {
        cb_wait_front(cb_out, M_block * N_tiles);
        uint32_t rp = get_read_ptr(cb_out);
        for (uint32_t m = 0; m < M_block; m++) {
            for (uint32_t n = 0; n < N_tiles; n++) {
                noc_async_write_page((mb * M_block + m) * N_tiles + n, out_acc, rp);
                rp += out_tile_bytes;
            }
        }
        noc_async_write_barrier();
        cb_pop_front(cb_out, M_block * N_tiles);
    }
}
