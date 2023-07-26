#include <stdint.h>

#include "dataflow_api.h"

void kernel_main() {
    const uint32_t in0_cb = get_compile_time_arg_val(0);
    const uint32_t in1_cb = get_compile_time_arg_val(1);
    uint32_t src0_addr = get_arg_val<uint32_t>(0);
    uint32_t src0_noc_x = get_arg_val<uint32_t>(1);
    uint32_t src0_noc_y = get_arg_val<uint32_t>(2);
    uint32_t src1_addr = get_arg_val<uint32_t>(3);
    uint32_t src1_noc_x = get_arg_val<uint32_t>(4);
    uint32_t src1_noc_y = get_arg_val<uint32_t>(5);
    uint32_t num_blocks = get_arg_val<uint32_t>(6);
    uint32_t in0_block_tile_cnt = get_arg_val<uint32_t>(7);
    uint32_t in1_block_tile_cnt = get_arg_val<uint32_t>(8);
    uint32_t in0_block_size_bytes = get_arg_val<uint32_t>(9);
    uint32_t in1_block_size_bytes = get_arg_val<uint32_t>(10);

    uint32_t l1_write_addr_in0;
    uint32_t l1_write_addr_in1;

    for (uint32_t i = 0; i < num_blocks; i++) {
        uint64_t src0_noc_addr = get_noc_addr(src0_noc_x, src0_noc_y, src0_addr);
        uint64_t src1_noc_addr = get_noc_addr(src1_noc_x, src1_noc_y, src1_addr);

        cb_reserve_back(in0_cb, in0_block_tile_cnt);
        cb_reserve_back(in1_cb, in1_block_tile_cnt);

        l1_write_addr_in0 = get_write_ptr(in0_cb);
        l1_write_addr_in1 = get_write_ptr(in1_cb);

        noc_async_read(src0_noc_addr, l1_write_addr_in0, in0_block_size_bytes);
        noc_async_read(src1_noc_addr, l1_write_addr_in1, in1_block_size_bytes);

        noc_async_read_barrier();
        auto ptr0 = reinterpret_cast<volatile tt_l1_ptr uint32_t*> (l1_write_addr_in0);
        auto ptr1 = reinterpret_cast<volatile tt_l1_ptr uint32_t*> (l1_write_addr_in1);

        cb_push_back(in0_cb, in0_block_tile_cnt);
        cb_push_back(in1_cb, in1_block_tile_cnt);

        src0_addr += in0_block_size_bytes;
        src1_addr += in1_block_size_bytes;
    }
}
