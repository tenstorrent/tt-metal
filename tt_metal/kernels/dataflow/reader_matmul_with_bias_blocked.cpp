#include <stdint.h>
#include "dataflow_api.h"

void kernel_main() {
    uint32_t src0_addr            = get_arg_val<uint32_t>(0);
    uint32_t src0_noc_x           = get_arg_val<uint32_t>(1);
    uint32_t src0_noc_y           = get_arg_val<uint32_t>(2);
    uint32_t src1_addr            = get_arg_val<uint32_t>(3);
    uint32_t src1_noc_x           = get_arg_val<uint32_t>(4);
    uint32_t src1_noc_y           = get_arg_val<uint32_t>(5);
    uint32_t num_blocks           = get_arg_val<uint32_t>(6);

    uint32_t in0_block_tile_cnt   = get_arg_val<uint32_t>(7);
    uint32_t in1_block_tile_cnt   = get_arg_val<uint32_t>(8);
    uint32_t in0_block_size_bytes = get_arg_val<uint32_t>(9);
    uint32_t in1_block_size_bytes = get_arg_val<uint32_t>(10);

    uint32_t with_bias            = get_arg_val<uint32_t>(11);
    uint32_t src2_addr;
    uint32_t src2_noc_x;
    uint32_t src2_noc_y;
    uint32_t in2_block_tile_cnt;
    uint32_t in2_block_size_bytes;

    if (with_bias) {
        src2_addr            = get_arg_val<uint32_t>(12);
        src2_noc_x           = get_arg_val<uint32_t>(13);
        src2_noc_y           = get_arg_val<uint32_t>(14);
        in2_block_tile_cnt   = get_arg_val<uint32_t>(15);
        in2_block_size_bytes = get_arg_val<uint32_t>(16);
    }

    constexpr uint32_t cb_id_in0 = 0;
    constexpr uint32_t cb_id_in1 = 1;
    constexpr uint32_t cb_id_in2 = 2;

    uint32_t l1_write_addr_in0;
    uint32_t l1_write_addr_in1;
    uint32_t l1_write_addr_in2;

    for(uint32_t i = 0; i < num_blocks; i++) {
        uint64_t src0_noc_addr = get_noc_addr(src0_noc_x, src0_noc_y, src0_addr);
        uint64_t src1_noc_addr = get_noc_addr(src1_noc_x, src1_noc_y, src1_addr);

        cb_reserve_back(cb_id_in0, in0_block_tile_cnt);
        cb_reserve_back(cb_id_in1, in1_block_tile_cnt);

        l1_write_addr_in0 = get_write_ptr(cb_id_in0);
        l1_write_addr_in1 = get_write_ptr(cb_id_in1);

        noc_async_read(src0_noc_addr, l1_write_addr_in0, in0_block_size_bytes);
        noc_async_read(src1_noc_addr, l1_write_addr_in1, in1_block_size_bytes);

        noc_async_read_barrier();

        cb_push_back(cb_id_in0, in0_block_tile_cnt);
        cb_push_back(cb_id_in1, in1_block_tile_cnt);

        src0_addr += in0_block_size_bytes;
        src1_addr += in1_block_size_bytes;
    }

    if (with_bias) {
        uint64_t src2_noc_addr = get_noc_addr(src2_noc_x, src2_noc_y, src2_addr);
        l1_write_addr_in2 = get_write_ptr(cb_id_in2);
        cb_reserve_back(cb_id_in2, in2_block_tile_cnt);
        noc_async_read(src2_noc_addr, l1_write_addr_in2, in2_block_size_bytes);
        noc_async_read_barrier();
        cb_push_back(cb_id_in2, in2_block_tile_cnt);
    }
}
