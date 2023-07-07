#include "dataflow_kernel_api.h"
#include "debug_print.h"

using u32 = std::uint32_t;

void kernel_main() {
    u32 dram_buffer_dst_addr  = *((volatile u32*)(L1_ARG_BASE));
    u32 dram_dst_noc_x        = *((volatile u32*)(L1_ARG_BASE+4));
    u32 dram_dst_noc_y        = *((volatile u32*)(L1_ARG_BASE+8));
    u32 num_tiles             = *((volatile u32*)(L1_ARG_BASE+12));
    u32 ARG0                  = *((volatile u32*)(L1_ARG_BASE+16));
    u32 x                     = *((volatile u32*)(L1_ARG_BASE+20));
    u32 y                     = *((volatile u32*)(L1_ARG_BASE+24));
    u32 do_raise              = *((volatile u32*)(L1_ARG_BASE+28));

    u32 operand = 16;

    DPRINT << WAIT{x*5 + y*1000}; // wait for a coreid-based signal to be raised by _nc kernel to ensure debug print ordering

    // TODO(AP): enabling this string DPRINT currently causes a 4GB vector to be returned from
    // std::vector<uint32_t> hex_vec = get_risc_binary(hex_file_path, riscv_id);
    DPRINT << "TestStr";
    DPRINT << 'B' << 'R' << '{' << x << ',' << y << '}' << ENDL();
    for (u32 a = 0; a < ARG0; a++)
        DPRINT << '+';
    DPRINT << ENDL();
    if (do_raise)
        // in _nc kernel we wait for this signal to sync multi-core debug print order on the host
        DPRINT << RAISE{x + y*20 + 20000};

    // single-tile chunks
    uint32_t chunk_size_bytes = get_tile_size(operand);
    uint32_t chunk_size_tiles = 1;

    for (uint32_t i = 0; i < num_tiles; i += chunk_size_tiles) {
        // DRAM NOC dst address
        std::uint64_t dram_buffer_dst_noc_addr = NOC_XY_ADDR(NOC_X(dram_dst_noc_x), NOC_Y(dram_dst_noc_y), dram_buffer_dst_addr);

        dataflow::cb_wait_front(operand, chunk_size_tiles);
        uint32_t l1_read_addr = dataflow::get_read_ptr(operand);

        ncrisc_noc_fast_write_any_len(loading_noc, NCRISC_WR_REG_CMD_BUF, l1_read_addr, dram_buffer_dst_noc_addr, chunk_size_bytes,
                            NOC_UNICAST_WRITE_VC, false, false, 1);

        // wait for all the writes to complete (ie acked)
        while (!ncrisc_noc_nonposted_writes_flushed(loading_noc));

        dataflow::cb_pop_front(operand, chunk_size_tiles);
        dram_buffer_dst_addr += chunk_size_bytes;
    }
}
