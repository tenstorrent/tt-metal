// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "ethernet/tunneling.h"

void kernel_main() {
    constexpr uint32_t test_range_bytes = 0x1000;  // Max is 8K (0x2000)
    constexpr uint32_t test_range_words = test_range_bytes / sizeof(uint32_t);

    // Read from local memory into the temp registers
#pragma gcc unroll 0
    for (uint32_t i = 0; i < 1; ++i) {
        asm volatile(
            "li t4, 0xFFB00000\n\t"  // local_mem = 0xFFB00000
            "lw a1, 0(t4)\n\t"       // local_mem[0]
            "lw a0, 16(t4)\n\t"      // local_mem[4]
            "lw a3, 32(t4)\n\t"      // local_mem[8]
            :
            :
            : "a1", "a0", "a3");
    }
    // // Force storage in memory - no register allocation
    // volatile std::uint32_t local_eth_l1_src_addr = get_arg_val<uint32_t>(0);   // 88640
    // volatile std::uint32_t remote_eth_l1_dst_addr = get_arg_val<uint32_t>(1);  // 88640
    // volatile std::uint32_t num_bytes = get_arg_val<uint32_t>(2);               // 16

    // constexpr uint32_t num_bytes_per_send = get_compile_time_arg_val(0);
    // constexpr uint32_t num_bytes_per_send_word_size = get_compile_time_arg_val(1);

    // uint32_t num_bytes_sent = 0;
    // while (num_bytes_sent < num_bytes) {
    //     while (true) {
    //         eth_txq_reg_read(0, ETH_TXQ_CMD);
    //         if (((eth_txq_reg_read(0, ETH_TXQ_STATUS) >> ETH_TXQ_STATUS_CMD_ONGOING_BIT) & 0x1) == 0) {
    //             break;
    //         }
    //     }
    //     eth_txq_reg_write(0, ETH_TXQ_TRANSFER_START_ADDR, ((num_bytes_sent + local_eth_l1_src_addr) >> 4) << 4);
    //     eth_txq_reg_write(0, ETH_TXQ_DEST_ADDR, ((num_bytes_sent + remote_eth_l1_dst_addr) >> 4) << 4);
    //     eth_txq_reg_write(0, ETH_TXQ_TRANSFER_SIZE_BYTES, num_bytes_per_send_word_size << 4);
    //     eth_txq_reg_write(0, ETH_TXQ_CMD, ETH_TXQ_CMD_START_DATA);
    //     num_bytes_sent += num_bytes_per_send;
    // }
    // erisc_info->channels[0].bytes_sent += num_bytes;
}
