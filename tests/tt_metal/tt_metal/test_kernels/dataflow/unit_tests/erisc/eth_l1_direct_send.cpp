// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "ethernet/tt_eth_api.h"
#include "ethernet/tunneling.h"
#include "debug/ring_buffer.h"
#include "ethernet/dataflow_api.h"

void kernel_main() {
    // constexpr uint32_t q_num = 0;
    // constexpr uint32_t src_word_addr = 0x20000 >> 4;
    // constexpr uint32_t dest_word_addr = 0x50000 >> 4;
    // constexpr uint32_t num_words = 64 >> 4;

    // L1 memory range: 0x0 to 0x70000
    constexpr uint32_t l1_start_addr = 0x0;
    constexpr uint32_t l1_end_addr = 0x70000;
    constexpr uint32_t bank_size = 16;  // Each bank is 16B

    // eth_txq_reg_write(1, ETH_TXQ_DATA_PACKET_ACCEPT_AHEAD, 32);

    for (int j = 0; j < 10; ++j) {
        // Stride through L1 memory in 16B increments to access all banks
        for (uint32_t addr = l1_start_addr; addr < l1_end_addr; addr += bank_size) {
            // Read from current address in the stride
            volatile uint32_t* ptr = reinterpret_cast<volatile uint32_t*>(addr);
            [[maybe_unused]] volatile uint32_t ld1 = ptr[0];  // Read first 4 bytes of 16B bank
            [[maybe_unused]] volatile uint32_t ld2 = ptr[1];  // Read second 4 bytes of 16B bank
            [[maybe_unused]] volatile uint32_t ld3 = ptr[2];  // Read third 4 bytes of 16B bank
            [[maybe_unused]] volatile uint32_t ld4 = ptr[3];  // Read fourth 4 bytes of 16B bank
        }

        // while (internal_::eth_txq_is_busy(q_num)) {}
        // // Keep the original eth_txq operations
        // eth_txq_reg_write(q_num, ETH_TXQ_DEST_ADDR, STREAM_REG_ADDR(14,
        // STREAM_REMOTE_DEST_BUF_SPACE_AVAILABLE_UPDATE_REG_INDEX)); eth_txq_reg_write(q_num, ETH_TXQ_REMOTE_REG_DATA,
        // 1 << REMOTE_DEST_BUF_WORDS_FREE_INC); eth_txq_reg_write(q_num, ETH_TXQ_CMD, ETH_TXQ_CMD_START_REG);
    }
}
