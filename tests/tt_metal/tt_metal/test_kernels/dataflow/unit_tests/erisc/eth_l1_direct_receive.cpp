// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "ethernet/tt_eth_api.h"
#include "ethernet/tunneling.h"
#include "debug/ring_buffer.h"
#include "ethernet/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t q_num = 0;
    constexpr uint32_t src_word_addr = 0x20000 >> 4;
    constexpr uint32_t dest_word_addr = 0x50000 >> 4;
    constexpr uint32_t num_words = 64 >> 4;
    // std::uint32_t num_bytes = get_arg_val<uint32_t>(0);

    // eth_wait_for_bytes(num_bytes);
    // eth_receiver_done();

    for (int j = 0; j < 1000; ++j) {
        while (internal_::eth_txq_is_busy(q_num)) {
            volatile uint32_t* ptr = reinterpret_cast<volatile uint32_t*>(0x20000);
            for (int i = 0; i < 100; ++i) {
                [[maybe_unused]] volatile uint32_t ld1 = ptr[0];
                [[maybe_unused]] volatile uint32_t ld2 = ptr[4];
            }
        }
        eth_txq_reg_write(q_num, ETH_TXQ_TRANSFER_START_ADDR, src_word_addr << 4);
        eth_txq_reg_write(q_num, ETH_TXQ_DEST_ADDR, dest_word_addr << 4);
        eth_txq_reg_write(q_num, ETH_TXQ_TRANSFER_SIZE_BYTES, num_words << 4);
        eth_txq_reg_write(q_num, ETH_TXQ_CMD, ETH_TXQ_CMD_START_DATA);
    }
}
