// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "eth_fw_api.h"

void kernel_main() {
    std::uint32_t local_eth_l1_src_addr = get_arg_val<uint32_t>(0);
    std::uint32_t remote_eth_l1_dst_addr = get_arg_val<uint32_t>(1);
    std::uint32_t num_bytes = get_arg_val<uint32_t>(2);

    constexpr uint32_t num_bytes_per_send = get_compile_time_arg_val(0);
    constexpr uint32_t num_bytes_per_send_word_size = get_compile_time_arg_val(1);

    volatile eth_api_table_t* eth_api_table = (volatile eth_api_table_t*)(MEM_SYSENG_ETH_API_TABLE);
    reinterpret_cast<void (*)(uint32_t)>(
        (uint32_t)(((eth_api_table_t*)(MEM_SYSENG_ETH_API_TABLE))->eth_link_status_check_ptr))(0xFFFFFFFF);

    volatile eth_live_status_t* link_status = (volatile eth_live_status_t*)(ETH_RESULTS_BASE_ADDR_LIVE_STATUS);

    volatile tt_l1_ptr uint32_t* port_status_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(0x7CC04);

    DPRINT << "pre-send" << ENDL();
    DPRINT << "retrain_count " << link_status->retrain_count << ENDL();
    DPRINT << "port_status " << port_status_ptr[0] << ENDL();
    DPRINT << "rx_link_up " << link_status->rx_link_up << ENDL();
    // DPRINT << "frames_txd " << link_status->frames_txd << ENDL();
    // DPRINT << "frames_txd_ok " << link_status->frames_txd_ok << ENDL();
    // DPRINT << "frames_txd_badfcs " << link_status->frames_txd_badfcs << ENDL();
    // DPRINT << "bytes_txd " << link_status->bytes_txd << ENDL();
    // DPRINT << "bytes_txd_ok " << link_status->bytes_txd_ok << ENDL();
    // DPRINT << "frames_rxd_ok " << link_status->frames_rxd_ok << ENDL();
    // DPRINT << "frames_rxd_badfcs " << link_status->frames_rxd_badfcs << ENDL();
    // DPRINT << "frames_rxd_dropped " << link_status->frames_rxd_dropped << ENDL();
    // DPRINT << "bytes_rxd " << link_status->bytes_rxd << ENDL();
    // DPRINT << "bytes_rxd_ok " << link_status->bytes_rxd_ok << ENDL();
    // DPRINT << "bytes_rxd_badfcs " << link_status->bytes_rxd_badfcs << ENDL();
    // DPRINT << "bytes_rxd_dropped " << link_status->bytes_rxd_dropped << ENDL();
    // DPRINT << "corr_cw " << link_status->corr_cw << ENDL();
    // DPRINT << "uncorr_cw " << link_status->uncorr_cw << ENDL();

    eth_send_bytes(
        local_eth_l1_src_addr, remote_eth_l1_dst_addr, num_bytes, num_bytes_per_send, num_bytes_per_send_word_size);

    reinterpret_cast<void (*)(uint32_t)>(
        (uint32_t)(((eth_api_table_t*)(MEM_SYSENG_ETH_API_TABLE))->eth_link_status_check_ptr))(0xFFFFFFFF);
    DPRINT << "post-send" << ENDL();
    DPRINT << "retrain_count " << link_status->retrain_count << ENDL();
    DPRINT << "port_status " << port_status_ptr[0] << ENDL();
    DPRINT << "rx_link_up " << link_status->rx_link_up << ENDL();
    // DPRINT << "frames_txd " << link_status->frames_txd << ENDL();
    // DPRINT << "frames_txd_ok " << link_status->frames_txd_ok << ENDL();
    // DPRINT << "frames_txd_badfcs " << link_status->frames_txd_badfcs << ENDL();
    // DPRINT << "bytes_txd " << link_status->bytes_txd << ENDL();
    // DPRINT << "bytes_txd_ok " << link_status->bytes_txd_ok << ENDL();
    // DPRINT << "frames_rxd_ok " << link_status->frames_rxd_ok << ENDL();
    // DPRINT << "frames_rxd_badfcs " << link_status->frames_rxd_badfcs << ENDL();
    // DPRINT << "frames_rxd_dropped " << link_status->frames_rxd_dropped << ENDL();
    // DPRINT << "bytes_rxd " << link_status->bytes_rxd << ENDL();
    // DPRINT << "bytes_rxd_ok " << link_status->bytes_rxd_ok << ENDL();
    // DPRINT << "bytes_rxd_badfcs " << link_status->bytes_rxd_badfcs << ENDL();
    // DPRINT << "bytes_rxd_dropped " << link_status->bytes_rxd_dropped << ENDL();
    // DPRINT << "corr_cw " << link_status->corr_cw << ENDL();
    // DPRINT << "uncorr_cw " << link_status->uncorr_cw << ENDL();

    eth_wait_for_receiver_done();
}
