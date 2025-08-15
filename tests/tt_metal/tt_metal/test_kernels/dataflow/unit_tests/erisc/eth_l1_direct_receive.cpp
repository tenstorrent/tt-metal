// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "eth_fw_api.h"

/**
 * Any two RISC processors cannot use the same CMD_BUF
 * non_blocking APIs shouldn't be mixed with slow noc.h APIs
 * explicit flushes need to be used since the calls are non-blocking
 * */

void kernel_main() {
    std::uint32_t num_bytes = get_arg_val<uint32_t>(0);

    volatile eth_api_table_t* eth_api_table = (volatile eth_api_table_t*)(MEM_SYSENG_ETH_API_TABLE);
    reinterpret_cast<void (*)(uint32_t)>(
        (uint32_t)(((eth_api_table_t*)(MEM_SYSENG_ETH_API_TABLE))->eth_link_status_check_ptr))(0xFFFFFFFF);

    volatile eth_live_status_t* link_status = (volatile eth_live_status_t*)(ETH_RESULTS_BASE_ADDR_LIVE_STATUS);
    volatile tt_l1_ptr uint32_t* port_status_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(0x7CC04);

    DPRINT << "pre-receive" << ENDL();
    DPRINT << "port_status " << port_status_ptr[0] << ENDL();
    DPRINT << "retrain_count " << link_status->retrain_count << ENDL();
    DPRINT << "rx_link_up " << link_status->rx_link_up << ENDL();
    // DPRINT << "frames_rxd " << link_status->frames_rxd << ENDL();
    // DPRINT << "frames_rxd_ok " << link_status->frames_rxd_ok << ENDL();
    // DPRINT << "frames_rxd_badfcs " << link_status->frames_rxd_badfcs << ENDL();
    // DPRINT << "frames_rxd_dropped " << link_status->frames_rxd_dropped << ENDL();
    // DPRINT << "bytes_rxd " << link_status->bytes_rxd << ENDL();
    // DPRINT << "bytes_rxd_ok " << link_status->bytes_rxd_ok << ENDL();
    // DPRINT << "bytes_rxd_badfcs " << link_status->bytes_rxd_badfcs << ENDL();
    // DPRINT << "bytes_rxd_dropped " << link_status->bytes_rxd_dropped << ENDL();
    // DPRINT << "corr_cw " << link_status->corr_cw << ENDL();
    // DPRINT << "uncorr_cw " << link_status->uncorr_cw << ENDL();

    eth_wait_for_bytes(num_bytes);
    eth_receiver_done();
}
