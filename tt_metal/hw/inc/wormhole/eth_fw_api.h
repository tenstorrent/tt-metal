// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>

#define MEM_SYSENG_BOOT_RESULTS_BASE 0x1EC0

struct boot_results_t {
    uint32_t reserved_0_4[5];
    uint32_t link_status;
    uint32_t reserved_6;
    uint32_t retrain_cnt;
    uint32_t reserved_8_15[8];
    uint32_t packet_send_0;
    uint32_t packet_send_1;
    uint32_t packet_send_2;
    uint32_t packet_send_3;
    uint32_t packet_recv_0;
    uint32_t packet_recv_1;
    uint32_t packet_recv_2;
    uint32_t packet_recv_3;
    uint32_t reserved_24_44[21];
    uint32_t rx_link_up_time;
    uint32_t dummy_packet_time;
    uint32_t crc_err;
    uint32_t reserved_48;
    uint32_t pcs_faults;
    uint32_t retrain_trigger_link;
    uint32_t retrain_trigger_crc;
    uint32_t corr_cw_hi;
    uint32_t corr_cw_lo;
    uint32_t uncorr_cw_hi;
    uint32_t uncorr_cw_lo;
    uint32_t reserved_56_63[8];
    uint32_t local_board_id_lo;
    uint32_t local_board_id_hi;
    uint32_t local_x_y_coord;
    uint32_t local_rack_shelf;
    uint32_t local_eth_id;
    uint32_t local_board_type;
    uint32_t spare_0;
    uint32_t ack_local;
    uint32_t remote_board_id_lo;
    uint32_t remote_board_id_hi;
    uint32_t remote_x_y_coord;
    uint32_t remote_rack_shelf;
    uint32_t remote_eth_id;
    uint32_t remote_board_type;
    uint32_t spare_1;
    uint32_t ack_remote;
};

#if defined(KERNEL_BUILD) || defined(FW_BUILD)
#include "ethernet/erisc.h"

FORCE_INLINE bool is_link_up() {
    // Collected when FW/Fabric is idle and context switches to base FW
    volatile boot_results_t* link_stats = (volatile boot_results_t*)(MEM_SYSENG_BOOT_RESULTS_BASE);
    return link_stats->link_status == 0x6;
}
#endif
