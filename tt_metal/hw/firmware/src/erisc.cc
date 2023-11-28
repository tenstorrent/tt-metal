// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "eth_l1_address_map.h"
#include "ethernet/dataflow_api.h"
#include "noc_nonblocking_api.h"
#include "noc_parameters.h"
#include "risc_attribs.h"
#include "tt_eth_api.h"

#ifdef __cplusplus
extern "C" {
#endif

void ApplicationHandler(void) __attribute__((__section__(".init")));

#ifdef __cplusplus
}
#endif


void __attribute__((section("erisc_l1_code"))) ApplicationHandler(void) {
    rtos_context_switch_ptr = (void (*)())RtosTable[0];

    noc_init();

    int32_t src_addr = eth_l1_mem::address_map::ERISC_APP_RESERVED_BASE;
    int32_t dst_addr = eth_l1_mem::address_map::ERISC_APP_RESERVED_BASE;

    uint32_t mode = erisc_info->mode;
    uint32_t num_loops = erisc_info->num_bytes >> 4;
    if (mode == 0) {
        // Ethernet Send
        erisc_info->bytes_sent = 0;
        for (uint32_t i = 0; i < num_loops; i++) {
            eth_send_packet(0, i + (src_addr >> 4), i + (dst_addr >> 4), 1);
        }
        erisc_info->bytes_sent = erisc_info->num_bytes;
        eth_send_packet(0, ((uint32_t)(&(erisc_info->bytes_sent))) >> 4, ((uint32_t)(&(erisc_info->bytes_sent))) >> 4, 1);
        uint64_t start_time = eth_read_wall_clock();
        while (erisc_info->bytes_sent != 0) {
            RISC_POST_STATUS(0x10000001 | (erisc_info->bytes_sent << 12));
        }
    } else if (mode == 1) {
        // Ethernet Receive
        uint64_t start_time = eth_read_wall_clock();
        while (erisc_info->bytes_sent != erisc_info->num_bytes) {
            RISC_POST_STATUS(0x10000002 | (erisc_info->bytes_sent << 12));
        }
        erisc_info->bytes_sent = 0;
        eth_send_packet(0, ((uint32_t)(&(erisc_info->bytes_sent))) >> 4, ((uint32_t)(&(erisc_info->bytes_sent))) >> 4, 1);
    } else {
        while (true) RISC_POST_STATUS(0x1234DEAD);
    }
    flag_disable[0] = 0;
}
