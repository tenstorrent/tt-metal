// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/hw/inc/ethernet/tunneling.h"

void kernel_main() {
    erisc_info->unused_arg0 = 21300000;
    erisc_info->unused_arg1 = 21400000;
    erisc_info->unused_arg2 = 21500000;
    constexpr bool sender_is_issue_path = get_compile_time_arg_val(0);

    uint32_t consumer_noc_encoding = uint32_t(NOC_XY_ENCODING(7, 10)); // soon to be unified
    uint32_t producer_noc_encoding = uint32_t(NOC_XY_ENCODING(2, 10));
    uint32_t eth_router_noc_encoding = uint32_t(NOC_XY_ENCODING(my_x[0], my_y[0]));

    volatile tt_l1_ptr uint32_t *eth_src_db_semaphore_addr = reinterpret_cast<volatile tt_l1_ptr uint32_t *>(eth_get_semaphore(0));
    volatile tt_l1_ptr uint32_t *eth_dst_db_semaphore_addr = reinterpret_cast<volatile tt_l1_ptr uint32_t *>(eth_get_semaphore(1));

    const uint8_t buffer_id = sender_is_issue_path? (uint8_t)CQTunnelPath::ISSUE : (uint8_t)CQTunnelPath::COMPLETION;
    const uint8_t other_buffer_id = sender_is_issue_path? (uint8_t)CQTunnelPath::COMPLETION : (uint8_t)CQTunnelPath::ISSUE;
    db_cb_config_t *eth_src_db_cb_config = sender_is_issue_path ? get_local_db_cb_config(eth_l1_mem::address_map::ISSUE_CQ_CB_BASE, false) :
                                                                  get_local_db_cb_config(eth_l1_mem::address_map::COMPLETION_CQ_CB_BASE, false);
    db_cb_config_t *eth_dst_db_cb_config = sender_is_issue_path ? get_local_db_cb_config(eth_l1_mem::address_map::ISSUE_CQ_CB_BASE, false) :
                                                                  get_local_db_cb_config(eth_l1_mem::address_map::COMPLETION_CQ_CB_BASE, false);

    int i = 0;
   // while(true) {
   //     erisc_info->unused_arg1 = 99999990 + buffer_id;
   //     internal_::risc_context_switch();
   // }
  while(routing_info->routing_enabled) {
    while(routing_info->routing_enabled && eth_src_db_semaphore_addr[0] == 0 && routing_info->fd_buffer_msgs[buffer_id].bytes_sent != 1) {
        internal_::risc_context_switch();
    }
    if (!routing_info->routing_enabled) {
        break;
    }
    if (eth_src_db_semaphore_addr[0] != 0) {
      // Acquired src for issue
    erisc_info->unused_arg0 = 21300001 + i * 1000;
      eth_tunnel_src_forward_one_cmd<buffer_id, other_buffer_id, sender_is_issue_path>(eth_src_db_cb_config, producer_noc_encoding);
    erisc_info->unused_arg0 = 21300002 + i * 1000;
    }
    if (routing_info->fd_buffer_msgs[other_buffer_id].bytes_sent ==1 ) {
      // shouldn't be here
      erisc_info->unused_arg0 = 77777777;
      internal_::risc_context_switch();
    }
    if (routing_info->fd_buffer_msgs[buffer_id].bytes_sent == 1) {
      // Dst got data from src issue
    erisc_info->unused_arg0 = 21300003 + i * 1000;
      eth_tunnel_dst_forward_one_cmd<buffer_id, other_buffer_id, sender_is_issue_path>(eth_dst_db_cb_config, consumer_noc_encoding);
    erisc_info->unused_arg0 = 21300004 + i * 1000;
    }
    i++;
  }
}
