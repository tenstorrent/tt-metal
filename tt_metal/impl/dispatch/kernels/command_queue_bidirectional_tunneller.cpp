// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/hw/inc/ethernet/tunneling.h"

void kernel_main() {
    erisc_info->unused_arg0 = 21300000;
    erisc_info->unused_arg1 = 21400000;
    erisc_info->unused_arg2 = 21500000;
    constexpr bool sender_is_issue_path = get_compile_time_arg_val(0);

    uint32_t consumer_noc_encoding = uint32_t(NOC_XY_ENCODING(CONSUMER_NOC_X, CONSUMER_NOC_Y)); // soon to be unified
    uint32_t producer_noc_encoding = uint32_t(NOC_XY_ENCODING(PRODUCER_NOC_X, PRODUCER_NOC_Y));

    volatile tt_l1_ptr uint32_t *eth_src_db_semaphore_addr = reinterpret_cast<volatile tt_l1_ptr uint32_t *>(eth_get_semaphore(0));
    volatile tt_l1_ptr uint32_t *eth_dst_db_semaphore_addr = reinterpret_cast<volatile tt_l1_ptr uint32_t *>(eth_get_semaphore(1));

    const uint8_t buffer_id = sender_is_issue_path? (uint8_t)CQTunnelPath::ISSUE : (uint8_t)CQTunnelPath::COMPLETION;
    const uint8_t other_buffer_id = sender_is_issue_path? (uint8_t)CQTunnelPath::COMPLETION : (uint8_t)CQTunnelPath::ISSUE;
    db_cb_config_t *eth_src_db_cb_config = sender_is_issue_path ? get_local_db_cb_config(eth_l1_mem::address_map::ISSUE_CQ_CB_BASE) :
                                                                  get_local_db_cb_config(eth_l1_mem::address_map::COMPLETION_CQ_CB_BASE);
    db_cb_config_t *eth_dst_db_cb_config = sender_is_issue_path ? get_local_db_cb_config(eth_l1_mem::address_map::ISSUE_CQ_CB_BASE) :
                                                                  get_local_db_cb_config(eth_l1_mem::address_map::COMPLETION_CQ_CB_BASE);

    int i = 0;
   // while(true) {
   //     erisc_info->unused_arg1 = 99999990 + buffer_id;
   //     internal_::risc_context_switch();
   // }
  while(routing_info->routing_enabled) {
    // Implement yielding if SENDER is not ISSUE, this may help with devices getting commands first
    while(routing_info->routing_enabled && eth_src_db_semaphore_addr[0] == 0 && routing_info->fd_buffer_msgs[buffer_id].bytes_sent != 1
        && routing_info->fd_buffer_msgs[other_buffer_id].bytes_sent != 1) {
        internal_::risc_context_switch();
    }
    if (!routing_info->routing_enabled) {
        break;
    }
    if (routing_info->fd_buffer_msgs[buffer_id].bytes_sent == 1) {
      // Dst got data from src issue
    erisc_info->unused_arg0 = 21300003 + 10 * buffer_id;
    erisc_info->unused_arg2 = CONSUMER_NOC_X << 16 | CONSUMER_NOC_Y;
      eth_tunnel_dst_forward_one_cmd<buffer_id, other_buffer_id, sender_is_issue_path>(eth_dst_db_cb_config, consumer_noc_encoding);
    erisc_info->unused_arg0 = 21300004;
    }

    if (routing_info->fd_buffer_msgs[other_buffer_id].bytes_sent == 1) {
      // Dst got data from src completion
      while(true) {
          erisc_info->unused_arg1 = 99999999;
          internal_::risc_context_switch();
      }
    erisc_info->unused_arg0 = 21300005 + 10 * other_buffer_id;
      eth_tunnel_dst_forward_one_cmd<other_buffer_id, buffer_id, sender_is_issue_path>(eth_dst_db_cb_config, consumer_noc_encoding);
    erisc_info->unused_arg0 = 21300006;
    }
    if (eth_src_db_semaphore_addr[0] != 0) {
      // Acquired src for issue or completion
    erisc_info->unused_arg0 = 21300001 + 10 * buffer_id;
      eth_tunnel_src_forward_one_cmd<buffer_id, other_buffer_id, sender_is_issue_path>(eth_src_db_cb_config, producer_noc_encoding);
    erisc_info->unused_arg0 = 21300002;
    }
    i++;
  }
}
