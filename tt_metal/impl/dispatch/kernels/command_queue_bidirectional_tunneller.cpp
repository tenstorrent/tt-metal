// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/hw/inc/ethernet/tunneling.h"

void kernel_main() {
    constexpr bool sender_is_issue_path = get_compile_time_arg_val(0);
    constexpr uint32_t num_tensix_cmd_slots = get_compile_time_arg_val(1);

    uint32_t consumer_noc_encoding = uint32_t(NOC_XY_ENCODING(CONSUMER_NOC_X, CONSUMER_NOC_Y));
    uint32_t producer_noc_encoding = uint32_t(NOC_XY_ENCODING(PRODUCER_NOC_X, PRODUCER_NOC_Y));

    volatile tt_l1_ptr uint32_t *eth_src_db_semaphore_addr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t *>(eth_get_semaphore(0));
    volatile tt_l1_ptr uint32_t *eth_dst_db_semaphore_addr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t *>(eth_get_semaphore(1));

    const uint8_t buffer_id = sender_is_issue_path ? 0 : 1;
    const uint8_t other_buffer_id = sender_is_issue_path ? 1 : 0;
    db_cb_config_t *eth_src_db_cb_config = sender_is_issue_path
                                               ? get_local_db_cb_config(eth_l1_mem::address_map::ISSUE_CQ_CB_BASE)
                                               : get_local_db_cb_config(eth_l1_mem::address_map::COMPLETION_CQ_CB_BASE);
    db_cb_config_t *eth_dst_db_cb_config = sender_is_issue_path
                                               ? get_local_db_cb_config(eth_l1_mem::address_map::COMPLETION_CQ_CB_BASE)
                                               : get_local_db_cb_config(eth_l1_mem::address_map::ISSUE_CQ_CB_BASE);

    uint32_t remote_issue_cmd_slots = 0;

    while (routing_info->routing_enabled) {
        // Implement yielding if SENDER is not ISSUE, this may help with devices getting commands first
        while (routing_info->routing_enabled && eth_src_db_semaphore_addr[0] == 0 &&
               routing_info->fd_buffer_msgs[buffer_id].bytes_sent != 1 &&
               routing_info->fd_buffer_msgs[other_buffer_id].bytes_sent != 1) {
            internal_::risc_context_switch();
        }
        if (!routing_info->routing_enabled) {
            break;
        }
        // can make two copies of this kernel, one for SRC is issue, one for SRC is completion
        if constexpr(not sender_is_issue_path) {
            // Service completion path first
            // Acquired src for completion
            if (eth_src_db_semaphore_addr[0] != 0) {
                eth_tunnel_src_forward_one_cmd<buffer_id, other_buffer_id, sender_is_issue_path>(
                    eth_src_db_cb_config, producer_noc_encoding, &remote_issue_cmd_slots);
            }
          //  }
            if (remote_issue_cmd_slots == num_tensix_cmd_slots) {
                continue;
            }
            // Cmd valid in issue DST
            if (routing_info->fd_buffer_msgs[other_buffer_id].bytes_sent == 1) {
                eth_tunnel_dst_forward_one_cmd<other_buffer_id, buffer_id, sender_is_issue_path>(
                    eth_dst_db_cb_config, consumer_noc_encoding, &remote_issue_cmd_slots);
            }
        } else {
            // Service completion path first
            // Cmd valid in completion DST
            if (routing_info->fd_buffer_msgs[other_buffer_id].bytes_sent == 1) {
                eth_tunnel_dst_forward_one_cmd<other_buffer_id, buffer_id, sender_is_issue_path>(
                    eth_dst_db_cb_config, consumer_noc_encoding, &remote_issue_cmd_slots);
            }
            if (remote_issue_cmd_slots == num_tensix_cmd_slots) {
                continue;
            }

            // Acquired src for issue
            if (eth_src_db_semaphore_addr[0] != 0) {
                eth_tunnel_src_forward_one_cmd<buffer_id, other_buffer_id, sender_is_issue_path>(
                    eth_src_db_cb_config, producer_noc_encoding, &remote_issue_cmd_slots);
            }
        }
    }
}
