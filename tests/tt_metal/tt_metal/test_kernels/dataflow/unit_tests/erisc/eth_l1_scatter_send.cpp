// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

void kernel_main() {
    std::uint32_t local_eth_l1_src_addr = get_arg_val<uint32_t>(0);
    std::uint32_t remote_eth_l1_dst_addr_0 = get_arg_val<uint32_t>(1);
    std::uint32_t remote_eth_l1_dst_addr_1 = get_arg_val<uint32_t>(2);
    std::uint32_t remote_eth_l1_dst_addr_2 = get_arg_val<uint32_t>(3);
    std::uint32_t remote_eth_l1_dst_addr_3 = get_arg_val<uint32_t>(4);
    std::uint32_t num_bytes = get_arg_val<uint32_t>(5);

    reset_erisc_info();
    eth_send_bytes(local_eth_l1_src_addr, remote_eth_l1_dst_addr_0, num_bytes);
    eth_send_bytes(local_eth_l1_src_addr, remote_eth_l1_dst_addr_1, num_bytes);
    eth_send_bytes(local_eth_l1_src_addr, remote_eth_l1_dst_addr_2, num_bytes);
    eth_send_bytes(local_eth_l1_src_addr, remote_eth_l1_dst_addr_3, num_bytes);
    eth_wait_for_receiver_done();
}
