// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "debug/assert.h"

#if defined(COMPILE_FOR_TRISC)
#include "compute_kernel_api/common.h"

namespace NAMESPACE {
void MAIN {
#else
#include "dataflow_api.h"

void kernel_main() {
#endif
#if !defined(UCK_CHLKC_MATH)
    constexpr uint32_t remote_cb_id = get_compile_time_arg_val(0);

    auto& remote_receiver_cb_interface = get_remote_receiver_cb_interface(remote_cb_id);

    uint32_t arg_idx = 0;
    uint32_t config_idx = 0;
    bool pass = true;
    // config_addr
    uint32_t config_addr = get_arg_val<uint32_t>(arg_idx++);
    pass &= remote_receiver_cb_interface.config_ptr == config_addr;
    ASSERT(pass);
    volatile tt_l1_ptr uint32_t* config_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(config_addr);
    // is_sender
    bool is_sender = get_arg_val<uint32_t>(arg_idx++);
    pass &= config_ptr[config_idx++] == is_sender;
    ASSERT(pass);
    // num_receivers
    config_idx++;  // Skip num_receivers
    // fifo_start_addr
    uint32_t fifo_start_addr = get_arg_val<uint32_t>(arg_idx++);
    pass &= config_ptr[config_idx++] == fifo_start_addr;
    ASSERT(pass);
    pass &= remote_receiver_cb_interface.fifo_start_addr == fifo_start_addr;
    ASSERT(pass);
    // fifo_size
    uint32_t fifo_size = get_arg_val<uint32_t>(arg_idx++);
    pass &= config_ptr[config_idx++] == fifo_size;
    ASSERT(pass);
    // fifo_ptr
    uint32_t fifo_ptr = get_arg_val<uint32_t>(arg_idx++);
    pass &= config_ptr[config_idx++] == fifo_ptr;
    ASSERT(pass);
    // remote_noc_xy_addr
    uint32_t remote_noc_xy_addr = config_ptr[config_idx++];
    pass &= remote_receiver_cb_interface.sender_noc_x == get_arg_val<uint32_t>(arg_idx++);
    ASSERT(pass);
    pass &= remote_receiver_cb_interface.sender_noc_y == get_arg_val<uint32_t>(arg_idx++);
    ASSERT(pass);

    // aligned_pages_acked_addr
    uint32_t aligned_pages_sent_addr = config_ptr[config_idx++];
    pass &= remote_receiver_cb_interface.aligned_pages_acked_ptr == aligned_pages_sent_addr + L1_ALIGNMENT;
    ASSERT(pass);
    volatile tt_l1_ptr uint32_t* pages_sent_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(aligned_pages_sent_addr);
    pass &= *pages_sent_ptr == get_arg_val<uint32_t>(arg_idx++);
    ASSERT(pass);
    volatile tt_l1_ptr uint32_t* pages_acked_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(remote_receiver_cb_interface.aligned_pages_acked_ptr);
    pass &= *pages_sent_ptr == get_arg_val<uint32_t>(arg_idx++);
    ASSERT(pass);
    // fifo_rd_ptr
    pass &= remote_receiver_cb_interface.fifo_rd_ptr == get_arg_val<uint32_t>(arg_idx++);
    ASSERT(pass);
    // fifo_limit_page_aligned
    pass &= remote_receiver_cb_interface.fifo_limit_page_aligned == get_arg_val<uint32_t>(arg_idx++);
    ASSERT(pass);
    // fifo_page_size
    pass &= remote_receiver_cb_interface.fifo_page_size == get_arg_val<uint32_t>(arg_idx++);
    ASSERT(pass);

    // Hang if watcher not enabled
    while (!pass);
#endif
}
#if defined(COMPILE_FOR_TRISC)
}  // namespace NAMESPACE
#endif
