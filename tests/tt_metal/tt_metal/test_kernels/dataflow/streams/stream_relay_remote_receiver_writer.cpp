// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "dataflow_api.h"

#include "tt_metal/hw/inc/wormhole/noc/noc_overlay_parameters.h"

void kernel_main() {
  uint32_t arg_idx = 0;

  constexpr uint32_t msg_hdr_size = get_compile_time_arg_val(0);

  uint32_t output_buffer_addr = get_arg_val<uint32_t>(arg_idx++);
  uint32_t cb_page_size = get_arg_val<uint32_t>(arg_idx++);
  uint32_t num_pages = get_arg_val<uint32_t>(arg_idx++);

  uint32_t write_page_size = cb_page_size - msg_hdr_size;
  const InterleavedAddrGen<true> dest_addr_gen = {
      .bank_base_address = output_buffer_addr, .page_size = write_page_size};

  auto cb = tt::CB::c_in0;
  for (uint32_t i = 0; i < num_pages; i++) {
    cb_wait_front(cb, 1);
    // NOTE THAT msg_hdr_size is doubled on host side to maintain alignment for DRAM reads/writes in THIS TEST ONLY
    uint32_t src_start = get_read_ptr(cb) + msg_hdr_size;

    uint64_t dst_noc_addr = get_noc_addr(i, dest_addr_gen);
    noc_async_write(src_start, dst_noc_addr, write_page_size);

    noc_async_write_barrier();
    cb_pop_front(cb, 1);
  }

}
