// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "dataflow_api.h"

void kernel_main() {
  uint32_t adjacent_noc_x = get_arg_val<uint32_t>(0);
  uint32_t adjacent_noc_y = get_arg_val<uint32_t>(1);
  uint32_t adjacent_noc_cb1 = get_arg_val<uint32_t>(2);
  uint32_t iter_count = get_arg_val<uint32_t>(3);

  constexpr uint32_t cb0_id = 0;
  uint32_t single_tile_size_bytes = get_tile_size(cb0_id);

  uint32_t cb0_addr;
  cb_reserve_back(cb0_id, 1);
  cb0_addr = get_write_ptr(cb0_id);

  for (uint32_t i = 0; i < iter_count; i++) {
    uint64_t target_noc_addr =
        get_noc_addr(adjacent_noc_x, adjacent_noc_y, adjacent_noc_cb1);

    noc_async_read(target_noc_addr, cb0_addr, single_tile_size_bytes);
    noc_async_read_barrier();
  }
}
