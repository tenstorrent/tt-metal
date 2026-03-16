// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/cpp/ttnn/kernel_lib/matmul_1d_dataflow_helpers.hpp"

void kernel_main() {
    uint32_t dst_addr = get_arg_val<uint32_t>(0);
    uint32_t Mt = get_arg_val<uint32_t>(1);
    uint32_t Nt = get_arg_val<uint32_t>(3);  // index 2 is Kt (unused by writer), index 3 is Nt

    constexpr uint32_t cb_id_out0 = 16;

    dataflow_kernel_lib::write_matmul_tiles<cb_id_out0>(dst_addr, Mt, Nt);
}
