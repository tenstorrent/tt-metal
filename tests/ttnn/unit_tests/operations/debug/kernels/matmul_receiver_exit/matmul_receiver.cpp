// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Component harness: the output CB is ready when the receiver finishes its input handshake.
#define kernel_main receiver_kernel_main
#include "ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/reader_bmm_tile_layout_in1_receiver_writer_padding.cpp"
#undef kernel_main

void kernel_main() {
    cb_reserve_back(tt::CBIndex::c_16, 1);
    cb_push_back(tt::CBIndex::c_16, 1);
    receiver_kernel_main();
}
