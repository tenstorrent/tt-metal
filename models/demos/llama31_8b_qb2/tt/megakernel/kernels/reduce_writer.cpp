// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#define kernel_main native_reduce_writer_main
#include "ttnn/cpp/ttnn/operations/experimental/ccl/reduce_scatter_minimal_direct/device/kernels/reduce_scatter_minimal_direct_writer.cpp"
#undef kernel_main
void kernel_main() {
    native_reduce_writer_main();
    // close_finish resets the fabric teardown semaphore and publishes the
    // producer cursor. Reopening the same connections is supported by the
    // current adapter and preserves counter state across these two phases.
    noc_semaphore_set(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(13)), 1);
    native_reduce_writer_main();
}
