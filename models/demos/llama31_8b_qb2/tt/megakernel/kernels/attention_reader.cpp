// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#ifndef QB2_ENTRY
#define QB2_ENTRY kernel_main
#endif
#define kernel_main native_attention_reader
#include "ttnn/cpp/ttnn/operations/transformer/sdpa_decode/device/kernels/dataflow/reader_decode_all.cpp"
#undef kernel_main
#include "tools/profiler/kernel_profiler.hpp"
void QB2_ENTRY() {
    {
        DeviceZoneScopedN("ATTENTION-WAIT-KV");
        noc_semaphore_wait(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(4)), 1);
    }
    native_attention_reader();
}
