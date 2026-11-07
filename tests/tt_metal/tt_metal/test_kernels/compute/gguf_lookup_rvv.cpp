// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// GGUF grid lookups on the pack thread (TRISC2), for test_gguf_page_helpers.cpp and test_trisc2_rvv.cpp: each page of
// c_0 through lookup_grid into c_16 and, with ComputeConfig::enable_trisc2_rvv, through lookup_grid_rvv into c_17.
// The unpack thread waits for the pages and sends their address. Runtime arg: page count. Define: GGUF_FMT.

#include <cstdint>

#include "api/compute/cb_api.h"
#include "api/compute/common.h"
#if defined(TRISC_PACK)
#include "experimental/gguf_lookup_rvv.h"
#endif

void kernel_main() {
    const uint32_t pages = get_arg_val<uint32_t>(0);
    UNPACK(({
        cb_wait_front(tt::CBIndex::c_0, pages);
        ckernel::mailbox_write(
            ckernel::ThreadId::PackThreadId, get_local_cb_interface(tt::CBIndex::c_0).fifo_rd_ptr << cb_addr_shift);
    }));
#if defined(TRISC_PACK)
    using namespace experimental::gguf;
    constexpr auto FMT = static_cast<Format>(GGUF_FMT);
    constexpr uint32_t page_bytes = page_layout<FMT>().rows * 64;
    const uint32_t in = ckernel::mailbox_read(ckernel::ThreadId::UnpackThreadId);
    cb_reserve_back(tt::CBIndex::c_16, pages);
    cb_reserve_back(tt::CBIndex::c_17, pages);
    const uint32_t out[2] = {
        get_local_cb_interface(tt::CBIndex::c_16).fifo_wr_ptr << cb_addr_shift,
        get_local_cb_interface(tt::CBIndex::c_17).fifo_wr_ptr << cb_addr_shift};
    asm volatile("fence" ::: "memory");
    for (uint32_t p = 0; p < pages; p++) {
        const auto* indices = reinterpret_cast<const uint32_t*>(in + p * page_bytes + index_offset<FMT>());
        lookup_grid<FMT>(indices, level_table<FMT>(), reinterpret_cast<uint32_t*>(out[0] + p * page_bytes));
#if defined(__riscv_vector)
        lookup_grid_rvv<FMT>(indices, level_table<FMT>(), reinterpret_cast<uint32_t*>(out[1] + p * page_bytes));
#endif
    }
    asm volatile("fence" ::: "memory");
    cb_push_back(tt::CBIndex::c_16, pages);
    cb_push_back(tt::CBIndex::c_17, pages);
#endif
    UNPACK((cb_pop_front(tt::CBIndex::c_0, pages)));
}
