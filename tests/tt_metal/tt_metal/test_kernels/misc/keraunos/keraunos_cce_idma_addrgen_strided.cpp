// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Strided IDMA copy driven by the overlay address generator on a Keraunos CCE hart. Same shape as
// data_movement/quasar_examples/quasar_idma/kernels/idma_1d_strided_example.cpp, with 64-bit bases from
// runtime args instead of Tensix L1 compile-time addresses.
//
// Element i moves from src + i * src_stride to dst + i * dst_stride. Address generator 0 feeds command
// buffer 0 (paired_cmdbuf(ADDRGEN_0) == CMDBUF_0).
//
// Status words, as in cce_idma_copy.cpp: 0 started, 1 the first popped source address (low 32 bits),
// 2 issued, 4 acked.
//
// Runtime args:
//   0, 1: source base, low and high 32 bits
//   2, 3: destination base, low and high 32 bits
//   4:    element size in bytes
//   5:    number of elements
//   6:    source stride in bytes
//   7:    destination stride in bytes
//   8:    status block, local SRAM offset

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "dev_mem_map.h"
#include "internal/tt-2xx/quasar/overlay/addrgen_api.hpp"
#include "internal/tt-2xx/quasar/overlay/cmdbuff_api.hpp"

using namespace overlay;

namespace {
uint64_t arg_u64(uint32_t index) {
    return (static_cast<uint64_t>(get_arg_val<uint32_t>(index + 1)) << 32) | get_arg_val<uint32_t>(index);
}
}  // namespace

void kernel_main() {
    const uint64_t src = arg_u64(0);
    const uint64_t dst = arg_u64(2);
    const uint32_t elem_size = get_arg_val<uint32_t>(4);
    const uint32_t count = get_arg_val<uint32_t>(5);
    const uint64_t src_stride = get_arg_val<uint32_t>(6);
    const uint64_t dst_stride = get_arg_val<uint32_t>(7);
    volatile uint32_t* status =
        reinterpret_cast<volatile uint32_t*>(MEM_L1_UNCACHED_BASE + get_arg_val<uint32_t>(8));

    reset_cmdbuf_0();
    reset_addrgen_0();
    setup_src_base_start_addrgen_0(src);
    setup_src_inner_loop_addrgen_0(src_stride, static_cast<uint64_t>(count) * src_stride);
    setup_dest_base_start_addrgen_0(dst);
    setup_dest_inner_loop_addrgen_0(dst_stride, static_cast<uint64_t>(count) * dst_stride);
    status[1] = static_cast<uint32_t>(peek_src_addrgen_0());

    idma_setup_as_copy_cmdbuf_0(false);
    setup_vcs_cmdbuf_0(NocVcs::WRITE);
    setup_trids_cmdbuf_0();
    set_len_cmdbuf_0(elem_size);
    status[0] = 0x1D3A0001;

    for (uint32_t i = 0; i < count; i++) {
        while (idma_free_space_cmdbuf_0() == 0) {
        }
        push_both_addrgen_0();
        issue_cmdbuf_0();
    }
    status[2] = 0x1D3A0002;

    while (!idma_acked_cmdbuf_0()) {
    }
    status[4] = 0x1D3A0003;
}
