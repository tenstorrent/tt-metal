// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

// IDMA L1 accumulate on a Keraunos CCE hart with every knob from the runtime args, for finding the setting that adds
// (cce-dv l1_atomic_accum.cpp: fmt 3 INT32, 4 INT32_NORMAL, 5 UINT32, 6 INT8, 7 UINT8; operation 9).
//
// Status words (uint32, uncached alias): 0 started, 1 completed repeats, 4 done.
//
// Runtime args:
//   0, 1: source address, low and high 32 bits
//   2, 3: destination address, low and high 32 bits
//   4:    length in bytes
//   5:    repeats
//   6:    L1 atomic format
//   7:    L1 atomic operation
//   8:    flags: bit 0 wrapping_en, bit 1 disable_sat
//   9:    status block, local SRAM offset

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "dev_mem_map.h"
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
    const uint32_t len = get_arg_val<uint32_t>(4);
    const uint32_t repeats = get_arg_val<uint32_t>(5);
    const uint32_t fmt = get_arg_val<uint32_t>(6);
    const uint32_t op = get_arg_val<uint32_t>(7);
    const uint32_t flags = get_arg_val<uint32_t>(8);
    volatile uint32_t* status =
        reinterpret_cast<volatile uint32_t*>(MEM_L1_UNCACHED_BASE + get_arg_val<uint32_t>(9));

    status[0] = 0x1D3A0001;
    for (uint32_t i = 0; i < repeats; i++) {
        reset_cmdbuf<CMDBUF_0>();
        idma_setup_as_atomic_accum_cmdbuf<CMDBUF_0>((flags & 1u) != 0);
        l1_atomic_instr_cmdbuf<CMDBUF_0>(fmt, (flags & 2u) != 0, op);
        setup_ongoing_cmdbuf<CMDBUF_0>({});
        setup_vcs_cmdbuf<CMDBUF_0>(NocVcs::WRITE);
        setup_trids_cmdbuf<CMDBUF_0>(CMDBUF_DEF_TRID);
        set_src_cmdbuf<CMDBUF_0>(src);
        set_dest_cmdbuf<CMDBUF_0>(dst);
        set_len_cmdbuf<CMDBUF_0>(len);
        issue_cmdbuf<CMDBUF_0>();
        while (!idma_acked_cmdbuf<CMDBUF_0>()) {
        }
        status[1] = i + 1;
    }
    status[4] = 0x1D3A0003;
}
