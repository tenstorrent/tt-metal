// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

// One or more IDMA copies on a Keraunos CCE hart, through the Quasar overlay command buffer API.
//
// The kernel records its progress in a small status block in its own SRAM so the host can tell how far
// it got even when a wait never finishes. Status words (uint32, through the uncached alias):
//   0: kStarted once the command buffer is programmed
//   1: idma_free_space before the first issue
//   2: kIssued once every copy is issued
//   3: outstanding iDMA acks for the programmed trid right after the last issue
//   4: kAcked once the wait completes
//
// Compile-time args:
//   0: command buffer, 0 or 1
// Runtime args:
//   0, 1: source address, low and high 32 bits
//   2, 3: destination address, low and high 32 bits
//   4:    length of one copy in bytes
//   5:    VC set: 0 = NocVcs::READ, 1 = NocVcs::WRITE (what idma_copy_cmdbuf and SysEng firmware use)
//   6:    wait: 0 = idma_acked_cmdbuf (programmed trid), 1 = all_idma_acked_cmdbuf
//   7:    status block, local SRAM offset
//   8:    number of copies; copy i moves src + i * len to dst + i * len
//   9:    1 = issue through the composite idma_copy_cmdbuf helper instead of the step-by-step sequence

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "dev_mem_map.h"
#include "internal/tt-2xx/quasar/overlay/cmdbuff_api.hpp"

using namespace overlay;

namespace {
constexpr uint32_t kStarted = 0x1D3A0001;
constexpr uint32_t kIssued = 0x1D3A0002;
constexpr uint32_t kAcked = 0x1D3A0003;

constexpr CmdBuf kCmdBuf = get_compile_time_arg_val(0) == 0 ? CMDBUF_0 : CMDBUF_1;

uint64_t arg_u64(uint32_t index) {
    return (static_cast<uint64_t>(get_arg_val<uint32_t>(index + 1)) << 32) | get_arg_val<uint32_t>(index);
}
}  // namespace

void kernel_main() {
    const uint64_t src = arg_u64(0);
    const uint64_t dst = arg_u64(2);
    const uint32_t len = get_arg_val<uint32_t>(4);
    const NocVcs vcs = get_arg_val<uint32_t>(5) != 0 ? NocVcs::WRITE : NocVcs::READ;
    const bool wait_all = get_arg_val<uint32_t>(6) != 0;
    volatile uint32_t* status =
        reinterpret_cast<volatile uint32_t*>(MEM_L1_UNCACHED_BASE + get_arg_val<uint32_t>(7));
    const uint32_t copies = get_arg_val<uint32_t>(8);
    const bool use_helper = get_arg_val<uint32_t>(9) != 0;

    if (use_helper) {
        status[0] = kStarted;
        status[1] = static_cast<uint32_t>(idma_free_space_cmdbuf<kCmdBuf>());
        for (uint32_t i = 0; i < copies; i++) {
            idma_copy_cmdbuf<kCmdBuf>(src + static_cast<uint64_t>(i) * len, dst + static_cast<uint64_t>(i) * len, len);
        }
    } else {
        reset_cmdbuf<kCmdBuf>();
        idma_setup_as_copy_cmdbuf<kCmdBuf>(false);
        setup_ongoing_cmdbuf<kCmdBuf>({.src_addr = true, .dest_addr = true});
        setup_vcs_cmdbuf<kCmdBuf>(vcs);
        setup_trids_cmdbuf<kCmdBuf>();
        set_src_cmdbuf<kCmdBuf>(src);
        set_dest_cmdbuf<kCmdBuf>(dst);
        set_len_cmdbuf<kCmdBuf>(len);
        status[0] = kStarted;
        status[1] = static_cast<uint32_t>(idma_free_space_cmdbuf<kCmdBuf>());
        for (uint32_t i = 0; i < copies; i++) {
            while (idma_free_space_cmdbuf<kCmdBuf>() == 0) {
            }
            issue_cmdbuf<kCmdBuf>();
        }
    }
    status[2] = kIssued;
    status[3] = static_cast<uint32_t>(__builtin_riscv_ttrocc_cmdbuf_idma_tr_ack(kCmdBuf));

    if (wait_all) {
        while (!all_idma_acked_cmdbuf<kCmdBuf>()) {
        }
    } else {
        while (!idma_acked_cmdbuf<kCmdBuf>()) {
        }
    }
    status[4] = kAcked;
}
