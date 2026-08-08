// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Always-on ethernet firmware execution-stage breadcrumb.
//
// Unlike WAYPOINT() (which is compiled out unless -DWATCHER_ENABLED), set_eth_fw_stage() writes
// unconditionally to a dedicated reserved eth-L1 word (MEM_*ERISC_FW_STAGE_BASE, carved from the
// tail of the eth mailbox region) so the coarse stage of the ethernet firmware stack (base FW vs
// Metal application FW vs a launched kernel) can be recovered offline via a plain UMD L1 read,
// without a watcher build or a live repro. It is exposed to the host as HalL1MemAddrType::FW_STAGE.
// See EthFwStage in hostdev/dev_msgs.h and the host-side decoder in tt_metal/llrt/llrt.cpp.
//
// One slot is written per ethernet processor (index from get_hw_thread_idx(): 0 = erisc/ierisc,
// 1 = subordinate erisc/ierisc).

#include "hostdev/dev_msgs.h"  // EthFwStage
#include "internal/hw_thread.h"

#if defined(COMPILE_FOR_ERISC)
#define ETH_FW_STAGE_BASE_ADDR MEM_AERISC_FW_STAGE_BASE
#elif defined(COMPILE_FOR_IDLE_ERISC)
#define ETH_FW_STAGE_BASE_ADDR MEM_IERISC_FW_STAGE_BASE
#endif

#ifdef ETH_FW_STAGE_BASE_ADDR
inline void set_eth_fw_stage(EthFwStage stage) {
    volatile uint32_t* stage_slots = reinterpret_cast<volatile uint32_t*>(ETH_FW_STAGE_BASE_ADDR);
    stage_slots[internal_::get_hw_thread_idx()] = static_cast<uint32_t>(stage);
}
#else
// Non-ethernet build: no-op so the header can be included unconditionally.
inline void set_eth_fw_stage(EthFwStage) {}
#endif
