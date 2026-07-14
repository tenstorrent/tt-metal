// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Always-on ethernet firmware execution-stage breadcrumb.
//
// Unlike WAYPOINT() (which is compiled out unless -DWATCHER_ENABLED), set_eth_fw_stage() writes
// unconditionally to a reserved eth-L1 word (MEM_ERISC_FW_STAGE_BASE, which on Blackhole is
// base FW's boot_results_t.eth_status.spare[0..1]) so the coarse stage of the ethernet firmware
// stack (base FW vs Metal application FW vs a launched kernel) can be recovered offline via a plain
// UMD L1 read, without a watcher build or a live repro. It is exposed to the host as
// HalL1MemAddrType::FW_STAGE. See EthFwStage in hostdev/dev_msgs.h and the host-side decoder in
// tt_metal/llrt/llrt.cpp.
//
// One slot is written per ethernet processor (index from get_hw_thread_idx(): 0 = erisc/ierisc,
// 1 = subordinate erisc/ierisc). Arches that don't define MEM_ERISC_FW_STAGE_BASE get a no-op.

#include "dev_mem_map.h"
#include "hostdev/dev_msgs.h"  // EthFwStage
#include "internal/hw_thread.h"

#if defined(MEM_ERISC_FW_STAGE_BASE) && (defined(COMPILE_FOR_ERISC) || defined(COMPILE_FOR_IDLE_ERISC))
inline void set_eth_fw_stage(EthFwStage stage) {
    volatile uint32_t* stage_slots = reinterpret_cast<volatile uint32_t*>(MEM_ERISC_FW_STAGE_BASE);
    stage_slots[internal_::get_hw_thread_idx()] = static_cast<uint32_t>(stage);
}
#else
inline void set_eth_fw_stage(EthFwStage) {}
#endif
