// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Always-on ethernet firmware execution-stage breadcrumb.
//
// Unlike WAYPOINT() (which is compiled out unless -DWATCHER_ENABLED), set_eth_fw_stage() writes
// unconditionally to a reserved eth-L1 word (MEM_SYSENG_ETH_FW_STAGE, which on Blackhole is
// base FW's eth_status_t.eth_fw_stage) so the coarse stage of the ethernet firmware
// stack (base FW vs Metal application FW vs a launched kernel) can be recovered offline via a plain
// UMD L1 read, without a watcher build or a live repro. It is exposed to the host as
// HalL1MemAddrType::FW_STAGE. See eth_fw_stage_e in the Blackhole eth_fw_api.h and the host-side decoder
// in tt_metal/llrt/llrt.cpp.
//
// One slot is written per ethernet processor (index from get_hw_thread_idx(): 0 = erisc/ierisc,
// 1 = subordinate erisc/ierisc). Arches that don't define MEM_SYSENG_ETH_FW_STAGE get a no-op.

#include "eth_fw_api.h"
#include "internal/hw_thread.h"

#if defined(MEM_SYSENG_ETH_FW_STAGE) && (defined(COMPILE_FOR_ERISC) || defined(COMPILE_FOR_IDLE_ERISC))
inline void set_eth_fw_stage(eth_fw_stage_e stage) {
    auto* fw_stage = reinterpret_cast<volatile eth_fw_stage_t*>(MEM_SYSENG_ETH_FW_STAGE);
    fw_stage->stage[internal_::get_hw_thread_idx()] = static_cast<uint32_t>(stage);
}
#else
// A macro rather than an empty function: eth_fw_stage_e is only defined on arches that implement the breadcrumb.
#define set_eth_fw_stage(stage) ((void)0)
#endif
