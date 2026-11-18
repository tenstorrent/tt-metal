// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// How the router writes to the host, for the host_rdma channel: the credits it returns to the host that injected the
// packets. The router names only HostWriter and HostWriteTarget, so the transport behind them can change (another
// path to host memory, a doorbell, a mailbox on another core) without touching the router's channel code.

#include <cstdint>

#include "internal/risc_attribs.h"
#include "tt_metal/fabric/hw/inc/edm_fabric/router_data_cache.hpp"
#include <tt-metalium/experimental/fabric/fabric_edm_types.hpp>

namespace tt::tt_fabric::host_rdma {

// Where a word for the host goes: a node and an address there, both as the host published them in the channel's
// HostRdmaChannelInfo. `enabled` is false until the host has published a target.
struct HostWriteTarget {
    uint32_t noc_xy;
    uint64_t addr;
    bool enabled;
};

// The channel's target, read from its connection info (which the host writes, so the read skips the data cache).
template <bool RISC_CPU_DATA_CACHE_ENABLED>
FORCE_INLINE HostWriteTarget read_host_write_target(uint32_t connection_info_addr) {
    router_invalidate_l1_cache<RISC_CPU_DATA_CACHE_ENABLED>();
    auto* info = reinterpret_cast<volatile tt_l1_ptr HostRdmaChannelInfo*>(connection_info_addr);
    HostWriteTarget target;
    target.enabled = info->credit_enabled != 0;
    target.noc_xy = info->credit_noc_xy;
    target.addr = (static_cast<uint64_t>(info->credit_addr_hi) << 32) | info->credit_addr_lo;
    return target;
}

// At router init: no target until the host publishes one (a target left by an earlier run may point at host memory
// that run has since freed).
FORCE_INLINE void clear_host_write_target(uint32_t connection_info_addr) {
    auto* info = reinterpret_cast<volatile tt_l1_ptr HostRdmaChannelInfo*>(connection_info_addr);
    info->credit_enabled = 0;
    info->credit_addr_lo = 0;
    info->credit_addr_hi = 0;
    info->credit_noc_xy = 0;
}

// A NoC write through the PCIe core into pinned host memory: the 64-bit form of the write (the target address's high
// half in NOC_RET_ADDR_MID, the PCIe core's encoding as the node), posted. The command buffer is programmed whole for
// the write and its high address half cleared after, so a later on-chip write on the same buffer is not routed to the
// host.
struct PcieNocHostWriter {
    template <uint8_t WRITE_NOC_MODE, uint32_t WRITE_CMD_BUF>
    static FORCE_INLINE void write_word(uint8_t noc, uint32_t src_l1_addr, const HostWriteTarget& target) {
        while (!noc_cmd_buf_ready(noc, WRITE_CMD_BUF)) {
        }
        noc_write_init_state<WRITE_CMD_BUF>(noc, NOC_UNICAST_WRITE_VC);
        noc_wwrite_with_state<
            WRITE_NOC_MODE,
            WRITE_CMD_BUF,
            CQ_NOC_SNDL,
            CQ_NOC_SEND,
            CQ_NOC_WAIT,
            /*update_counter=*/true,
            /*posted=*/true>(noc, src_l1_addr, target.noc_xy, target.addr, sizeof(uint32_t), 1);
#ifdef ARCH_BLACKHOLE
        while (!noc_cmd_buf_ready(noc, WRITE_CMD_BUF)) {
        }
        noc_cmd_buf_clear_ret_addr_mid(noc, WRITE_CMD_BUF);
#endif
    }
};

// The router's path to the host.
using HostWriter = PcieNocHostWriter;

}  // namespace tt::tt_fabric::host_rdma
