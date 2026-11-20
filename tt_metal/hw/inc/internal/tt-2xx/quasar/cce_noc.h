// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// NOC calls on a Keraunos CCE (a Quasar Ethernet core), behind noc_nonblocking_api_v2.h. A CCE has no NOC endpoint:
// a NOC coordinate resolves to an HSIO tile and CCE by rule or through the host-written KeraunosNocChannels
// (hostdev/keraunos_noc_channels.h), its address comes from KeraunosSpaRoutes, a copy is one iDMA transfer on command
// buffer 0, and a word write or increment is a CPU store or load and store. Ethernet sends use
// command buffer 1 (eth_cce_send.h).

#pragma once

#include <cstdint>

#include "dev_mem_map.h"

#ifndef KERAUNOS_CCE
#error "cce_noc.h is for a Keraunos CCE (Quasar Ethernet core) only"
#endif

#include "api/debug/assert.h"
#include "hostdev/keraunos_noc_channels.h"
#include "hostdev/keraunos_spa_routes.h"
#include "internal/risc_attribs.h"
#include "internal/tt-2xx/quasar/overlay/cmdbuff_api.hpp"
#include "noc_parameters.h"

namespace internal_::cce_noc {

// What halt() records in KeraunosNocChannels::fault; the detail goes to fault_detail.
enum class Fault : uint32_t {
    Unmapped = 0xCCEEF101,     // detail: the NOC address. Its coordinate names no core, or the channels are not valid
    OutsideSram = 0xCCEEF103,  // detail: the flat address, outside both views of its CCE's SRAM
    Unsupported = 0xCCEEF104,  // detail: the line of noc_nonblocking_api_v2.h that called unsupported()
};

constexpr overlay::CmdBuf kCmdBuf = overlay::CMDBUF_0;
constexpr uint32_t kTrid = overlay::CMDBUF_DEF_TRID;
// chippy's write request VC (cmdbuf_accel.h CMDBUF_WR_REQ_VC), which selects the iDMA backend; overlay's is 1.
constexpr uint32_t kReqVc = 2;
constexpr uint32_t kRespVc = overlay::CMDBUF_WR_RESP_VC;

FORCE_INLINE volatile KeraunosNocChannels& channels() {
    return *reinterpret_cast<volatile KeraunosNocChannels*>(
        static_cast<uintptr_t>(MEM_IERISC_NOC_CHANNELS_BASE + MEM_L1_UNCACHED_BASE));
}

// This core's KeraunosSpaRoutes, where experimental/keraunos_cce.h reads them (that header is for idle Ethernet
// kernels only).
FORCE_INLINE const volatile KeraunosSpaRoutes& spa_routes() {
    return *reinterpret_cast<const volatile KeraunosSpaRoutes*>(
        static_cast<uintptr_t>(MEM_IERISC_SPA_ROUTES_BASE + MEM_L1_UNCACHED_BASE));
}

// The CCE this image is built for, within its tile.
constexpr uint32_t my_eth_cce = ETH_CCE_LOCAL_BASE / MEM_IERISC_L1_SIZE;

// Stops this hart: records @p fault and @p detail in the channels, where the host reads them, then asserts and spins.
// Not a trap: the CCE firmware installs no trap vector.
[[noreturn, gnu::cold, gnu::noinline]] inline void halt(Fault fault, uint64_t detail) {
    volatile KeraunosNocChannels& r = channels();
    r.fault_detail = detail;
    r.fault = static_cast<uint32_t>(fault);
    ASSERT(false);
    while (true) {
    }
}

// A NOC call a CCE does not serve: multicast, stateful command buffers, NOC atomics other than an increment. @p line identifies the caller.
[[noreturn, gnu::cold, gnu::noinline]] inline void unsupported(uint32_t line) { halt(Fault::Unsupported, line); }

// The core a NOC address names, and the address's local part.
struct Target {
    uint32_t hsio;
    uint32_t eth_cce;
    uint64_t local;  // bits 35:0 of the NOC address, in the target's own view
};

// The core the unicast NOC address @p noc_addr names: by its own coordinate, or by an Ethernet channel's through the
// channels. Halts with Fault::Unmapped when bits 63:48 are set, the coordinate names no core or the channels are not
// valid. Read-only; never blocks.
FORCE_INLINE Target lookup(uint64_t noc_addr) {
    const uint64_t xy = noc_addr >> NOC_ADDR_LOCAL_BITS;
    if ((xy >> (2 * NOC_ADDR_NODE_ID_BITS)) == 0) {
        const int32_t core = keraunos_noc_core(channels(), xy & NOC_NODE_ID_MASK, xy >> NOC_ADDR_NODE_ID_BITS);
        if (core >= 0) {
            return {
                keraunos_noc_channel_hsio(core), keraunos_noc_channel_eth_cce(core), noc_addr & NOC_LOCAL_ADDR_MASK};
        }
    }
    halt(Fault::Unmapped, noc_addr);
}

// The SRAM offset of @p flat, a flat address of CCE @p eth_cce of a tile in that CCE's own view, either one: cached
// from MEM_IERISC_ETH_CCE_BASE(eth_cce), uncached MEM_L1_UNCACHED_OFFSET above that. Halts with Fault::OutsideSram
// outside both.
FORCE_INLINE uint64_t sram_offset(uint64_t flat, uint32_t eth_cce) {
    const uint64_t cached = flat >= MEM_IERISC_UNCACHED_BASE ? flat - MEM_L1_UNCACHED_OFFSET : flat;
    // Below the CCE's base the difference wraps, so one comparison refuses both sides.
    const uint64_t offset = cached - MEM_IERISC_ETH_CCE_BASE(eth_cce);
    if (offset >= MEM_IERISC_L1_SIZE) {
        halt(Fault::OutsideSram, flat);
    }
    return offset;
}

// The iDMA address of @p target: the system address of its SRAM (KeraunosSpaRoutes), for this CCE's own SRAM too.
FORCE_INLINE uint64_t dma_addr(const Target& target) {
    const volatile KeraunosSpaRoutes& r = spa_routes();
    return r.hsio_sram_base + target.hsio * r.hsio_tile_stride + target.eth_cce * r.eth_cce_sram_size +
           sram_offset(target.local, target.eth_cce);
}

// The iDMA address of @p noc_addr.
FORCE_INLINE uint64_t dma_addr(uint64_t noc_addr) { return dma_addr(lookup(noc_addr)); }

// The iDMA address of @p flat, a flat address in this CCE's SRAM, either view.
FORCE_INLINE uint64_t local_dma_addr(uint32_t flat) { return dma_addr(Target{spa_routes().hsio, my_eth_cce, flat}); }

// The address a CPU load or store uses for @p noc_addr: on this tile, either CCE, the uncached view of its local part
// (every CCE sees both of its tile's SRAMs uncached, CCE by CCE from MEM_IERISC_UNCACHED_BASE); on another tile the
// system address. Either way the access reaches SRAM, where the target's harts and the iDMA see it, as a NOC write
// would; a store through a cached view can stay in this hart's data cache.
FORCE_INLINE uintptr_t cpu_addr(uint64_t noc_addr) {
    const Target target = lookup(noc_addr);
    return target.hsio == spa_routes().hsio ? MEM_IERISC_UNCACHED_BASE + MEM_IERISC_ETH_CCE_BASE(target.eth_cce) +
                                                  sram_offset(target.local, target.eth_cce)
                                            : dma_addr(target);
}

// Issues one iDMA copy of @p len bytes between two iDMA addresses, programming the whole command buffer as chippy's
// perform_dma_transfer() does, and returns once the buffer has taken it. Completion is copies_done(). The source must
// already be in SRAM: the copy reads SRAM, not this hart's cache.
FORCE_INLINE void copy(uint64_t src, uint64_t dst, uint32_t len) {
    overlay::reset_cmdbuf<kCmdBuf>();
    overlay::idma_setup_as_copy_cmdbuf<kCmdBuf>();
    __builtin_riscv_ttrocc_cmdbuf_wr_reg(kCmdBuf, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_REQ_VC_REG_OFFSET / 8, kReqVc);
    __builtin_riscv_ttrocc_cmdbuf_wr_reg(kCmdBuf, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_RESP_VC_REG_OFFSET / 8, kRespVc);
    overlay::setup_trids_cmdbuf<kCmdBuf>(kTrid);
    overlay::set_src_cmdbuf<kCmdBuf>(src, 0, 0, 0);
    overlay::set_dest_cmdbuf<kCmdBuf>(dst, 0, 0, 0);
    overlay::set_len_cmdbuf<kCmdBuf>(len);
    overlay::issue_cmdbuf<kCmdBuf>();
    while (overlay::idma_free_space_cmdbuf<kCmdBuf>() == 0) {
    }
}

// True once every copy() this hart issued is acked; then fences, so later loads see what the copies wrote.
FORCE_INLINE bool copies_done() {
    if (!overlay::idma_acked_cmdbuf<kCmdBuf>(kTrid)) {
        return false;
    }
    asm volatile("fence iorw, iorw" ::: "memory");
    return true;
}

}  // namespace internal_::cce_noc
