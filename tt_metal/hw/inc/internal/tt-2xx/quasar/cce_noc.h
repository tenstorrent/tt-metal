// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// NOC calls on a Keraunos CCE (a Quasar Ethernet core), behind noc_nonblocking_api_v2.h. A CCE has no NOC endpoint:
// a NOC coordinate resolves through the host-written CceNocRoutes (hostdev/cce_spa_routes.h), a copy is one iDMA
// transfer on command buffer 0, and a word write or increment is a CPU store or load and store. Ethernet sends use
// command buffer 1 (eth_cce_send.h).

#pragma once

#if !(defined(COMPILE_FOR_ERISC) || defined(COMPILE_FOR_IDLE_ERISC))
#error "cce_noc.h is for a Keraunos CCE (Quasar Ethernet core) only"
#endif

#include <cstdint>

#include "api/debug/assert.h"
#include "dev_mem_map.h"
#include "hostdev/cce_spa_routes.h"
#include "internal/risc_attribs.h"
#include "internal/tt-2xx/quasar/overlay/cmdbuff_api.hpp"
#include "noc_parameters.h"

namespace internal_::cce_noc {

// What halt() records in CceNocRoutes::fault; the detail goes to fault_detail.
enum class Fault : uint32_t {
    Unmapped = 0xCCEEF101,        // detail: the NOC address. Its coordinate is not in the routes, or they are invalid
    OtherCceOfTile = 0xCCEEF102,  // detail: the NOC address
    OutsideSram = 0xCCEEF103,     // detail: the flat address, above both views of this CCE's SRAM
    Unsupported = 0xCCEEF104,     // detail: the line of noc_nonblocking_api_v2.h that called unsupported()
};

constexpr overlay::CmdBuf kCmdBuf = overlay::CMDBUF_0;
constexpr uint32_t kTrid = overlay::CMDBUF_DEF_TRID;
// chippy's write request VC (cmdbuf_accel.h CMDBUF_WR_REQ_VC), which selects the iDMA backend; overlay's is 1.
constexpr uint32_t kReqVc = 2;
constexpr uint32_t kRespVc = overlay::CMDBUF_WR_RESP_VC;

FORCE_INLINE volatile CceNocRoutes& routes() {
    return *reinterpret_cast<volatile CceNocRoutes*>(
        static_cast<uintptr_t>(MEM_IERISC_NOC_ROUTES_BASE + MEM_IERISC_UNCACHED_BASE));
}

// Stops this hart: records @p fault and @p detail in the routes, where the host reads them, then asserts and spins.
// Not a trap: the CCE firmware installs no trap vector.
[[noreturn, gnu::cold, gnu::noinline]] inline void halt(Fault fault, uint64_t detail) {
    volatile CceNocRoutes& r = routes();
    r.fault_detail = detail;
    r.fault = static_cast<uint32_t>(fault);
    ASSERT(false);
    while (true) {
    }
}

// A NOC call a CCE does not serve: multicast, posted writes, stateful command buffers, NOC atomics other than an
// increment. @p line identifies the caller.
[[noreturn, gnu::cold, gnu::noinline]] inline void unsupported(uint32_t line) { halt(Fault::Unsupported, line); }

struct Target {
    CceNocRouteKind kind;
    uint64_t sram_base;
    uint64_t local;  // bits 35:0 of the NOC address
};

// The route of the unicast NOC address @p noc_addr. Halts with Fault::Unmapped when bits 63:48 are set, the
// coordinate has no route or the routes are not valid, and with Fault::OtherCceOfTile on this tile's other CCE.
// Read-only; never blocks.
FORCE_INLINE Target lookup(uint64_t noc_addr) {
    const uint64_t xy = noc_addr >> NOC_ADDR_LOCAL_BITS;
    volatile CceNocRoutes& r = routes();
    const uint32_t num_routes = r.num_routes;
    if (r.valid == CCE_NOC_ROUTES_VALID && num_routes <= CCE_NOC_ROUTES_MAX &&
        (xy >> (2 * NOC_ADDR_NODE_ID_BITS)) == 0) {
        const uint8_t x = xy & NOC_NODE_ID_MASK;
        const uint8_t y = xy >> NOC_ADDR_NODE_ID_BITS;
        for (uint32_t i = 0; i < num_routes; i++) {
            if (r.routes[i].x == x && r.routes[i].y == y) {
                const CceNocRouteKind kind = r.routes[i].kind;
                if (kind == CceNocRouteKind::THIS_TILE_OTHER_CCE) {
                    halt(Fault::OtherCceOfTile, noc_addr);
                }
                return {kind, r.routes[i].sram_base, noc_addr & NOC_LOCAL_ADDR_MASK};
            }
        }
    }
    halt(Fault::Unmapped, noc_addr);
}

// The SRAM offset of @p flat, a flat address of this CCE in either view. Halts with Fault::OutsideSram above both.
FORCE_INLINE uint64_t sram_offset(uint64_t flat) {
    const uint64_t offset = flat >= MEM_IERISC_UNCACHED_BASE ? flat - MEM_IERISC_UNCACHED_BASE : flat;
    if (offset >= MEM_IERISC_L1_SIZE) {
        halt(Fault::OutsideSram, flat);
    }
    return offset;
}

// The iDMA address of @p noc_addr: its route's SRAM base plus the SRAM offset of its local part.
FORCE_INLINE uint64_t dma_addr(uint64_t noc_addr) {
    const Target target = lookup(noc_addr);
    return target.sram_base + sram_offset(target.local);
}

// The iDMA address of @p flat, a flat address in this CCE's SRAM, either view.
FORCE_INLINE uint64_t local_dma_addr(uint32_t flat) {
    volatile CceNocRoutes& r = routes();
    return dma_addr(NOC_XY_ADDR(r.self_x, r.self_y, flat));
}

// The address a CPU load or store uses for @p noc_addr: on this CCE its local part as given (either view), on another
// tile the system address.
FORCE_INLINE uintptr_t cpu_addr(uint64_t noc_addr) {
    const Target target = lookup(noc_addr);
    if (target.kind == CceNocRouteKind::THIS_CCE) {
        return target.local;
    }
    return target.sram_base + sram_offset(target.local);
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
