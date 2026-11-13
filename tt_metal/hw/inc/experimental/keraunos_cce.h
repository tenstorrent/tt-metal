// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// DRAFT for review (keraunos_plan.md Phase 5): device API for kernels on a Keraunos HSIO CCE.
//
// A Keraunos idle-ethernet core is one HSIO tile with two CCEs (eight harts each). EthernetConfig
// processor RISCV_0 is CCE0 hart 0 and RISCV_1 is CCE1 hart 0, the way Blackhole's erisc0/erisc1 share
// one ethernet core. This header hides three things a kernel should not hard-code:
//   - which CCE it runs on, and where each CCE's SRAM and uncached alias sit in its local view;
//   - the package SPA of every reachable memory (other HSIO tiles, Mimir CCE SRAM, Mimir GDDR), which
//     come from the host-written KeraunosSpaRoutes table;
//   - the overlay command-buffer sequence that moves data (the Quasar IDMA path in cmdbuff_api.hpp).
//
// Naming follows the dataflow API: addresses are computed by *_addr(), transfers are async_* and
// complete at a barrier, and a single 32-bit store is write32 (Blackhole fabric's inline write).
//
// Status on MMK emulation (keraunos_worklog.md):
//   working   own SRAM, peer CCE SRAM, other HSIO tiles' SRAM; CPU and IDMA, reads and writes
//   parked    every Mimir address (K0 D2D2 -> M1 D2D1): stores retire but do not land, loads hang
//   untested  atomic_inc (no remote atomic path verified yet)

#pragma once

#include <stdint.h>

#include "dev_mem_map.h"
#include "hostdev/keraunos_spa_routes.h"
#include "internal/hw_thread.h"
#include "internal/risc_attribs.h"
#include "internal/tt-2xx/quasar/overlay/cmdbuff_api.hpp"

#if defined(ARCH_QUASAR) && defined(COMPILE_FOR_IDLE_ERISC)

namespace experimental::keraunos {

// --- Identity ------------------------------------------------------------------------------------

// The CCE this image runs on (0 or 1).
constexpr uint32_t my_eth_cce() { return ETH_PHYSICAL_CCE_ID; }
constexpr uint32_t peer_eth_cce() { return 1 - ETH_PHYSICAL_CCE_ID; }
// Hart within this CCE (0-7).
inline uint32_t my_hart() { return internal_::get_hw_thread_idx() / MEM_IERISC_ETH_CCES; }

inline volatile tt_l1_ptr KeraunosSpaRoutes* spa_routes() {
    return reinterpret_cast<volatile tt_l1_ptr KeraunosSpaRoutes*>(MEM_IERISC_UNCACHED_BASE + MEM_IERISC_SPA_ROUTES_BASE);
}
// This core's HSIO tile.
inline uint32_t my_hsio() { return spa_routes()->hsio; }

// --- Local addresses on this tile ----------------------------------------------------------------
// Both CCEs see the tile the same way, so these are the same numbers on either CCE, and the same as
// the host's core offset. Use the uncached form for anything another agent (host, peer CCE, a remote
// writer) reads or writes: the CCE L2 does not snoop inbound writes.

constexpr uint64_t eth_cce_sram_addr(uint32_t eth_cce, uint32_t offset) { return MEM_IERISC_ETH_CCE_BASE(eth_cce) + offset; }
constexpr uint64_t eth_cce_sram_uncached_addr(uint32_t eth_cce, uint32_t offset) {
    return MEM_IERISC_UNCACHED_BASE + MEM_IERISC_ETH_CCE_BASE(eth_cce) + offset;
}
constexpr uint64_t own_sram_addr(uint32_t offset) { return eth_cce_sram_addr(my_eth_cce(), offset); }
constexpr uint64_t peer_sram_addr(uint32_t offset) { return eth_cce_sram_addr(peer_eth_cce(), offset); }

// --- Package addresses (SPA) -----------------------------------------------------------------------
// What an IDMA transfer, or a store that leaves the tile, uses.

inline uint64_t hsio_sram_addr(uint32_t hsio, uint32_t eth_cce, uint32_t offset) {
    volatile tt_l1_ptr KeraunosSpaRoutes* r = spa_routes();
    return r->hsio_sram_base + hsio * r->hsio_tile_stride + eth_cce * r->eth_cce_sram_size + offset;
}
// This tile's own SRAM as an SPA (an IDMA source or destination on this tile).
inline uint64_t local_spa_addr(uint32_t eth_cce, uint32_t offset) { return hsio_sram_addr(my_hsio(), eth_cce, offset); }
// Parked on MMK: see the status block above.
inline uint64_t mimir_cce_sram_addr(uint32_t global_mimir_cce, uint64_t offset) {
    volatile tt_l1_ptr KeraunosSpaRoutes* r = spa_routes();
    return r->mimir_cce_sram_base + global_mimir_cce * r->mimir_cce_sram_stride + offset;
}
// Parked on MMK: see the status block above.
inline uint64_t mimir_gddr_addr(uint32_t mimir, uint64_t offset) {
    if (mimir >= KERAUNOS_SPA_ROUTES_MAX_MIMIRS) {
        __builtin_trap();
    }
    return spa_routes()->mimir_gddr_base[mimir] + offset;
}

// --- Transfers ------------------------------------------------------------------------------------
// Bulk moves are IDMA copies on this hart's command buffer, each tagged with a transaction id so a
// sender can wait for one channel's data and leave others in flight (Blackhole's noc write with trid).
// Both addresses are SPAs (or this hart's local view of its own tile).

inline void async_copy(uint64_t src, uint64_t dst, uint32_t len, uint32_t trid = overlay::CMDBUF_DEF_TRID) {
    using namespace overlay;
    while (idma_free_space_cmdbuf<CMDBUF_0>() == 0) {
    }
    idma_copy_cmdbuf<CMDBUF_0>(src, dst, len, trid);
}
// The fabric-facing names: a write pushes local data out, a read pulls remote data in.
inline void async_write(uint32_t local_offset, uint64_t dst, uint32_t len, uint32_t trid = overlay::CMDBUF_DEF_TRID) {
    async_copy(local_spa_addr(my_eth_cce(), local_offset), dst, len, trid);
}
inline void async_read(uint64_t src, uint32_t local_offset, uint32_t len, uint32_t trid = overlay::CMDBUF_DEF_TRID) {
    async_copy(src, local_spa_addr(my_eth_cce(), local_offset), len, trid);
}
// Waits until every copy issued with `trid` has been acknowledged.
inline void barrier(uint32_t trid) {
    while (!overlay::idma_acked_cmdbuf<overlay::CMDBUF_0>(trid)) {
    }
}
// Waits until every copy this hart issued has been acknowledged.
inline void barrier_all() {
    while (!overlay::all_idma_acked_cmdbuf<overlay::CMDBUF_0>()) {
    }
}

// One 32-bit store to any address (local, peer, other tile, SPA), ordered before later stores.
inline void write32(uint64_t addr, uint32_t value) {
    *reinterpret_cast<volatile uint32_t*>(addr) = value;
    asm volatile("fence w, w" ::: "memory");
}
inline uint32_t read32(uint64_t addr) { return *reinterpret_cast<volatile uint32_t*>(addr); }

// Remote increment for credits. NOT YET VERIFIED on Keraunos (keraunos_plan.md Phase 3); until it is,
// fabric code uses monotonic counters written with write32 (Blackhole's counter-mode credits).
void atomic_inc(uint64_t addr, uint32_t increment);

// --- Peer CCE sync --------------------------------------------------------------------------------
// Flags live in CCE0's SRAM and are touched only through the uncached alias, so both CCEs agree
// without cache maintenance (Blackhole's wait_for_other_local_erisc).

inline void peer_signal(uint32_t flag_offset, uint32_t value) { write32(eth_cce_sram_uncached_addr(0, flag_offset), value); }
inline void peer_wait(uint32_t flag_offset, uint32_t value) {
    while (read32(eth_cce_sram_uncached_addr(0, flag_offset)) != value) {
    }
}

}  // namespace experimental::keraunos

#endif
