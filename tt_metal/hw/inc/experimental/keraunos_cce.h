// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Device API for kernels on a Keraunos HSIO CCE (keraunos_plan.md Phase 5).
//
// A Keraunos HSIO tile has two CCEs. Each is its own idle-ethernet core with eight harts: CCE0 is descriptor
// core (4, hsio) and CCE1 is (5, hsio). A kernel is compiled once per CCE, so the same source gets different
// local and peer SRAM bases from the build defines ETH_CCE_LOCAL_BASE and ETH_CCE_PEER_BASE. This header hides:
//   - which CCE a kernel runs on, and where each CCE's SRAM and uncached alias sit in its own view;
//   - the package SPA of every reachable memory (other HSIO tiles, Mimir CCE SRAM, Mimir GDDR), which comes
//     from the host-written KeraunosSpaRoutes table, so a kernel holds no package constants;
//   - the overlay command-buffer sequences that move data (the Quasar IDMA path in cmdbuff_api.hpp).
//
// Naming follows the dataflow and fabric APIs: addresses come from *_addr(), bulk transfers are async_* and
// complete at a barrier, and a single 32-bit store is write32.
//
// Verified on the MMK emulation (KeraunosPaths*, KeraunosFromMimir*, KeraunosApi*):
//   own SRAM, peer CCE SRAM, other HSIO tiles' SRAM, Mimir CCE SRAM (M0 and M1, CCE0 and CCE1), Mimir GDDR
//   (M1 direct, M0 through M1), from CCE0 and from CCE1, by CPU load/store and by IDMA, reads and writes.
// Not verified: atomic_inc (no remote atomic path has been checked; see atomic_inc below).

#pragma once

#include <stdint.h>

#include "dev_mem_map.h"
#include "hostdev/keraunos_spa_routes.h"
#include "internal/hw_thread.h"
#include "internal/risc_attribs.h"
#include "internal/tt-2xx/quasar/overlay/cmdbuff_api.hpp"

#if defined(KERAUNOS_CCE)

namespace experimental::keraunos {

// --- Identity ------------------------------------------------------------------------------------

// The CCE this image was built for (0 or 1): ETH_CCE_LOCAL_BASE is 0 for CCE0 and one SRAM size for CCE1.
constexpr uint32_t my_eth_cce() { return ETH_CCE_LOCAL_BASE / MEM_IERISC_L1_SIZE; }
constexpr uint32_t peer_eth_cce() { return 1 - my_eth_cce(); }
// Hart within this CCE (0-7).
inline uint32_t my_hart() { return internal_::get_hw_thread_idx(); }

// The route table sits in this CCE's SRAM and is written by the host (through the uncached alias, since the host
// write is not snooped by the hart's cache).
inline volatile tt_l1_ptr KeraunosSpaRoutes* spa_routes() {
    return reinterpret_cast<volatile tt_l1_ptr KeraunosSpaRoutes*>(MEM_L1_UNCACHED_BASE + MEM_IERISC_SPA_ROUTES_BASE);
}
// This core's HSIO tile.
inline uint32_t my_hsio() { return spa_routes()->hsio; }

// --- Local addresses on this tile ----------------------------------------------------------------
// Every CCE sees the tile's two SRAMs at fixed native addresses: CCE0 at [0, 4 MiB) and CCE1 at [4 MiB, 8 MiB),
// each with an uncached alias 8 MiB above. Offsets are CCE-relative (what the host and the HAL use). Use the
// uncached form for anything another agent (host, peer CCE, a remote writer) reads or writes: the CCE L2 does not
// snoop inbound writes.

constexpr uint64_t eth_cce_sram_addr(uint32_t eth_cce, uint32_t offset) {
    return MEM_IERISC_ETH_CCE_BASE(eth_cce) + offset;
}
constexpr uint64_t eth_cce_sram_uncached_addr(uint32_t eth_cce, uint32_t offset) {
    return MEM_IERISC_UNCACHED_BASE + MEM_IERISC_ETH_CCE_BASE(eth_cce) + offset;
}
constexpr uint64_t own_sram_addr(uint32_t offset) { return eth_cce_sram_addr(my_eth_cce(), offset); }
constexpr uint64_t own_sram_uncached_addr(uint32_t offset) { return eth_cce_sram_uncached_addr(my_eth_cce(), offset); }
constexpr uint64_t peer_sram_addr(uint32_t offset) { return eth_cce_sram_addr(peer_eth_cce(), offset); }
constexpr uint64_t peer_sram_uncached_addr(uint32_t offset) { return eth_cce_sram_uncached_addr(peer_eth_cce(), offset); }

// --- Package addresses (SPA) -----------------------------------------------------------------------
// What an IDMA transfer, or a load or store that leaves the tile, uses. Offsets are CCE-relative.

inline uint64_t hsio_sram_addr(uint32_t hsio, uint32_t eth_cce, uint32_t offset) {
    volatile tt_l1_ptr KeraunosSpaRoutes* r = spa_routes();
    return r->hsio_sram_base + hsio * r->hsio_tile_stride + eth_cce * r->eth_cce_sram_size + offset;
}
// This tile's own SRAM as an SPA (an IDMA source or destination on this tile).
inline uint64_t local_spa_addr(uint32_t eth_cce, uint32_t offset) { return hsio_sram_addr(my_hsio(), eth_cce, offset); }
// A Mimir CCE's SRAM, numbered globally across the package (Mimir 0's CCEs, then Mimir 1's), over the D2D BAR1.
inline uint64_t mimir_cce_sram_addr(uint32_t global_mimir_cce, uint64_t offset) {
    volatile tt_l1_ptr KeraunosSpaRoutes* r = spa_routes();
    return r->mimir_cce_sram_base + global_mimir_cce * r->mimir_cce_sram_stride + offset;
}
// Mimir `mimir`'s GDDR at `offset` into its address space. Mimir 1 is on this package's D2D link; Mimir 0 is
// reached through Mimir 1.
inline uint64_t mimir_gddr_addr(uint32_t mimir, uint64_t offset) {
    if (mimir >= KERAUNOS_SPA_ROUTES_MAX_MIMIRS) {
        __builtin_trap();
    }
    return spa_routes()->mimir_gddr_base[mimir] + offset;
}

// --- Transfers ------------------------------------------------------------------------------------
// Bulk moves are IDMA copies on this hart's command buffer 0. Each is tagged with a transaction id so a sender
// can wait for one channel's data and leave others in flight (Blackhole's noc write with trid). A copy whose source
// is remote must use the read VC set; one whose source is local uses the write VC set (what the SysEng firmware
// and the overlay helpers use). Both addresses are SPAs, or an address in this hart's own view of this tile.

inline void async_write(uint64_t src, uint64_t dst, uint32_t len, uint32_t trid = overlay::CMDBUF_DEF_TRID) {
    using namespace overlay;
    while (idma_free_space_cmdbuf<CMDBUF_0>() == 0) {
    }
    idma_copy_cmdbuf<CMDBUF_0>(src, dst, len, trid);
}
inline void async_read(uint64_t src, uint64_t dst, uint32_t len, uint32_t trid = overlay::CMDBUF_DEF_TRID) {
    using namespace overlay;
    while (idma_free_space_cmdbuf<CMDBUF_0>() == 0) {
    }
    reset_cmdbuf<CMDBUF_0>();
    idma_setup_as_copy_cmdbuf<CMDBUF_0>(false);
    setup_ongoing_cmdbuf<CMDBUF_0>({.src_addr = true, .dest_addr = true});
    setup_vcs_cmdbuf<CMDBUF_0>(NocVcs::READ);
    setup_trids_cmdbuf<CMDBUF_0>(trid);
    set_src_cmdbuf<CMDBUF_0>(src);
    set_dest_cmdbuf<CMDBUF_0>(dst);
    set_len_cmdbuf<CMDBUF_0>(len);
    issue_cmdbuf<CMDBUF_0>();
}
// Waits until every copy issued with `trid` has been acknowledged.
inline void barrier(uint32_t trid = overlay::CMDBUF_DEF_TRID) {
    while (!overlay::idma_acked_cmdbuf<overlay::CMDBUF_0>(trid)) {
    }
}
// Waits until every copy this hart issued has been acknowledged.
inline void barrier_all() {
    while (!overlay::all_idma_acked_cmdbuf<overlay::CMDBUF_0>()) {
    }
}

// One 32-bit store or load to any address (own or peer SRAM, other tile, Mimir). A store is ordered before later
// stores. A load across D2D returns the remote data; stores to a SPA may retire before they land, so check
// remote writes by reading back through another path or with a barrier on an IDMA.
inline void write32(uint64_t addr, uint32_t value) {
    *reinterpret_cast<volatile uint32_t*>(addr) = value;
    asm volatile("fence w, w" ::: "memory");
}
inline uint32_t read32(uint64_t addr) { return *reinterpret_cast<volatile uint32_t*>(addr); }

// Remote atomic increment is not provided. The IDMA atomic accumulate (idma_l1_atomic_accum_cmdbuf) has not been
// shown to work across tiles or D2D on the emulation (KeraunosAtomics.* are DISABLED). Until it is, fabric code
// uses monotonic counters written with write32 (the Blackhole counter-mode credits).

// --- Peer CCE sync --------------------------------------------------------------------------------
// Flags live in CCE0's SRAM and are touched only through the uncached alias, so both CCEs agree without cache
// maintenance (Blackhole's wait_for_other_local_erisc). CCE0 reaches them in its own SRAM, CCE1 in its peer window.

inline void peer_signal(uint32_t flag_offset, uint32_t value) {
    write32(eth_cce_sram_uncached_addr(0, flag_offset), value);
}
inline void peer_wait(uint32_t flag_offset, uint32_t value) {
    while (read32(eth_cce_sram_uncached_addr(0, flag_offset)) != value) {
    }
}

}  // namespace experimental::keraunos

#endif
