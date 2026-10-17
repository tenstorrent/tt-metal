// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// kernels/packet_injection/metal_producer_transport.hpp
//
// The L1-visibility, NoC and cycle-counter operations every Metal-launched
// packet-injection producer shares. A target transport derives from it and
// adds its identity, L1 bound, header hook and wall counter.

#ifndef X280_KERNELS_PACKET_INJECTION_METAL_PRODUCER_TRANSPORT_HPP
#define X280_KERNELS_PACKET_INJECTION_METAL_PRODUCER_TRANSPORT_HPP

#include <stdint.h>

#include "packet_injection_abi.h"
#include "../../include/rt/measurement.h"

#if defined(__riscv)
#include "api/dataflow/dataflow_api.h"
#include "../shared/metal_bounded_wait.hpp"
#endif

namespace packet_injection {

/**
 * Producer transport operations over the tt-metal dataflow API, all static.
 * A host build has no device to reach: every write is a no-op, every barrier
 * succeeds and read_cpu() reports failure.
 */
struct MetalProducerTransport {
    /** Drops cached L1 lines so a peer's NoC write to this core is observed. */
    static inline void invalidate_cache() {
#if defined(__riscv)
        invalidate_l1_cache();
#endif
    }

    /**
     * Waits until every write this core issued on `noc_index` has completed, or
     * `budget` ticks of the core's wall clock (the clock read_timer() reads)
     * pass. Returns false on expiry, with the writes still outstanding.
     */
    static inline bool barrier(uint32_t noc_index, uint64_t budget) {
#if defined(__riscv)
        return metal_wait::write_barrier(noc_index, budget);
#else
        (void)noc_index;
        (void)budget;
        return true;
#endif
    }

    /**
     * Issues one PI_SLOT_BYTES write from local L1 `src_l1_addr` to
     * `dst_lim_addr` on the tile at TRANSLATED (`peer_x`, `peer_y`). Returns
     * once issued; barrier() is what waits for completion.
     */
    static inline void issue_header_write(
        uint32_t src_l1_addr, uint32_t peer_x, uint32_t peer_y, uint32_t dst_lim_addr, uint32_t noc_index) {
#if defined(__riscv)
        uint64_t dst_noc = get_noc_addr(peer_x, peer_y, dst_lim_addr, noc_index);
        noc_async_write(src_l1_addr, dst_noc, PI_SLOT_BYTES, noc_index);
#else
        (void)src_l1_addr;
        (void)peer_x;
        (void)peer_y;
        (void)dst_lim_addr;
        (void)noc_index;
#endif
    }

    /** As issue_header_write(), for one 4-byte word. */
    static inline void issue_u32_write(
        uint32_t src_l1_addr, uint32_t peer_x, uint32_t peer_y, uint32_t dst_lim_addr, uint32_t noc_index) {
#if defined(__riscv)
        uint64_t dst_noc = get_noc_addr(peer_x, peer_y, dst_lim_addr, noc_index);
        noc_async_write(src_l1_addr, dst_noc, sizeof(uint32_t), noc_index);
#else
        (void)src_l1_addr;
        (void)peer_x;
        (void)peer_y;
        (void)dst_lim_addr;
        (void)noc_index;
#endif
    }

    /** Native per-RISC cycles into `*ticks`; returns zero when the read fails. */
    static inline int read_cpu(uint64_t* ticks) {
#if defined(__riscv) && __riscv_xlen == 32
        return measurement_cpu_cycles(ticks);
#else
        (void)ticks;
        return 0;
#endif
    }
};

}  // namespace packet_injection

#endif  // X280_KERNELS_PACKET_INJECTION_METAL_PRODUCER_TRANSPORT_HPP
