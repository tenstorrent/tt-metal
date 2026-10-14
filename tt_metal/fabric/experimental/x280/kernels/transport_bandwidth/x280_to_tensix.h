// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

/*
 * kernels/transport_bandwidth/x280_to_tensix.h -- X280 -> Tensix member
 * of the transport_bandwidth suite.
 *
 * X280 is the initiator. It programs one paired-range NOC TLB slot at the
 * Tensix worker's L1 destination, then issues bursts of individual u32
 * stores through the window. Tensix is a passive addressable target -- no
 * Tensix kernel is submitted; L1 controllers accept NOC writes whether or
 * not the worker's harts are running application code.
 *
 * Every issue moves `transfer_size` bytes as `transfer_size / 4` back-to-
 * back u32 stores. BUG-4 forbids memcpy through a NOC window; the burst is
 * a straight sequence of `sw` instructions, the shape this member
 * measures. Every
 * store against the uncached window becomes its own NoC request
 * (TLBWindows.md), so `transfer_size` is a burst of 4-byte requests, not
 * one packet.
 *
 * TLB mode is ordering mode 2, Posted Writes (`NOC_ORDER_POSTED`);
 * see BlackholeA0/L2CPUTile/TLBWindows.md. A fenced read is not a completion
 * barrier, so host L1 readback verifies delivery.
 *
 * Timing:
 *   The firmware publishes both CLINT `mtime` (50 MHz, PLL-independent)
 *   and `rdcycle` (at the running PLL) from the same bracket. `mtime`
 *   is the primary because it is PLL-independent and needs no
 *   PLL conversion on the host; `rdcycle` is the
 *   same-bracket secondary, published as a cross-check between two
 *   independent clocks. `rdcycle` does not advance while the hart stalls
 *   on memory or a BRESP (MEASUREMENT.md
 *   "Unconfirmed hardware assumptions", item 5); the two counters have
 *   counted in lockstep on every run so far.
 *   See rt/x280_measurement.h for the fenced helpers and the design's
 *   Measurement discipline for the rule the choice satisfies.
 *
 * See x280/CHARACTERIZATION_UBENCH_DESIGN.md § Transport bandwidth.
 */
#ifndef X280_TRANSPORT_BW_X280_TO_TENSIX_H
#define X280_TRANSPORT_BW_X280_TO_TENSIX_H

#include <stdint.h>

#include "../../include/x280_noc_coord.h"
#include "transport_bandwidth_common.h"
#include "ubench_section_load.h"

/* --------------------------------------------------------------------------
 * Backend arguments (LIM-resident)
 * --------------------------------------------------------------------------
 * peer_dst_addr is the Tensix worker's L1 address in the same address view
 * the TLB reaches. The host chooses a 2 MiB-aligned base such that this
 * address, and any offsets the timed loop touches within transfer_size
 * bytes of it, all live inside one 2 MiB TLB window.
 *
 * peer_tlb_index picks the paired-range slot to program. The image
 * verifies the index is within X280_TLB_PAIRED_BASE ..
 * X280_TLB_PAIRED_BASE + X280_TLB_PAIRED_COUNT before touching it.
 */
typedef struct {
    uint64_t peer_dst_addr;    /* Tensix L1 destination base */
    uint64_t own_staging_addr; /* LIM: per-word source pattern for issue() */
    uint64_t own_result_addr;  /* LIM: transport_bw_result_t destination */
    x280_noc_coord_t peer_noc; /* Tensix worker, virtual (UMD TRANSLATED) */
    uint32_t peer_tlb_index;   /* X280_TLB_PAIRED_BASE + n */
    uint32_t reserved;
} transport_bw_x280_to_tensix_backend_args_t;

typedef struct {
    uint32_t cookie_common;
    uint32_t pad0;
    transport_bw_protocol_args_t common;
    uint32_t cookie_backend;
    uint32_t pad1;
    transport_bw_x280_to_tensix_backend_args_t backend;
} transport_bw_x280_to_tensix_section_t;

typedef transport_bw_x280_to_tensix_section_t transport_bw_x280_to_tensix_args_block_t;

/* The largest listed transfer_size. The host sizes the LIM staging vector and
 * the Tensix L1 destination to it. */
#define TB_X2T_STAGING_MAX_BYTES 256u

/* --------------------------------------------------------------------------
 * Static assertions
 * -------------------------------------------------------------------------- */
UBENCH_LAYOUT_ASSERT(
    sizeof(transport_bw_x280_to_tensix_backend_args_t) == 40u, "backend args are 40 bytes: 3 x u64 + 4 x u32");
UBENCH_LAYOUT_ASSERT(
    offsetof(transport_bw_x280_to_tensix_backend_args_t, peer_dst_addr) == 0u, "peer_dst_addr is the first field");
UBENCH_LAYOUT_ASSERT(
    offsetof(transport_bw_x280_to_tensix_backend_args_t, peer_noc) == 24u,
    "u32 tail begins after the three u64 addresses");

UBENCH_LAYOUT_ASSERT(
    offsetof(transport_bw_x280_to_tensix_section_t, cookie_common) == 0u, "common cookie is at offset 0");
UBENCH_LAYOUT_ASSERT(
    offsetof(transport_bw_x280_to_tensix_section_t, common) == 8u,
    "common section follows cookie + 4B pad on 8B alignment");

UBENCH_LAYOUT_ASSERT(
    sizeof(transport_bw_x280_to_tensix_section_t) == 80u,
    "section is 80 bytes: cookie + pad + common (24) + cookie + pad + backend (40)");

#endif /* X280_TRANSPORT_BW_X280_TO_TENSIX_H */
