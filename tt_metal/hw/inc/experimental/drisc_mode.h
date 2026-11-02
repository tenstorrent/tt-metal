// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdint.h>
#include <stdbool.h>

#include "noc_parameters.h"

namespace experimental {

/*
  DRISC NIU Mode Configuration (Blackhole)

  Each DRISC has two NIUs (one per NOC instance). Bit 15 of NIU_CFG_0
  (NIU_CFG_0_AXI_SUBORDINATE_ENABLE) selects the NIU mode:

    NOC2AXI (bit=1): hardware default at cold boot.
        - Incoming NOC traffic is routed by address: accesses in the
          DRISC L1 range land in DRISC L1; accesses in the DRAM range
          are forwarded over AXI to DRAM (so Tensix reads/writes DRAM
          directly through this endpoint).
        - DRISC cannot initiate NOC transactions.

    Stream (bit=0):
        - DRISC can initiate NOC transactions.
        - NOC traffic terminates at DRISC L1.
        - Tensix cannot access DRAM directly through this endpoint;
          DRAM traffic must go through the DRISC L1 + DMA path.

  Who sets the mode:
    DRISC firmware, once per boot, and nobody else. Kernels never switch
    modes -- a kernel that did would have to restore the mode before
    returning, and the restore races the receivers still writing credits
    back into DRISC L1 (a receiver that has not drained loses its tail
    credits across the switch). Firmware picking the mode once removes
    both the restore and the race.

  Which NIU ends up in which mode:
    A DRAM view names one preferred endpoint subchannel per NOC; only
    the NIU on that subchannel forwards that NOC's DRAM accesses over
    AXI. Every other NIU on a DRAM core is unused by the DRAM path, so
    firmware puts it in stream mode. The host passes the set to keep in
    NOC2AXI as core_info.noc2axi_niu_mask (bit N = NIU N), computed from
    the soc descriptor's dram_views by
    metal_SocDescriptor::get_dram_endpoint_noc_mask.

    On Blackhole that leaves NIU 0 in stream mode on every DRISC Metal
    owns, because get_metal_dram_cores hands back no core whose NOC0 bit
    is set. NIU 1 is in stream mode too on the subchannels that are no
    view's NOC1 endpoint. A kernel initiates NOC traffic on the NOC it
    was built for (CreateKernel(DramConfig{.noc = ...})), so NOC0 works
    anywhere and NOC1 only where it is free.

  NOC addressing note (on-chip, kernel-initiated):
    In NOC2AXI mode the bottom 8 GB of NIU address space maps to GDDR,
    so a plain local address routes to DRAM, not DRISC L1. To target
    DRISC L1, add the offset 0x2000000000 to place the address outside
    the 8 GB GDDR window, routing it to L1 instead. That tagged address
    also reaches L1 in stream mode, which is why host and Tensix access
    to DRISC L1 (mailboxes, watcher, sockets) is mode-independent.

  Register persistence:
    NIU_CFG_0 persists across program runs; only a chip reset
    (tt-smi -r) restores the NOC2AXI default. Firmware boot therefore
    writes both NIUs unconditionally rather than assuming the cold-boot
    value.
*/

//////////////////////////////////////////////////////////////////
/////////////////// Local API (DRISC only) ///////////////////////
//////////////////////////////////////////////////////////////////
#ifdef COMPILE_FOR_DRISC
/*
  Local API: a DRISC inspects its own NIU (kernels and firmware), or sets
  it (firmware only).

  Parameters:
    noc: NIU instance (0 or 1). Defaults to noc_index.
*/

#include "noc_nonblocking_api.h"
#include "internal/dataflow/dataflow_api_common.h"

inline __attribute__((always_inline)) bool drisc_is_noc2axi_mode(uint8_t noc = noc_index) {
    uint32_t cfg = NOC_CFG_READ_REG(noc, NIU_CFG_0);
    return (cfg >> NIU_CFG_0_AXI_SUBORDINATE_ENABLE) & 0x1;
}

// FW_BUILD (not the usual KERNEL_BUILD || FW_BUILD "device build" pair) on purpose:
// the mode is boot state, not per-kernel state, so this must not compile into a
// kernel. See "Who sets the mode" above.
#ifdef FW_BUILD

// Put each NIU in its permanent mode: NOC2AXI for the NIUs named in
// noc2axi_niu_mask (bit N = NIU N), stream mode for the rest. Called once per
// firmware boot; no later caller changes either NIU. Inserting the mask bit
// keeps one straight-line path per NIU -- an if/else here more than doubles the
// code this costs in the DRISC firmware window.
inline __attribute__((always_inline)) void drisc_init_niu_modes(uint8_t noc2axi_niu_mask) {
    for (uint8_t noc = 0; noc < NUM_NOCS; noc++) {
        uint32_t cfg = NOC_CFG_READ_REG(noc, NIU_CFG_0) & ~(1u << NIU_CFG_0_AXI_SUBORDINATE_ENABLE);
        cfg |= static_cast<uint32_t>((noc2axi_niu_mask >> noc) & 0x1) << NIU_CFG_0_AXI_SUBORDINATE_ENABLE;
        NOC_CFG_WRITE_REG(noc, NIU_CFG_0, cfg);
    }
}

#endif  // FW_BUILD

#endif  // COMPILE_FOR_DRISC

}  // namespace experimental
