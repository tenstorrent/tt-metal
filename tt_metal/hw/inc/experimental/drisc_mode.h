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

    NOC2AXI (bit=1): hardware default at cold boot; also forced at every
      DRISC firmware boot.
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

  NOC addressing note (on-chip, kernel-initiated):
    In NOC2AXI mode the bottom 8 GB of NIU address space maps to GDDR,
    so a plain local address routes to DRAM, not DRISC L1. To target
    DRISC L1, add the offset 0x2000000000 to place the address outside
    the 8 GB GDDR window, routing it to L1 instead.
    In stream mode all inbound NOC traffic terminates at L1; use local
    addresses without the tag.

  Register persistence (IMPORTANT):
    NIU_CFG_0 persists across program runs; only a chip reset
    (tt-smi -r) restores the NOC2AXI default. For deterministic
    startup, DRISC firmware boot unconditionally forces NOC2AXI mode
    on both NIUs. Kernels that flip a DRISC into stream mode should
    restore NOC2AXI before returning so subsequent programs see the
    expected default.

  API (all inline, defined in this header):
    Local API: drisc_set_* / drisc_is_*
      Guarded by COMPILE_FOR_DRISC. Usable from DRISC firmware and
      DRISC kernels to configure the DRISC's own NIU.
*/

//////////////////////////////////////////////////////////////////
/////////////////// Local API (DRISC only) ///////////////////////
//////////////////////////////////////////////////////////////////
#ifdef COMPILE_FOR_DRISC
/*
  Local API: a DRISC configures its own NIU.

  Parameters:
    noc: NIU instance (0 or 1). Defaults to noc_index.
*/

#include "noc_nonblocking_api.h"
#include "internal/dataflow/dataflow_api_common.h"

inline __attribute__((always_inline)) void drisc_set_stream_mode(uint8_t noc = noc_index) {
    uint32_t cfg = NOC_CFG_READ_REG(noc, NIU_CFG_0);
    NOC_CFG_WRITE_REG(noc, NIU_CFG_0, cfg & ~(1u << NIU_CFG_0_AXI_SUBORDINATE_ENABLE));
}

inline __attribute__((always_inline)) void drisc_set_noc2axi_mode(uint8_t noc = noc_index) {
    uint32_t cfg = NOC_CFG_READ_REG(noc, NIU_CFG_0);
    NOC_CFG_WRITE_REG(noc, NIU_CFG_0, cfg | (1u << NIU_CFG_0_AXI_SUBORDINATE_ENABLE));
}

inline __attribute__((always_inline)) bool drisc_is_noc2axi_mode(uint8_t noc = noc_index) {
    uint32_t cfg = NOC_CFG_READ_REG(noc, NIU_CFG_0);
    return (cfg >> NIU_CFG_0_AXI_SUBORDINATE_ENABLE) & 0x1;
}

// Apply to both NIU instances.
inline __attribute__((always_inline)) void drisc_set_stream_mode_all(void) {
    drisc_set_stream_mode(0);
    drisc_set_stream_mode(1);
}

inline __attribute__((always_inline)) void drisc_set_noc2axi_mode_all(void) {
    drisc_set_noc2axi_mode(0);
    drisc_set_noc2axi_mode(1);
}

#endif  // COMPILE_FOR_DRISC

}  // namespace experimental
