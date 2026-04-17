// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdint.h>
#include <stdbool.h>

#include "noc_parameters.h"

/*
  DRISC NIU Mode Configuration (Blackhole)

  Each DRISC has two NIUs (one per NOC instance). Bit 15 of NIU_CFG_0
  (NIU_CFG_0_AXI_SUBORDINATE_ENABLE) selects the NIU mode:

    NOC2AXI (bit=1): hardware default at cold boot; also forced at every
      DRISC firmware boot.
        - Tensix can directly read/write DRAM through this endpoint.
        - NOC traffic bypasses DRISC L1 and goes straight to DRAM.
        - DRISC cannot initiate NOC transactions.

    Stream (bit=0):
        - DRISC can initiate NOC transactions (e.g., drive the DMA
          engine for L1<->DRAM transfers).
        - NOC traffic terminates at DRISC L1, so other cores can
          read/write that L1.
        - Tensix cannot access DRAM directly through this endpoint;
          DRAM traffic must go through the DRISC L1/DMA path.

  Register persistence (IMPORTANT):
    NIU_CFG_0 persists across program runs; only a chip reset
    (tt-smi -r) restores the NOC2AXI default. For deterministic
    startup, DRISC firmware boot unconditionally forces NOC2AXI mode
    on both NIUs.

  APIs (all inline, defined in this header):
    1. Local API: drisc_set_* / drisc_is_*
       Guarded by COMPILE_FOR_DRISC. Usable from DRISC firmware and
       DRISC kernels.
    2. Remote API: drisc_remote_set_* / drisc_remote_is_*
       Guarded by KERNEL_BUILD && !COMPILE_FOR_TRISC. Usable from any
       data-movement kernel (NOC-capable); not from TRISC compute.
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

//////////////////////////////////////////////////////////////////
/////////////////// Remote API (kernels only) ////////////////////
//////////////////////////////////////////////////////////////////
#if defined(KERNEL_BUILD) && !defined(COMPILE_FOR_TRISC)
/*
  Remote API: any data-movement kernel configures a remote DRISC NIU
  via NOC read-modify-write. Useful when a DM kernel (e.g., on Tensix)
  needs DRISC L1 access (stream mode) rather than direct DRAM (NOC2AXI mode).

  Parameters:
    drisc_noc_x, drisc_noc_y: NOC coords of the target DRISC.
    noc:                      NOC instance; selects both transport
                              and which NIU on the target (one per NOC).
                              Defaults to noc_index.
    scratch_l1_addr:          Caller-owned L1 address on the caller
                              core; destination for the NIU_CFG_0
                              readback. Safe to reuse a buffer the
                              caller will overwrite later, since each
                              call completes before returning.
*/

#include "api/dataflow/dataflow_api.h"

// Caller-side address of NIU_CFG_0 for the given NIU instance (0 or 1).
inline __attribute__((always_inline)) uint32_t drisc_remote_niu_cfg_local_addr(uint8_t noc) {
    return NOC_CFG(NIU_CFG_0) + (noc * NOC_INSTANCE_OFFSET);
}

// Read NIU_CFG_0 from a remote DRISC; value lands in *scratch_l1_addr
// and is returned.
inline __attribute__((always_inline)) uint32_t drisc_remote_read_niu_cfg(
    uint32_t drisc_noc_x, uint32_t drisc_noc_y, uint32_t scratch_l1_addr, uint8_t noc = noc_index) {
    uint64_t niu_addr = get_noc_addr(drisc_noc_x, drisc_noc_y, drisc_remote_niu_cfg_local_addr(noc), noc);
    noc_async_read(niu_addr, scratch_l1_addr, sizeof(uint32_t), noc);
    noc_async_read_barrier(noc);
    return *reinterpret_cast<volatile uint32_t*>(scratch_l1_addr);
}

// Put the remote DRISC NIU into stream mode (DRISC L1 reachable via NOC).
inline __attribute__((always_inline)) void drisc_remote_set_stream_mode(
    uint32_t drisc_noc_x, uint32_t drisc_noc_y, uint32_t scratch_l1_addr, uint8_t noc = noc_index) {
    uint32_t cfg = drisc_remote_read_niu_cfg(drisc_noc_x, drisc_noc_y, scratch_l1_addr, noc);
    cfg &= ~(1u << NIU_CFG_0_AXI_SUBORDINATE_ENABLE);
    uint64_t niu_addr = get_noc_addr(drisc_noc_x, drisc_noc_y, drisc_remote_niu_cfg_local_addr(noc), noc);
    noc_inline_dw_write(niu_addr, cfg, 0xF, noc);
    noc_async_write_barrier(noc);
}

// Put the remote DRISC NIU into NOC2AXI mode (direct DRAM via this
// endpoint; DRISC L1 unreachable over NOC).
inline __attribute__((always_inline)) void drisc_remote_set_noc2axi_mode(
    uint32_t drisc_noc_x, uint32_t drisc_noc_y, uint32_t scratch_l1_addr, uint8_t noc = noc_index) {
    uint32_t cfg = drisc_remote_read_niu_cfg(drisc_noc_x, drisc_noc_y, scratch_l1_addr, noc);
    cfg |= (1u << NIU_CFG_0_AXI_SUBORDINATE_ENABLE);
    uint64_t niu_addr = get_noc_addr(drisc_noc_x, drisc_noc_y, drisc_remote_niu_cfg_local_addr(noc), noc);
    noc_inline_dw_write(niu_addr, cfg, 0xF, noc);
    noc_async_write_barrier(noc);
}

// Returns true if the remote DRISC NIU is currently in NOC2AXI mode.
inline __attribute__((always_inline)) bool drisc_remote_is_noc2axi_mode(
    uint32_t drisc_noc_x, uint32_t drisc_noc_y, uint32_t scratch_l1_addr, uint8_t noc = noc_index) {
    uint32_t cfg = drisc_remote_read_niu_cfg(drisc_noc_x, drisc_noc_y, scratch_l1_addr, noc);
    return (cfg >> NIU_CFG_0_AXI_SUBORDINATE_ENABLE) & 0x1;
}

#endif  // KERNEL_BUILD && !COMPILE_FOR_TRISC
