// SPDX-License-Identifier: Apache-2.0
//
// Blackhole active-eth RDMA SW-L1 region map — the chip-side half of the contract.
// Included by the RISC1 RDMA kernel AND the host SDK (TtRdmaEndpoint) so no address
// can drift between FW and host. All values are ETH-tile L1 byte offsets.
//
// HARD RULE: never write >= MEM_SYSENG_RESERVED_BASE (0x70000). Base FW code/data,
// boot_results (0x7CC00), the api-table (~0x7CF00) and mailboxes (0x7D000) live there;
// writing that region BRICKS THE LINK. (This is the region erisc_ports.sh *reads*.)
//
// Grounding: tt-metal blackhole dev_mem_map.h; active_erisc.cc enter_reset/resume_from_reset;
// docs/tt-rdma-v1/tt-rdma-blackhole-port.md §3.1 + tt-rdma-fw-arch-rx.md; and the BH.0 appendix
// of tt-rdma-bh-bf3-impl-plan.md.
#pragma once

#include "dev_mem_map.h"  // MEM_ETH_SIZE, MEM_SYSENG_RESERVED_BASE, MEM_AERISC_*, MEM_ERISC_*
#include "tt_rdma_wire.h"

// RISC0 reset save-area: active_erisc.cc enter_reset() copies RISC0 GPRs + 8 KB local mem here
// (MEM_ERISC_L1_TEMP_STORAGE = 0x00040000), resume_from_reset() restores from it. DO NOT reuse.
#define TT_RDMA_RISC0_RESET_SAVE_BASE 0x00040000u
#define TT_RDMA_RISC0_RESET_SAVE_SIZE 0x00002000u  // 8 KB (2048 words)

// ---- The one TUNABLE. PIN THIS IN BH.0 against the real build's low/high watermarks. ----
// Placed above the RISC0 reset save (>= 0x42000) and below the top reserved region (< 0x70000).
// The window [0x42000 .. 0x70000) is ~184 KB — far more than the WH map (0x4000..0x2B000).
// BH.0 must confirm: (a) the low FW/kernel-config region ends below TT_RDMA_L1_BASE, and
// (b) if normal tt-metal dispatch is kept, TT_RDMA_L1_END also clears the fabric-router /
//     barrier / app-sync reserved region just under 0x70000 (see the extra assert below).
#ifndef TT_RDMA_L1_BASE
#define TT_RDMA_L1_BASE 0x00042000u
#endif

// ---- Region layout (offsets from TT_RDMA_L1_BASE) — mirror of the WH map, re-based & grown. ----
#define TT_RDMA_RX_RING_OFF 0x00000u
#define TT_RDMA_RX_RING_SIZE 0x04000u  // 16 KB (>= WH's 14 KB); BUF_WRAP replacement, RX-classifier(TCAM 0x1AF6)-fed

#define TT_RDMA_MR_TABLE_OFF (TT_RDMA_RX_RING_OFF + TT_RDMA_RX_RING_SIZE)
#define TT_RDMA_MR_SLOTS 64u                            // grown from WH's 16 (free on BH; NCCL/UCX need it)
#define TT_RDMA_MR_TABLE_SIZE (TT_RDMA_MR_SLOTS * 32u)  // 2 KB

#define TT_RDMA_RCB_OFF (TT_RDMA_MR_TABLE_OFF + TT_RDMA_MR_TABLE_SIZE)
#define TT_RDMA_RCB_SIZE 0x00200u  // RCB header + doorbells + dbg counters

#define TT_RDMA_WQE_DESCR_OFF (TT_RDMA_RCB_OFF + TT_RDMA_RCB_SIZE)
#define TT_RDMA_WQE_DESCR_SIZE 0x00400u  // 64 x 16 B

#define TT_RDMA_WQE_PAYLOAD_OFF (TT_RDMA_WQE_DESCR_OFF + TT_RDMA_WQE_DESCR_SIZE)
#define TT_RDMA_WQE_SLOTS 32u
#define TT_RDMA_WQE_SLOT_BYTES 4096u
#define TT_RDMA_WQE_PAYLOAD_SIZE (TT_RDMA_WQE_SLOTS * TT_RDMA_WQE_SLOT_BYTES)  // 128 KB

#define TT_RDMA_TX_BUF0_OFF (TT_RDMA_WQE_PAYLOAD_OFF + TT_RDMA_WQE_PAYLOAD_SIZE)
#define TT_RDMA_TX_BUF_BYTES 4096u
#define TT_RDMA_TX_BUF1_OFF (TT_RDMA_TX_BUF0_OFF + TT_RDMA_TX_BUF_BYTES)

#define TT_RDMA_READ_CORR_OFF (TT_RDMA_TX_BUF1_OFF + TT_RDMA_TX_BUF_BYTES)
#define TT_RDMA_READ_CORR_SIZE 0x00800u  // 2 KB READ tag->{local_mr,offset,len} table

#define TT_RDMA_REGION_END (TT_RDMA_READ_CORR_OFF + TT_RDMA_READ_CORR_SIZE)
#define TT_RDMA_L1_END (TT_RDMA_L1_BASE + TT_RDMA_REGION_END)

// Absolute addresses for consumers:
#define TT_RDMA_RX_RING_ADDR (TT_RDMA_L1_BASE + TT_RDMA_RX_RING_OFF)
#define TT_RDMA_MR_TABLE_ADDR (TT_RDMA_L1_BASE + TT_RDMA_MR_TABLE_OFF)
#define TT_RDMA_RCB_ADDR (TT_RDMA_L1_BASE + TT_RDMA_RCB_OFF)
#define TT_RDMA_WQE_DESCR_ADDR (TT_RDMA_L1_BASE + TT_RDMA_WQE_DESCR_OFF)
#define TT_RDMA_WQE_PAYLOAD_ADDR (TT_RDMA_L1_BASE + TT_RDMA_WQE_PAYLOAD_OFF)
#define TT_RDMA_TX_BUF0_ADDR (TT_RDMA_L1_BASE + TT_RDMA_TX_BUF0_OFF)
#define TT_RDMA_TX_BUF1_ADDR (TT_RDMA_L1_BASE + TT_RDMA_TX_BUF1_OFF)
#define TT_RDMA_READ_CORR_ADDR (TT_RDMA_L1_BASE + TT_RDMA_READ_CORR_OFF)

// NOTE: the SEND landing ring (RxWqeRing) is NOT in L1 — it is DMA-pushed to a host hugepage
// at +128 KB (reverse of the TX DMA-pull). See tt-rdma-fw-arch-rx.md / host-sdk.md.

// ---- Compile-time guards (the link-bricking one is #1). ----
#ifdef __cplusplus
#define TT_RDMA_SASSERT static_assert
#else
#define TT_RDMA_SASSERT _Static_assert
#endif
// 1. Must not run into the top base-FW/mailbox reserved region -> would brick the link.
TT_RDMA_SASSERT(
    TT_RDMA_L1_END <= MEM_SYSENG_RESERVED_BASE,
    "TT-RDMA L1 region overruns MEM_SYSENG_RESERVED_BASE (0x70000) - bricks the link");
// 2. Must not overlap the RISC0 reset save at 0x40000..0x42000.
TT_RDMA_SASSERT(
    TT_RDMA_L1_BASE >= (TT_RDMA_RISC0_RESET_SAVE_BASE + TT_RDMA_RISC0_RESET_SAVE_SIZE) ||
        TT_RDMA_L1_END <= TT_RDMA_RISC0_RESET_SAVE_BASE,
    "TT-RDMA L1 region overlaps the RISC0 reset save (0x40000..0x42000)");
// 3. (Enable when keeping normal tt-metal dispatch) also clear the fabric-router/barrier region:
//    TT_RDMA_SASSERT(TT_RDMA_L1_END <= MEM_ERISC_FABRIC_ROUTER_RESERVED_BASE, "...");
