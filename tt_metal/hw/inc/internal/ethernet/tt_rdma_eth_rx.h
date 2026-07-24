// SPDX-License-Identifier: Apache-2.0
//
// Blackhole ETH-SS raw-L2 RX contract for on-core (RISC1) TT-RDMA kernels (M-1b).
//
// Register offsets ported from bh-erisc-fpga:
//   - src/common/api/eth_init.cpp  (eth_rxq_init: the golden raw-RX setup)
//   - src/common/registers/eth_core_a_reg.h  (RXQ offsets + ETH_RXQ_CTRL bitfields)
// NOTE: the BH RXQ block base is 0xFFB94000 (NOT the WH gateway's 0xFFB92000 in
// erisc_cmac_gw.cpp — WH/BH differ here; use the BH value).
//
// Raw mode: incoming frames (after MAC/L2 strip) are written contiguously into the
// L1 buffer at BUF_START_WORD_ADDR; BUF_PTR advances by the words received. The MAC
// RX router (eth_init.cpp eth_ctrl_init) sends bcast->RXQ0, mcast->RXQ1, unicast
// ("other")->RXQ2, so a UNICAST 0x1AF6 frame from the peer lands on RXQ2 — the queue
// we use for RDMA (symmetric with the TX queue), which base FW doesn't consume on a
// NIC/EXTERNAL core.
#pragma once

#include <stdint.h>

#ifndef TT_ETH_REG32
#define TT_ETH_REG32(a) (*(volatile tt_l1_ptr uint32_t*)(uintptr_t)(a))
#endif

// ---- RXQ block: per-queue register file (base + q*stride) ----
#define TT_ETH_RXQ0_BASE 0xFFB94000u
#define TT_ETH_RXQ_STRIDE 0x1000u
#define TT_ETH_RXQ_CTRL 0x00u                 // bit1 = packet_mode, bit2 = buf_wrap
#define TT_ETH_RXQ_BUF_PTR 0x08u              // words written by HW so far (advances on RX)
#define TT_ETH_RXQ_BUF_START_WORD_ADDR 0x0Cu  // L1 WORD address (byte>>4) where frames land
#define TT_ETH_RXQ_BUF_SIZE_WORDS 0x10u
#define TT_ETH_RXQ_PACKET_DROP_CNT 0x4Cu
#define TT_ETH_RXQ_CTRL_RAW_NOWRAP 0x0u  // packet_mode=0, buf_wrap=0 -> raw, fill from start

// The queue TT-RDMA receives on. Base FW routes unicast -> RXQ2; keep it symmetric with TX.
#ifndef TT_RDMA_RX_QUEUE
#define TT_RDMA_RX_QUEUE 2u
#endif

// Put a queue into raw-L2 RX mode, landing frames at l1_buf_byte_addr (16-B aligned).
// Reconfigures from whatever base FW left (it sets all queues to packet mode at init).
static inline void tt_rdma_rxq_init_raw(uint32_t q, uint32_t l1_buf_byte_addr, uint32_t buf_size_bytes) {
    const uint32_t qb = TT_ETH_RXQ0_BASE + q * TT_ETH_RXQ_STRIDE;
    TT_ETH_REG32(qb + TT_ETH_RXQ_BUF_START_WORD_ADDR) = l1_buf_byte_addr >> 4;  // word address
    TT_ETH_REG32(qb + TT_ETH_RXQ_BUF_PTR) = 0u;
    TT_ETH_REG32(qb + TT_ETH_RXQ_BUF_SIZE_WORDS) = buf_size_bytes >> 4;
    TT_ETH_REG32(qb + TT_ETH_RXQ_CTRL) = TT_ETH_RXQ_CTRL_RAW_NOWRAP;  // raw mode
}

static inline uint32_t tt_rdma_rxq_bufptr(uint32_t q) {
    return TT_ETH_REG32(TT_ETH_RXQ0_BASE + q * TT_ETH_RXQ_STRIDE + TT_ETH_RXQ_BUF_PTR);
}
static inline uint32_t tt_rdma_rxq_dropcnt(uint32_t q) {
    return TT_ETH_REG32(TT_ETH_RXQ0_BASE + q * TT_ETH_RXQ_STRIDE + TT_ETH_RXQ_PACKET_DROP_CNT);
}
