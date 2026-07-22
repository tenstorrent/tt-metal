// SPDX-License-Identifier: Apache-2.0
//
// Blackhole ETH-SS raw-L2 TX contract for on-core (RISC1) TT-RDMA kernels.
//
// A tt-metal active-eth JIT kernel cannot link the bh-erisc base FW, so the raw
// send sequence is inlined here — the same self-contained pattern the shipped
// gateway kernel uses (tt_metal/hw/firmware/src/erisc_cmac_gw.cpp).
//
// Register offsets are ported from bh-erisc-fpga:
//   - src/common/api/eth_ss.cpp  (eth_send_raw: the golden send sequence)
//   - src/common/registers/eth_core_a_reg.h  (offsets + INSERT_CTL bitfields)
// and cross-checked against erisc_cmac_gw.cpp (WH and BH agree on the TXQ block:
// base 0xFFB90000, stride 0x1000, CMD 0x04 / START_RAW bit0, START_ADDR 0x14,
// SIZE 0x18). See docs/tt-rdma-v1/tt-rdma-blackhole-port.md §1 (eth_send_raw is
// the spec-named RISC1 external/gateway TX path) + tt-rdma-wire-protocol-v1.md.
//
// COEXISTENCE: base FW owns TXQ0 (chip-info / telemetry). A TT-RDMA kernel MUST
// use a different queue (default q2) + its own TXPKT_CFG row so it never touches
// base FW's link-maintenance state. SA on all 10 TXPKT rows is set once by base
// FW (eth_init.cpp eth_ctrl_init); we only set DA + ethertype + INSERT_CTL.
#pragma once

#include <stdint.h>

// ---- TXQ block: per-queue register file (base + q*stride) ----
#define TT_ETH_TXQ0_BASE 0xFFB90000u
#define TT_ETH_TXQ_STRIDE 0x1000u
#define TT_ETH_TXQ_CTRL 0x00u
#define TT_ETH_TXQ_CMD 0x04u                  // bit0 = transfer_start_raw
#define TT_ETH_TXQ_STATUS 0x08u               // bit16 = cmd_ongoing (busy)
#define TT_ETH_TXQ_TRANSFER_START_ADDR 0x14u  // L1 byte address of the frame body
#define TT_ETH_TXQ_TRANSFER_SIZE_BYTES 0x18u  // frame body byte count (hdr + payload)
#define TT_ETH_TXQ_TXPKT_CFG_SEL_SW 0x80u     // which TXPKT_CFG row this queue emits with
#define TT_ETH_TXQ_CMD_START_RAW 0x1u
#define TT_ETH_TXQ_STATUS_CMD_ONGOING 0x00010000u

// ---- ETH_CTRL TXPKT_CFG rows: 10 rows, stride 0x80, row N = base + N*0x80 ----
// (row2 INSERT_CTL is 0xFFB98300 in eth_core_a_reg.h -> row0 = 0xFFB98200.)
#define TT_ETH_TXPKT_CFG0_BASE 0xFFB98200u
#define TT_ETH_TXPKT_CFG_STRIDE 0x80u
#define TT_ETH_TXPKT_INSERT_CTL 0x00u  // vlan/l3/l4/roce insert flags; 0 = raw L2 only
#define TT_ETH_TXPKT_MAC_SA_LO 0x10u
#define TT_ETH_TXPKT_MAC_SA_HI 0x14u
#define TT_ETH_TXPKT_MAC_DA_LO 0x18u
#define TT_ETH_TXPKT_MAC_DA_HI 0x1Cu
#define TT_ETH_TXPKT_ETHERTYPE 0x20u  // low 16 bits = ethertype on the wire

// The queue TT-RDMA uses (base FW owns q0). Overridable at compile time.
#ifndef TT_RDMA_TX_QUEUE
#define TT_RDMA_TX_QUEUE 2u
#endif

#define TT_ETH_REG32(a) (*(volatile tt_l1_ptr uint32_t*)(uintptr_t)(a))

// Configure a TXPKT_CFG row for raw TT-RDMA egress: dest MAC + ethertype, no
// upper-layer header insertion, and point the queue at this row. Call ONCE
// before the first send (dest MAC / ethertype are static for the link).
// dst_mac is packed like eth_send_raw: hi = mac>>32, lo = mac & 0xffffffff
// (the MAC hardware does the on-wire byte-swap).
static inline void tt_rdma_txpkt_config(uint32_t q, uint64_t dst_mac, uint16_t ethertype) {
    const uint32_t row = TT_ETH_TXPKT_CFG0_BASE + q * TT_ETH_TXPKT_CFG_STRIDE;
    TT_ETH_REG32(row + TT_ETH_TXPKT_MAC_DA_HI) = (uint32_t)(dst_mac >> 32);
    TT_ETH_REG32(row + TT_ETH_TXPKT_MAC_DA_LO) = (uint32_t)(dst_mac & 0xFFFFFFFFu);
    TT_ETH_REG32(row + TT_ETH_TXPKT_ETHERTYPE) = (uint32_t)ethertype;
    TT_ETH_REG32(row + TT_ETH_TXPKT_INSERT_CTL) = 0u;  // raw L2: [DA][SA][ethertype][payload]
    TT_ETH_REG32(TT_ETH_TXQ0_BASE + q * TT_ETH_TXQ_STRIDE + TT_ETH_TXQ_TXPKT_CFG_SEL_SW) = q;
}

// Fire one raw L2 frame: body = [32-B TT-RDMA header][payload] already in L1 at
// l1_src_byte_addr, nbytes total. Blocks until the queue is idle first. The
// MAC/ethertype are prepended by the TXPKT row configured above.
static inline void tt_rdma_send_raw(uint32_t q, uint32_t l1_src_byte_addr, uint32_t nbytes) {
    const uint32_t qb = TT_ETH_TXQ0_BASE + q * TT_ETH_TXQ_STRIDE;
    while (TT_ETH_REG32(qb + TT_ETH_TXQ_STATUS) & TT_ETH_TXQ_STATUS_CMD_ONGOING) {
        // wait for any in-flight transfer on THIS queue to drain (never q0's)
    }
    TT_ETH_REG32(qb + TT_ETH_TXQ_TRANSFER_START_ADDR) = l1_src_byte_addr;
    TT_ETH_REG32(qb + TT_ETH_TXQ_TRANSFER_SIZE_BYTES) = nbytes;
    TT_ETH_REG32(qb + TT_ETH_TXQ_CMD) = TT_ETH_TXQ_CMD_START_RAW;
}
