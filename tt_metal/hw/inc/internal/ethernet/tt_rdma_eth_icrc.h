// SPDX-License-Identifier: Apache-2.0
//
// Blackhole ETH-CTRL ROCE_ICRC hardware CRC engine — register map for on-core probes.
//
// Ported from bh-erisc-fpga src/common/registers/eth_core_a_reg.h (ROCE_ICRC block,
// base 0xFFB98000). This is a RoCEv2 invariant-CRC (iCRC) offload: it computes a
// CRC-32 (poly 0x04C11DB7 — the SAME polynomial as TT-RDMA header_cksum, by design,
// see tt_rdma_hdr_build.h) inline on the RX datapath, with a programmable init, a
// per-direction input/output bit-order (reflection) select, an optional RX check
// (compare received vs calculated), and a 64-byte header bytemask.
//
// GOAL of exposing it: offload the TT-RDMA header CRC off the RISC hot path. The open
// question this header lets a probe answer empirically is whether the engine ENGAGES on
// our raw 0x1AF6 frames (it is built for IPv4/UDP:4791/BTH RoCE framing) and, if so, what
// CTRL bit-order/init makes RX_CALCULATED == the software tt_rdma_crc32 over bytes [0..27].
//
// SAFETY: on EXTERNAL/raw rails base FW does not run RoCE, so these registers are idle.
// Default CTRL (0x30700000) has RX_CHECK_EN=0 — the engine only observes. Leave check
// disabled until a probe proves HW==SW, so a mis-tuned engine can never drop live frames.
#pragma once

#include <stdint.h>

#ifndef TT_ETH_REG32
#define TT_ETH_REG32(a) (*(volatile tt_l1_ptr uint32_t*)(uintptr_t)(a))
#endif

#define TT_ETH_CTRL_BASE 0xFFB98000u

#define TT_ICRC_CTRL (TT_ETH_CTRL_BASE + 0x100u)                    // RX_CHECK_EN[8] + bit-order selects
#define TT_ICRC_TX_INIT (TT_ETH_CTRL_BASE + 0x104u)                 // TX CRC seed (default 0xC704DD7B)
#define TT_ICRC_RX_INIT (TT_ETH_CTRL_BASE + 0x108u)                 // RX CRC seed (default 0xC704DD7B)
#define TT_ICRC_RX_RECEIVED (TT_ETH_CTRL_BASE + 0x10Cu)             // CRC value extracted from the received packet
#define TT_ICRC_RX_CALCULATED (TT_ETH_CTRL_BASE + 0x110u)           // CRC the engine computed over the received packet
#define TT_ICRC_HDR_MASK(i) (TT_ETH_CTRL_BASE + 0x160u + (i) * 4u)  // i in [0..15], 64 B of header bytemask

// CTRL bitfields (eth_core_a_reg.h:6222+). Bit-order selects are 4-bit mux codes whose
// exact reflection semantics are NOT in ../docs — a probe must sweep them empirically.
#define TT_ICRC_CTRL_RX_CHECK_EN (1u << 8)
#define TT_ICRC_CTRL_RX_IN_ORDER_SHIFT 16
#define TT_ICRC_CTRL_RX_OUT_ORDER_SHIFT 20
#define TT_ICRC_CTRL_TX_IN_ORDER_SHIFT 24
#define TT_ICRC_CTRL_TX_OUT_ORDER_SHIFT 28
#define TT_ICRC_CTRL_ORDER_MASK 0xFu
#define TT_ICRC_CTRL_DEFAULT 0x30700000u  // POR value (RX_CHECK_EN off)

static inline uint32_t tt_icrc_rd(uint32_t addr) { return TT_ETH_REG32(addr); }
static inline void tt_icrc_wr(uint32_t addr, uint32_t v) { TT_ETH_REG32(addr) = v; }
