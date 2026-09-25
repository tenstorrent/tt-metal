// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TT_ETH_SS_REGS_H
#define TT_ETH_SS_REGS_H

///////////////
// ETH Params

#define NUM_ECC_SOURCES (5 + 4 * 3 + 2)
#ifdef ARCH_BLACKHOLE
#define NUM_ETH_QUEUES 3
#else
#define NUM_ETH_QUEUES 2
#endif

#define ETH_CTRL_REGS_START 0xFFB94000
// Write to start ERISC IRAM load.
// Write value: word address for the start of binary in L1.
// Read value:  bit 0 = status (1=ongoing, 0=complete), bits [17:1] = currend read address.
#define ETH_CORE_IRAM_LOAD ((0x30 + NUM_ECC_SOURCES * 4) + 0x1C)

//////////////////
// RISC debug regs
#define ETH_RISC_REGS_START 0xFFB10000

#define ETH_RISC_RESET 0x21B0
#define ETH_RISC_WALL_CLOCK_0 0x21F0
#define ETH_RISC_WALL_CLOCK_1 0x21F4
#define ETH_RISC_WALL_CLOCK_1_AT 0x21F8

#ifdef ARCH_BLACKHOLE
//////////////////////////////
// eth_ctrl PTP timer A: CFR counts refclk ticks from power-on, 64NS is the PTP time in ns; reading a counter's LO
// half latches its HI half.
#define ETH_PTP_TIMER_REGS_START 0xFFB98800
#define ETH_PTP_TIMER_CTRL 0x00  // [0] timer_en
#define ETH_PTP_TIMER_FUTURE_CFR_LO 0x04
#define ETH_PTP_TIMER_FUTURE_CFR_HI 0x08
#define ETH_PTP_TIMER_FUTURE_PTI 0x0C  // [23:0] per-tick increment, 8 integer ns bits . 16 fractional
#define ETH_PTP_TIMER_FUTURE_TIMESTAMP_LO 0x10
#define ETH_PTP_TIMER_FUTURE_TIMESTAMP_HI 0x14
#define ETH_PTP_TIMER_UPDATE_PTI 0x20        // [0]
#define ETH_PTP_TIMER_UPDATE_TIMESTAMP 0x24  // [0]
#define ETH_PTP_TIMER_UPDATE_STAT 0x40
#define ETH_PTP_TIMER_PTI_STAT 0x44  // [23:0] per-tick increment in use
#define ETH_PTP_TIMER_CFR_LO 0x50
#define ETH_PTP_TIMER_CFR_HI 0x54
#define ETH_PTP_TIMER_64NS_LO 0x60
#define ETH_PTP_TIMER_64NS_HI 0x64

//////////////////////////////
// eth_ctrl TX header table: entry i at ETH_TXPKT_CFG_REGS_START + i * ETH_TXPKT_CFG_REGS_SIZE, for 0 <= i < 10
#define ETH_TXPKT_CFG_REGS_START 0xFFB98200
#define ETH_TXPKT_CFG_REGS_SIZE 0x80
#define ETH_TXPKT_CFG_INSERT_CTL 0x00
#define ETH_TXPKT_CFG_CUSTOM_HDR 0x04
#define ETH_TXPKT_CFG_MAC_SA_LO 0x10
#define ETH_TXPKT_CFG_MAC_SA_HI 0x14
#define ETH_TXPKT_CFG_MAC_DA_LO 0x18
#define ETH_TXPKT_CFG_MAC_DA_HI 0x1C
#define ETH_TXPKT_CFG_ETHERTYPE 0x20
#define ETH_TXPKT_CFG_VLAN1 0x24
#define ETH_TXPKT_CFG_VLAN2 0x28

//////////////////////////////
// RX classifier: TCAM flow lookup and flow table (row 64 of the flow table is the NO_MATCH_* registers)
#define ETH_RX_CLASSIFIER_REGS_START 0xFFB9C000
#define ETH_RX_CLASSIFIER_TCAM_ROW_MAPPING 0xC00  // + 4 * row
#define ETH_RX_CLASSIFIER_NO_MATCH_ACTIONS 0xD04
#define ETH_RX_CLASSIFIER_TCAM_ROW_UPDATE 0xD40
#define ETH_RX_CLASSIFIER_TCAM_TUPLE_TYPE_WRITE 0xD80
#define ETH_RX_CLASSIFIER_TCAM_SA_WRITE 0xD90  // 4 words
#define ETH_RX_CLASSIFIER_TCAM_DA_WRITE 0xDA0  // 4 words
#define ETH_RX_CLASSIFIER_TCAM_NON_IP_ADDR_FLAGS_WRITE 0xDB0
#define ETH_RX_CLASSIFIER_TCAM_ETHERTYPE_WRITE 0xDC0
#define ETH_RX_CLASSIFIER_TCAM_PRIORITY_WRITE 0xDC4
#define ETH_RX_CLASSIFIER_TCAM_UPDATE 0xDF0
#define ETH_RX_CLASSIFIER_FTABLE_LABELS 0xE80
#define ETH_RX_CLASSIFIER_FTABLE_ACTIONS 0xE84
#define ETH_RX_CLASSIFIER_FTABLE_VLAN 0xE88
#define ETH_RX_CLASSIFIER_FTABLE_SW_METADATA 0xE8C
#define ETH_RX_CLASSIFIER_FTABLE_UPDATE 0xEA0

//////////////////////////////
// RX classifier timestamp handling: the RX timestamp FIFO
#define ETH_RX_TH_REGS_START 0xFFB9D800
#define ETH_RX_TH_TS_LOW 0x00
#define ETH_RX_TH_TS_HIGH 0x04
#define ETH_RX_TH_TS_LABEL 0x08
#define ETH_RX_TH_STATUS 0x10

//////////////////////////////
// MAC (Rianta RSm410)
#define ETH_MAC_REGS_START 0xFFBA0000
#define ETH_MAC_TX_CFG 0x2200
#define ETH_MAC_TX_DELAY 0x2218  // [15:0] added to every TX timestamp
#define ETH_MAC_TX_INT 0x2288    // write 1 to clear
#define ETH_MAC_TX_INT_RAW 0x2290
#define ETH_MAC_TS_FIFO_FULL_THRESH 0x2300
#define ETH_MAC_TS_FIFO_0 0x2E00  // TX timestamp FIFO, 4 words per entry; reading word 0 pops; empty reads 0xFFFFFFFF
#define ETH_MAC_TS_FIFO_1 0x2E04
#define ETH_MAC_TS_FIFO_2 0x2E08
#define ETH_MAC_TS_FIFO_3 0x2E0C
#endif

//////////////////////////////
// TX queue 0/1 controllers
#define ETH_TXQ0_REGS_START 0xFFB90000
#define ETH_TXQ_REGS_SIZE 0x1000
#define ETH_TXQ_REGS_SIZE_BIT 12

//////////////////////////////
// RX queue controllers
#define ETH_RXQ_REGS_SIZE 0x1000

// TXQ_CTRL[0]: set to enable packet resend mode (must be set on both sides)
// TXQ_CTRL[1]: reserved, should be 0
// TXQ_CTRL[2]: 0 = use Length field, 1 = use Type field (from ETH_TXQ_ETH_TYPE)
// TXQ_CTRL[3]: set to disable drop notification, timeout-only for resend
#define ETH_TXQ_CTRL 0x0
#define ETH_TXQ_CTRL_KEEPALIVE (0x1 << 0)
#define ETH_TXQ_CTRL_USE_TYPE (0x1 << 2)
#define ETH_TXQ_CTRL_DIS_DROP (0x1 << 3)

// TXQ_CMD should be written as one-hot.
// TXQ_CMD[0]: issue raw transfer (no resend)
// TXQ_CMD[1]: issue packet transfer (with resend)
// TXQ_CMD[2]: issue remote reg write (TXQ0 only)
// TXQ_CMD[3]: MAC queue flush
// TXQ_CMD read returns 1 if command ongoing, 0 if ready for next command.
#define ETH_TXQ_CMD 0x4
#define ETH_TXQ_CMD_START_RAW (0x1 << 0)
#define ETH_TXQ_CMD_START_DATA (0x1 << 1)
#define ETH_TXQ_CMD_START_REG (0x1 << 2)
#define ETH_TXQ_CMD_FLUSH (0x1 << 3)

#define ETH_TXQ_STATUS 0x8  // IMPROVE: document (misc. internal bits for debug)
#define ETH_TXQ_STATUS_CMD_ONGOING_BIT \
    0x10  // On Blackhole bit 16 of the ETH_TXQ_STATUS register indicates whether a packer transfer (raw/data/reg write)
          // is ongoing
#define ETH_TXQ_MAX_PKT_SIZE_BYTES 0xC  // Max ethernet payload size (default = 1500 bytes)
#define ETH_TXQ_BURST_LEN 0x10          // Value to drive on ati_q#_pbl output (default = 8)
#define ETH_TXQ_TRANSFER_START_ADDR \
    0x14  // Start source address (byte address, should be 16-byte aligned for packet transfers)
#define ETH_TXQ_TRANSFER_SIZE_BYTES 0x18  // Transfer size in bytes (should be multiple of 16 for packet transfers)
#define ETH_TXQ_DEST_ADDR \
    0x1C  // Remote destination address for (packet/register transfer only, should be 16-byte aligned)
#define ETH_TXQ_CTRL_WORD 0x20           // Reserved
#define ETH_TXQ_TRANSFER_CNT 0x30        // Number of issued transfers
#define ETH_TXQ_PKT_START_CNT 0x34       // Number of issued packets
#define ETH_TXQ_PKT_END_CNT 0x3C         // Number of sent packets
#define ETH_TXQ_WORD_CNT 0x40            // Number of send 16-byte words
#define ETH_TXQ_REMOTE_REG_DATA 0x44     // Write data for remote register write
#define ETH_TXQ_REMOTE_SEQ_TIMEOUT 0x48  // Timeout for resend if no sequence number acks received
#define ETH_TXQ_LOCAL_SEQ_UPDATE_TIMEOUT \
    0x4C                               // Timeout for sending sequence number update packet if no other traffic issued
#define ETH_TXQ_DEST_MAC_ADDR_HI 0x50  // Destination MAC address [47:32]
#define ETH_TXQ_DEST_MAC_ADDR_LO 0x54  // Destination MAC address [31:0]
#define ETH_TXQ_SRC_MAC_ADDR_HI 0x58   // Source MAC address [47:32]
#define ETH_TXQ_SRC_MAC_ADDR_LO 0x5C   // Source MAC address [31:0]
#define ETH_TXQ_ETH_TYPE 0x60          // Type field for outgoing packets (used if TXQ_CTRL[2]=1)
#define ETH_TXQ_MIN_PACKET_SIZE_WORDS \
    0x64  // Minimal packet size (in 16-byte words); padding added (and dropped at destination) for smaller packets
#define ETH_TXQ_RESEND_CNT 0x68                // Number of resend start events
#define ETH_TXQ_DATA_PACKET_ACCEPT_AHEAD 0x6C  // Number of packets to accept before previous ones sent
#ifdef ARCH_BLACKHOLE
#define ETH_TXQ_TXPKT_CFG_SEL_SW 0x80  // TX header table entry for each kind of software-issued packet
#define ETH_TXQ_TIMESTAMP 0x90         // timestamp command and in-frame timestamp offset
#define ETH_TXQ_RX_TIMESTAMP_LO 0x94   // two-step timestamp tag, returned in the MAC's TX timestamp FIFO
#define ETH_TXQ_RX_TIMESTAMP_HI 0x98
#endif

#endif
