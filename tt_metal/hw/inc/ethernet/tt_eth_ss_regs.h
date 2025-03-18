// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
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

//////////////////////////////
// TX queue 0/1 controllers
#define ETH_TXQ0_REGS_START 0xFFB90000
#define ETH_TXQ_REGS_SIZE 0x1000
#define ETH_TXQ_REGS_SIZE_BIT 12

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

#define ETH_TXQ_STATUS 0x8              // IMPROVE: document (misc. internal bits for debug)
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

#endif
