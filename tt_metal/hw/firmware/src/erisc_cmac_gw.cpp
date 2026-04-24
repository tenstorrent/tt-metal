// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Zero-copy gateway kernel for the external CMAC port.
//
// Runs on the erisc core after erisc_cmac_simple has achieved PCS lock.
// The host signals entry to this loop by writing kModeMagic to kModeAddr.
//
// TX (host → CMAC): host writes frame bytes directly to TX_BUF0/1 via PCIe,
//   then sets kTxSizeAddr.  This kernel arms the TX queue and clears kTxSizeAddr.
//   Erisc touches only control registers — the frame bytes never pass through
//   a software copy loop.
//
// RX (CMAC → host): CMAC DMA delivers frames into a ping-pong pair of buffers
//   (kPacketBuf at 0x4000 and kPacketBuf1 at 0x6C00).  When a frame completes,
//   this kernel re-arms the RX queue to write into the OTHER buffer, then
//   publishes kRxBufSelAddr (which buffer holds the frame) and kRxWpAddr (word
//   count).  Host reads directly from the published buffer via PCIe MMIO.
//   No ack from the host is required before the next frame starts — the firmware
//   has already switched buffers, eliminating the single-buffer stall.
//
// L1 layout shared with ExternalIfaceSender (external_iface_sender.hpp):
//   0x1F40  kTxSizeAddr   — host→erisc: frame size in bytes (0 = idle)
//   0x1F44  kTxBufSelAddr — host→erisc: 0 = TX_BUF0 (0x6000), 1 = TX_BUF1 (0x6600)
//   0x1F48  kRxWpAddr     — erisc→host: word count written into the completed buffer
//   0x1F4C  kRxRpAddr     — (unused in ping-pong mode; kept for ABI compatibility)
//   0x1F50  kModeAddr     — host writes kModeMagic to switch to data-path loop
//   0x1F54  kRxBufSelAddr — erisc→host: 0 = frame is in kPacketBuf, 1 = kPacketBuf1
//   0x4000  kPacketBuf    — CMAC RX DMA target buf 0 (host reads directly via PCIe)
//   0x6000  kTxBuf0       — host writes TX frame here
//   0x6600  kTxBuf1       — TX ping-pong partner
//   0x6C00  kPacketBuf1   — CMAC RX DMA target buf 1 (4 KB; after TX_BUF1 which ends at 0x6BF0)

#include <cstdint>

// TX queue register offsets (from eth_cmac_init.h)
#define ETH_TXQ0_REGS_START 0xFFB90000u
#define ETH_TXQ_REGS_SIZE 0x1000u
#define ETH_TXQ_CMD 0x4u
#define ETH_TXQ_CMD_START_RAW 0x1u
#define ETH_TXQ_TRANSFER_START_ADDR 0x14u
#define ETH_TXQ_TRANSFER_SIZE_BYTES 0x18u

// RX queue register offsets
#define ETH_RXQ0_REGS_START 0xFFB92000u
#define ETH_RXQ_REGS_SIZE 0x1000u
#define ETH_RXQ_BUF_PTR 0x8u
#define ETH_RXQ_BUF_START_WORD_ADDR 0xCu

#define REG32(addr) (*reinterpret_cast<volatile uint32_t*>(addr))

static inline void txq_write(uint32_t q, uint32_t off, uint32_t val) {
    REG32(ETH_TXQ0_REGS_START + q * ETH_TXQ_REGS_SIZE + off) = val;
}
static inline uint32_t rxq_read(uint32_t q, uint32_t off) {
    return REG32(ETH_RXQ0_REGS_START + q * ETH_RXQ_REGS_SIZE + off);
}
static inline void rxq_write(uint32_t q, uint32_t off, uint32_t val) {
    REG32(ETH_RXQ0_REGS_START + q * ETH_RXQ_REGS_SIZE + off) = val;
}

// Mailbox addresses
static constexpr uint32_t kTxSizeAddr = 0x1F40u;
static constexpr uint32_t kTxBufSelAddr = 0x1F44u;
static constexpr uint32_t kRxWpAddr = 0x1F48u;
// kRxRpAddr (0x1F4C) kept for ABI compatibility; not used in ping-pong mode.
static constexpr uint32_t kRxBufSelAddr = 0x1F54u;  // erisc→host: which RX buf holds the frame

// TX buffer addresses
static constexpr uint32_t kTxBuf0 = 0x6000u;
static constexpr uint32_t kTxBuf1 = 0x6600u;

// RX ping-pong buffer addresses (word addresses for RXQ register programming)
static constexpr uint32_t kPacketBuf = 0x4000u;   // RX buf 0 (8 KB: 0x4000–0x5FFF)
static constexpr uint32_t kPacketBuf1 = 0x6C00u;  // RX buf 1 (4 KB: 0x6C00–0x7BFF; after TX_BUF1 end 0x6BF0)

// Size of each RX buffer in 32-bit words (4 KB = 1024 words)
static constexpr uint32_t kRxBufSizeWords = 1024u;

void kernel_main() {
    uint8_t rx_buf_idx = 0;  // which RX buffer the RX queue is currently writing into
    uint32_t last_rx_ptr = rxq_read(0, ETH_RXQ_BUF_PTR);

    while (true) {
        // ── TX: host filled a buffer and wrote the frame size ──────────────
        uint32_t sz = REG32(kTxSizeAddr);
        if (sz != 0) {
            uint32_t buf = (REG32(kTxBufSelAddr) == 0) ? kTxBuf0 : kTxBuf1;
            txq_write(0, ETH_TXQ_TRANSFER_START_ADDR, buf);
            txq_write(0, ETH_TXQ_TRANSFER_SIZE_BYTES, sz);
            txq_write(0, ETH_TXQ_CMD, ETH_TXQ_CMD_START_RAW);
            REG32(kTxSizeAddr) = 0;  // signal host: slot available
        }

        // ── RX: CMAC DMA advanced the buffer pointer (ping-pong) ───────────
        uint32_t cur = rxq_read(0, ETH_RXQ_BUF_PTR);
        if (cur != last_rx_ptr) {
            // Switch the RX queue to the OTHER buffer before publishing to the
            // host.  CMAC DMA can immediately start filling the next buffer
            // while the host reads the current one via PCIe — no stall.
            uint8_t next_idx = rx_buf_idx ^ 1u;
            uint32_t next_buf_word_addr =
                (next_idx == 0) ? (kPacketBuf / sizeof(uint32_t)) : (kPacketBuf1 / sizeof(uint32_t));

            // Re-arm RX queue to write into the next buffer.
            rxq_write(0, ETH_RXQ_BUF_START_WORD_ADDR, next_buf_word_addr);
            rxq_write(0, ETH_RXQ_BUF_PTR, 0);  // reset write pointer

            // Publish to host: which buffer holds the completed frame and its
            // word count.  kRxBufSelAddr must be written before kRxWpAddr so
            // the host sees a consistent view when it polls kRxWpAddr.
            REG32(kRxBufSelAddr) = rx_buf_idx;
            REG32(kRxWpAddr) = cur;  // word count written into the old buffer

            rx_buf_idx = next_idx;
            last_rx_ptr = 0;  // BUF_PTR was reset above
        }
    }
}
