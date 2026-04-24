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
// RX (CMAC → host): CMAC DMA delivers frames into PACKET_BUF (kPacketBuf).
//   This kernel publishes the updated ETH_RXQ_BUF_PTR to kRxWpAddr with a
//   single 4-byte write.  Host reads frame bytes directly from kPacketBuf via
//   PCIe, then writes the consumed pointer back to kRxRpAddr.
//
// L1 layout shared with ExternalIfaceSender (external_iface_sender.hpp):
//   0x1F40  kTxSizeAddr   — host→erisc: frame size in bytes (0 = idle)
//   0x1F44  kTxBufSelAddr — host→erisc: 0 = TX_BUF0 (0x6000), 1 = TX_BUF1 (0x6600)
//   0x1F48  kRxWpAddr     — erisc→host: ETH_RXQ_BUF_PTR word value (new frame ready)
//   0x1F4C  kRxRpAddr     — host→erisc: consumed pointer (host finished reading)
//   0x4000  kPacketBuf    — CMAC RX DMA target (host reads directly via PCIe)
//   0x6000  kTxBuf0       — host writes TX frame here
//   0x6600  kTxBuf1       — ping-pong partner

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

#define REG32(addr) (*reinterpret_cast<volatile uint32_t*>(addr))

static inline void txq_write(uint32_t q, uint32_t off, uint32_t val) {
    REG32(ETH_TXQ0_REGS_START + q * ETH_TXQ_REGS_SIZE + off) = val;
}
static inline uint32_t rxq_read(uint32_t q, uint32_t off) {
    return REG32(ETH_RXQ0_REGS_START + q * ETH_RXQ_REGS_SIZE + off);
}

// Mailbox addresses
static constexpr uint32_t kTxSizeAddr = 0x1F40u;
static constexpr uint32_t kTxBufSelAddr = 0x1F44u;
static constexpr uint32_t kRxWpAddr = 0x1F48u;
static constexpr uint32_t kRxRpAddr = 0x1F4Cu;

static constexpr uint32_t kTxBuf0 = 0x6000u;
static constexpr uint32_t kTxBuf1 = 0x6600u;

void kernel_main() {
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

        // ── RX: CMAC DMA advanced the buffer pointer ────────────────────────
        uint32_t cur = rxq_read(0, ETH_RXQ_BUF_PTR);
        if (cur != last_rx_ptr) {
            REG32(kRxWpAddr) = cur;  // publish to host (single 4-byte write)
            last_rx_ptr = cur;
            // Wait for host to signal consumed before accepting next frame.
            // Host writes the same cur value to kRxRpAddr when done reading.
            //
            // SINGLE-BUFFER STALL: while the host holds kRxRpAddr unacknowledged,
            // CMAC continues DMAing into PACKET_BUF and wraps, corrupting the
            // unread frame.  At line rate (120 ns/frame) the host must ack before
            // the next frame lands — this is feasible for the monitoring use-case
            // but will drop frames under sustained traffic.
            // TODO: add a second PACKET_BUF (ping-pong at 0x4000/0x6C00) so CMAC
            // can fill buf[1] while host reads buf[0].  Requires a 4 KB L1 slot
            // at 0x6C00 (currently free) and a buf_sel word in the mailbox block.
            while (REG32(kRxRpAddr) != cur) {
                // While waiting, still service TX so we don't stall the TX path.
                uint32_t tx_sz = REG32(kTxSizeAddr);
                if (tx_sz != 0) {
                    uint32_t buf = (REG32(kTxBufSelAddr) == 0) ? kTxBuf0 : kTxBuf1;
                    txq_write(0, ETH_TXQ_TRANSFER_START_ADDR, buf);
                    txq_write(0, ETH_TXQ_TRANSFER_SIZE_BYTES, tx_sz);
                    txq_write(0, ETH_TXQ_CMD, ETH_TXQ_CMD_START_RAW);
                    REG32(kTxSizeAddr) = 0;
                }
            }
        }
    }
}
