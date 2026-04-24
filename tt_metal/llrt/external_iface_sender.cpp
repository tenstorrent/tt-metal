// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#include "llrt/external_iface_sender.hpp"

#include <chrono>
#include <cstdint>
#include <span>
#include <thread>
#include <vector>

#include "impl/context/metal_context.hpp"
#include <llrt/tt_cluster.hpp>
#include <umd/device/types/cluster_descriptor_types.hpp>

namespace tt::llrt {

// ---------------------------------------------------------------------------
// Helper: return a const reference to the singleton Cluster.
// ---------------------------------------------------------------------------
static const ::tt::Cluster& cluster() { return ::tt::tt_metal::MetalContext::instance().get_cluster(); }

// ---------------------------------------------------------------------------
// Helper: read a single uint32_t from erisc L1.
// ---------------------------------------------------------------------------
static uint32_t read_word(ChipId chip_id, CoreCoord core, uint32_t addr) {
    auto vec = cluster().read_core(chip_id, core, addr, sizeof(uint32_t));
    return vec[0];
}

// ---------------------------------------------------------------------------
// Helper: write a single uint32_t to erisc L1 without write-combining
//         (important for doorbell semantics).
// ---------------------------------------------------------------------------
static void write_word_immediate(ChipId chip_id, CoreCoord core, uint32_t addr, uint32_t value) {
    std::vector<uint32_t> buf{value};
    cluster().write_core_immediate(chip_id, core, buf, addr);
}

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------
ExternalIfaceSender::ExternalIfaceSender(ChipId chip_id, CoreCoord virtual_eth_core) :
    chip_id_(chip_id), virtual_core_(virtual_eth_core) {}

// ---------------------------------------------------------------------------
// is_link_up
// ---------------------------------------------------------------------------
bool ExternalIfaceSender::is_link_up() const { return read_word(chip_id_, virtual_core_, kTrainStatusAddr) == 0u; }

// ---------------------------------------------------------------------------
// wait_for_link
// ---------------------------------------------------------------------------
bool ExternalIfaceSender::wait_for_link(uint32_t timeout_ms) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);

    while (!is_link_up()) {
        if (std::chrono::steady_clock::now() >= deadline) {
            return false;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    // PCS lock achieved — switch firmware to data-path mode.
    write_word_immediate(chip_id_, virtual_core_, kModeAddr, kModeMagic);
    return true;
}

// ---------------------------------------------------------------------------
// send
// ---------------------------------------------------------------------------
bool ExternalIfaceSender::send(std::span<const uint8_t> buf) {
    if (buf.size() > kMaxFrameBytes) {
        return false;
    }

    // Check that the previous TX has been consumed (doorbell cleared).
    if (read_word(chip_id_, virtual_core_, kTxSizeAddr) != 0u) {
        return false;
    }

    // Select the ping-pong TX buffer.
    const uint32_t tx_addr = (tx_buf_sel_ == 0) ? kTxBuf0 : kTxBuf1;

    // Write frame bytes directly into erisc L1 (data path — no erisc copy).
    cluster().write_core(chip_id_, virtual_core_, std::span<const uint8_t>(buf), tx_addr);

    // Write buffer selector.
    write_word_immediate(chip_id_, virtual_core_, kTxBufSelAddr, static_cast<uint32_t>(tx_buf_sel_));

    // Write frame size — this is the doorbell that arms the TX DMA.
    write_word_immediate(chip_id_, virtual_core_, kTxSizeAddr, static_cast<uint32_t>(buf.size()));

    // Advance ping-pong index for next call.
    tx_buf_sel_ ^= 1;
    return true;
}

// ---------------------------------------------------------------------------
// rx_watermark
// ---------------------------------------------------------------------------
uint32_t ExternalIfaceSender::rx_watermark() const { return read_word(chip_id_, virtual_core_, kRxWpAddr); }

// ---------------------------------------------------------------------------
// read_rx
// ---------------------------------------------------------------------------
void ExternalIfaceSender::read_rx(uint32_t word_offset, std::span<uint8_t> out) const {
    const uint32_t addr = kPacketBuf + word_offset * 4u;
    cluster().read_core(out.data(), static_cast<uint32_t>(out.size()), tt_cxy_pair(chip_id_, virtual_core_), addr);
}

// ---------------------------------------------------------------------------
// rx_consume
// ---------------------------------------------------------------------------
void ExternalIfaceSender::rx_consume(uint32_t consumed_wp) {
    write_word_immediate(chip_id_, virtual_core_, kRxRpAddr, consumed_wp);
}

}  // namespace tt::llrt
