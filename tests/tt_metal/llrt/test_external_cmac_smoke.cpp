// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Smoke test for the external CMAC port TX data path.
//
// What it verifies:
//   1. ExternalIfaceSender::wait_for_link() returns true within 10 s, meaning
//      erisc_cmac_simple achieved PCS lock and the firmware mode-switch magic
//      word was written.
//   2. A 60-byte minimal TT-link Ethernet frame (EtherType 0x1AF4) is accepted
//      by send() without error.
//   3. If the FPGA BAR device file is accessible (TT_FPGA_BAR env var or the
//      default /dev/xdma0_user), the stat_cls_passed register at AXI-Lite
//      offset 0x430 increments by exactly 1.
//
// Skip conditions (exit 0 with a message):
//   - TT_METAL_EXTERNAL_CMAC_PORTS is not set or empty (no CMAC port configured).
//   - The FPGA BAR device cannot be opened (the TX-path assertion is still made).
//
// Usage:
//   TT_METAL_EXTERNAL_CMAC_PORTS=0:14 ./test_external_cmac_smoke
//   TT_METAL_EXTERNAL_CMAC_PORTS=0:14 TT_FPGA_BAR=/dev/xdma0_user ./test_external_cmac_smoke
//   Optional peer MAC: TT_PEER_MAC=01:02:03:04:05:06 (default: FF×6 broadcast)

#include <array>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <span>
#include <string>
#include <thread>
#include <vector>

// POSIX for FPGA BAR access
#include <fcntl.h>
#include <sys/types.h>
#include <unistd.h>

#include <tt_stl/assert.hpp>
#include <tt-logger/tt-logger.hpp>

#include "impl/context/metal_context.hpp"
#include "llrt/external_iface_sender.hpp"
#include "llrt/rtoptions.hpp"
#include "llrt/tt_cluster.hpp"

// -------------------------------------------------------------------------
// Minimal Ethernet frame helpers
// -------------------------------------------------------------------------

// Parse "AA:BB:CC:DD:EE:FF" env var into a 6-byte array.  Returns false if
// the env var is not set or cannot be parsed (caller uses default).
static bool parse_mac_env(const char* env_name, std::array<uint8_t, 6>& out) {
    const char* val = std::getenv(env_name);
    if (!val) {
        return false;
    }
    unsigned int bytes[6] = {};
    if (std::sscanf(val, "%x:%x:%x:%x:%x:%x", &bytes[0], &bytes[1], &bytes[2], &bytes[3], &bytes[4], &bytes[5]) != 6) {
        log_warning(tt::LogTest, "Could not parse {} = '{}', using default MAC.", env_name, val);
        return false;
    }
    for (int i = 0; i < 6; i++) {
        out[i] = static_cast<uint8_t>(bytes[i]);
    }
    return true;
}

// Build a 60-byte minimal Ethernet frame (no FCS) with EtherType 0x1AF4.
// Layout: [dst 6B][src 6B][EtherType 2B][payload 46B of 0x00]
static std::vector<uint8_t> build_tt_link_frame(
    const std::array<uint8_t, 6>& dst_mac, const std::array<uint8_t, 6>& src_mac) {
    std::vector<uint8_t> frame(60, 0x00u);
    // Destination MAC
    std::memcpy(frame.data(), dst_mac.data(), 6);
    // Source MAC
    std::memcpy(frame.data() + 6, src_mac.data(), 6);
    // EtherType 0x1AF4 (big-endian)
    frame[12] = 0x1Au;
    frame[13] = 0xF4u;
    // Payload bytes [14..59] remain 0x00 (minimum padding to 60 bytes).
    return frame;
}

// -------------------------------------------------------------------------
// FPGA BAR read helper
// -------------------------------------------------------------------------

// Open the FPGA AXI-Lite BAR device file.  Returns -1 on failure.
static int open_fpga_bar() {
    const char* bar_path = std::getenv("TT_FPGA_BAR");
    if (!bar_path) {
        bar_path = "/dev/xdma0_user";
    }
    int fd = ::open(bar_path, O_RDONLY);
    if (fd < 0) {
        log_warning(
            tt::LogTest,
            "Could not open FPGA BAR device '{}' (errno={}). "
            "Skipping FPGA stat_cls_passed check — TX path will still be asserted.",
            bar_path,
            errno);
    }
    return fd;
}

// Read a 32-bit register from the FPGA AXI-Lite BAR at the given byte offset.
static uint32_t fpga_read32(int fd, off_t offset) {
    uint32_t val = 0;
    ssize_t n = ::pread(fd, &val, sizeof(val), offset);
    TT_FATAL(n == static_cast<ssize_t>(sizeof(val)), "pread from FPGA BAR at offset 0x{:x} failed", offset);
    return val;
}

// -------------------------------------------------------------------------
// main
// -------------------------------------------------------------------------

int main() {
    // ------------------------------------------------------------------
    // Step 1 — Check whether external CMAC ports are configured.
    // ------------------------------------------------------------------
    const auto& rtopts = tt::tt_metal::MetalContext::instance().rtoptions();

    if (!rtopts.has_external_cmac_ports()) {
        log_info(
            tt::LogTest,
            "SKIP: TT_METAL_EXTERNAL_CMAC_PORTS is not set. "
            "No external CMAC port configured — nothing to test.");
        return 0;
    }

    // Use the first configured port.
    const auto& ports = rtopts.get_external_cmac_ports();
    const auto [chip_id_signed, eth_chan] = *ports.begin();
    const tt::ChipId chip_id = static_cast<tt::ChipId>(chip_id_signed);

    log_info(tt::LogTest, "External CMAC smoke test: chip_id={}, eth_chan={}", chip_id, eth_chan);

    // ------------------------------------------------------------------
    // Step 2 — Resolve eth_chan → virtual core coordinate.
    // ------------------------------------------------------------------
    const auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();
    const auto& soc_desc = cluster.get_soc_desc(chip_id);

    CoreCoord logical_core = soc_desc.get_eth_core_for_channel(eth_chan, CoordSystem::LOGICAL);
    CoreCoord virtual_core =
        cluster.get_virtual_coordinate_from_logical_coordinates(chip_id, logical_core, CoreType::ETH);

    log_info(
        tt::LogTest,
        "Resolved eth_chan={} → logical ({},{}) → virtual ({},{})",
        eth_chan,
        logical_core.x,
        logical_core.y,
        virtual_core.x,
        virtual_core.y);

    // ------------------------------------------------------------------
    // Step 3 — Construct ExternalIfaceSender.
    // ------------------------------------------------------------------
    tt::llrt::ExternalIfaceSender sender(chip_id, virtual_core);

    // ------------------------------------------------------------------
    // Step 4 — Wait for PCS lock (10 s timeout).
    // ------------------------------------------------------------------
    log_info(tt::LogTest, "Waiting for PCS link (up to 10 s) ...");
    bool link_up = sender.wait_for_link(10000);
    TT_FATAL(
        link_up,
        "wait_for_link() timed out after 10 s — PCS lock not achieved on chip {} eth_chan {}",
        chip_id,
        eth_chan);
    log_info(tt::LogTest, "PCS link up. Firmware mode-switch magic word written.");

    // ------------------------------------------------------------------
    // Step 5 — Build a minimal 60-byte TT-link Ethernet frame.
    // ------------------------------------------------------------------
    std::array<uint8_t, 6> dst_mac = {0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF};  // broadcast default
    std::array<uint8_t, 6> src_mac = {0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0x00};

    parse_mac_env("TT_PEER_MAC", dst_mac);  // override dst from env if set

    std::vector<uint8_t> frame = build_tt_link_frame(dst_mac, src_mac);
    TT_FATAL(frame.size() == 60, "Internal error: frame size {} != 60", frame.size());

    // ------------------------------------------------------------------
    // Step 6 — Open the FPGA BAR and read the baseline stat_cls_passed.
    // ------------------------------------------------------------------
    // stat_cls_passed register in tt_link_regs at AXI-Lite offset 0x430.
    static constexpr off_t kStatClsPassedOffset = 0x430;

    int fpga_fd = open_fpga_bar();
    uint32_t baseline_stat = 0;
    if (fpga_fd >= 0) {
        baseline_stat = fpga_read32(fpga_fd, kStatClsPassedOffset);
        log_info(tt::LogTest, "FPGA stat_cls_passed baseline = {}", baseline_stat);
    }

    // ------------------------------------------------------------------
    // Step 7 — Send the frame.
    // ------------------------------------------------------------------
    bool sent = sender.send(std::span<const uint8_t>(frame));
    TT_FATAL(sent, "send() returned false — TX ring busy or frame too large");
    log_info(tt::LogTest, "Frame sent successfully ({} bytes, EtherType=0x1AF4).", frame.size());

    // ------------------------------------------------------------------
    // Step 8 — Wait 1 ms for the FPGA classifier to process the frame.
    // ------------------------------------------------------------------
    std::this_thread::sleep_for(std::chrono::milliseconds(1));

    // ------------------------------------------------------------------
    // Step 9 — Verify FPGA stat_cls_passed incremented by 1.
    // ------------------------------------------------------------------
    if (fpga_fd >= 0) {
        uint32_t new_stat = fpga_read32(fpga_fd, kStatClsPassedOffset);
        log_info(tt::LogTest, "FPGA stat_cls_passed after send = {} (expected {})", new_stat, baseline_stat + 1);
        TT_FATAL(
            new_stat == baseline_stat + 1,
            "stat_cls_passed did not increment: before={}, after={}. "
            "Check FPGA classifier and TT-link EtherType filtering.",
            baseline_stat,
            new_stat);
        ::close(fpga_fd);
    }

    // ------------------------------------------------------------------
    // Step 10 — Report success.
    // ------------------------------------------------------------------
    log_info(tt::LogTest, "PASS: external CMAC smoke test");
    return 0;
}
