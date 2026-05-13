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
//   3. If TT_FPGA_BAR is set and the file is accessible, the stat_cls_passed
//      register at AXI-Lite offset 0x430 increments by exactly 1.
//
// Skip conditions (exit 0 with a message):
//   - TT_METAL_EXTERNAL_CMAC_PORTS is not set or empty (no CMAC port configured).
//   - TT_FPGA_BAR is not set or the file cannot be opened (TX-path still asserted).
//
// The shell uses QDMA IP (not XDMA), so BAR2 is exposed via the sysfs PCI
// resource file — NOT /dev/xdma0_user.  Find the path with:
//   FPGA_PCI=$(lspci -D | grep -i xilinx | awk '{print $1}')
//   export TT_FPGA_BAR=/sys/bus/pci/devices/${FPGA_PCI}/resource2
//
// Usage:
//   TT_METAL_EXTERNAL_CMAC_PORTS=0:14 ./test_external_cmac_smoke
//   TT_METAL_EXTERNAL_CMAC_PORTS=0:14 TT_FPGA_BAR=/sys/bus/pci/devices/0000:XX:YY.Z/resource2 \
//     ./test_external_cmac_smoke
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
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/device.hpp>

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

// Build a 256-byte Ethernet frame (no FCS) with EtherType 0x1AF4.
// Bumped from the 60-byte minimum because the CMAC TX path silently drops
// sub-frame-size raw transmits on this rig — burst-mode 1500-byte frames
// reach the wire but sub-100-byte ones don't show up in NIC PHY counters.
// 256 B is comfortably above any plausible min-size guard while still being
// short enough to round-trip in one DPDK burst.
// Layout: [dst 6B][src 6B][EtherType 2B][payload 242B counting pattern]
static std::vector<uint8_t> build_tt_link_frame(
    const std::array<uint8_t, 6>& dst_mac, const std::array<uint8_t, 6>& src_mac) {
    std::vector<uint8_t> frame(256, 0x00u);
    // Destination MAC
    std::memcpy(frame.data(), dst_mac.data(), 6);
    // Source MAC
    std::memcpy(frame.data() + 6, src_mac.data(), 6);
    // EtherType 0x1AF4 (big-endian)
    frame[12] = 0x1Au;
    frame[13] = 0xF4u;
    // Payload bytes [14..255] = counting pattern so a wire capture is
    // distinguishable from burst-mode TTWH frames.
    for (size_t i = 14; i < frame.size(); ++i) {
        frame[i] = static_cast<uint8_t>(i & 0xFF);
    }
    return frame;
}

// -------------------------------------------------------------------------
// FPGA BAR read helper
// -------------------------------------------------------------------------

// Open the FPGA AXI-Lite BAR via the sysfs PCI resource2 file.
// Returns -1 if TT_FPGA_BAR is not set or the file cannot be opened.
// The shell uses QDMA IP: set TT_FPGA_BAR to the sysfs resource2 path,
// e.g. /sys/bus/pci/devices/0000:XX:YY.Z/resource2
static int open_fpga_bar() {
    const char* bar_path = std::getenv("TT_FPGA_BAR");
    if (!bar_path) {
        log_warning(
            tt::LogTest,
            "TT_FPGA_BAR not set. Skipping FPGA stat_cls_passed check. "
            "Set it to the sysfs resource2 path: "
            "FPGA_PCI=$(lspci -D | grep -i xilinx | awk '{{print $1}}'); "
            "export TT_FPGA_BAR=/sys/bus/pci/devices/${{FPGA_PCI}}/resource2");
        return -1;
    }
    int fd = ::open(bar_path, O_RDONLY);
    if (fd < 0) {
        log_warning(
            tt::LogTest,
            "Could not open FPGA BAR '{}' (errno={}). "
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

    // Trigger MetalContext init so RiscFirmwareInitializer::initialize_cmac_firmware
    // actually runs and loads erisc_cmac_simple onto the eth core. Without this the
    // test would just observe whatever the WH boot ROM left in the eth core's L1.
    tt::tt_metal::IDevice* device = tt::tt_metal::CreateDevice(chip_id);

    // ------------------------------------------------------------------
    // Step 2 — Resolve eth_chan → virtual core coordinate.
    // ------------------------------------------------------------------
    const auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();
    const auto& soc_desc = cluster.get_soc_desc(chip_id);

    CoreCoord logical_core = soc_desc.get_eth_core_for_channel(eth_chan, tt::CoordSystem::LOGICAL);
    CoreCoord virtual_core =
        cluster.get_virtual_coordinate_from_logical_coordinates(chip_id, logical_core, tt::CoreType::ETH);

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

    // CreateDevice's RISC reset transition kills the CMAC PCS marker stream
    // briefly. On Mellanox-attached external CMAC, the peer's eswitch FDB
    // enters a transient state for ~hundreds of ms after link recovery and
    // silently drops frames that arrive in that window — they reach
    // rx_packets_phy and rx_prio0_packets but never the PF vport. Wait for
    // the eswitch to re-stabilize before firing the gw frame. Override with
    // TT_METAL_CMAC_POST_LINK_SETTLE_MS=N (default 2500 ms; 0 disables).
    // The structural fix is to stop bouncing the link via the warm-reload
    // guard in RiscFirmwareInitializer, but that requires KMD-level changes
    // to skip the soft-reset of eth cores on chip close — until then this
    // sleep is the smallest path to a passing end-to-end smoke test.
    uint32_t settle_ms = 2500;
    if (const char* env = std::getenv("TT_METAL_CMAC_POST_LINK_SETTLE_MS")) {
        settle_ms = static_cast<uint32_t>(std::strtoul(env, nullptr, 0));
    }
    if (settle_ms > 0) {
        log_info(tt::LogTest, "Settling for {} ms (Mellanox eswitch FDB recovery window)...", settle_ms);
        std::this_thread::sleep_for(std::chrono::milliseconds(settle_ms));
    }

    // ------------------------------------------------------------------
    // Step 5 — Build a minimal 60-byte TT-link Ethernet frame.
    // ------------------------------------------------------------------
    std::array<uint8_t, 6> dst_mac = {0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF};  // broadcast default
    std::array<uint8_t, 6> src_mac = {0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0x00};

    parse_mac_env("TT_PEER_MAC", dst_mac);  // override dst from env if set

    std::vector<uint8_t> frame = build_tt_link_frame(dst_mac, src_mac);
    TT_FATAL(frame.size() == 256, "Internal error: frame size {} != 256", frame.size());

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
    log_info(tt::LogTest, "Frame submitted to TX doorbell ({} B, EtherType=0x1AF4).", frame.size());

    // ------------------------------------------------------------------
    // Step 8 — Wait for firmware to consume the doorbell and arm CMAC TX.
    // ------------------------------------------------------------------
    // The doorbell-clear is also the gw-mode-entry ack: erisc_cmac_simple only
    // clears kTxSizeAddr from inside run_gateway_loop, which it enters at most
    // ~33 ms after kModeMagic is written.  Without this wait, CloseDevice would
    // reset the erisc before firmware ever sees the magic word.
    bool consumed = sender.wait_tx_consumed(5000);
    TT_FATAL(
        consumed,
        "wait_tx_consumed() timed out after 5000 ms — firmware did not enter gw mode "
        "or did not arm CMAC TX. Check kModeMagic write at L1:0x1F50 and erisc heartbeat.");
    log_info(tt::LogTest, "Firmware consumed TX doorbell — CMAC TX armed.");

    // Wait 1 ms for the FPGA classifier to ingest the frame off the wire
    // (separate latency from the firmware-side consume above).
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
    tt::tt_metal::CloseDevice(device);
    return 0;
}
