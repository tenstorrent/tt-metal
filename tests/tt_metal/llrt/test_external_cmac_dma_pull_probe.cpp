// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Phase 3.1 plumbing probe for the WQE-ring v2 FW DMA-pull path.
//
// What it verifies:
//   1. ExternalIfaceSender::enable_fw_dma_pull() allocates a hugepage,
//      maps it for NoC DMA via UMD (PCIDevice::map_hugepage_to_noc), and
//      publishes the resulting NoC base address into RCB+0x20/+0x24.
//   2. FW's run_dma_pull_probe() (entered when RCB.mode == 2) issues a
//      single ncrisc_noc_fast_read from that NoC address into L1 0x3000.
//   3. The 64 B pre-loaded into the host hugepage end up bit-identical
//      in L1 0x3000.
//
// Pre-condition: the new erisc_cmac_simple.elf (with run_dma_pull_probe)
// must be deployed at tt_metal/hw/firmware/bin/erisc_cmac_simple.elf
// BEFORE the test runs — RiscFirmwareInitializer loads from there at
// CreateDevice time.
//
// Usage:
//   tt-smi -r
//   TT_METAL_EXTERNAL_CMAC_PORTS=0:14 ./test_external_cmac_dma_pull_probe

#include <array>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>

#include <tt_stl/assert.hpp>
#include <tt-logger/tt-logger.hpp>

#include "impl/context/metal_context.hpp"
#include "llrt/external_iface_sender.hpp"
#include "llrt/rtoptions.hpp"
#include "llrt/tt_cluster.hpp"
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/device.hpp>

namespace {

// Pre-load pattern: 16 u32 = 64 B. Distinctive values so a partial read
// (e.g. only the first 32 B landed) is easy to spot in the diff.
constexpr std::array<uint32_t, 16> kPattern = {
    0xCAFEBABEu,
    0xDEADBEEFu,
    0x11111111u,
    0x22222222u,
    0x33333333u,
    0x44444444u,
    0x55555555u,
    0x66666666u,
    0x77777777u,
    0x88888888u,
    0x99999999u,
    0xAAAAAAAAu,
    0xBBBBBBBBu,
    0xCCCCCCCCu,
    0xDDDDDDDDu,
    0xEEEEEEEEu,
};

// Hard-coded to match the host-side / FW-side constants in
// external_iface_sender.hpp / main_cmac.cc. Probed range is the first
// 64 B of the hugepage → L1 kDmaPullProbeDst..+kDmaPullProbeLen.
constexpr uint32_t kProbeDst = 0x3000;
constexpr uint32_t kProbeLen = 64;
constexpr uint32_t kWqeRcbAddr = 0x8400;
constexpr uint32_t kRcbFwStatusOff = 0x0C;
constexpr uint32_t kFwStatusProbeOk = 0xFEEDFACEu;
// FW polls these in main()'s housekeeping window (~every 131 ms) to
// decide whether to dispatch to a gw loop / probe. We write directly
// because the probe is wire-independent — we skip ExternalIfaceSender::
// wait_for_link, which would block on PCS lock that may not be available
// in bench setups.
constexpr uint32_t kGwModeAddr = 0x1F50;
constexpr uint32_t kGwModeMagic = 0xDA7ADA7Au;

}  // namespace

int main() {
    const auto& rtopts = tt::tt_metal::MetalContext::instance().rtoptions();
    if (!rtopts.has_external_cmac_ports()) {
        log_info(tt::LogTest, "SKIP: TT_METAL_EXTERNAL_CMAC_PORTS unset");
        return 0;
    }
    const auto& ports = rtopts.get_external_cmac_ports();
    const auto [chip_id_signed, eth_chan] = *ports.begin();
    const tt::ChipId chip_id = static_cast<tt::ChipId>(chip_id_signed);

    log_info(tt::LogTest, "DMA-pull probe: chip={}, eth_chan={}", chip_id, eth_chan);

    tt::tt_metal::IDevice* device = tt::tt_metal::CreateDevice(chip_id);
    (void)device;

    const auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();
    const auto& soc_desc = cluster.get_soc_desc(chip_id);
    CoreCoord logical_core = soc_desc.get_eth_core_for_channel(eth_chan, tt::CoordSystem::LOGICAL);
    CoreCoord virtual_core =
        cluster.get_virtual_coordinate_from_logical_coordinates(chip_id, logical_core, tt::CoreType::ETH);
    log_info(tt::LogTest, "Resolved eth_chan={} → virtual ({},{})", eth_chan, virtual_core.x, virtual_core.y);

    tt::llrt::ExternalIfaceSender sender(chip_id, virtual_core);

    // Order is load-bearing (same as enable_wqe_ring): the FW samples
    // RCB.mode only at gw-mode entry, which happens when wait_for_link()
    // writes GW_MODE_MAGIC. So enable_fw_dma_pull() must run first.
    TT_FATAL(
        sender.enable_fw_dma_pull(),
        "enable_fw_dma_pull() failed — check hugepages (cat /proc/meminfo | grep HugePages) "
        "and KMD version (≥ 2.0.0 for map_buffer_to_noc support)");
    log_info(
        tt::LogTest, "enable_fw_dma_pull() OK; hugepage VA={}, NoC base published to RCB", sender.dma_pull_buffer());

    // Pre-load the pattern into the first 64 B of the hugepage. FW will
    // pull from offset 0 of the NoC-mapped region.
    auto* dst = static_cast<uint32_t*>(sender.dma_pull_buffer());
    TT_FATAL(dst != nullptr, "dma_pull_buffer() returned nullptr after successful enable");
    for (size_t i = 0; i < kPattern.size(); ++i) {
        dst[i] = kPattern[i];
    }
    // Fence so the FW NoC read sees the populated values rather than a
    // store buffer in flight on the host CPU. mmap'd hugepages are normal
    // cacheable memory; the IOMMU/PCIe controller will snoop, but only
    // for retired stores.
    __sync_synchronize();
    log_info(tt::LogTest, "Pre-loaded {} B of pattern into hugepage offset 0", kProbeLen);

    // The Phase 3.1 probe is wire-independent: it's a NoC PCIe read from
    // host hugepage into L1, with no CMAC involvement. So bypass PCS lock
    // (wait_for_link) and trigger gw-mode dispatch by writing GW_MODE_MAGIC
    // directly. FW samples kGwModeAddr in its ~131 ms housekeeping window
    // and dispatches to run_dma_pull_probe because mode==2.
    {
        std::vector<uint32_t> magic{kGwModeMagic};
        cluster.write_core_immediate(chip_id, virtual_core, magic, kGwModeAddr);
    }
    log_info(tt::LogTest, "Wrote GW_MODE_MAGIC to L1:0x{:x} — FW should dispatch within ~131 ms.", kGwModeAddr);

    // Poll fw_status until the probe completes. The NoC read should land
    // within microseconds of gw-mode entry, but the heartbeat housekeeping
    // window runs every ~131 ms, so up to one such window of slack before
    // FW even enters the probe. 5 s is a generous bound.
    const auto poll_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    uint32_t fw_status = 0;
    while (std::chrono::steady_clock::now() < poll_deadline) {
        auto v = cluster.read_core(chip_id, virtual_core, kWqeRcbAddr + kRcbFwStatusOff, 4);
        fw_status = v[0];
        if (fw_status == kFwStatusProbeOk) {
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    TT_FATAL(
        fw_status == kFwStatusProbeOk,
        "FW probe did not signal completion: RCB.fw_status=0x{:x} (expected 0x{:x}). "
        "Either FW isn't running run_dma_pull_probe (wrong ELF deployed?) or the NoC "
        "read hung on the PCIe NIU.",
        fw_status,
        kFwStatusProbeOk);
    log_info(tt::LogTest, "FW signalled probe completion (fw_status=0x{:x}).", fw_status);

    // Read FW-side diagnostics at L1 0x3100 first so we can interpret a
    // mismatch below. The probe wrote: lo, hi, noc_xy_local_addr[0], OUT
    // before issue, OUT after issue, then 4 words of L1[0x3000..0x300C]
    // from FW's view.
    {
        auto d = cluster.read_core(chip_id, virtual_core, 0x3100, 36);
        log_info(
            tt::LogTest,
            "FW diag: src_lo=0x{:08x} src_hi=0x{:08x} my_noc_xy_hi=0x{:08x} "
            "out_before=0x{:x} out_after_issue=0x{:x}",
            d[0],
            d[1],
            d[2],
            d[3],
            d[4]);
        log_info(
            tt::LogTest,
            "FW view of L1 0x3000..0x300C post-flush: {:08x} {:08x} {:08x} {:08x}",
            d[5],
            d[6],
            d[7],
            d[8]);
    }

    // Read back 64 B from L1 0x3000 and compare to the pre-loaded pattern.
    auto words = cluster.read_core(chip_id, virtual_core, kProbeDst, kProbeLen);
    TT_FATAL(
        words.size() == kPattern.size(), "read_core returned {} words, expected {}", words.size(), kPattern.size());

    bool all_match = true;
    for (size_t i = 0; i < kPattern.size(); ++i) {
        if (words[i] != kPattern[i]) {
            log_error(
                tt::LogTest,
                "MISMATCH at word {} (L1 0x{:x}): expected 0x{:08x}, got 0x{:08x}",
                i,
                kProbeDst + i * 4,
                kPattern[i],
                words[i]);
            all_match = false;
        }
    }

    if (!all_match) {
        log_error(tt::LogTest, "FAIL: probe pattern did not transit host → L1 intact.");
        return 1;
    }

    log_info(
        tt::LogTest,
        "PASS: all {} words at L1 0x{:x} match the host-prefilled pattern. "
        "Phase 3.1 plumbing verified — FW NoC PCIe read against host hugepage works.",
        kPattern.size(),
        kProbeDst);
    return 0;
}
