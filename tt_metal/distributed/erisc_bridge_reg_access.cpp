// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Register access in one TU so tt_cluster.hpp does not spread. Contract §7.1a: WriteToDeviceL1
// does not reach stream register space and does not say so.

#include "tt_metal/distributed/erisc_bridge_doorbell.hpp"

#include <tt-metalium/device.hpp>
#include "impl/context/metal_context.hpp"
#include "llrt/tt_cluster.hpp"

#include <chrono>
#include <exception>
#include <limits>
#include <string>

namespace tt::tt_fabric::erisc_bridge {

namespace {
// write_reg/read_reg take a VIRTUAL coord; the logical one addresses the wrong core silently.
tt_cxy_pair target_of(tt::tt_metal::IDevice* dev, const tt::tt_metal::CoreCoord& logical, tt::CoreType ct) {
    return tt_cxy_pair(dev->id(), dev->virtual_core_from_logical_core(logical, ct));
}
}  // namespace

std::uint32_t reg_read(
    tt::tt_metal::IDevice* dev, const tt::tt_metal::CoreCoord& logical, tt::CoreType ct, std::uint32_t addr) {
    std::uint32_t v = 0;
    tt::tt_metal::MetalContext::instance().get_cluster().read_reg(&v, target_of(dev, logical, ct), addr);
    return v;
}

bool reg_write(
    tt::tt_metal::IDevice* dev,
    const tt::tt_metal::CoreCoord& logical,
    tt::CoreType ct,
    std::uint32_t addr,
    std::uint32_t val) {
    // False on failure, not an exception: every caller checks the result and records
    // first_error(), so returning true unconditionally made those branches dead code.
    try {
        tt::tt_metal::MetalContext::instance().get_cluster().write_reg(&val, target_of(dev, logical, ct), addr);
    } catch (const std::exception&) {
        return false;
    }
    return true;
}

bool write_bytes_to_l1(
    tt::tt_metal::IDevice* dev,
    const tt::tt_metal::CoreCoord& logical,
    std::uint32_t addr,
    const void* src,
    std::uint32_t bytes,
    tt::CoreType ct) {
    if (src == nullptr || bytes == 0) {
        return false;
    }
    // write_core, not write_reg: L1 is memory. On Blackhole this is PIO through a TLB window --
    // supports_dma_operations() is WORMHOLE_B0 only (issue #22957), which is the ~292 MB/s.
    try {
        tt::tt_metal::MetalContext::instance().get_cluster().write_core(src, bytes, target_of(dev, logical, ct), addr);
    } catch (const std::exception&) {
        return false;  // see reg_write
    }
    return true;
}

bool read_bytes_from_l1(
    tt::tt_metal::IDevice* dev,
    const tt::tt_metal::CoreCoord& logical,
    std::uint32_t addr,
    void* dst,
    std::uint32_t bytes,
    tt::CoreType ct) {
    if (dst == nullptr || bytes == 0) {
        return false;
    }
    try {
        tt::tt_metal::MetalContext::instance().get_cluster().read_core(dst, bytes, target_of(dev, logical, ct), addr);
    } catch (const std::exception&) {
        return false;  // see reg_write
    }
    return true;
}

ClockAnchor measure_clock_anchor(
    tt::tt_metal::IDevice* dev, const tt::tt_metal::CoreCoord& eth_logical, std::uint32_t samples) {
    ClockAnchor a;
    if (dev == nullptr || samples == 0) {
        a.why = "measure_clock_anchor: no device";
        return a;
    }
    // From the device, never assumed. get_device_aiclk returns MHz, which IS cycles per
    // microsecond -- the unit works out only because the clock is quoted in MHz.
    const int aiclk_mhz = tt::tt_metal::MetalContext::instance().get_cluster().get_device_aiclk(dev->id());
    if (aiclk_mhz <= 0) {
        a.why = "measure_clock_anchor: device reported no AICLK";
        return a;
    }
    a.cyc_per_us = static_cast<double>(aiclk_mhz);

    double best_rtt_us = std::numeric_limits<double>::max();
    for (std::uint32_t k = 0; k < samples; ++k) {
        const auto t1 = std::chrono::steady_clock::now();
        const std::uint32_t c = reg_read(dev, eth_logical, tt::CoreType::ETH, kEriscWallClockLo);
        const auto t2 = std::chrono::steady_clock::now();
        const double rtt_us = std::chrono::duration<double, std::micro>(t2 - t1).count();
        // The TIGHTEST bracket wins, not the average: a slow read brackets the truth more
        // loosely, and averaging it in would widen the error bar rather than narrow it.
        if (rtt_us < best_rtt_us) {
            best_rtt_us = rtt_us;
            a.cyc = c;
            a.host_us = std::chrono::duration<double, std::micro>(t1.time_since_epoch()).count() + rtt_us / 2.0;
        }
    }
    if (best_rtt_us == std::numeric_limits<double>::max()) {
        a.why = "measure_clock_anchor: no read completed";
        return a;
    }
    a.uncertainty_us = best_rtt_us / 2.0;

    // AICLK scales at runtime. A run whose conversion rate moved underneath it has not been
    // measured, it has been guessed, so say so rather than report the number.
    const int aiclk_after = tt::tt_metal::MetalContext::instance().get_cluster().get_device_aiclk(dev->id());
    if (aiclk_after != aiclk_mhz) {
        a.why = "measure_clock_anchor: AICLK moved from " + std::to_string(aiclk_mhz) + " to " +
                std::to_string(aiclk_after) + " MHz during calibration";
        return a;
    }
    a.ok = true;
    return a;
}

}  // namespace tt::tt_fabric::erisc_bridge
