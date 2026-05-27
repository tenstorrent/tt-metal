// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>

#include <umd/device/types/cluster_descriptor_types.hpp>

namespace tt::tt_metal::experimental::x280 {

// Bit i = 1 selects physical L2CPU index i (0..3) for boot. Default boots
// every enabled L2CPU on the chip; the actual booted set is the intersection
// of `l2cpu_mask` and the chip's non-harvested L2CPUs.
constexpr std::uint8_t kAllL2CpuMask = 0xFu;

// Which memory region the X280 fetches its first instruction from. Both
// modes load the same baseline "sentinel + WFI" idle firmware shape, but
// from different physical artifacts whose linker scripts differ:
//   * Lim  - firmware-lim-idle.bin (ld/x280.ld), loads at LIM 0x08000000.
//            Available immediately at reset; no DRAM controller dependency.
//   * Dram - firmware-idle.bin     (ld/x280-dram.ld), loads at the L2CPU-
//            local DRAM aperture 0x400030000000. Mirrors the pyluwen path
//            in x280/host/boot_idle_x280.py.
// The caller picks which one to boot; the runtime resolves the firmware
// path and sentinel/trap constants from a per-region profile.
enum class BootRegion : std::uint8_t {
    Lim,
    Dram,
};

// Boots the baseline idle firmware on the selected (non-harvested) L2CPUs
// of `device_id`. Resolves runtime/hw/firmware/blackhole/x280/<artifact>
// from the tt-metal install layout (artifact name picked from `region`),
// programs PLL4, releases reset, and polls the firmware's sentinel.
// Throws via TT_THROW on any failure.
//
// `region` chooses which firmware artifact to use and where to load it:
//   * BootRegion::Lim  (default) - firmware-lim-idle.bin into LIM.
//   * BootRegion::Dram           - firmware-idle.bin into L2CPU-local DRAM.
//
// `l2cpu_mask` is a 4-bit mask selecting which physical L2CPUs to boot:
//   bit 0 -> L2CPU 0, bit 1 -> L2CPU 1, bit 2 -> L2CPU 2, bit 3 -> L2CPU 3.
// Defaults to `kAllL2CpuMask` (0xF). Bits above bit 3 are ignored. L2CPUs
// selected by the mask but not enabled on the chip (harvested) are silently
// skipped.
//
// Preconditions:
//   * MetalContext is initialized (Cluster + UMD have already opened all
//     devices and called start_device, so ARC FW has set the chip's power
//     state to BUSY).
//   * `device_id`'s arch is Blackhole (otherwise no-op, returns 0).
//   * No other process / pyluwen instance is driving the chip.
//
// Returns the number of L2CPUs successfully booted. Returns 0 if the chip
// is not Blackhole, the firmware artifact is missing, no L2CPUs are enabled,
// or `l2cpu_mask` selects no enabled L2CPUs.
std::size_t boot_idle(
    tt::ChipId device_id, BootRegion region = BootRegion::Lim, std::uint8_t l2cpu_mask = kAllL2CpuMask);

// Same, but lets the caller override the firmware binary path (useful for
// experiments that ship a non-default x280 firmware). The caller must pass
// the `region` that matches the binary's linker script: the binary itself
// encodes the load address, but the runtime needs to know which load/sentinel
// addresses and sentinel value to drive against it.
std::size_t boot_idle(
    tt::ChipId device_id,
    BootRegion region,
    const std::filesystem::path& firmware_bin,
    std::uint8_t l2cpu_mask = kAllL2CpuMask);

}  // namespace tt::tt_metal::experimental::x280
