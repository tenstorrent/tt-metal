// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "firmware_capability.hpp"

#include <tt-logger/tt-logger.hpp>
#include "tt_stl/assert.hpp"

namespace tt::tt_metal {

namespace {

void compare_requested_and_actual_capabilities(
    tt::ARCH arch,
    const FirmwareVersions& fw_versions,
    const FirmwareCapabilityRequest& requested,
    FirmwareCapabilityResult& resolved) {
    resolved.enable_2_erisc_mode = requested.enable_2_erisc_mode;
    // DRAM programmable cores are a Blackhole-only feature; off unless the BLACKHOLE case below
    // confirms the firmware supports them.
    resolved.dram_programmable_cores = false;
    // Only Blackhole's HAL acts on this, but it feeds the build key, so clearing it off-Blackhole would
    // desync from the offline-compiled build key and miss every precompiled binary.
    resolved.eth_ptp_trace = requested.eth_ptp_trace;

    switch (arch) {
        case tt::ARCH::BLACKHOLE: {
            constexpr tt::umd::SemVer k_min_2_erisc_version(1, 7, 0);
            // If ethernet firmware cannot be queried assume it's ok
            // This occurs on the simulator
            const bool eth_fw_ok = !fw_versions.eth_fw || (fw_versions.eth_fw.value() >= k_min_2_erisc_version);
            if (requested.enable_2_erisc_mode && !eth_fw_ok) {
                log_warning(
                    tt::LogLLRuntime,
                    "Blackhole multi erisc mode requires ethernet firmware version {} or higher, but detected version "
                    "{}. Automatically falling back to single erisc mode for compatibility.",
                    k_min_2_erisc_version.to_string(),
                    fw_versions.eth_fw ? fw_versions.eth_fw->to_string() : "unknown");
                resolved.enable_2_erisc_mode = false;
            }

            // Earlier firmware places the syseng firmware on a DRAM core the application wants to use,
            // so enabling DRAM programmable cores there would collide with syseng (see #45751). Unlike
            // eth_fw above, a missing bundle version is treated as NOT ok -- enabling on unknown firmware
            // is the risky case we are guarding against.
            constexpr tt::umd::FirmwareBundleVersion k_min_dram_cores_version(19, 12, 0, 0);
            const bool bundle_fw_ok =
                fw_versions.firmware_bundle && (fw_versions.firmware_bundle.value() >= k_min_dram_cores_version);
            resolved.dram_programmable_cores = requested.dram_programmable_cores && bundle_fw_ok;

            // debug_buf_t gained its scratchpad in 1.8.0; stamping into older firmware would land on
            // live syseng state. Unlike the bundle version, an unknown eth version is ok: it is only
            // unset when discovery inspected no ethernet core, and a core metal never sees is a core
            // we never stamp.
            constexpr tt::umd::SemVer k_min_eth_ptp_trace_version(1, 8, 0);
            const bool ptp_trace_fw_ok =
                !fw_versions.eth_fw || (fw_versions.eth_fw.value() >= k_min_eth_ptp_trace_version);
            if (requested.eth_ptp_trace && !ptp_trace_fw_ok) {
                log_warning(
                    tt::LogLLRuntime,
                    "Blackhole PTP trace stamping requires ethernet firmware version {} or higher, but detected "
                    "version {}. Disabling PTP trace stamping.",
                    k_min_eth_ptp_trace_version.to_string(),
                    fw_versions.eth_fw ? fw_versions.eth_fw->to_string() : "unknown");
            }
            resolved.eth_ptp_trace = requested.eth_ptp_trace && ptp_trace_fw_ok;
            break;
        }
        case tt::ARCH::WORMHOLE_B0:
        case tt::ARCH::QUASAR: break;
        case tt::ARCH::Invalid:
        default: TT_THROW("Unsupported arch {} for firmware capability check", arch);
    }
}

}  // namespace

bool check_firmware_capabilities(
    tt::ARCH arch,
    const FirmwareVersions& fw_versions,
    const FirmwareCapabilityRequest& requested,
    FirmwareCapabilityResult& resolved_out) {
    compare_requested_and_actual_capabilities(arch, fw_versions, requested, resolved_out);
    return resolved_out.enable_2_erisc_mode == requested.enable_2_erisc_mode &&
           resolved_out.dram_programmable_cores == requested.dram_programmable_cores &&
           resolved_out.eth_ptp_trace == requested.eth_ptp_trace;
}

}  // namespace tt::tt_metal
