// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
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
    resolved = {requested.enable_2_erisc_mode};

    switch (arch) {
        case tt::ARCH::BLACKHOLE: {
            constexpr tt::umd::semver_t k_min_2_erisc_version(1, 7, 0);
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
    return resolved_out.enable_2_erisc_mode == requested.enable_2_erisc_mode;
}

}  // namespace tt::tt_metal
