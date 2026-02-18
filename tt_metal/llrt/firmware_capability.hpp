// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include <umd/device/types/arch.hpp>
#include <umd/device/utils/semver.hpp>

namespace tt::tt_metal {

struct FirmwareVersions {
    std::optional<tt::umd::semver_t> eth_fw;
};

struct FirmwareCapabilityRequest {
    bool enable_2_erisc_mode = false;
};

struct FirmwareCapabilityResult {
    bool enable_2_erisc_mode = false;
};

/**
 * Check if the firmware versions are compatible with the requested capabilities.
 *
 * @param arch Platform architecture (from cluster/rtoptions, not HAL).
 * @param fw_versions Available firmware versions (e.g. from cluster driver).
 * @param requested Requested feature flags and integer parameters.
 * @param resolved_out Output: supported capabilities which may be less than what was requested
 * @return true if resolved equals requested; false if any reduction was applied.
 */
bool check_firmware_capabilities(
    tt::ARCH arch,
    const FirmwareVersions& fw_versions,
    const FirmwareCapabilityRequest& requested,
    FirmwareCapabilityResult& resolved_out);

}  // namespace tt::tt_metal
