// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// harness/targets/drisc_endpoint.hpp -- DRISC endpoint selection from SoC descriptor.

#ifndef X280_HARNESS_TARGETS_DRISC_ENDPOINT_HPP
#define X280_HARNESS_TARGETS_DRISC_ENDPOINT_HPP

#include <tt-metalium/core_coord.hpp>
#include "llrt/metal_soc_descriptor.hpp"

#include <algorithm>
#include <cstdint>
#include <optional>

namespace x280::harness::targets {

using tt::tt_metal::CoreCoord;

// Blackhole reserves each DRAM view's NOC0 worker endpoint for the syseng
// firmware; get_metal_dram_cores is tt-metal's own enumeration of the DRAM
// cores a kernel may launch on, with that endpoint excluded. Returns the
// lowest such endpoint of DRAM bank 0 as the LOGICAL coordinate
// CreateKernel(DramConfig) takes: logical.y indexes dram_bank_endpoint_coords,
// not the raw subchannel id. std::nullopt when every endpoint of the bank is
// firmware-owned.
inline std::optional<CoreCoord> select_drisc_endpoint(const metal_SocDescriptor& soc_desc) {
    const std::vector<CoreCoord> launchable = soc_desc.get_metal_dram_cores(tt::CoordSystem::TRANSLATED);
    const auto& endpoints = soc_desc.dram_bank_endpoint_coords.at(0);
    for (uint32_t i = 0; i < endpoints.size(); ++i) {
        if (std::find(launchable.begin(), launchable.end(), endpoints[i]) != launchable.end()) {
            return CoreCoord{0, i};
        }
    }
    return std::nullopt;
}

}  // namespace x280::harness::targets

#endif  // X280_HARNESS_TARGETS_DRISC_ENDPOINT_HPP
