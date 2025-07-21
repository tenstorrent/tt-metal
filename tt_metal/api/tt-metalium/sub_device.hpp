// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cstdint>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/hal_types.hpp>
#include <tt_stl/span.hpp>

namespace tt::tt_metal {

class SubDevice {
public:
    SubDevice(const std::array<CoreRangeSet, NumHalProgrammableCoreTypes>& cores);
    SubDevice(tt::stl::Span<const CoreRangeSet> cores);
    SubDevice(std::array<CoreRangeSet, NumHalProgrammableCoreTypes>&& cores);

    SubDevice(const SubDevice& sub_device) = default;
    SubDevice& operator=(const SubDevice& sub_device) = default;

    SubDevice(SubDevice&& sub_device) noexcept = default;
    SubDevice& operator=(SubDevice&& sub_device) noexcept = default;

    bool has_core_type(HalProgrammableCoreType core_type) const;
    uint32_t num_cores(HalProgrammableCoreType core_type) const;
    const std::array<CoreRangeSet, NumHalProgrammableCoreTypes>& cores() const;
    const CoreRangeSet& cores(HalProgrammableCoreType core_type) const;

private:
    void validate() const;

    // These are logical coords from the original device grid
    // There is no remapping of logical coords
    std::array<CoreRangeSet, NumHalProgrammableCoreTypes> cores_;
};

}  // namespace tt::tt_metal
