// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cstdint>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/hal_types.hpp>
#include <tt_stl/span.hpp>

namespace tt::tt_metal {

// Validation that depends on the target device (e.g. which core types its HAL supports) happens when the
// SubDevice is applied to a device, in SubDeviceManager::validate_sub_devices().
class SubDeviceImpl {
public:
    // Constructors for internal tt_metal/ use
    explicit SubDeviceImpl(const std::array<CoreRangeSet, NumHalProgrammableCoreTypes>& cores);
    explicit SubDeviceImpl(std::array<CoreRangeSet, NumHalProgrammableCoreTypes>&& cores);
    explicit SubDeviceImpl(ttsl::Span<const CoreRangeSet> cores);

    // Copy/move semantics
    SubDeviceImpl(const SubDeviceImpl&) = default;
    SubDeviceImpl& operator=(const SubDeviceImpl&) = default;
    SubDeviceImpl(SubDeviceImpl&&) noexcept = default;
    SubDeviceImpl& operator=(SubDeviceImpl&&) noexcept = default;

    // Query methods
    bool has_core_type(HalProgrammableCoreType core_type) const;
    uint32_t num_cores(HalProgrammableCoreType core_type) const;
    const std::array<CoreRangeSet, NumHalProgrammableCoreTypes>& cores() const;
    const CoreRangeSet& cores(HalProgrammableCoreType core_type) const;

private:
    void validate() const;

    std::array<CoreRangeSet, NumHalProgrammableCoreTypes> cores_;
};

}  // namespace tt::tt_metal
