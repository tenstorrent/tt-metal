// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cstdint>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/hal_types.hpp>
#include <tt_stl/span.hpp>
#include "context/metal_env_impl.hpp"

namespace tt::tt_metal {

class SubDeviceImpl {
public:
    // Constructors for internal tt_metal/ use
    explicit SubDeviceImpl(
        tt::tt_metal::MetalEnvImpl* env, const std::array<CoreRangeSet, NumHalProgrammableCoreTypes>& cores);
    explicit SubDeviceImpl(
        tt::tt_metal::MetalEnvImpl* env, std::array<CoreRangeSet, NumHalProgrammableCoreTypes>&& cores);
    explicit SubDeviceImpl(tt::tt_metal::MetalEnvImpl* env, tt::stl::Span<const CoreRangeSet> cores);

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
    tt::tt_metal::MetalEnvImpl* env_;
};

}  // namespace tt::tt_metal
