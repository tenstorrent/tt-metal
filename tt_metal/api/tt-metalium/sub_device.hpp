// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cstdint>
#include <memory>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/hal_types.hpp>
#include <tt_stl/span.hpp>

namespace tt::tt_metal {

// Forward declaration
class SubDeviceImpl;

class SubDevice {
public:
    // Public constructor (for external use)
    explicit SubDevice(tt::stl::Span<const CoreRangeSet> cores);

    // Internal constructor (for tt_metal/ internal use)
    SubDevice(SubDeviceImpl&& impl);

    // Special member functions
    SubDevice(const SubDevice& other);
    SubDevice& operator=(const SubDevice& other);
    SubDevice(SubDevice&& other) noexcept;
    SubDevice& operator=(SubDevice&& other) noexcept;
    ~SubDevice();

    // Query methods
    const CoreRangeSet& cores(HalProgrammableCoreType core_type) const;

    SubDeviceImpl* impl();
    const SubDeviceImpl* impl() const;

private:
    std::unique_ptr<SubDeviceImpl> pimpl_;
};

}  // namespace tt::tt_metal
