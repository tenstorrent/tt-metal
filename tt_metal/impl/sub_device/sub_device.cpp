// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <array>
#include <cstdint>

#include <assert.hpp>
#include <core_coord.hpp>
#include <sub_device.hpp>
#include <hal.hpp>
#include <span.hpp>

namespace tt::tt_metal {

SubDevice::SubDevice(const std::array<CoreRangeSet, NumHalProgrammableCoreTypes>& cores) : cores_(cores) {
    this->validate();
}

SubDevice::SubDevice(tt::stl::Span<const CoreRangeSet> cores) {
    TT_FATAL(cores.size() <= this->cores_.size(), "Too many core types for SubDevice");
    std::copy(cores.begin(), cores.end(), this->cores_.begin());
    this->validate();
}

SubDevice::SubDevice(std::array<CoreRangeSet, NumHalProgrammableCoreTypes>&& cores) : cores_(std::move(cores)) {
    this->validate();
}

void SubDevice::validate() const {
    auto num_core_types = hal.get_programmable_core_type_count();
    for (uint32_t i = num_core_types; i < NumHalProgrammableCoreTypes; ++i) {
        TT_FATAL(
            this->cores_[i].empty(),
            "CoreType {} is not allowed in SubDevice",
            static_cast<HalProgrammableCoreType>(i));
    }
    TT_FATAL(
        this->cores_[static_cast<uint32_t>(HalProgrammableCoreType::IDLE_ETH)].empty(),
        "CoreType IDLE_ETH is not allowed in SubDevice");
}

bool SubDevice::has_core_type(HalProgrammableCoreType core_type) const {
    return !this->cores_[static_cast<uint32_t>(core_type)].empty();
}

uint32_t SubDevice::num_cores(HalProgrammableCoreType core_type) const {
    return this->cores_[static_cast<uint32_t>(core_type)].num_cores();
}

const std::array<CoreRangeSet, NumHalProgrammableCoreTypes>& SubDevice::cores() const { return this->cores_; }

const CoreRangeSet& SubDevice::cores(HalProgrammableCoreType core_type) const {
    return this->cores_[static_cast<uint32_t>(core_type)];
}

}  // namespace tt::tt_metal
