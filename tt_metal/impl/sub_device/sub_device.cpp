// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt_stl/fmt.hpp>
#include <tt_stl/assert.hpp>
#include <core_coord.hpp>
#include <sub_device.hpp>
#include <tt_stl/span.hpp>
#include <algorithm>
#include <array>
#include <cstdint>
#include <utility>

#include "hal_types.hpp"
#include "impl/context/metal_context.hpp"
#include "impl/context/metal_env_accessor.hpp"
#include "impl/sub_device/sub_device_impl.hpp"

namespace tt::tt_metal {

// SubDeviceImpl implementation

tt::tt_metal::MetalEnvImpl* get_env_or_default(tt::tt_metal::MetalEnvImpl* env) {
    if (env == nullptr) {
        return &MetalEnvAccessor(MetalContext::instance(DEFAULT_CONTEXT_ID).get_env()).impl();
    }
    return env;
}

SubDeviceImpl::SubDeviceImpl(
    tt::tt_metal::MetalEnvImpl* env, const std::array<CoreRangeSet, NumHalProgrammableCoreTypes>& cores) :
    cores_(cores), env_(get_env_or_default(env)) {
    this->validate();
}

SubDeviceImpl::SubDeviceImpl(tt::tt_metal::MetalEnvImpl* env, tt::stl::Span<const CoreRangeSet> cores) :
    env_(get_env_or_default(env)) {
    TT_FATAL(cores.size() <= this->cores_.size(), "Too many core types for SubDevice");
    std::copy(cores.begin(), cores.end(), this->cores_.begin());
    this->validate();
}

SubDeviceImpl::SubDeviceImpl(
    tt::tt_metal::MetalEnvImpl* env, std::array<CoreRangeSet, NumHalProgrammableCoreTypes>&& cores) :
    cores_(std::move(cores)), env_(get_env_or_default(env)) {
    this->validate();
}

void SubDeviceImpl::validate() const {
    TT_ASSERT(env_ != nullptr, "Missing MetalEnv for this SubDevice");
    auto num_core_types = env_->get_hal().get_programmable_core_type_count();
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

bool SubDeviceImpl::has_core_type(HalProgrammableCoreType core_type) const {
    return !this->cores_[static_cast<uint32_t>(core_type)].empty();
}

uint32_t SubDeviceImpl::num_cores(HalProgrammableCoreType core_type) const {
    return this->cores_[static_cast<uint32_t>(core_type)].num_cores();
}

const std::array<CoreRangeSet, NumHalProgrammableCoreTypes>& SubDeviceImpl::cores() const { return this->cores_; }

const CoreRangeSet& SubDeviceImpl::cores(HalProgrammableCoreType core_type) const {
    return this->cores_[static_cast<uint32_t>(core_type)];
}

// SubDevice implementation

SubDevice::SubDevice(tt::stl::Span<const CoreRangeSet> cores) :
    pimpl_(std::make_unique<SubDeviceImpl>(nullptr, cores)) {}

SubDevice::SubDevice(SubDeviceImpl&& impl) : pimpl_(std::make_unique<SubDeviceImpl>(std::move(impl))) {}

SubDevice::SubDevice(const SubDevice& other) :
    pimpl_(other.pimpl_ ? std::make_unique<SubDeviceImpl>(*other.pimpl_) : nullptr) {}

SubDevice& SubDevice::operator=(const SubDevice& other) {
    if (this != &other) {
        pimpl_ = other.pimpl_ ? std::make_unique<SubDeviceImpl>(*other.pimpl_) : nullptr;
    }
    return *this;
}

SubDevice::SubDevice(SubDevice&& other) noexcept = default;

SubDevice& SubDevice::operator=(SubDevice&& other) noexcept = default;

SubDevice::~SubDevice() = default;

const CoreRangeSet& SubDevice::cores(HalProgrammableCoreType core_type) const { return pimpl_->cores(core_type); }

SubDeviceImpl* SubDevice::impl() { return pimpl_.get(); }

const SubDeviceImpl* SubDevice::impl() const { return pimpl_.get(); }

}  // namespace tt::tt_metal
