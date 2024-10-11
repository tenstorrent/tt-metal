// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/impl/device/device_key.hpp"

#include "tt_metal/impl/device/device_pool.hpp"

namespace tt::tt_metal {

auto v1::DeviceKey::operator->() const -> Device * { return static_cast<Device *>(*this); }

v1::DeviceKey::operator Device *() const {
    const auto loc = this->index();
    const auto &devices = DevicePool::instance().devices;
    const auto size = devices.size();
    TT_FATAL(loc < size, "Invalid Device key {}; Expected index less than {}", loc, size);
    return devices[loc].get();
}

}  // namespace tt::tt_metal
