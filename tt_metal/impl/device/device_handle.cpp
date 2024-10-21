// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/impl/device/device_handle.hpp"

#include "tt_metal/impl/device/device_pool.hpp"

namespace tt::tt_metal {

auto v1::DeviceHandle::operator->() const -> Device * { return static_cast<Device *>(*this); }

v1::DeviceHandle::operator Device *() const {
    TT_FATAL(this->key.version() & 1, "Invalid DeviceHandle; Expected valid key version");
    const auto loc = this->key.index();
    const auto &devices = DevicePool::instance().devices;
    const auto size = devices.size();
    TT_FATAL(loc < size, "Invalid DeviceHandle {}; Expected index less than {}", loc, size);
    return devices[loc].get();
}

}  // namespace tt::tt_metal
