// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tt_stl/slotmap.hpp"

namespace tt {

class DevicePool;

namespace tt_metal {
inline namespace v0 {

class Device;

}  // namespace v0

namespace v1 {

struct DeviceKey : stl::Key<std ::uint16_t, 12> {
    using Key::Key;
};

class DeviceHandle {
    friend DevicePool;

    DeviceKey key;

    DeviceHandle(DeviceKey key) : key(key) {}

   public:
    DeviceHandle() = default;

    // TODO remove with v0
    auto operator->() const -> Device *;

    // TODO remove with v0
    operator Device *() const;
};

}  // namespace v1
}  // namespace tt_metal
}  // namespace tt
