// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tt_metal/api/types.hpp"
#include "tt_stl/slotmap.hpp"

namespace tt::tt_metal {
inline namespace v0 {

class Device;

}  // namespace v0

namespace v1 {

struct DeviceKey : stl::Key<std ::uint16_t, 12> {
    using Key::Key;

    DeviceKey(DeviceHandle handle) : Key(handle.key) {}

    // TODO remove with v0
    auto operator->() const -> Device *;

    // TODO remove with v0
    operator Device *() const;
};

}  // namespace v1
}  // namespace tt::tt_metal
