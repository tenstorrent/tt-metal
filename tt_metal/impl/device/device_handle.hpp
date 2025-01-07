// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tt_stl/slotmap.hpp"

namespace tt {

class DevicePool;

namespace tt_metal {
inline namespace v0 {

class IDevice;

}  // namespace v0

namespace v1 {

struct DeviceKey : stl::Key<std ::uint16_t, 12> {
    using Key::Key;
};

}  // namespace v1
}  // namespace tt_metal
}  // namespace tt
