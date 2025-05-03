// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "device.hpp"

#include <core/ttnn_all_includes.hpp>

namespace ttml::core {

Device::Device(int device_index) : m_device(tt::tt_metal::distributed::MeshDevice::create_unit_mesh(device_index)) {
}

[[nodiscard]] tt::tt_metal::IDevice& Device::get_device() {
    assert(m_device);
    return *m_device;
}
}  // namespace ttml::core
