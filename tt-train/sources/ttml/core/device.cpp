// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "device.hpp"

#include <core/ttnn_all_includes.hpp>

namespace {
void device_deleter(tt::tt_metal::IDevice* device) {
    assert(device != nullptr);
    tt::tt_metal::CloseDevice(device);
};
}  // namespace

namespace ttml::core {

Device::Device(int device_index) :
    m_device(std::unique_ptr<tt::tt_metal::IDevice, void (*)(tt::tt_metal::IDevice*)>(
        tt::tt_metal::CreateDevice(device_index), &device_deleter)) {
}

[[nodiscard]] tt::tt_metal::IDevice& Device::get_device() {
    assert(m_device);
    return *m_device;
}
}  // namespace ttml::core
