// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>

#include <core/ttnn_all_includes.hpp>


namespace ttml::core {
// should I implement pimpl or its fine
class Device {
public:
    explicit Device(int device_index);
    Device(IDevice&& device) = default;
    Device(const IDevice&) = delete;

    IDevice& operator=(const IDevice&) = delete;
    IDevice& operator=(IDevice&&) = default;
    ~Device() = default;

    [[nodiscard]] tt::tt_metal::IDevice& get_device();

private:
    std::unique_ptr<tt::tt_metal::Device, void (*)(tt::tt_metal::IDevice*)> m_device;
};
}  // namespace ttml::core
