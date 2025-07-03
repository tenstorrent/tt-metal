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
    Device(Device&& device) = default;
    Device(const Device&) = delete;

    Device& operator=(const Device&) = delete;
    Device& operator=(Device&&) = default;
    ~Device() = default;

    [[nodiscard]] tt::tt_metal::IDevice& get_device();

private:
    std::shared_ptr<tt::tt_metal::distributed::MeshDevice> m_device;
};
}  // namespace ttml::core
