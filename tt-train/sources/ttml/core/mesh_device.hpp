// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <core/ttnn_all_includes.hpp>
#include <memory>

namespace ttml::core {
// should I implement pimpl or its fine
class MeshDevice {
public:
    explicit MeshDevice(tt::tt_metal::MeshShape shape);
    MeshDevice(MeshDevice&& device) = default;
    MeshDevice(const MeshDevice&) = delete;

    MeshDevice& operator=(const MeshDevice&) = delete;
    MeshDevice& operator=(MeshDevice&&) = default;
    ~MeshDevice();

    [[nodiscard]] ttnn::MeshDevice& get_device();

private:
    std::shared_ptr<ttnn::MeshDevice> m_mesh_device;
};
}  // namespace ttml::core
