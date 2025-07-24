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
    explicit MeshDevice(const tt::tt_metal::distributed::MeshShape& shape, const std::vector<int>& device_ids);
    MeshDevice(MeshDevice&& device) = default;
    MeshDevice(const MeshDevice&) = delete;

    MeshDevice& operator=(const MeshDevice&) = delete;
    MeshDevice& operator=(MeshDevice&&) = default;
    ~MeshDevice();

    [[nodiscard]] ttnn::distributed::MeshDevice& get_device();

private:
    std::shared_ptr<ttnn::distributed::MeshDevice> m_mesh_device;
};
}  // namespace ttml::core
