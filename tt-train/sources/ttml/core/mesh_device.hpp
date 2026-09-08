// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <core/ttnn_all_includes.hpp>
#include <memory>
#include <tt-metalium/experimental/context/metal_env.hpp>

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
    [[nodiscard]] std::shared_ptr<ttnn::distributed::MeshDevice> get_device_ptr() const;

private:
    // A MetalEnv reads TT_MESH_GRAPH_DESC_PATH and its fabric topology once, when it is
    // constructed, so owning one per open is what lets the mesh graph descriptor selected by
    // enable_fabric() differ between opens in a single process. Declared before m_mesh_device
    // so it outlives it.
    std::unique_ptr<tt::tt_metal::MetalEnv> m_env;
    std::shared_ptr<ttnn::distributed::MeshDevice> m_mesh_device;
};
}  // namespace ttml::core
