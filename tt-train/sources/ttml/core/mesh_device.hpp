// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <core/ttnn_all_includes.hpp>
#include <memory>

namespace ttml::core {
// should I implement pimpl or its fine
class MeshDevice {
public:
    // ``num_command_queues`` is forwarded to ``ttnn::distributed::open_mesh_device``. Default 1 to
    // match the historical behavior; set to 2 for workflows that need cross-CQ overlap (e.g.
    // ``ThreadedWeightBridge``: main-thread CQ0 for ``ttnn.copy`` + ``record_event``, background
    // thread CQ1 for ``ttnn.to_torch`` / ``ttnn.copy_host_to_device_tensor``).
    explicit MeshDevice(
        const tt::tt_metal::distributed::MeshShape& shape,
        const std::vector<int>& device_ids,
        std::size_t num_command_queues = 1);
    MeshDevice(MeshDevice&& device) = default;
    MeshDevice(const MeshDevice&) = delete;

    MeshDevice& operator=(const MeshDevice&) = delete;
    MeshDevice& operator=(MeshDevice&&) = default;
    ~MeshDevice();

    [[nodiscard]] ttnn::distributed::MeshDevice& get_device();
    [[nodiscard]] std::shared_ptr<ttnn::distributed::MeshDevice> get_device_ptr() const;

private:
    std::shared_ptr<ttnn::distributed::MeshDevice> m_mesh_device;
};
}  // namespace ttml::core
