// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include <tt-metalium/mesh_coord.hpp>

namespace tt::tt_metal::distributed {

class MeshDevice;

class MeshEventImpl {
public:
    MeshEventImpl(uint32_t id, MeshDevice* device, uint32_t mesh_cq_id, const MeshCoordinateRange& device_range);

    MeshEventImpl(const MeshEventImpl& other) = default;

    MeshEventImpl& operator=(const MeshEventImpl& other) = default;

    MeshEventImpl(MeshEventImpl&& other) noexcept = default;
    MeshEventImpl& operator=(MeshEventImpl&& other) noexcept = default;

    // Accessors
    uint32_t id() const;
    MeshDevice* device() const;
    uint32_t mesh_cq_id() const;
    const MeshCoordinateRange& device_range() const;

private:
    uint32_t id_ = 0;
    MeshDevice* device_ = nullptr;
    uint32_t mesh_cq_id_ = 0;
    MeshCoordinateRange device_range_;
};

}  // namespace tt::tt_metal::distributed