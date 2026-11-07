// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <utility>
#include <vector>

#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/mesh_event.hpp>
#include <tt-metalium/sub_device_types.hpp>

namespace tt::tt_metal::distributed {

class MeshDevice;

class MeshEventImpl {
public:
    MeshEventImpl(uint32_t id, MeshDevice* device, uint32_t mesh_cq_id, const MeshCoordinateRange& device_range);

    uint32_t id() const { return id_; }
    MeshDevice* device() const { return device_; }
    uint32_t mesh_cq_id() const { return mesh_cq_id_; }
    const MeshCoordinateRange& device_range() const { return device_range_; }
    // Sub-devices whose prior work this event waited for when recorded. Empty if not recorded by a queue.
    const std::vector<SubDeviceId>& sub_device_ids() const { return sub_device_ids_; }
    void set_sub_device_ids(std::vector<SubDeviceId> sub_device_ids) { sub_device_ids_ = std::move(sub_device_ids); }

private:
    uint32_t id_ = 0;
    MeshDevice* device_ = nullptr;
    uint32_t mesh_cq_id_ = 0;
    MeshCoordinateRange device_range_;
    std::vector<SubDeviceId> sub_device_ids_;
};

}  // namespace tt::tt_metal::distributed
