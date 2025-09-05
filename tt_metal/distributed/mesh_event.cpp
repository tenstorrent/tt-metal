// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <mesh_event.hpp>

#include "mesh_device.hpp"

namespace tt::tt_metal::distributed {

MeshEvent::MeshEvent(uint32_t id, MeshDevice* device, uint32_t mesh_cq_id, const MeshCoordinateRange& device_range) :
    id_(id), device_(device), mesh_cq_id_(mesh_cq_id), device_range_(device_range) {}

uint32_t MeshEvent::id() const { return id_; }
MeshDevice* MeshEvent::device() const { return device_; }
uint32_t MeshEvent::mesh_cq_id() const { return mesh_cq_id_; }
const MeshCoordinateRange& MeshEvent::device_range() const { return device_range_; }

void MeshEvent::synchronize() {
    if (device_->using_slow_dispatch()) {
        return;
    }
    for (const auto& coord : device_range_) {
        auto physical_device = device_->get_device(coord);
        while (physical_device->sysmem_manager().get_last_completed_event(mesh_cq_id_) < id_);
    }
}

bool MeshEvent::query() {
    const auto& device_range = device_range_;
    auto& sysmem_manager = device_->get_device(*(device_range.begin()))->sysmem_manager();
    bool event_completed = sysmem_manager.get_last_completed_event(mesh_cq_id_) >= id_;
    return event_completed;
}

std::ostream& operator<<(std::ostream& os, const MeshEvent& event) {
    os << "MeshEvent(id=" << event.id() << ", device_id=" << event.device()->id()
       << ", mesh_cq_id=" << event.mesh_cq_id() << ", device_range=" << event.device_range() << ")";
    return os;
}

}  // namespace tt::tt_metal::distributed
