// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <mesh_event.hpp>
#include "mesh_event_impl.hpp"

#include "mesh_device.hpp"

namespace tt::tt_metal::distributed {

// MeshEventImpl implementation
MeshEventImpl::MeshEventImpl(
    uint32_t id, MeshDevice* device, uint32_t mesh_cq_id, const MeshCoordinateRange& device_range) :
    id_(id), device_(device), mesh_cq_id_(mesh_cq_id), device_range_(device_range) {}

uint32_t MeshEventImpl::id() const { return id_; }
MeshDevice* MeshEventImpl::device() const { return device_; }
uint32_t MeshEventImpl::mesh_cq_id() const { return mesh_cq_id_; }
const MeshCoordinateRange& MeshEventImpl::device_range() const { return device_range_; }

// MeshEvent implementation
MeshEvent::MeshEvent(std::unique_ptr<MeshEventImpl> impl) : pimpl_(std::move(impl)) {}

MeshEvent::~MeshEvent() = default;

MeshEvent::MeshEvent(const MeshEvent& other) : pimpl_(std::make_unique<MeshEventImpl>(*other.pimpl_)) {}

MeshEvent& MeshEvent::operator=(const MeshEvent& other) {
    if (this != &other) {
        pimpl_ = std::make_unique<MeshEventImpl>(*other.pimpl_);
    }
    return *this;
}

MeshEvent::MeshEvent(MeshEvent&& other) noexcept = default;

MeshEvent& MeshEvent::operator=(MeshEvent&& other) noexcept = default;

uint32_t MeshEvent::id() const { return pimpl_->id(); }
MeshDevice* MeshEvent::device() const { return pimpl_->device(); }
uint32_t MeshEvent::mesh_cq_id() const { return pimpl_->mesh_cq_id(); }
const MeshCoordinateRange& MeshEvent::device_range() const { return pimpl_->device_range(); }

std::ostream& operator<<(std::ostream& os, const MeshEvent& event) {
    os << "MeshEvent(id=" << event.id() << ", device_id=" << event.device()->id()
       << ", mesh_cq_id=" << event.mesh_cq_id() << ", device_range=" << event.device_range() << ")";
    return os;
}

}  // namespace tt::tt_metal::distributed
