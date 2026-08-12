// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <mesh_event.hpp>

#include "mesh_device.hpp"
#include "mesh_event_impl.hpp"

namespace tt::tt_metal::distributed {

MeshEventImpl::MeshEventImpl(
    uint32_t id, MeshDevice* device, uint32_t mesh_cq_id, const MeshCoordinateRange& device_range) :
    id_(id), device_(device), mesh_cq_id_(mesh_cq_id), device_range_(device_range) {}

MeshEvent::MeshEvent(std::unique_ptr<MeshEventImpl> impl) : pimpl_(std::move(impl)) {}

MeshEvent::MeshEvent(const MeshEvent& other) :
    pimpl_(other.pimpl_ ? std::make_unique<MeshEventImpl>(*other.pimpl_) : nullptr) {}

MeshEvent& MeshEvent::operator=(const MeshEvent& other) {
    if (this != &other) {
        pimpl_ = other.pimpl_ ? std::make_unique<MeshEventImpl>(*other.pimpl_) : nullptr;
    }
    return *this;
}

MeshEvent::MeshEvent(MeshEvent&& other) noexcept = default;
MeshEvent& MeshEvent::operator=(MeshEvent&& other) noexcept = default;
MeshEvent::~MeshEvent() = default;

MeshDevice* MeshEvent::device() const { return pimpl_->device(); }

MeshEventImpl& MeshEvent::impl() { return *pimpl_; }
const MeshEventImpl& MeshEvent::impl() const { return *pimpl_; }

MeshEvent make_mesh_event(
    uint32_t id, MeshDevice* device, uint32_t mesh_cq_id, const MeshCoordinateRange& device_range) {
    return MeshEvent(std::make_unique<MeshEventImpl>(id, device, mesh_cq_id, device_range));
}

std::ostream& operator<<(std::ostream& os, const MeshEvent& event) {
    os << "MeshEvent(id=" << event.impl().id() << ", device_id=" << event.device()->id()
       << ", mesh_cq_id=" << event.impl().mesh_cq_id() << ", device_range=" << event.impl().device_range() << ")";
    return os;
}

}  // namespace tt::tt_metal::distributed
